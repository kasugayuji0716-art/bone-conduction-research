"""
KD Teacher 用: Whisper small を気導マイク音声でファインチューニング

KD Adapter（script 34）のTeacherとして使用するため、
気導マイク（acoustic）音声でWhisperをFTする。

Pretrained Whisper を Teacher にした問題点:
  → TAPSドメインを知らない → 引き寄せる方向が不明確

本スクリプトで学習したモデルを Teacher にすることで:
  → TAPSの韓国語・話者特性を気導マイク側から学習済み
  → より適切な「目標表現」を提供できる

入力:  data/raw/taps/acoustic/{train,dev}/
ラベル: data/raw/taps/metadata_{train,dev}.csv
出力:  checkpoints/whisper_acoustic_finetuned/

実行:
  python3 scripts/35_finetune_whisper_acoustic.py
  python3 scripts/35_finetune_whisper_acoustic.py --epochs 20 --batch_size 16
"""

import csv
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

BASE_DIR  = Path(__file__).parent.parent
TAPS_DIR  = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT_DIR  = BASE_DIR / 'checkpoints' / 'whisper_acoustic_finetuned'
MODEL_ID  = 'openai/whisper-small'
TARGET_SR = 16000


class AcousticMicDataset(Dataset):
    def __init__(self, split: str, processor: WhisperProcessor):
        self.processor = processor
        self.samples   = []

        meta_path = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                path = TAPS_DIR / 'acoustic' / split / \
                       f"{row['speaker_id']}_{row['sentence_id']}.wav"
                if path.exists():
                    self.samples.append({
                        'audio_path': str(path),
                        'text':       row['text'],
                    })

        print(f'  [{split}] {len(self.samples)} サンプル（気導マイク）')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, _ = sf.read(s['audio_path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        input_features = self.processor.feature_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors='pt'
        ).input_features[0]

        labels = self.processor.tokenizer(
            s['text'], return_tensors='pt'
        ).input_ids[0]

        return {'input_features': input_features, 'labels': labels}


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f['input_features'] for f in features])

        label_ids = [f['labels'] for f in features]
        max_len   = max(l.shape[0] for l in label_ids)
        padded    = []
        for l in label_ids:
            pad = torch.full((max_len - l.shape[0],), -100, dtype=torch.long)
            padded.append(torch.cat([l, pad]))

        return {'input_features': input_features, 'labels': torch.stack(padded)}


def make_compute_metrics(processor):
    cer_metric = evaluate.load('cer')

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {'cer': cer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int,   default=500)
    parser.add_argument('--patience',     type=int,   default=3)
    args = parser.parse_args()

    print(f'モデル  : {MODEL_ID}')
    print(f'入力    : 気導マイク（acoustic）')
    print(f'出力    : {CKPT_DIR}')
    print(f'用途    : KD Adapter の Teacher モデル')
    print(f'epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}\n')

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language='korean', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    train_ds = AcousticMicDataset('train', processor)
    dev_ds   = AcousticMicDataset('dev',   processor)
    collator = WhisperDataCollator(processor=processor)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='cer',
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=50,
        report_to='none',
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        processing_class=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(CKPT_DIR))
    processor.save_pretrained(str(CKPT_DIR))

    print(f'\n完了。Teacher モデル保存先: {CKPT_DIR}')
    print(f'\n次のステップ:')
    print(f'  python3 scripts/36_finetune_whisper_kd_v2.py  # 改良KDで再学習')


if __name__ == '__main__':
    main()
