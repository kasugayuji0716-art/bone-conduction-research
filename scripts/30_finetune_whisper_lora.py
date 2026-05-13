"""
フェーズ3: Whisper Encoder+Decoder LoRA ファインチューニング

Encoder-only Adapter（29番）との比較用。
LoRA-Whisper (Interspeech 2024, arXiv:2406.06619) に倣い、
Encoder・Decoder 両方の Attention 層に LoRA を挿入して学習する。

Adapter との違い:
  Adapter → Encoder のみ変更、Decoder は多言語のまま → cross-lingual 転移しやすい（仮説）
  LoRA    → Encoder + Decoder 両方変更 → Decoder が韓国語に特化 → 転移しにくい（仮説）

学習: 韓国語 TAPS データ（Adapter と同条件）
評価: 32_evaluate_crosslingual.py で Korean + French VibraVox を評価

出力:
  checkpoints/whisper_encoder_decoder_lora/  -- LoRA adapter weights (peft 形式)

実行:
  pip install peft
  python3 scripts/30_finetune_whisper_lora.py
  python3 scripts/30_finetune_whisper_lora.py --r 16 --epochs 20 --batch_size 16
"""

import csv
import json
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
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError('peft が必要です: pip install peft')
import evaluate

BASE_DIR  = Path(__file__).parent.parent
TAPS_DIR  = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT_DIR  = BASE_DIR / 'checkpoints' / 'whisper_encoder_decoder_lora'
MODEL_ID  = 'openai/whisper-small'
TARGET_SR = 16000


# ── Dataset（22番と同一）────────────────────────────────────────
class ThroatMicDataset(Dataset):
    def __init__(self, split: str, processor: WhisperProcessor):
        self.processor = processor
        self.samples   = []

        meta_path = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                path = TAPS_DIR / 'throat' / split / \
                       f"{row['speaker_id']}_{row['sentence_id']}.wav"
                if path.exists():
                    self.samples.append({'audio_path': str(path),
                                         'text': row['text']})

        print(f'  [{split}] {len(self.samples)} サンプル')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, sr = sf.read(s['audio_path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        input_features = self.processor.feature_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors='pt'
        ).input_features[0]

        labels = self.processor.tokenizer(
            s['text'], return_tensors='pt'
        ).input_ids[0]

        return {'input_features': input_features, 'labels': labels}


# ── Data Collator ──────────────────────────────────────────────
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

        return {'input_features': input_features,
                'labels': torch.stack(padded)}


# ── CER 計算 ───────────────────────────────────────────────────
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


# ── メイン ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r',              type=int,   default=16,
                        help='LoRA rank（LoRA-Whisper 2024 では 32）')
    parser.add_argument('--lora_alpha',     type=int,   default=32)
    parser.add_argument('--target_modules', type=str,   default='q_proj,v_proj',
                        help='LoRA を適用するモジュール名（カンマ区切り）')
    parser.add_argument('--epochs',         type=int,   default=20)
    parser.add_argument('--batch_size',     type=int,   default=16)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--warmup_steps',   type=int,   default=200)
    parser.add_argument('--patience',       type=int,   default=3)
    args = parser.parse_args()

    target_modules = [m.strip() for m in args.target_modules.split(',')]

    print(f'モデル         : {MODEL_ID}')
    print(f'LoRA rank      : {args.r},  alpha={args.lora_alpha}')
    print(f'target_modules : {target_modules}')
    print(f'出力           : {CKPT_DIR}')
    print(f'epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}\n')

    # モデル・プロセッサ準備
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language='korean', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    # LoRA 設定（Encoder + Decoder 両方に適用される）
    # task_type を指定しないことで peft 0.19+ / transformers 5.x の互換性問題を回避
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # データセット
    train_ds = ThroatMicDataset('train', processor)
    dev_ds   = ThroatMicDataset('dev',   processor)
    collator = WhisperDataCollator(processor=processor)

    # 学習設定
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=True,
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

    # LoRA adapter 重みのみ保存（peft 形式）
    model.save_pretrained(str(CKPT_DIR))
    processor.save_pretrained(str(CKPT_DIR / 'processor'))

    # 設定を別途保存（評価スクリプト参照用）
    config = {
        'r': args.r,
        'lora_alpha': args.lora_alpha,
        'target_modules': target_modules,
        'base_model': MODEL_ID,
    }
    with open(CKPT_DIR / 'lora_config_custom.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\n完了。')
    print(f'  LoRA adapter : {CKPT_DIR}')
    print(f'  Processor    : {CKPT_DIR / "processor"}')


if __name__ == '__main__':
    main()
