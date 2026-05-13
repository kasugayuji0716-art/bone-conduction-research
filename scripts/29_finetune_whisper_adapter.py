"""
フェーズ3: Whisper Encoder-only Adapter ファインチューニング

Whisper本体の重みを完全凍結し、Encoderの各Transformer層の後に
小さなAdapterモジュールのみを挿入・学習する。

Adapterの構造:
  x → LayerNorm → Linear(512→r) → GELU → Linear(r→512) → + x（残差接続）
  初期化: 出力層をゼロ初期化 → 学習開始時は恒等変換

学習: 韓国語TAPSデータ（Full FTと同条件）
評価: 韓国語テストセット（Full FTとの比較）
     → 後に日本語喉マイク音声での転移性評価（30_evaluate_japanese.py）

出力:
  checkpoints/whisper_encoder_adapter/adapter_weights.pt  -- Adapterのみ
  checkpoints/whisper_encoder_adapter/adapter_config.json -- 設定
  checkpoints/whisper_encoder_adapter/processor/          -- Processor

実行:
  python3 scripts/29_finetune_whisper_adapter.py
  python3 scripts/29_finetune_whisper_adapter.py --r 64 --epochs 20 --batch_size 16
"""

import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
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
CKPT_DIR  = BASE_DIR / 'checkpoints' / 'whisper_encoder_adapter'
MODEL_ID  = 'openai/whisper-small'
TARGET_SR = 16000
D_MODEL   = 512  # Whisper small の隠れ層次元


# ── Adapter モジュール ────────────────────────────────────────
class Adapter(nn.Module):
    """
    ボトルネック型Adapter: 512 → r → 512（残差接続付き）
    出力層をゼロ初期化することで学習開始時は恒等変換
    """
    def __init__(self, d_model: int = D_MODEL, r: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model)
        # ゼロ初期化 → 学習開始時は恒等変換（Whisperの動作を変えない）
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


# ── Adapter挿入済みEncoder層 ──────────────────────────────────
class WhisperEncoderLayerWithAdapter(nn.Module):
    """元のEncoder層をラップしてAdapterを後段に付加する"""

    def __init__(self, original_layer: nn.Module, adapter: Adapter):
        super().__init__()
        self.layer   = original_layer
        self.adapter = adapter

    def forward(self, hidden_states, attention_mask=None,
                layer_head_mask=None, output_attentions=False):
        outputs      = self.layer(hidden_states, attention_mask,
                                  layer_head_mask, output_attentions)
        hidden_states = self.adapter(outputs[0])
        return (hidden_states,) + outputs[1:]


# ── Adapterを挿入してWhisperを準備 ───────────────────────────
def build_adapter_model(model: WhisperForConditionalGeneration,
                        r: int) -> WhisperForConditionalGeneration:
    # 1. Encoderの各層にAdapterを挿入
    layers = model.model.encoder.layers
    for i in range(len(layers)):
        adapter = Adapter(d_model=D_MODEL, r=r)
        layers[i] = WhisperEncoderLayerWithAdapter(layers[i], adapter)

    # 2. Adapter以外の全パラメータを凍結
    for name, param in model.named_parameters():
        param.requires_grad = ('adapter' in name)

    # 3. 学習可能パラメータ数を表示
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'総パラメータ数    : {total:,}')
    print(f'学習可能パラメータ: {train:,} ({train / total * 100:.2f}%)')

    return model


# ── Dataset ──────────────────────────────────────────────────
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


# ── Data Collator ─────────────────────────────────────────────
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


# ── CER 計算 ──────────────────────────────────────────────────
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
    parser.add_argument('--r',            type=int,   default=64,   help='Adapterボトルネック次元')
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-3, help='Adapterの学習率（Full FTより大きくてOK）')
    parser.add_argument('--warmup_steps', type=int,   default=100)
    parser.add_argument('--patience',     type=int,   default=3)
    args = parser.parse_args()

    print(f'モデル      : {MODEL_ID}')
    print(f'Adapter r   : {args.r}')
    print(f'出力        : {CKPT_DIR}')
    print(f'epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}\n')

    # モデル・プロセッサ準備
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language='korean', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    # Adapter挿入・凍結
    print('Adapterを挿入中...')
    model = build_adapter_model(model, r=args.r)

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

    # Adapterの重みのみ保存（軽量）
    adapter_state = {
        name: param
        for name, param in model.named_parameters()
        if 'adapter' in name
    }
    torch.save(adapter_state, CKPT_DIR / 'adapter_weights.pt')

    # 設定を保存（推論時にAdapterを再構築するために必要）
    config = {'r': args.r, 'd_model': D_MODEL,
              'num_encoder_layers': len(model.model.encoder.layers)}
    with open(CKPT_DIR / 'adapter_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Processorも保存
    processor.save_pretrained(str(CKPT_DIR / 'processor'))

    print(f'\n完了。')
    print(f'  Adapter重み : {CKPT_DIR / "adapter_weights.pt"}')
    print(f'  設定        : {CKPT_DIR / "adapter_config.json"}')


if __name__ == '__main__':
    main()
