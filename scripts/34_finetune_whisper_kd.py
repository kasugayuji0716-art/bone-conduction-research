"""
フェーズ3: Whisper Encoder Adapter + Knowledge Distillation

Teacher: Pretrained Whisper-small（気導マイク入力、完全凍結）
Student: Whisper-small + Encoder Adapter（Adapterのみ学習、喉マイク入力）

Loss = α × CE損失（デコーダ出力 vs テキストラベル）
     + β × KD損失（Student Encoder出力 vs Teacher Encoder出力の距離）

BAF-Netとの差異:
  BAF-Net → 推論時に気導マイクが必須（デュアルマイク）
  本手法  → 学習時のみペアデータを使用、推論時は喉マイク単体でOK

出力:
  checkpoints/whisper_kd_adapter/adapter_weights_kd.pt
  checkpoints/whisper_kd_adapter/adapter_config.json
  checkpoints/whisper_kd_adapter/processor/

実行:
  python3 scripts/34_finetune_whisper_kd.py
  python3 scripts/34_finetune_whisper_kd.py --r 64 --alpha 1.0 --beta 0.5 --kd_loss cosine
"""

import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import cer as compute_cer

BASE_DIR  = Path(__file__).parent.parent
TAPS_DIR  = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT_DIR  = BASE_DIR / 'checkpoints' / 'whisper_kd_adapter'
MODEL_ID  = 'openai/whisper-small'
TARGET_SR = 16000
D_MODEL   = 768  # Whisper small


# ── Adapter（29番と同一定義）──────────────────────────────────
class Adapter(nn.Module):
    def __init__(self, d_model: int = D_MODEL, r: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class WhisperEncoderLayerWithAdapter(nn.Module):
    def __init__(self, original_layer: nn.Module, adapter: Adapter):
        super().__init__()
        self.layer   = original_layer
        self.adapter = adapter

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = self.layer(hidden_states, attention_mask, **kwargs)
        return self.adapter(hidden_states)


def build_student(r: int) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    layers = model.model.encoder.layers
    for i in range(len(layers)):
        adapter  = Adapter(d_model=D_MODEL, r=r)
        layers[i] = WhisperEncoderLayerWithAdapter(layers[i], adapter)

    # Adapter以外を凍結
    for name, param in model.named_parameters():
        param.requires_grad = ('adapter' in name)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Student 総パラメータ    : {total:,}')
    print(f'Student 学習可能パラメータ: {train:,} ({train/total*100:.2f}%)')
    return model


# ── ペアデータセット ──────────────────────────────────────────
class TAPSPairedDataset(Dataset):
    """
    同一発話の (喉マイク, 気導マイク, テキスト) ペアを返す
    """
    def __init__(self, split: str, processor: WhisperProcessor):
        self.processor = processor
        self.samples   = []

        meta_path = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                spk = row['speaker_id']
                sid = row['sentence_id']
                throat  = TAPS_DIR / 'throat'   / split / f'{spk}_{sid}.wav'
                acoustic = TAPS_DIR / 'acoustic' / split / f'{spk}_{sid}.wav'
                if throat.exists() and acoustic.exists():
                    self.samples.append({
                        'throat_path':   str(throat),
                        'acoustic_path': str(acoustic),
                        'text':          row['text'],
                    })

        print(f'  [{split}] ペアサンプル数: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        def load_wav(path):
            wav, _ = sf.read(path, dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav

        throat_wav   = load_wav(s['throat_path'])
        acoustic_wav = load_wav(s['acoustic_path'])

        throat_feat = self.processor.feature_extractor(
            throat_wav, sampling_rate=TARGET_SR, return_tensors='pt'
        ).input_features[0]

        acoustic_feat = self.processor.feature_extractor(
            acoustic_wav, sampling_rate=TARGET_SR, return_tensors='pt'
        ).input_features[0]

        labels = self.processor.tokenizer(
            s['text'], return_tensors='pt'
        ).input_ids[0]

        return {
            'throat_features':   throat_feat,
            'acoustic_features': acoustic_feat,
            'labels':            labels,
            'text':              s['text'],
        }


# ── Data Collator ─────────────────────────────────────────────
@dataclass
class PairedDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        throat_feats   = torch.stack([f['throat_features']   for f in features])
        acoustic_feats = torch.stack([f['acoustic_features'] for f in features])

        label_ids = [f['labels'] for f in features]
        max_len   = max(l.shape[0] for l in label_ids)
        padded    = []
        for l in label_ids:
            pad = torch.full((max_len - l.shape[0],), -100, dtype=torch.long)
            padded.append(torch.cat([l, pad]))

        texts = [f['text'] for f in features]

        return {
            'throat_features':   throat_feats,
            'acoustic_features': acoustic_feats,
            'labels':            torch.stack(padded),
            'texts':             texts,
        }


# ── KD損失 ───────────────────────────────────────────────────
def kd_loss_fn(student_enc: torch.Tensor,
               teacher_enc: torch.Tensor,
               loss_type: str = 'mse') -> torch.Tensor:
    """
    student_enc, teacher_enc: [batch, time, d_model]
    time次元で平均を取ってから距離を計算する
    """
    s = student_enc.mean(dim=1)  # [batch, d_model]
    t = teacher_enc.mean(dim=1)  # [batch, d_model]

    if loss_type == 'cosine':
        return (1 - F.cosine_similarity(s, t, dim=-1)).mean()
    else:  # mse
        return F.mse_loss(s, t)


# ── Encoder出力を取得するフック ──────────────────────────────
class EncoderOutputHook:
    def __init__(self):
        self.output = None

    def hook_fn(self, module, input, output):
        if hasattr(output, 'last_hidden_state'):
            self.output = output.last_hidden_state
        elif isinstance(output, tuple):
            self.output = output[0]
        else:
            self.output = output

    def register(self, model: WhisperForConditionalGeneration):
        # Encoder最終層の後にフックを登録
        handle = model.model.encoder.register_forward_hook(self.hook_fn)
        return handle


# ── 評価（CER計算）────────────────────────────────────────────
def evaluate_cer(model, processor, dataloader, device, max_batches=None):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            feats = batch['throat_features'].to(device)
            ids   = model.generate(feats, language='ko', task='transcribe',
                                   max_new_tokens=225)
            preds = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            hyps.extend(preds)
            refs.extend(batch['texts'])
    if not refs:
        return 1.0
    return compute_cer(refs, hyps)


# ── メイン ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r',            type=int,   default=64)
    parser.add_argument('--alpha',        type=float, default=1.0,
                        help='CE損失の重み')
    parser.add_argument('--beta',         type=float, default=0.5,
                        help='KD損失の重み')
    parser.add_argument('--kd_loss',      type=str,   default='cosine',
                        choices=['mse', 'cosine'])
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int,   default=100)
    parser.add_argument('--patience',     type=int,   default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'デバイス: {device}')
    print(f'設定: r={args.r}, α={args.alpha}, β={args.beta}, '
          f'kd_loss={args.kd_loss}, lr={args.lr}')

    # ── モデル準備 ────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language='korean', task='transcribe')

    print('\nTeacher（Pretrained Whisper, 凍結）を準備中...')
    teacher = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    print('Student（Whisper + Adapter）を準備中...')
    student = build_student(r=args.r)
    student.config.forced_decoder_ids = None
    student.config.suppress_tokens    = []
    student.to(device)

    # Encoder出力フック
    teacher_hook = EncoderOutputHook()
    student_hook = EncoderOutputHook()
    teacher_handle = teacher_hook.register(teacher)
    student_handle = student_hook.register(student)

    # ── データ準備 ────────────────────────────────────────
    print('\nデータセット準備中...')
    train_ds = TAPSPairedDataset('train', processor)
    dev_ds   = TAPSPairedDataset('dev',   processor)
    collator = PairedDataCollator(processor=processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collator,
                              num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=8,
                              shuffle=False, collate_fn=collator,
                              num_workers=4, pin_memory=True)

    # ── オプティマイザ・スケジューラ ──────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 学習ループ ────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_cer     = float('inf')
    patience_cnt = 0
    global_step  = 0

    print(f'\n学習開始（最大{args.epochs}エポック、patience={args.patience}）\n')

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_ce_loss  = 0.0
        epoch_kd_loss  = 0.0
        epoch_total    = 0.0

        for step, batch in enumerate(train_loader):
            throat_feats   = batch['throat_features'].to(device)
            acoustic_feats = batch['acoustic_features'].to(device)
            labels         = batch['labels'].to(device)

            # ── Teacher forward（Encoderのみ、no_grad）────
            with torch.no_grad():
                teacher(input_features=acoustic_feats,
                        decoder_input_ids=torch.zeros(
                            acoustic_feats.size(0), 1, dtype=torch.long
                        ).to(device))
            teacher_enc = teacher_hook.output.detach()  # [B, T, D]

            # ── Student forward（CE損失 + Encoder出力取得）
            outputs = student(input_features=throat_feats, labels=labels)
            ce_loss = outputs.loss

            student_enc = student_hook.output  # [B, T, D]

            # ── KD損失 ──────────────────────────────────
            kd_loss = kd_loss_fn(student_enc, teacher_enc, args.kd_loss)

            total_loss = args.alpha * ce_loss + args.beta * kd_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            epoch_ce_loss += ce_loss.item()
            epoch_kd_loss += kd_loss.item()
            epoch_total   += total_loss.item()
            global_step   += 1

            if (step + 1) % 50 == 0:
                print(f'  Epoch {epoch} [{step+1}/{len(train_loader)}] '
                      f'CE={ce_loss.item():.4f}  KD={kd_loss.item():.4f}  '
                      f'Total={total_loss.item():.4f}  '
                      f'lr={scheduler.get_last_lr()[0]:.2e}')

        n = len(train_loader)
        print(f'\n[Epoch {epoch}] 平均損失: '
              f'CE={epoch_ce_loss/n:.4f}  KD={epoch_kd_loss/n:.4f}  '
              f'Total={epoch_total/n:.4f}')

        # ── 検証（CER）──────────────────────────────────
        dev_cer = evaluate_cer(student, processor, dev_loader, device, max_batches=30)
        print(f'[Epoch {epoch}] Dev CER（推定）: {dev_cer:.4f}')

        if dev_cer < best_cer:
            best_cer     = dev_cer
            patience_cnt = 0
            # ── チェックポイント保存 ──────────────────
            adapter_state = {
                name: param
                for name, param in student.named_parameters()
                if 'adapter' in name
            }
            torch.save(adapter_state, CKPT_DIR / 'adapter_weights_kd.pt')
            print(f'  ✓ ベストモデル更新 (CER={best_cer:.4f}) → 保存')
        else:
            patience_cnt += 1
            print(f'  patience: {patience_cnt}/{args.patience}')
            if patience_cnt >= args.patience:
                print(f'\nEarly stopping（epoch {epoch}）')
                break

        student.train()

    # ── フック解除 ────────────────────────────────────────
    teacher_handle.remove()
    student_handle.remove()

    # ── 設定・Processorを保存 ──────────────────────────────
    config = {
        'r':       args.r,
        'd_model': D_MODEL,
        'alpha':   args.alpha,
        'beta':    args.beta,
        'kd_loss': args.kd_loss,
        'best_dev_cer': best_cer,
        'num_encoder_layers': len(student.model.encoder.layers),
    }
    with open(CKPT_DIR / 'adapter_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    processor.save_pretrained(str(CKPT_DIR / 'processor'))

    print(f'\n完了。Best Dev CER: {best_cer:.4f}')
    print(f'  Adapter重み : {CKPT_DIR / "adapter_weights_kd.pt"}')
    print(f'  設定        : {CKPT_DIR / "adapter_config.json"}')
    print(f'\n次のステップ:')
    print(f'  python3 scripts/32_evaluate_crosslingual.py  '
          f'# crosslingual評価（KDモデルを追加）')


if __name__ == '__main__':
    main()
