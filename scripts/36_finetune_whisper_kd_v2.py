"""
フェーズ3: KD Adapter v2（改良版）

v1（script 34）からの変更点:
  1. Teacher: Pretrained Whisper → 気導マイクFT済みWhisper（script 35の出力）
     → TAPSドメインを知っており、より適切な「目標表現」を提供
  2. β: 0.5 → 2.0（KD損失の重みを4倍に。v1ではKDがCEの1/50しか効かなかった）
  3. Layer-wise KD: Encoder最終層のみ → 全Encoderレイヤーの平均
     → 各層の表現を段階的に引き寄せる

学習:
  Teacher: checkpoints/whisper_acoustic_finetuned/（35番で学習）
  Student: Whisper-small + Encoder Adapter（Adapterのみ学習）
  Loss = α×CE損失 + β×KD損失（全Encoderレイヤーのcosine距離平均）

出力:
  checkpoints/whisper_kd_adapter_v2/

実行:
  python3 scripts/36_finetune_whisper_kd_v2.py
  python3 scripts/36_finetune_whisper_kd_v2.py --beta 5.0 --layerwise
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

BASE_DIR      = Path(__file__).parent.parent
TAPS_DIR      = BASE_DIR / 'data' / 'raw' / 'taps'
TEACHER_CKPT  = BASE_DIR / 'checkpoints' / 'whisper_acoustic_finetuned'
CKPT_DIR      = BASE_DIR / 'checkpoints' / 'whisper_kd_adapter_v2'
MODEL_ID      = 'openai/whisper-small'
TARGET_SR     = 16000
D_MODEL       = 768


# ── Adapter（29番と同一）─────────────────────────────────────
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
        layers[i] = WhisperEncoderLayerWithAdapter(layers[i], Adapter(D_MODEL, r))

    for name, param in model.named_parameters():
        param.requires_grad = ('adapter' in name)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Student 学習可能パラメータ: {train:,} / {total:,} ({train/total*100:.2f}%)')
    return model


# ── ペアデータセット ──────────────────────────────────────────
class TAPSPairedDataset(Dataset):
    def __init__(self, split: str, processor: WhisperProcessor):
        self.processor = processor
        self.samples   = []

        meta_path = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                spk = row['speaker_id']
                sid = row['sentence_id']
                throat   = TAPS_DIR / 'throat'   / split / f'{spk}_{sid}.wav'
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

        def load(path):
            wav, _ = sf.read(path, dtype='float32')
            return wav.mean(axis=1) if wav.ndim > 1 else wav

        def feat(wav):
            return self.processor.feature_extractor(
                wav, sampling_rate=TARGET_SR, return_tensors='pt'
            ).input_features[0]

        return {
            'throat_features':   feat(load(s['throat_path'])),
            'acoustic_features': feat(load(s['acoustic_path'])),
            'labels': self.processor.tokenizer(
                s['text'], return_tensors='pt').input_ids[0],
            'text': s['text'],
        }


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

        return {
            'throat_features':   throat_feats,
            'acoustic_features': acoustic_feats,
            'labels':            torch.stack(padded),
            'texts':             [f['text'] for f in features],
        }


# ── Layer-wise KD フック ─────────────────────────────────────
class LayerwiseHook:
    """全Encoderレイヤーの出力を収集する"""
    def __init__(self):
        self.outputs = []
        self._handles = []

    def register(self, model: WhisperForConditionalGeneration):
        self.outputs = []
        for layer in model.model.encoder.layers:
            # WhisperEncoderLayerWithAdapter の場合は .layer、通常層はそのまま
            target = layer.layer if hasattr(layer, 'layer') else layer
            handle = target.register_forward_hook(self._hook)
            self._handles.append(handle)

    def _hook(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self.outputs.append(h)

    def reset(self):
        self.outputs = []

    def remove(self):
        for h in self._handles:
            h.remove()


def layerwise_kd_loss(student_layers: list, teacher_layers: list) -> torch.Tensor:
    """各Encoderレイヤーのcosine距離を平均"""
    n = min(len(student_layers), len(teacher_layers))
    losses = []
    for s, t in zip(student_layers[:n], teacher_layers[:n]):
        s_mean = s.mean(dim=1)      # [B, D]
        t_mean = t.detach().mean(dim=1)
        losses.append((1 - F.cosine_similarity(s_mean, t_mean, dim=-1)).mean())
    return torch.stack(losses).mean()


def last_layer_kd_loss(student_layers: list, teacher_layers: list) -> torch.Tensor:
    s = student_layers[-1].mean(dim=1)
    t = teacher_layers[-1].detach().mean(dim=1)
    return (1 - F.cosine_similarity(s, t, dim=-1)).mean()


# ── 評価 ─────────────────────────────────────────────────────
def evaluate_cer(model, processor, loader, device, max_batches=30):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids  = model.generate(
                batch['throat_features'].to(device),
                language='ko', task='transcribe', max_new_tokens=225
            )
            hyps.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
            refs.extend(batch['texts'])
    return compute_cer(refs, hyps) if refs else 1.0


# ── メイン ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r',          type=int,   default=64)
    parser.add_argument('--alpha',      type=float, default=1.0,  help='CE損失の重み')
    parser.add_argument('--beta',       type=float, default=2.0,  help='KD損失の重み（v1の4倍）')
    parser.add_argument('--layerwise',  action='store_true',       help='全層KD（デフォルト: 最終層のみ）')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--patience',   type=int,   default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kd_mode = 'layer-wise（全層）' if args.layerwise else '最終層のみ'
    print(f'デバイス : {device}')
    print(f'Teacher  : {TEACHER_CKPT}')
    print(f'KD mode  : {kd_mode}')
    print(f'α={args.alpha}, β={args.beta}, r={args.r}, lr={args.lr}\n')

    if not TEACHER_CKPT.exists():
        raise FileNotFoundError(
            f'Teacher が見つかりません: {TEACHER_CKPT}\n'
            f'先に python3 scripts/35_finetune_whisper_acoustic.py を実行してください'
        )

    # ── モデル準備 ────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language='korean', task='transcribe')

    print('Teacher（気導マイクFT済みWhisper）を読み込み中...')
    teacher = WhisperForConditionalGeneration.from_pretrained(str(TEACHER_CKPT))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    print('Student（Whisper + Adapter）を準備中...')
    student = build_student(r=args.r)
    student.to(device)

    # フック登録
    teacher_hook = LayerwiseHook()
    student_hook = LayerwiseHook()
    teacher_hook.register(teacher)
    student_hook.register(student)

    # ── データ ───────────────────────────────────────────
    print('\nデータセット準備中...')
    train_ds = TAPSPairedDataset('train', processor)
    dev_ds   = TAPSPairedDataset('dev',   processor)
    collator = PairedDataCollator(processor=processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator,
                              num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds, batch_size=8,
                              shuffle=False, collate_fn=collator,
                              num_workers=4, pin_memory=True)

    # ── オプティマイザ ──────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr)
    total_steps  = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return max(0.0, 1 - (step - args.warmup_steps) /
                   max(1, total_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 学習ループ ────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_cer, patience_cnt, step = float('inf'), 0, 0

    print(f'\n学習開始（最大{args.epochs}エポック, patience={args.patience}）\n')

    for epoch in range(1, args.epochs + 1):
        student.train()
        ep_ce = ep_kd = ep_tot = 0.0

        for batch in train_loader:
            throat_feats   = batch['throat_features'].to(device)
            acoustic_feats = batch['acoustic_features'].to(device)
            labels         = batch['labels'].to(device)

            # Teacher forward（フックで各層出力を収集）
            teacher_hook.reset()
            with torch.no_grad():
                teacher(input_features=acoustic_feats,
                        decoder_input_ids=torch.zeros(
                            acoustic_feats.size(0), 1, dtype=torch.long, device=device))
            t_layers = [h.detach() for h in teacher_hook.outputs]

            # Student forward
            student_hook.reset()
            outputs  = student(input_features=throat_feats, labels=labels)
            ce_loss  = outputs.loss
            s_layers = student_hook.outputs

            # KD損失
            if args.layerwise:
                kd_loss = layerwise_kd_loss(s_layers, t_layers)
            else:
                kd_loss = last_layer_kd_loss(s_layers, t_layers)

            total_loss = args.alpha * ce_loss + args.beta * kd_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            ep_ce  += ce_loss.item()
            ep_kd  += kd_loss.item()
            ep_tot += total_loss.item()
            step   += 1

            if step % 50 == 0:
                print(f'  E{epoch} [{step%len(train_loader) or len(train_loader)}'
                      f'/{len(train_loader)}] '
                      f'CE={ce_loss.item():.4f}  KD={kd_loss.item():.4f}  '
                      f'Total={total_loss.item():.4f}  '
                      f'lr={scheduler.get_last_lr()[0]:.2e}')

        n = len(train_loader)
        print(f'\n[Epoch {epoch}] CE={ep_ce/n:.4f}  KD={ep_kd/n:.4f}  '
              f'Total={ep_tot/n:.4f}  (KD/CE比={ep_kd/ep_ce:.3f})')

        dev_cer = evaluate_cer(student, processor, dev_loader, device)
        print(f'[Epoch {epoch}] Dev CER: {dev_cer:.4f}')

        if dev_cer < best_cer:
            best_cer, patience_cnt = dev_cer, 0
            adapter_state = {k: v for k, v in student.named_parameters()
                             if 'adapter' in k}
            torch.save(adapter_state, CKPT_DIR / 'adapter_weights_kd_v2.pt')
            print(f'  ✓ ベスト更新 (CER={best_cer:.4f}) → 保存')
        else:
            patience_cnt += 1
            print(f'  patience: {patience_cnt}/{args.patience}')
            if patience_cnt >= args.patience:
                print(f'\nEarly stopping（epoch {epoch}）')
                break

        student.train()

    teacher_hook.remove()
    student_hook.remove()

    config = {
        'r': args.r, 'd_model': D_MODEL,
        'alpha': args.alpha, 'beta': args.beta,
        'layerwise': args.layerwise,
        'teacher': str(TEACHER_CKPT),
        'best_dev_cer': best_cer,
        'num_encoder_layers': len(student.model.encoder.layers),
    }
    with open(CKPT_DIR / 'adapter_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    processor.save_pretrained(str(CKPT_DIR / 'processor'))

    print(f'\n完了。Best Dev CER: {best_cer:.4f}')
    print(f'  Adapter重み : {CKPT_DIR / "adapter_weights_kd_v2.pt"}')
    print(f'\n次のステップ:')
    print(f'  python3 scripts/32_evaluate_crosslingual.py  # 評価（F_kd_v2を追加）')


if __name__ == '__main__':
    main()
