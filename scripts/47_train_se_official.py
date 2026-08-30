"""
スクリプト47: TAPS公式SE-Conformerを使ったASR-aware再学習
script 41と同じ損失だが、SEモデルにTAPS公式実装（torchaudio.ConformerLayer）を使用。
公式実装との公平な比較を可能にする。

使い方（DNN PC）:
    # SI-SDRのみ（公式forwardベースライン）
    python scripts/47_train_se_official.py --lambda_asr 0.0 --tag si_sdr_official

    # ASR-aware（公式forward）
    python scripts/47_train_se_official.py --lambda_asr 1.0 --tag asr_aware_official

    # 評価
    python scripts/48_eval_official.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as Ta
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperModel

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TARGET_SR = 16000
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# TAPS公式SE-Conformerのハイパーパラメータ（チェックポイントから特定）
TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)


# ══════════════════════════════════════════════════════════════
# Whisper log-mel（微分可能）
# ══════════════════════════════════════════════════════════════

class WhisperLogMel(nn.Module):
    N_FRAMES = 3000

    def __init__(self, sr=16000):
        super().__init__()
        self.mel = Ta.MelSpectrogram(
            sample_rate=sr, n_fft=400, win_length=400, hop_length=160,
            n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
            window_fn=torch.hann_window, normalized=False,
        )

    def forward(self, x):
        mel = self.mel(x)
        log_mel = torch.log10(mel.clamp(min=1e-10))
        max_val = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, max_val - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        if log_mel.shape[-1] < self.N_FRAMES:
            log_mel = F.pad(log_mel, (0, self.N_FRAMES - log_mel.shape[-1]))
        else:
            log_mel = log_mel[..., :self.N_FRAMES]
        return log_mel


# ══════════════════════════════════════════════════════════════
# 損失関数
# ══════════════════════════════════════════════════════════════

def si_sdr_loss(est, target):
    est    = est    - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot    = (est * target).sum(dim=-1, keepdim=True)
    norm2  = (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
    proj   = (dot / norm2) * target
    noise  = est - proj
    sdr    = 10 * torch.log10((proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8) + 1e-8)
    return -sdr.mean()


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class TAPSPairDataset(Dataset):
    def __init__(self, split, max_sec=15.0):
        self.samples = []
        meta = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sid, uid = row['speaker_id'], row['sentence_id']
                t = TAPS_DIR / 'throat'   / split / f'{sid}_{uid}.wav'
                a = TAPS_DIR / 'acoustic' / split / f'{sid}_{uid}.wav'
                if t.exists() and a.exists():
                    self.samples.append((t, a))
        self.max_len = int(max_sec * TARGET_SR)
        print(f'  [{split}] {len(self.samples)} pairs')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_path, a_path = self.samples[idx]
        t_wav, _ = sf.read(t_path, dtype='float32')
        a_wav, _ = sf.read(a_path, dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)
        min_len = min(len(t_wav), len(a_wav), self.max_len)
        return torch.from_numpy(t_wav[:min_len]), torch.from_numpy(a_wav[:min_len])


def collate_fn(batch):
    t_wavs, a_wavs = zip(*batch)
    max_len = max(t.shape[0] for t in t_wavs)
    def pad(wavs):
        return torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in wavs])
    return pad(t_wavs), pad(a_wavs)


# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_asr', type=float, default=1.0)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--patience',   type=int,   default=3)
    parser.add_argument('--tag',        type=str,   default='asr_aware_official',
                        help='チェックポイント保存ディレクトリ名')
    args = parser.parse_args()

    ckpt_dir = BASE_DIR / 'checkpoints' / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== TAPS公式SE-Conformer ASR-aware再学習 ===')
    print(f'λ={args.lambda_asr} | tag={args.tag} | device={DEVICE}\n')

    # ── SE モデル（TAPS公式実装、pretrainedで初期化）──
    from models.seconformer import seconformer as TAPSSeconformer
    se_model = TAPSSeconformer(**TAPS_SE_CONFIG).to(DEVICE)
    state = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    se_model.load_state_dict(state)
    n_params = sum(p.numel() for p in se_model.parameters() if p.requires_grad)
    print(f'SE-Conformer (official): pretrained loaded ({n_params/1e6:.1f}M params)')

    # ── Whisper encoder（凍結）──
    whisper = WhisperModel.from_pretrained('openai/whisper-small')
    whisper_enc = whisper.encoder.to(DEVICE).eval()
    for p in whisper_enc.parameters():
        p.requires_grad_(False)
    print('Whisper encoder: frozen')

    # ── log-mel ──
    log_mel_fn = WhisperLogMel().to(DEVICE)

    # ── データ ──
    print('\nData:')
    train_ds = TAPSPairDataset('train')
    dev_ds   = TAPSPairDataset('dev')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(se_model.parameters(), lr=args.lr)

    log_path = ckpt_dir / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_recon', 'val_feat'])

    best_val, patience_cnt = float('inf'), 0

    for epoch in range(1, args.epochs + 1):

        # ── Train ──
        se_model.train()
        train_losses = []

        for t_wav, a_wav in train_dl:
            t_wav = t_wav.to(DEVICE)  # (B, T)
            a_wav = a_wav.to(DEVICE)

            # SE: expects (B, T) or (B, 1, T)
            se_out = se_model(t_wav)  # (B, 1, T)
            se_out = se_out.squeeze(1)  # (B, T)

            min_len = min(se_out.shape[-1], a_wav.shape[-1])
            se_t = se_out[..., :min_len]
            a_t  = a_wav[..., :min_len]

            l_recon = si_sdr_loss(se_t, a_t)

            if args.lambda_asr > 0:
                mel_se  = log_mel_fn(se_t)
                mel_air = log_mel_fn(a_t)
                with torch.no_grad():
                    feat_air = whisper_enc(mel_air).last_hidden_state
                feat_se = whisper_enc(mel_se).last_hidden_state
                l_feat = F.l1_loss(feat_se, feat_air)
            else:
                l_feat = torch.zeros(1, device=DEVICE)

            loss = l_recon + args.lambda_asr * l_feat

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(se_model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Val ──
        se_model.eval()
        val_losses, val_recons, val_feats = [], [], []

        with torch.no_grad():
            for t_wav, a_wav in dev_dl:
                t_wav = t_wav.to(DEVICE)
                a_wav = a_wav.to(DEVICE)
                se_out = se_model(t_wav).squeeze(1)
                min_len = min(se_out.shape[-1], a_wav.shape[-1])
                se_t = se_out[..., :min_len]
                a_t  = a_wav[..., :min_len]
                l_recon = si_sdr_loss(se_t, a_t)
                if args.lambda_asr > 0:
                    mel_se  = log_mel_fn(se_t)
                    mel_air = log_mel_fn(a_t)
                    feat_air = whisper_enc(mel_air).last_hidden_state
                    feat_se  = whisper_enc(mel_se).last_hidden_state
                    l_feat = F.l1_loss(feat_se, feat_air)
                else:
                    l_feat = torch.zeros(1, device=DEVICE)
                val_losses.append((l_recon + args.lambda_asr * l_feat).item())
                val_recons.append(l_recon.item())
                val_feats.append(l_feat.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_recon  = float(np.mean(val_recons))
        val_feat   = float(np.mean(val_feats))

        marker = ' <- best' if val_loss < best_val else ''
        print(f'Epoch {epoch:2d} | train={train_loss:.4f} | val={val_loss:.4f} '
              f'(recon={val_recon:.4f}, feat={val_feat:.4f}){marker}')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_recon, val_feat])

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(se_model.state_dict(), ckpt_dir / 'best.th')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'Early stopping (patience={args.patience})')
                break

    print(f'\nDone. Best val loss: {best_val:.4f}')
    print(f'Model: {ckpt_dir / "best.th"}')


if __name__ == '__main__':
    main()
