"""
スクリプト41: ASR-aware SE再学習
ハイブリッド損失: SI-SDR + λ × Whisper encoder 特徴量マッチング

動機:
    script 38でSEのencoder距離↑ → CER↑の相関を確認済み。
    → Whisper encoderの特徴量空間で気導音声に近い表現を出力するように
      SEを直接学習することで、CERを改善する。

損失:
    L_total = L_SI-SDR(ŝ, s_air) + λ × (1 - cosine_sim(enc(ŝ), enc(s_air)))
    L_SI-SDR : SE出力と気導音声の波形レベル類似度
    L_feat   : Whisper encoder空間での類似度（微分可能）

入力:
    data/raw/taps/throat/train/     ← SE入力
    data/raw/taps/acoustic/train/   ← SI-SDR目標・特徴量マッチング目標
出力:
    checkpoints/se_asr_aware/best.th
    checkpoints/se_asr_aware/training_log.csv

使い方（DNN PC）:
    python scripts/41_train_se_asr_aware.py
    python scripts/41_train_se_asr_aware.py --lambda_asr 0.5
    python scripts/41_train_se_asr_aware.py --lambda_asr 0.0  # SI-SDRのみ（TAPSベースライン相当）
"""

import argparse
import csv
import math
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
CKPT_DIR       = BASE_DIR / 'checkpoints' / 'se_asr_aware'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TARGET_SR = 16000
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════
# SE-Conformer（script 37と同一アーキテクチャ）
# ══════════════════════════════════════════════════════════════

class _SEConformerFFN(nn.Module):
    def __init__(self, d=512, r=64):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, r), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(r, d),
        )

    def forward(self, x):
        return self.sequential(x)


class _SEConformerConvModule(nn.Module):
    def __init__(self, d=512, k=15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.sequential = nn.Sequential(
            nn.Conv1d(d, d * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, k, padding=k // 2, groups=d),
            nn.BatchNorm1d(d), nn.SiLU(), nn.Conv1d(d, d, 1),
        )

    def forward(self, x):           # x: (B, T, d)
        r = x
        x = self.layer_norm(x).transpose(1, 2)
        x = self.sequential(x).transpose(1, 2)
        return x + r


class _SEConformerBlock(nn.Module):
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.ffn1                 = _SEConformerFFN(d)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.self_attn            = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv_module          = _SEConformerConvModule(d)
        self.ffn2                 = _SEConformerFFN(d)
        self.final_layer_norm     = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        r = x
        x = self.self_attn_layer_norm(x)
        a, _ = self.self_attn(x, x, x)
        x = r + a
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)


class SEConformerModel(nn.Module):
    CH = [1, 64, 128, 256, 512]; K = 8; S = 4
    RESAMPLE = 4; FLOOR = 1e-3

    def __init__(self, heads=4, n_conf=4):
        super().__init__()
        enc, dec = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            ic, hc = self.CH[i], self.CH[i + 1]
            enc.append(nn.Sequential(
                nn.Conv1d(ic, hc, self.K, stride=self.S), nn.ReLU(),
                nn.Conv1d(hc, hc * 2, 1), nn.GLU(dim=1),
            ))
        for i in range(3, -1, -1):
            ic, oc = self.CH[i + 1], self.CH[i]
            dec.append(nn.Sequential(
                nn.Conv1d(ic, ic * 2, 1), nn.GLU(dim=1), nn.ReLU(),
                nn.ConvTranspose1d(ic, max(oc, 1), self.K, stride=self.S),
            ))
        self.encoder    = enc
        self.conformers = nn.ModuleList([_SEConformerBlock(self.CH[-1], heads) for _ in range(n_conf)])
        self.decoder    = dec

    def valid_length(self, length: int) -> int:
        length = math.ceil(length * self.RESAMPLE)
        for _ in range(len(self.CH) - 1):
            length = math.ceil((length - self.K) / self.S) + 1
            length = max(length, 1)
        for _ in range(len(self.CH) - 1):
            length = (length - 1) * self.S + self.K
        return int(math.ceil(length / self.RESAMPLE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from models.demucs import upsample2, downsample2
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2: x = x.unsqueeze(1)
        # x: (B, 1, T)
        mono   = x.mean(dim=1, keepdim=True)
        std    = mono.std(dim=-1, keepdim=True)
        x      = x / (self.FLOOR + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        x = upsample2(upsample2(x))
        skips = []
        for enc in self.encoder:
            x = enc(x); skips.append(x)
        x = x.permute(0, 2, 1)
        for conf in self.conformers:
            x = conf(x)
        x = x.permute(0, 2, 1)
        for dec in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = dec(x)
        x = downsample2(downsample2(x))
        x = x[..., :length]
        out = std * x          # (B, 1, T)
        return out.squeeze(1)  # (B, T)  ← バッチ次元を保持


# ══════════════════════════════════════════════════════════════
# Whisper log-mel（微分可能）
# ══════════════════════════════════════════════════════════════

class WhisperLogMel(nn.Module):
    """Whisperと同一パラメータのlog-mel変換（微分可能）"""
    N_FRAMES = 3000   # 30秒 @ 16kHz / hop=160

    def __init__(self, sr=16000):
        super().__init__()
        self.mel = Ta.MelSpectrogram(
            sample_rate=sr, n_fft=400, win_length=400, hop_length=160,
            n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
            window_fn=torch.hann_window, normalized=False,
        )

    def forward(self, x):   # x: (B, T)
        mel = self.mel(x)   # (B, 80, T')
        log_mel = torch.log10(mel.clamp(min=1e-10))
        max_val = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, max_val - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        # 30秒にpad/truncate
        if log_mel.shape[-1] < self.N_FRAMES:
            log_mel = F.pad(log_mel, (0, self.N_FRAMES - log_mel.shape[-1]))
        else:
            log_mel = log_mel[..., :self.N_FRAMES]
        return log_mel   # (B, 80, 3000)


# ══════════════════════════════════════════════════════════════
# 損失関数
# ══════════════════════════════════════════════════════════════

def si_sdr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant SDR損失（最小化）"""
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
    def __init__(self, split: str, max_sec: float = 15.0):
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
        print(f'  [{split}] {len(self.samples)} ペア')

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
    parser.add_argument('--lambda_asr', type=float, default=0.1,
                        help='ASR特徴量損失の重み λ（default: 0.1）\n'
                             '  0.0: SI-SDRのみ（TAPSベースライン相当）\n'
                             '  0.1: 弱いASR制約\n'
                             '  1.0: 強いASR制約')
    parser.add_argument('--feat_loss', type=str, default='l1',
                        choices=['cosine', 'l1'],
                        help='特徴量損失の種類: cosine（mean-pool後のcosine距離）'
                             'or l1（フレームごとのL1距離、Perceive&Predict準拠）')
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--patience',   type=int,   default=3)
    args = parser.parse_args()

    print(f'\n=== ASR-aware SE再学習 ===')
    print(f'λ_asr={args.lambda_asr} | epochs={args.epochs} | bs={args.batch_size} | device={DEVICE}\n')

    # ── SE モデル（TAPSのpretrainedで初期化）──
    se_model = SEConformerModel().to(DEVICE)
    state = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    se_model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in se_model.parameters() if p.requires_grad)
    print(f'SE-Conformer: pretrainedで初期化 ({n_params/1e6:.1f}M params)')

    # ── Whisper encoder（凍結）──
    whisper    = WhisperModel.from_pretrained('openai/whisper-small')
    whisper_enc = whisper.encoder.to(DEVICE).eval()
    for p in whisper_enc.parameters():
        p.requires_grad_(False)
    print(f'Whisper encoder: 凍結')

    # ── log-mel変換（微分可能）──
    log_mel_fn = WhisperLogMel().to(DEVICE)

    # ── データ ──
    print('\nデータ読み込み:')
    train_ds = TAPSPairDataset('train')
    dev_ds   = TAPSPairDataset('dev')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(se_model.parameters(), lr=args.lr)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── ログ ──
    log_path = CKPT_DIR / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_l_recon', 'val_l_feat'])

    best_val, patience_cnt = float('inf'), 0

    for epoch in range(1, args.epochs + 1):

        # ── 学習フェーズ ──
        se_model.train()
        train_losses = []

        for t_wav, a_wav in train_dl:
            t_wav = t_wav.to(DEVICE)   # (B, T)
            a_wav = a_wav.to(DEVICE)   # (B, T)

            # SE適用
            se_out  = se_model(t_wav)  # (B, T')
            min_len = min(se_out.shape[-1], a_wav.shape[-1])
            se_t    = se_out[..., :min_len]
            a_t     = a_wav[..., :min_len]

            # SI-SDR損失
            l_recon = si_sdr_loss(se_t, a_t)

            # ASR特徴量マッチング損失
            if args.lambda_asr > 0:
                mel_se  = log_mel_fn(se_t)                                       # (B, 80, 3000)
                mel_air = log_mel_fn(a_t)                                        # (B, 80, 3000)
                with torch.no_grad():
                    feat_air = whisper_enc(mel_air).last_hidden_state              # (B, T', D)
                feat_se  = whisper_enc(mel_se).last_hidden_state                   # (B, T', D)
                if args.feat_loss == 'l1':
                    l_feat = F.l1_loss(feat_se, feat_air)
                else:  # cosine
                    l_feat = 1 - F.cosine_similarity(
                        feat_se.mean(dim=1), feat_air.mean(dim=1), dim=-1).mean()
            else:
                l_feat = torch.zeros(1, device=DEVICE)

            loss = l_recon + args.lambda_asr * l_feat

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(se_model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── 検証フェーズ ──
        se_model.eval()
        val_losses, val_recons, val_feats = [], [], []

        with torch.no_grad():
            for t_wav, a_wav in dev_dl:
                t_wav = t_wav.to(DEVICE)
                a_wav = a_wav.to(DEVICE)
                se_out  = se_model(t_wav)
                min_len = min(se_out.shape[-1], a_wav.shape[-1])
                se_t    = se_out[..., :min_len]
                a_t     = a_wav[..., :min_len]
                l_recon = si_sdr_loss(se_t, a_t)
                if args.lambda_asr > 0:
                    mel_se  = log_mel_fn(se_t)
                    mel_air = log_mel_fn(a_t)
                    feat_air = whisper_enc(mel_air).last_hidden_state
                    feat_se  = whisper_enc(mel_se).last_hidden_state
                    if args.feat_loss == 'l1':
                        l_feat = F.l1_loss(feat_se, feat_air)
                    else:
                        l_feat = 1 - F.cosine_similarity(
                            feat_se.mean(dim=1), feat_air.mean(dim=1), dim=-1).mean()
                else:
                    l_feat = torch.zeros(1, device=DEVICE)
                val_losses.append((l_recon + args.lambda_asr * l_feat).item())
                val_recons.append(l_recon.item())
                val_feats.append(l_feat.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_recon  = float(np.mean(val_recons))
        val_feat   = float(np.mean(val_feats))

        marker = ' ← best' if val_loss < best_val else ''
        print(f'Epoch {epoch:2d} | train={train_loss:.4f} | val={val_loss:.4f} '
              f'(recon={val_recon:.4f}, feat={val_feat:.4f}){marker}')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_recon, val_feat])

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(se_model.state_dict(), CKPT_DIR / 'best.th')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'Early stopping (patience={args.patience})')
                break

    print(f'\n完了。最良val損失: {best_val:.4f}')
    print(f'モデル保存先: {CKPT_DIR / "best.th"}')


if __name__ == '__main__':
    main()
