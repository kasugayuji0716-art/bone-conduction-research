"""
フェーズ2: GTCRNを喉マイク→気導マイクでファインチューニング

使い方:
  python scripts/20_finetune_gtcrn.py
  python scripts/20_finetune_gtcrn.py --epochs 50 --batch_size 8 --lr 1e-4

入力: data/raw/taps/throat/train/*.wav  (8kHz → 16kHzリサンプル)
ターゲット: data/raw/taps/acoustic/train/*.wav (16kHz)
検証: data/raw/taps/{throat,acoustic}/dev/*.wav
出力: checkpoints/gtcrn_taps_finetuned.tar
"""

import os
import sys
import argparse
import glob
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from scipy.signal import resample_poly

# GTCRNのコードをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtcrn'))
from gtcrn import GTCRN
from loss import HybridLoss

# ─── 定数 ───────────────────────────────────────────────
# 注意: TAPSデータセットの喉マイク・気導マイクはともに16kHzで保存されている
# （CLAUDE.mdの「喉マイク8kHz」はデータセット説明の誤り）
TARGET_SR   = 16000   # GTCRNが期待するサンプルレート
N_FFT       = 512
HOP         = 256
WIN         = 512
SEGMENT_SEC = 4.0     # 学習時の切り出し長（秒）
SEGMENT_LEN = int(TARGET_SR * SEGMENT_SEC)

BASE_DIR    = Path(__file__).parent.parent
TAPS_DIR    = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT_DIR    = BASE_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

PRETRAINED  = BASE_DIR / 'gtcrn' / 'checkpoints' / 'model_trained_on_dns3.tar'
SAVE_PATH   = CKPT_DIR / 'gtcrn_taps_finetuned.tar'


# ─── Dataset ────────────────────────────────────────────
class TAPSDataset(Dataset):
    """喉マイク・気導マイクペアデータセット

    - 喉マイク (8kHz) を16kHzにリサンプリング
    - SEGMENT_LEN サンプルにランダムクロップ（短い発話はゼロパディング）
    """
    def __init__(self, split: str, segment_len: int = SEGMENT_LEN, augment: bool = False):
        self.segment_len = segment_len
        self.augment = augment

        throat_dir   = TAPS_DIR / 'throat'   / split
        acoustic_dir = TAPS_DIR / 'acoustic' / split

        throat_files = sorted(glob.glob(str(throat_dir / '*.wav')))
        if not throat_files:
            raise FileNotFoundError(f"喉マイクWAVが見つかりません: {throat_dir}")

        self.pairs = []
        for t_path in throat_files:
            fname = os.path.basename(t_path)
            a_path = str(acoustic_dir / fname)
            if os.path.exists(a_path):
                self.pairs.append((t_path, a_path))

        print(f"  [{split}] {len(self.pairs)} ペア読み込み")

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # サンプルレートが異なる場合のみリサンプリング（念のため）
        if sr != TARGET_SR:
            wav = resample_poly(wav, TARGET_SR, sr).astype(np.float32)
        return wav

    def _crop_or_pad(self, wav):
        if len(wav) >= self.segment_len:
            # ランダムクロップ
            start = random.randint(0, len(wav) - self.segment_len)
            wav = wav[start:start + self.segment_len]
        else:
            # ゼロパディング
            pad = self.segment_len - len(wav)
            wav = np.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        t_path, a_path = self.pairs[idx]
        throat   = self._load(t_path)
        acoustic = self._load(a_path)

        # 長さをそろえてからクロップ（リサンプル後の誤差を吸収）
        min_len = min(len(throat), len(acoustic))
        throat   = throat[:min_len]
        acoustic = acoustic[:min_len]

        throat   = self._crop_or_pad(throat)
        acoustic = self._crop_or_pad(acoustic)

        return torch.from_numpy(throat), torch.from_numpy(acoustic)


# ─── STFT / iSTFT ───────────────────────────────────────
def rms_normalize(wav, eps=1e-8):
    """(B, L) → RMS=1 に正規化。スケールも返す"""
    rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
    return wav / rms, rms

def make_stft(wav, device):
    """wav: (B, L) → stft: (B, F, T, 2)"""
    window = torch.hann_window(WIN).pow(0.5).to(device)
    B = wav.shape[0]
    specs = []
    for i in range(B):
        s = torch.view_as_real(
            torch.stft(wav[i], N_FFT, HOP, WIN, window, return_complex=True)
        )
        specs.append(s)
    return torch.stack(specs, dim=0)  # (B, F, T, 2)


# ─── 学習・評価ループ ────────────────────────────────────
def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batch = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for throat, acoustic in loader:
            throat   = throat.to(device)
            acoustic = acoustic.to(device)

            # RMS正規化: 音量差を除去してスペクトル形状の変換だけを学習
            throat,  _ = rms_normalize(throat)
            acoustic, _ = rms_normalize(acoustic)

            noisy_stft = make_stft(throat,   device)  # (B,F,T,2)
            clean_stft = make_stft(acoustic, device)  # (B,F,T,2)

            pred_stft = model(noisy_stft)              # (B,F,T,2)

            # HybridLoss は (B, F, T, 2) を受け取る
            loss = loss_fn(pred_stft, clean_stft)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            n_batch += 1

    return total_loss / max(n_batch, 1)


# ─── メイン ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--num_workers',type=int,   default=2)
    parser.add_argument('--patience',   type=int,   default=5,
                        help='val lossが改善しなければ早期終了するエポック数')
    parser.add_argument('--resume',     type=str,   default=None,
                        help='チェックポイントパス（再開用）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ─ データ
    print("データセット準備中...")
    train_ds = TAPSDataset('train', augment=True)
    dev_ds   = TAPSDataset('dev',   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'),
                              drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'),
                              drop_last=False)

    # ─ モデル
    model = GTCRN().to(device)

    start_epoch = 1
    best_val_loss = float('inf')
    no_improve = 0
    log_rows = []

    if args.resume:
        print(f"チェックポイント再開: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch  = ckpt.get('epoch', 1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  epoch {start_epoch} から再開、best_val_loss={best_val_loss:.4f}")
    else:
        print(f"事前学習重みをロード: {PRETRAINED}")
        ckpt = torch.load(PRETRAINED, map_location=device)
        model.load_state_dict(ckpt['model'])

    loss_fn   = HybridLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    print(f"\n学習開始: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"  train={len(train_ds)}件, dev={len(dev_ds)}件")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss   = run_epoch(model, dev_loader,   loss_fn, optimizer, device, train=False)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        marker = " ✓" if improved else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")

        log_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if improved:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                'epoch':         epoch,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args':          vars(args),
            }, SAVE_PATH)
            print(f"  → 保存: {SAVE_PATH}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n早期終了: val lossが{args.patience}エポック改善なし")
                break

    # ─ ログCSV保存
    log_path = CKPT_DIR / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n完了。best val_loss={best_val_loss:.4f}")
    print(f"  モデル: {SAVE_PATH}")
    print(f"  ログ:   {log_path}")


if __name__ == '__main__':
    main()
