"""
スクリプト69: 複数SEの振幅統合を1つのSE-Conformerに蒸留する（ASRを一切使わない学習）

背景（results/FUSION_RESULTS_2026-09-28.md）
  ASR損失を使わない4つのSE（TAPS SE-Conformer / Demucs / TSTNN / CE0）の振幅を統合すると、学習に使っていない
  系列を含む9認識器すべてで TAPS より改善した。ただし SE を4つ動かすので推論コストが約4倍。
  統合出力を教師にして1つの SE-Conformer を学習し、TAPS と同じコストで効果が残るかを調べる。

教師: m4_med（4つの振幅の中央値、位相は TAPS。dev の最悪認識器で選んだ統合方法）。--target m4_amp も可
学習: TAPS pretrained SE-Conformer から初期化し、L1 + マルチ解像度STFT（TAPS 公式設定、script 51 と同じ）で教師に合わせる
モデル選択: dev の蒸留損失で早期終了（ASR を使わない）。15秒超の発話は除外（script 51 と同じ）

ステージ
  targets  train / dev の教師波形を data/processed/distill_<target>/<split>/<utt>.wav に保存（再開可能）
  train    学習し checkpoints/distill_<target>/best.th と training_log.csv を保存

2026-09-28 追記: train で作った教師では失敗した（dev 損失が TAPS 初期値から一度も下がらず）。統合の元の SE は
  いずれも TAPS train で学習済みで、train 発話では出力が気導音声に近く SE 同士の差も小さい（log-mel L1:
  TAPS–Demucs train 0.325 / dev 0.424、TAPS–気導 train 0.405 / dev 0.586）。つまり train の教師には
  「SE 固有の誤りが打ち消し合う」効果がほとんど含まれない。→ --train_split dev で、SE が見ていない dev 発話の
  教師を使う（dev 10話者のうち8話者で学習、2話者で早期終了。評価は test のみ）

使い方（GPU PC）
    python scripts/69_distill_fusion.py targets [--target m4_med]
    python scripts/69_distill_fusion.py train   [--target m4_med] [--epochs 50 --batch_size 8 --lr 3e-4]
    python scripts/69_distill_fusion.py train   --train_split dev --lr 1e-4 --epochs 100 --patience 10
  評価は script 66 の --set distill（6+3認識器）
"""

import argparse, csv, sys, time
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s51 = import_module('51_train_se_ce_loss')        # MultiResolutionSTFTLoss, TAPS_SE_CONFIG
s64 = import_module('64_retranscribe_all')        # load_samples
s66 = import_module('66_lambda_cross_asr')        # load_any_se
s67 = import_module('67_magnitude_fusion')        # stft / istft / SOURCES

DEVICE = s67.DEVICE
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
MAX_SEC = 15.0


def fused(outs, target):
    """outs: {taps, demucs, tstnn, ce0.0} の (T,) → 統合波形（位相は TAPS）"""
    n = min(v.shape[-1] for v in outs.values())
    S = {k: s67.stft(v[:n]) for k, v in outs.items()}
    A = torch.stack([S[k].abs() for k in s67.SOURCES])
    M = A.median(0).values if target == 'm4_med' else A.mean(0)
    return s67.istft(M * torch.exp(1j * S['taps'].angle()), n)


def target_dir(target, split):
    return BASE_DIR / 'data' / 'processed' / f'distill_{target}' / split


def make_targets(target):
    se = {c: s66.load_any_se(c) for c in s67.SOURCES}
    for split in ('train', 'dev'):
        out = target_dir(target, split)
        out.mkdir(parents=True, exist_ok=True)
        samples = s64.load_samples(split)
        t0 = time.time()
        for i, s in enumerate(samples):
            path = out / f"{s['utt']}.wav"
            if path.exists():
                continue
            wav, _ = sf.read(s['path'], dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                y = fused({k: m(x).reshape(-1) for k, m in se.items()}, target)
            sf.write(path, y, 16000, subtype='FLOAT')
            if (i + 1) % 200 == 0:
                torch.cuda.empty_cache()
                print(f'  {split} {i + 1}/{len(samples)}  {(time.time() - t0) / 60:.1f}min', flush=True)
        print(f'{split} targets done → {out}')


class DistillDataset(Dataset):
    def __init__(self, split, target, speakers=None):
        self.items = []
        excluded = 0
        tdir = target_dir(target, split)
        for s in s64.load_samples(split):
            if speakers is not None and s['spk'] not in speakers:
                continue
            info = sf.info(s['path'])
            if info.frames / info.samplerate > MAX_SEC:
                excluded += 1
                continue
            p = tdir / f"{s['utt']}.wav"
            if p.exists():
                self.items.append((s['path'], p))
        who = f' speakers={sorted(speakers)}' if speakers is not None else ''
        print(f'  [{split}] {len(self.items)} pairs (excluded {excluded} > {MAX_SEC}s){who}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, _ = sf.read(self.items[i][0], dtype='float32')
        y, _ = sf.read(self.items[i][1], dtype='float32')
        if x.ndim > 1:
            x = x.mean(axis=1)
        n = min(len(x), len(y))
        return torch.from_numpy(x[:n]), torch.from_numpy(y[:n])


def collate(batch):
    n = max(x.shape[0] for x, _ in batch)
    pad = lambda ws: torch.stack([F.pad(w, (0, n - w.shape[0])) for w in ws])
    xs, ys = zip(*batch)
    return pad(xs), pad(ys)


def train(args):
    ckpt = BASE_DIR / 'checkpoints' / f'distill_{args.target}{"_dev" if args.train_split == "dev" else ""}'
    ckpt.mkdir(parents=True, exist_ok=True)
    from models.seconformer import seconformer
    model = seconformer(**s51.TAPS_SE_CONFIG).to(DEVICE)
    st = torch.load(BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th', map_location=DEVICE,
                    weights_only=False)
    model.load_state_dict(st['model'] if isinstance(st, dict) and 'model' in st else st)
    stft_loss = s51.MultiResolutionSTFTLoss().to(DEVICE)

    def loss_fn(x, y):
        out = model(x).squeeze(1)
        n = min(out.shape[-1], y.shape[-1])
        return F.l1_loss(out[..., :n], y[..., :n]) + stft_loss(out[..., :n], y[..., :n])

    if args.train_split == 'dev':
        spk = sorted({s['spk'] for s in s64.load_samples('dev')})
        tr_ds = DistillDataset('dev', args.target, set(spk[:8]))
        dv_ds = DistillDataset('dev', args.target, set(spk[8:]))
    else:
        tr_ds, dv_ds = DistillDataset('train', args.target), DistillDataset('dev', args.target)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=2, pin_memory=True)
    dv = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=2, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    def evaluate():
        model.eval()
        with torch.no_grad():
            v = [loss_fn(x.to(DEVICE), y.to(DEVICE)).item() for x, y in dv]
        model.train()
        return float(np.mean(v))

    log = ckpt / 'training_log.csv'
    with open(log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'dev_loss', 'minutes'])
    best = evaluate()                       # epoch 0 = TAPS pretrained そのもの
    torch.save(model.state_dict(), ckpt / 'best.th')
    print(f'epoch 0 (TAPS pretrained): dev {best:.4f}', flush=True)
    with open(log, 'a', newline='') as f:
        csv.writer(f).writerow([0, '', f'{best:.5f}', 0])
    wait, t0 = 0, time.time()
    for ep in range(1, args.epochs + 1):
        losses = []
        for x, y in tr:
            loss = loss_fn(x.to(DEVICE), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        dev = evaluate()
        mark = ''
        if dev < best:
            best, wait, mark = dev, 0, ' <- best'
            torch.save(model.state_dict(), ckpt / 'best.th')
        else:
            wait += 1
        mins = (time.time() - t0) / 60
        print(f'epoch {ep}: train {np.mean(losses):.4f}  dev {dev:.4f}{mark}  ({mins:.0f}min)', flush=True)
        with open(log, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f'{np.mean(losses):.5f}', f'{dev:.5f}', f'{mins:.1f}'])
        if wait >= args.patience:
            print(f'early stop (patience={args.patience})')
            break
    print(f'best dev {best:.4f} → {ckpt / "best.th"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['targets', 'train'])
    ap.add_argument('--target', choices=['m4_med', 'm4_amp'], default='m4_med')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--train_split', choices=['train', 'dev'], default='train')
    a = ap.parse_args()
    make_targets(a.target) if a.stage == 'targets' else train(a)


if __name__ == '__main__':
    main()
