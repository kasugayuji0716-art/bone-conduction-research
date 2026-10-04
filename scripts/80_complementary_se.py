"""
スクリプト80: 相補学習 — 生成モデルの出力を土台に、Whisper に合わせる成分だけを学習する SE

推論時の融合（script 78: 生成モデルの出力と CE-SE の出力の振幅を半々で平均、位相は生成モデル）が
Whisper 系で最良だった。ここでは CE-SE（SE-Conformer）を、単体の出力ではなく
「生成モデルの出力と混ぜたあとの音」に Whisper-small の CE がかかるように追加学習する。
生成モデルは固定（学習しない）。

  損失 = 再構成（SE 出力 vs 気導音、L1 + 多重解像度 STFT） + λ × CE( Whisper-small( 融合音 ), 正解テキスト )
  融合音 = iSTFT( ((1−w)|G| + w|S|) · exp(j∠G) )   G: 生成モデルの出力、S: SE の出力
  --no_fuse: 対照。同じ dev データで、融合せずに SE 単体の出力に CE をかける（＝ dev で CE-SE を追加学習しただけ）

データ: 生成モデルは TAPS train で学習済みで、train の出力はほぼ正解の丸暗記になるため dev を使う
（10話者のうち8話者で学習、残り2話者で選択。script 79 と同じ分け方）。生成モデルの出力は保存済みの fm_taps（1回生成）。

使い方（GPU PC、venv）
    python scripts/80_complementary_se.py train --tag comp_ce10
    python scripts/80_complementary_se.py apply --tag comp_ce10 --src fm_taps_avg4 --out fm_avg4_comp_ce10
"""

import argparse
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
SE_WAV = BASE_DIR / 'data' / 'processed' / 'se_wav'
DEVICE = 'cuda'
N_FFT, HOP = 512, 128

s64 = import_module('64_retranscribe_all')
s79 = import_module('79_whisper_on_real_outputs')


def fuse(G_wav, S_wav, w, win):
    """(B, T) の2つの波形の振幅を混ぜ、位相は G（生成モデル）"""
    G = torch.stft(G_wav, N_FFT, HOP, window=win, return_complex=True)
    S = torch.stft(S_wav, N_FFT, HOP, window=win, return_complex=True)
    Y = ((1 - w) * G.abs() + w * S.abs()) * torch.exp(1j * G.angle())
    return torch.istft(Y, N_FFT, HOP, window=win, length=G_wav.shape[-1])


def train(args):
    s51 = import_module('51_train_se_ce_loss')
    torch.manual_seed(0); random.seed(0)
    ck = BASE_DIR / 'checkpoints' / args.tag; ck.mkdir(parents=True, exist_ok=True)
    se = s64.load_se(s64.SE_CKPTS[args.init]); se.train()
    stft_loss = s51.MultiResolutionSTFTLoss().to(DEVICE)
    ce = s79.whisper_ce()
    win = torch.hann_window(N_FFT, device=DEVICE)
    tr, va = s79.dev_split(args.max_sec)
    opt = torch.optim.Adam(se.parameters(), lr=args.lr, betas=(0.9, 0.99))

    def load(rows):
        xs, gs, ys = [], [], []
        for r in rows:
            name = f"{r['speaker_id']}_{r['sentence_id']}.wav"
            x, _ = sf.read(TAPS_DIR / 'throat' / 'dev' / name, dtype='float32')
            y, _ = sf.read(TAPS_DIR / 'acoustic' / 'dev' / name, dtype='float32')
            g, _ = sf.read(SE_WAV / args.gen / 'dev' / name, dtype='float32')
            n = min(len(x), len(y), len(g)); xs.append(x[:n]); ys.append(y[:n]); gs.append(g[:n])
        x, _ = s79.pad_batch(xs); y, _ = s79.pad_batch(ys); g, _ = s79.pad_batch(gs)
        return x, g, y, [r['text'] for r in rows]

    def step(rows, train_mode):
        x, g, y, texts = load(rows)
        s = se(x).reshape(x.shape[0], -1)[:, :x.shape[-1]]
        recon = F.l1_loss(s, y) + stft_loss(s, y)
        out = s if args.no_fuse else fuse(g, s, args.w, win)
        l_ce = ce(out, texts)
        return recon + args.lam * l_ce, l_ce

    def evaluate():
        se.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), args.batch_size):
                _, l = step(va[i:i + args.batch_size], False)
                k = len(va[i:i + args.batch_size]); tot += l.item() * k; n += k
        se.train()
        return tot / n

    best = evaluate()
    print(f'init={args.init} fuse={not args.no_fuse} w={args.w} lam={args.lam}  val CE（学習前）={best:.4f}', flush=True)
    torch.save(se.state_dict(), ck / 'best.th')
    for ep in range(1, args.epochs + 1):
        random.shuffle(tr); trl = []
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            loss, l = step(tr[i:i + args.batch_size], True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(se.parameters(), 5.0)
            opt.step(); trl.append(l.item())
        vl = evaluate()
        mark = ' <- best' if vl < best else ''
        print(f'epoch {ep:2d}  train CE={np.mean(trl):.4f}  val CE={vl:.4f}{mark}', flush=True)
        if vl < best:
            best = vl; torch.save(se.state_dict(), ck / 'best.th')
    print(f'done. best val CE={best:.4f}')


def apply(args):
    se = s64.load_se(BASE_DIR / 'checkpoints' / args.tag / 'best.th')
    win = torch.hann_window(N_FFT, device=DEVICE)
    out = SE_WAV / args.out / args.split; out.mkdir(parents=True, exist_ok=True)
    for smp in s64.load_samples(args.split):
        name = f"{smp['utt']}.wav"
        if (out / name).exists():
            continue
        x, sr = sf.read(smp['path'], dtype='float32')
        g, _ = sf.read(SE_WAV / args.src / args.split / name, dtype='float32')
        n = min(len(x), len(g))
        with torch.no_grad():
            xt = torch.from_numpy(x[:n])[None].to(DEVICE)
            s = se(xt).reshape(1, -1)[:, :n]
            y = s if args.no_fuse else fuse(torch.from_numpy(g[:n])[None].to(DEVICE), s, args.w, win)
        sf.write(out / name, y[0].cpu().numpy().astype(np.float32), sr)
    print(f'done → {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'apply'])
    ap.add_argument('--tag', default='comp_ce10')
    ap.add_argument('--init', default='ce10.0', help='初期値の SE（script 64 の SE_CKPTS の名前）')
    ap.add_argument('--gen', default='fm_taps', help='train: 土台にする生成モデルの出力（dev）')
    ap.add_argument('--src', default='fm_taps_avg4', help='apply: 土台にする生成モデルの出力')
    ap.add_argument('--out', default='')
    ap.add_argument('--split', default='test')
    ap.add_argument('--w', type=float, default=0.5)
    ap.add_argument('--lam', type=float, default=10.0)
    ap.add_argument('--no_fuse', action='store_true')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--max_sec', type=float, default=12.0)
    args = ap.parse_args()
    {'train': train, 'apply': apply}[args.stage](args)


if __name__ == '__main__':
    main()
