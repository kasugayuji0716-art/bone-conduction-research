"""
スクリプト89: 融合A（生成モデル4回平均 × CE-SE の振幅融合）の出力を教師にして、軽い SE-Conformer 1つに蒸留する

融合A は Whisper 系に最も効くが、4.3億パラメータの生成モデルを 20 段 × 4 回動かすので重い。
その出力を目標にして SE-Conformer（1210万パラメータ）を学習し、1回の順伝播で同じ効果がどこまで残るかを見る。

  損失 = L1(ŝ, 教師) + 多重解像度STFT(ŝ, 教師) + λ × CE(Whisper-small(ŝ), 正解テキスト)

データ: 生成モデルは TAPS train で学習済みで、train での出力はほぼ正解の丸暗記になるため、教師には使えない。
そこで TAPS dev（10話者）の融合A の出力（保存済み）を教師にし、8話者で学習、残り2話者で選択する（script 79・80 と同じ分け方）。
test は学習にも選択にも使わない。

使い方（GPU PC、venv）
    python scripts/89_distill_fusion.py --tag dfuse_l0 --lam 0
    python scripts/89_distill_fusion.py --tag dfuse_l1 --lam 1
評価は script 66（checkpoints/<tag>/best.th を SE として読む）
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
from dataset_cfg import RAW_DIR as TAPS_DIR, SE_WAV

s64 = import_module('64_retranscribe_all')
s79 = import_module('79_whisper_on_real_outputs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--init', default='ce10.0', help='初期値の SE（script 64 の SE_CKPTS の名前か checkpoints/ の名前）')
    ap.add_argument('--teacher', default='fm_avg4_mavg_ce10', help='教師にする保存済みの出力（se_wav/<teacher>/dev）')
    ap.add_argument('--lam', type=float, default=0.0, help='Whisper-small の CE 損失の重み（0 なら純粋な蒸留）')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--max_sec', type=float, default=12.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    s51 = import_module('51_train_se_ce_loss')
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    ck = BASE_DIR / 'checkpoints' / args.tag
    ck.mkdir(parents=True, exist_ok=True)
    init = s64.SE_CKPTS.get(args.init, BASE_DIR / 'checkpoints' / args.init / 'best.th')
    se = s64.load_se(init); se.train()
    stft_loss = s51.MultiResolutionSTFTLoss().to('cuda')
    ce = s79.whisper_ce() if args.lam > 0 else None
    tr, va = s79.dev_split(args.max_sec)
    opt = torch.optim.Adam(se.parameters(), lr=args.lr, betas=(0.9, 0.99))
    print(f'init={init} teacher={args.teacher} lam={args.lam}  train {len(tr)} / val {len(va)} 発話', flush=True)

    def load(rows):
        xs, ts = [], []
        for r in rows:
            name = f"{r['speaker_id']}_{r['sentence_id']}.wav"
            x, _ = sf.read(TAPS_DIR / 'throat' / 'dev' / name, dtype='float32')
            t, _ = sf.read(SE_WAV / args.teacher / 'dev' / name, dtype='float32')
            n = min(len(x), len(t)); xs.append(x[:n]); ts.append(t[:n])
        x, _ = s79.pad_batch(xs); t, _ = s79.pad_batch(ts)
        return x, t, [r['text'] for r in rows]

    def step(rows):
        x, t, texts = load(rows)
        s = se(x).reshape(x.shape[0], -1)[:, :x.shape[-1]]
        dist = F.l1_loss(s, t) + stft_loss(s, t)
        l_ce = ce(s, texts) if ce is not None else torch.zeros((), device=s.device)
        return dist + args.lam * l_ce, dist, l_ce

    def evaluate():
        se.eval(); tot, d_tot, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), args.batch_size):
                b = va[i:i + args.batch_size]
                loss, d, _ = step(b)
                tot += loss.item() * len(b); d_tot += d.item() * len(b); n += len(b)
        se.train()
        return tot / n, d_tot / n

    best, _ = evaluate(); bad = 0
    print(f'val（学習前）={best:.4f}', flush=True)
    torch.save(se.state_dict(), ck / 'best.th')
    for ep in range(1, args.epochs + 1):
        random.shuffle(tr); trl = []
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            loss, _, _ = step(tr[i:i + args.batch_size])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(se.parameters(), 5.0)
            opt.step(); trl.append(loss.item())
        vl, vd = evaluate()
        mark = ' <- best' if vl < best else ''
        print(f'epoch {ep:2d}  train={np.mean(trl):.4f}  val={vl:.4f}（蒸留 {vd:.4f}）{mark}', flush=True)
        if vl < best:
            best, bad = vl, 0
            torch.save(se.state_dict(), ck / 'best.th')
        else:
            bad += 1
            if bad >= args.patience:
                print('early stopping'); break
    print(f'done. best val={best:.4f} → {ck / "best.th"}')


if __name__ == '__main__':
    main()
