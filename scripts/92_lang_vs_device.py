"""
スクリプト92: TAPS → VibraVox の転移の失敗は、言語の違いか、機器（喉マイク）の違いか

気導マイクの音声に「喉マイクの周波数特性」をかけて疑似的な喉マイク音声を作り、言語と機器を別々に入れ替える。
そこに TAPS で適応した生成モデル（G_TAPS、韓国語・TAPS の喉マイク）と VibraVox で適応した生成モデル（G_VBX、仏語・VibraVox の喉マイク）を
学習なしで通し、どちらが効くかを見る。

  入力（BCR_DATASET の名前）            言語   機器の特性     G_TAPS から見て        G_VBX から見て
  taps                （TAPS test、本物）  韓国語 TAPS（本物）   一致                    言語・機器とも不一致
  taps_simtaps        TAPS 気導 × H_TAPS   韓国語 TAPS（疑似）   一致（疑似化の妥当性）   言語・機器とも不一致
  taps_simvbx         TAPS 気導 × H_VBX    韓国語 VibraVox（疑似）機器だけ不一致          言語だけ不一致
  vibravox_devreal    VibraVox dev、本物   仏語   VibraVox（本物）言語・機器とも不一致      一致
  vibravox_simvbx     VibraVox 気導 × H_VBX 仏語  VibraVox（疑似）言語・機器とも不一致      一致（疑似化の妥当性）
  vibravox_simtaps    VibraVox 気導 × H_TAPS 仏語 TAPS（疑似）   言語だけ不一致           機器だけ不一致

  H（機器の特性）: 同じ発話の喉マイクと気導マイクの長時間平均パワースペクトルの比（STFT 512/128、各データセットの train、
                  発話内で最大から 40 dB 以内のフレーム）。TAPS 側は 4 kHz 以上を 0 にする（8 kHz 収録）。
                  音量は各データセットの本物の喉マイクの RMS の中央値にそろえる
  限界: 平均的な周波数特性（線形・時不変）だけを真似る。喉マイク特有の非線形な歪み・時間変化・雑音は入らない。
        だから「一致（疑似化の妥当性）」のマスで本物並みに効くかを先に確かめ、疑似化が使えるかを判定する
  VibraVox の気導マイクは train と dev にしかないので、仏語側は dev（生成モデルの学習には使っていない）の 3 発話に 1 つ（841 発話）を使う

使い方（GPU PC、CPU で可）
    python scripts/92_lang_vs_device.py channel
    python scripts/92_lang_vs_device.py make
その後、各 BCR_DATASET で script 75 infer（--tag fm_taps / fm_vbx、--n_avg 1、--out g_taps / g_vbx）と script 66 asr --set langdev
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).parent.parent
RAW = BASE_DIR / 'data' / 'raw'
OUT = BASE_DIR / 'results' / 'lang_vs_device'
N_FFT, HOP, SR = 512, 128, 16000
WIN = torch.hann_window(N_FFT)
HZ = np.fft.rfftfreq(N_FFT, 1 / SR)


def stft(x):
    return torch.stft(torch.from_numpy(x), N_FFT, HOP, window=WIN, return_complex=True)


def pairs(ds, split, n, seed=0):
    rows = list(csv.DictReader(open(RAW / ds / f'metadata_{split}.csv', encoding='utf-8')))
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        name = f"{r['speaker_id']}_{r['sentence_id']}.wav"
        t, a = RAW / ds / 'throat' / split / name, RAW / ds / 'acoustic' / split / name
        if t.exists() and a.exists():
            out.append((t, a))
        if len(out) >= n:
            break
    return out


def channel(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for ds, n in [('taps', 1000), ('vibravox', 2000)]:
        pt, pa, rms = np.zeros(N_FFT // 2 + 1), np.zeros(N_FFT // 2 + 1), []
        for t, a in pairs(ds, 'train', n):
            xt, _ = sf.read(t, dtype='float32')
            xa, _ = sf.read(a, dtype='float32')
            m = min(len(xt), len(xa))
            T, A = stft(xt[:m]).abs().numpy() ** 2, stft(xa[:m]).abs().numpy() ** 2
            e = 10 * np.log10(A.sum(0) + 1e-12)
            keep = e >= e.max() - 40
            pt += T[:, keep].sum(1); pa += A[:, keep].sum(1)
            rms.append(np.sqrt(np.mean(xt ** 2)))
        g = 10 * np.log10((pt + 1e-12) / (pa + 1e-12))          # 喉 / 気導 の長時間平均パワー比（dB）
        res[ds] = dict(gain_db=g, rms=float(np.median(rms)), n=len(rms))
        bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)]
        print(ds, f'n={len(rms)} 喉マイクの RMS 中央値 {res[ds]["rms"]:.4f}')
        print('  喉/気導 [dB]: ' + ', '.join(f'{lo/1000:g}-{hi/1000:g}k {g[(HZ >= lo) & (HZ < hi)].mean():+.1f}' for lo, hi in bands))
    np.savez(OUT / 'channels.npz', taps_gain_db=res['taps']['gain_db'], vibravox_gain_db=res['vibravox']['gain_db'],
             taps_rms=res['taps']['rms'], vibravox_rms=res['vibravox']['rms'], hz=HZ)
    print('saved →', OUT / 'channels.npz')


def apply_channel(x, gain_db, cut4k, rms):
    X = stft(x)
    g = torch.from_numpy(10 ** (gain_db / 20)).float()[:, None]
    if cut4k:
        g[torch.from_numpy(HZ >= 4000)] = 0
    y = torch.istft(X * g, N_FFT, HOP, window=WIN, length=len(x)).numpy()
    return (y * rms / (np.sqrt(np.mean(y ** 2)) + 1e-9)).astype(np.float32)


def make(args):
    ch = np.load(OUT / 'channels.npz')
    H = {'taps': (ch['taps_gain_db'], True, float(ch['taps_rms'])), 'vbx': (ch['vibravox_gain_db'], False, float(ch['vibravox_rms']))}
    vbx_dev = list(csv.DictReader(open(RAW / 'vibravox' / 'metadata_dev.csv', encoding='utf-8')))[::3]
    jobs = [('taps_simtaps', 'taps', 'test', None, 'taps'), ('taps_simvbx', 'taps', 'test', None, 'vbx'),
            ('vibravox_simtaps', 'vibravox', 'dev', vbx_dev, 'taps'), ('vibravox_simvbx', 'vibravox', 'dev', vbx_dev, 'vbx'),
            ('vibravox_devreal', 'vibravox', 'dev', vbx_dev, None)]
    for name, src, split, rows, h in jobs:
        rows = rows if rows is not None else list(csv.DictReader(open(RAW / src / f'metadata_{split}.csv', encoding='utf-8')))
        dst = RAW / name / 'throat' / 'test'
        dst.mkdir(parents=True, exist_ok=True)
        kept = []
        for r in rows:
            fn = f"{r['speaker_id']}_{r['sentence_id']}.wav"
            srcf = RAW / src / ('throat' if h is None else 'acoustic') / split / fn
            if not srcf.exists():
                continue
            if not (dst / fn).exists():
                x, sr = sf.read(srcf, dtype='float32')
                y = x if h is None else apply_channel(x, *H[h])
                sf.write(dst / fn, y, SR)
            kept.append(r)
        with open(RAW / name / 'metadata_test.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(kept[0]))
            w.writeheader(); w.writerows(kept)
        print(f'{name}: {len(kept)} 発話 → {dst}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['channel', 'make'])
    args = ap.parse_args()
    {'channel': channel, 'make': make}[args.stage](args)


if __name__ == '__main__':
    main()
