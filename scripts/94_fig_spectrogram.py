"""論文用のスペクトログラム図: 喉マイク／生成×4／融合／気導マイク（TAPS test の 1 発話）

生成×4 は script 75 が保存した音声（data/processed/se_wav/fm_taps_avg4/test）。融合は保存済みの
data/processed/se_wav/fm_avg4_mavg_ce10/test を使い、無ければ CE-SE をその場で通して script 66 の mag_avg で作る

GPU PC には日本語フォントが無いので、2 段に分ける:
  GPU PC: python scripts/94_fig_spectrogram.py calc --utt p00_u00 --t0 0 --dur 4   → results/figures/spectrogram_gen_fusion.npz
  Mac:    python3 scripts/94_fig_spectrogram.py plot                               → results/figures/spectrogram_gen_fusion.png
"""
import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
NPZ = BASE / 'results/figures/spectrogram_gen_fusion.npz'
PNG = BASE / 'results/figures/spectrogram_gen_fusion.png'
PNG_WIDE = BASE / 'results/figures/spectrogram_gen_fusion_wide.png'   # 2 段幅、融合 − 生成×4 の差の段つき（paper_asj7）
SR = 16000
NAMES = ['喉マイク（処理なし）', '生成×4', '融合', '気導マイク（参考）']


def calc(args):
    import soundfile as sf
    import torch
    sys.path.insert(0, str(BASE / 'scripts'))
    s66 = import_module('66_lambda_cross_asr')

    def read(p):
        w, sr = sf.read(p, dtype='float32')
        assert sr == SR, (p, sr)
        return w.mean(axis=1) if w.ndim > 1 else w

    throat = read(BASE / f'data/raw/taps/throat/test/{args.utt}.wav')
    air = read(BASE / f'data/raw/taps/acoustic/test/{args.utt}.wav')
    gen = read(s66.SE_WAV / 'fm_taps_avg4' / 'test' / f'{args.utt}.wav')
    saved = s66.SE_WAV / 'fm_avg4_mavg_ce10' / 'test' / f'{args.utt}.wav'
    if saved.exists():   # 評価に使った融合の音声が保存されていればそれを使う
        fused = read(saved)
    else:
        ce_se = s66.load_any_se('ce10.0')
        with torch.no_grad():
            ce = ce_se(torch.from_numpy(throat).unsqueeze(0).to(s66.DEVICE)).squeeze().cpu().numpy().astype(np.float32)
        n = min(len(gen), len(ce))
        fused = s66.mag_avg(gen[:n], ce[:n])

    a, b = int(args.t0 * SR), int((args.t0 + args.dur) * SR)
    specs = []
    for w in [throat, gen, fused, air]:
        seg = torch.from_numpy(np.ascontiguousarray(w[a:b]))
        S = torch.stft(seg, 512, 128, window=torch.hann_window(512), return_complex=True).abs().numpy()
        specs.append(20 * np.log10(S + 1e-6))
    np.savez(NPZ, specs=np.stack(specs), t0=args.t0, utt=args.utt)
    print('→', NPZ)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(NPZ)
    specs, t0 = d['specs'], float(d['t0'])
    plt.rcParams['font.family'] = 'Hiragino Sans'
    fig, axes = plt.subplots(len(specs), 1, figsize=(3.4, 4.6), sharex=True)
    for ax, name, db in zip(axes, NAMES, specs):
        dur = (db.shape[1] - 1) * 128 / SR
        ax.imshow(db, origin='lower', aspect='auto', cmap='magma', vmin=db.max() - 70, vmax=db.max(),
                  extent=[t0, t0 + dur, 0, SR / 2000])
        ax.axhline(4, color='w', lw=0.5, ls='--')
        ax.set_ylabel('kHz', fontsize=7)
        ax.set_yticks([0, 4, 8])
        ax.tick_params(labelsize=7)
        ax.text(0.01, 0.95, name, transform=ax.transAxes, va='top', fontsize=7, color='w')
    axes[-1].set_xlabel('時間 (s)', fontsize=7)
    fig.tight_layout(pad=0.2, h_pad=0.2)
    fig.savefig(PNG, dpi=300)
    print('→', PNG)


def diff_masked(gen, fused, floor_db=50):
    """融合 − 生成×4（dB）。両方とも小さい（各図の最大から floor_db 以上低い）ビンは 0 にして、無音の揺らぎを隠す"""
    d = fused - gen
    low = np.maximum(gen, fused) < max(gen.max(), fused.max()) - floor_db
    d[low] = 0
    return d


def plot_wide(args):
    """2 段幅の版: 4 つの振幅スペクトログラムに、融合 − 生成×4 の差（dB）の段を加える"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(NPZ)
    specs, t0 = d['specs'], float(d['t0'])
    plt.rcParams['font.family'] = 'Hiragino Sans'
    rows = [(NAMES[0], specs[0]), (NAMES[1], specs[1]), (NAMES[2], specs[2]),
            ('融合 − 生成×4（dB）', diff_masked(specs[1], specs[2])), (NAMES[3], specs[3])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.9, 4.6), sharex=True)
    dur = (specs.shape[2] - 1) * 128 / SR
    for k, (ax, (name, db)) in enumerate(zip(axes, rows)):
        if k == 3:
            im = ax.imshow(db, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-10, vmax=10,
                           extent=[t0, t0 + dur, 0, SR / 2000])
            cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
            cb.ax.tick_params(labelsize=7)
            color = 'k'
        else:
            im = ax.imshow(db, origin='lower', aspect='auto', cmap='magma', vmin=db.max() - 70, vmax=db.max(),
                           extent=[t0, t0 + dur, 0, SR / 2000])
            cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
            cb.ax.tick_params(labelsize=7)
            color = 'w'
        ax.axhline(4, color=color, lw=0.5, ls='--')
        ax.set_ylabel('kHz', fontsize=8)
        ax.set_yticks([0, 4, 8])
        ax.tick_params(labelsize=8)
        box = dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8) if k == 3 else None
        ax.text(0.005, 0.93, name, transform=ax.transAxes, va='top', fontsize=8, color=color, bbox=box)
    axes[-1].set_xlabel('時間 (s)', fontsize=8)
    fig.tight_layout(pad=0.2, h_pad=0.2)
    fig.savefig(PNG_WIDE, dpi=300)
    print('→', PNG_WIDE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['calc', 'plot', 'plot_wide'])
    ap.add_argument('--utt', default='p00_u00')
    ap.add_argument('--t0', type=float, default=0.0, help='表示の開始（秒）')
    ap.add_argument('--dur', type=float, default=4.0, help='表示の長さ（秒）')
    args = ap.parse_args()
    {'calc': calc, 'plot': plot, 'plot_wide': plot_wide}[args.stage](args)


if __name__ == '__main__':
    main()
