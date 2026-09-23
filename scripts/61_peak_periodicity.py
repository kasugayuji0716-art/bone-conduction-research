"""
スクリプト61: CE-aware SE出力の櫛状ピークの周期性検証
仮説: CE SEの出力に約250 Hz間隔のピークが出る。
      SE-Conformerの最深層フレームレート（16k×resample 4 / stride 4^4 = 250 Hz）と一致
      → ConvTranspose由来のアーティファクト

検証内容:
  1. 各条件の平均パワースペクトル（test 100発話: 10話者×10発話）
  2. 移動中央値で包絡を除去した残差スペクトル
  3. 櫛スコア: 候補間隔Δ(100–600 Hz)ごとに k·Δ 位置の残差平均 → 最大となるΔ
  4. 250 Hz倍数位置の残差（ピーク高さ）を帯域別に集計
  5. λ依存性: λ=0 / 1 / 5 / 10、CE only、Enc L1 も比較

使い方（DNN PC）:
    python scripts/61_peak_periodicity.py
    python scripts/61_peak_periodicity.py --per_speaker 5   # 軽量版
"""

import argparse
import csv, sys, torch, numpy as np, soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter

BASE_DIR = Path(__file__).parent.parent
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)

SR = 16000
N_FFT = 2048          # 7.8 Hz分解能
HOP = 512
MEDIAN_BINS = 21      # 包絡除去用（約164 Hz幅）
CKPT = BASE_DIR / 'checkpoints'
CONDITIONS = [
    ('TAPS pretrained', BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
    ('CE λ=0 (recon only)', CKPT / 'ce_v2_lambda_0.0' / 'best.th'),
    ('CE λ=1', CKPT / 'ce_v2_lambda_1.0' / 'best.th'),
    ('CE λ=5', CKPT / 'ce_v2_lambda_5.0' / 'best.th'),
    ('CE λ=10', CKPT / 'ce_v2_lambda_10.0' / 'best.th'),
    ('CE only λ=10', CKPT / 'ce_v2_only_lambda_10.0' / 'best.th'),
    ('Enc L1 λ=5', CKPT / 'enc_v2_lambda_5.0' / 'best.th'),
]


def load_se(ckpt_path, device):
    from models.seconformer import seconformer
    model = seconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.to(device).eval()


def power_spectrum_sum(wav):
    """フレームごとのパワースペクトルの和とフレーム数を返す"""
    frames = np.lib.stride_tricks.sliding_window_view(wav, N_FFT)[::HOP]
    S = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=-1)) ** 2
    return S.sum(axis=0), S.shape[0]


def residual_db(mean_power):
    L = 10 * np.log10(np.maximum(mean_power, 1e-12))
    return L - median_filter(L, size=MEDIAN_BINS, mode='nearest')


def comb_score(resid, freqs, lo, hi, spacings):
    """各間隔Δについて、帯域[lo,hi]内の k·Δ 位置（±1 bin の最大）の残差平均"""
    df = freqs[1] - freqs[0]
    scores = []
    for d in spacings:
        ks = np.arange(np.ceil(lo / d), np.floor(hi / d) + 1)
        vals = []
        for f in ks * d:
            b = int(round(f / df))
            vals.append(resid[max(b - 1, 0):b + 2].max())
        scores.append(np.mean(vals) if vals else np.nan)
    return np.array(scores)


def peak_height(resid, freqs, spacing, lo, hi):
    return comb_score(resid, freqs, lo, hi, [spacing])[0]


def offgrid_height(resid, freqs, spacing, lo, hi):
    """対照: 格子の中間 (k+0.5)·Δ 位置の残差平均"""
    df = freqs[1] - freqs[0]
    fs = (np.arange(np.ceil(lo / spacing - 0.5), np.floor(hi / spacing - 0.5) + 1) + 0.5) * spacing
    return np.mean([resid[max(int(round(f / df)) - 1, 0):int(round(f / df)) + 2].max() for f in fs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_speaker', type=int, default=10)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = {}
    for label, ckpt in CONDITIONS:
        if ckpt.exists():
            models[label] = load_se(ckpt, device)
            print(f'  {label}: loaded')
        else:
            print(f'  {label}: NOT FOUND ({ckpt})')

    # test から話者ごとに先頭 per_speaker 発話
    by_spk = {}
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid, uid = row['speaker_id'], row['sentence_id']
            t = TAPS_DIR / 'throat' / 'test' / f'{sid}_{uid}.wav'
            a = TAPS_DIR / 'acoustic' / 'test' / f'{sid}_{uid}.wav'
            if t.exists() and a.exists():
                by_spk.setdefault(sid, []).append((t, a))
    samples = [p for sid in sorted(by_spk) for p in by_spk[sid][:args.per_speaker]]
    print(f'\n{len(samples)} utterances from {len(by_spk)} speakers')

    labels = ['Acoustic', 'Throat (No SE)'] + list(models)
    acc = {k: np.zeros(N_FFT // 2 + 1) for k in labels}
    cnt = {k: 0 for k in labels}

    for i, (t_path, a_path) in enumerate(samples):
        t_wav, _ = sf.read(t_path, dtype='float32')
        a_wav, _ = sf.read(a_path, dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)
        wavs = {'Acoustic': a_wav, 'Throat (No SE)': t_wav}
        x = torch.from_numpy(t_wav).unsqueeze(0).to(device)
        with torch.no_grad():
            for label, m in models.items():
                wavs[label] = m(x).squeeze().cpu().numpy()
        for k, w in wavs.items():
            s, n = power_spectrum_sum(w)
            acc[k] += s
            cnt[k] += n
        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{len(samples)}')

    freqs = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    spacings = np.arange(100, 601, 1.0)
    resid = {k: residual_db(acc[k] / cnt[k]) for k in labels}

    rows = []
    print(f'\n{"condition":<22} {"best Δ(4-8k)":>12} {"peak@250 4-8k":>14} '
          f'{"peak@250 0-4k":>14} {"off-grid 4-8k":>14}')
    for k in labels:
        sc = comb_score(resid[k], freqs, 4000, 7900, spacings)
        best = spacings[np.nanargmax(sc)]
        h_hi = peak_height(resid[k], freqs, 250, 4000, 7900)
        h_lo = peak_height(resid[k], freqs, 250, 250, 3900)
        # 対照: 250 Hz格子から半周期ずらした位置（125 Hzオフセット）
        off = offgrid_height(resid[k], freqs, 250, 4000, 7900)
        rows.append(dict(condition=k, best_spacing_hz=best,
                         peak250_4_8k_db=round(h_hi, 2),
                         peak250_0_4k_db=round(h_lo, 2),
                         offgrid_4_8k_db=round(off, 2)))
        print(f'{k:<22} {best:>10.0f}Hz {h_hi:>12.2f}dB {h_lo:>12.2f}dB {off:>12.2f}dB')

    out_csv = BASE_DIR / 'results' / 'peak_periodicity.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {out_csv}')

    # 図: 残差スペクトル（TAPS vs CE λ=10）と櫛スコア曲線
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    ax = axes[0]
    for k, c in [('Acoustic', 'green'), ('TAPS pretrained', 'blue'), ('CE λ=10', 'red')]:
        if k in resid:
            ax.plot(freqs / 1000, resid[k], color=c, lw=0.8, label=k)
    for f0 in np.arange(250, 8000, 250):
        ax.axvline(f0 / 1000, color='gray', lw=0.3, alpha=0.5)
    ax.set_xlim(0, 8)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Residual (dB)')
    ax.set_title(f'Detrended mean spectrum ({len(samples)} test utterances); gray = 250 Hz grid')
    ax.legend()

    ax = axes[1]
    for k in labels:
        if k in ('Throat (No SE)',):
            continue
        ax.plot(spacings, comb_score(resid[k], freqs, 4000, 7900, spacings), lw=1, label=k)
    ax.axvline(250, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Comb spacing Δ (Hz)')
    ax.set_ylabel('Mean residual at k·Δ, 4–8 kHz (dB)')
    ax.set_title('Comb score')
    ax.legend(fontsize=8)

    out_png = BASE_DIR / 'results' / 'figures' / 'peak_periodicity.png'
    fig.savefig(out_png, dpi=150)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
