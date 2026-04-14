"""
スペクトル分析：GTCRN適用前後の周波数特性変化
- 平均パワースペクトル（Before/After/Difference）
- 代表サンプルのスペクトログラム比較
- アーティファクト成分の可視化

出力: results/figures/spectrum_analysis.png
"""

import sys, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
import torch

BASE_DIR  = Path(__file__).parent.parent
TAPS_DIR  = BASE_DIR / 'data' / 'raw' / 'taps'
GTCRN_DIR = BASE_DIR / 'gtcrn'
FIG_DIR   = BASE_DIR / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
SR = 16000


# ── GTCRN ────────────────────────────────────────────────────
def load_gtcrn():
    sys.path.insert(0, str(GTCRN_DIR))
    from gtcrn import GTCRN
    model = GTCRN().eval()
    ckpt = torch.load(
        GTCRN_DIR / 'checkpoints' / 'model_trained_on_dns3.tar',
        map_location='cpu', weights_only=True
    )
    model.load_state_dict(ckpt['model'])
    return model


def apply_gtcrn(model, wav):
    x   = torch.from_numpy(wav.astype('float32'))
    win = torch.hann_window(512).pow(0.5)
    stft_c  = torch.stft(x, 512, 256, 512, win, return_complex=True)
    stft_in = torch.view_as_real(stft_c)
    with torch.no_grad():
        stft_out = model(stft_in[None])[0]
    out_c    = torch.view_as_complex(stft_out.contiguous())
    enhanced = torch.istft(out_c, 512, 256, 512, win)
    return enhanced.cpu().numpy().astype('float32')


# ── スペクトル計算 ────────────────────────────────────────────
def power_spectrum(wav, n_fft=2048):
    """平均パワースペクトル（dB）"""
    n = len(wav)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    frames = []
    hop = n_fft // 2
    for i in range(0, n - n_fft, hop):
        frame = wav[i:i + n_fft] * np.hanning(n_fft)
        ps = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(ps)
    avg = np.mean(frames, axis=0)
    avg_db = 10 * np.log10(avg + 1e-10)
    return freqs, avg_db


def spectrogram(wav, n_fft=512, hop=256):
    """スペクトログラム（dB）"""
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, len(wav) - n_fft, hop):
        frame = wav[i:i + n_fft] * win
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    S = np.array(frames).T
    return 10 * np.log10(S + 1e-10)


# ── メイン ────────────────────────────────────────────────────
def main():
    # サンプル収集（p00 全50件）
    samples = []
    for meta in [TAPS_DIR / 'metadata_test.csv', TAPS_DIR / 'metadata.csv']:
        if not meta.exists():
            continue
        with open(meta, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['speaker_id'] != 'p00':
                    continue
                sid  = row['sentence_id']
                p1   = TAPS_DIR / 'throat' / 'test' / f'p00_{sid}.wav'
                p2   = TAPS_DIR / 'throat' / f'p00_{sid}.wav'
                path = p1 if p1.exists() else (p2 if p2.exists() else None)
                if path:
                    samples.append({'sid': sid, 'path': path})
        if samples:
            break

    print(f'サンプル数: {len(samples)}')

    print('GTCRNロード中...')
    gtcrn = load_gtcrn()

    # 全サンプルのスペクトルを蓄積
    ps_before_all, ps_after_all, ps_diff_all = [], [], []
    ref_sample = None  # 代表サンプル（最初の1件）

    for i, s in enumerate(samples):
        wav, _ = sf.read(str(s['path']), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_se = apply_gtcrn(gtcrn, wav)

        freqs, ps_b = power_spectrum(wav)
        _,     ps_a = power_spectrum(wav_se)
        ps_before_all.append(ps_b)
        ps_after_all.append(ps_a)
        ps_diff_all.append(ps_a - ps_b)

        if ref_sample is None:
            ref_sample = {'wav': wav, 'wav_se': wav_se, 'sid': s['sid']}

        if (i + 1) % 10 == 0:
            print(f'  処理中: {i+1}/{len(samples)}')

    # 平均
    ps_before = np.mean(ps_before_all, axis=0)
    ps_after  = np.mean(ps_after_all,  axis=0)
    ps_diff   = np.mean(ps_diff_all,   axis=0)

    # ── 図作成 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    BLUE   = '#1565C0'
    GREEN  = '#2E7D32'
    RED    = '#C62828'
    ORANGE = '#E65100'
    GRAY   = '#546E7A'

    freq_khz = freqs / 1000

    # ── (A) 平均パワースペクトル：Before vs After ──────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(freq_khz, ps_before, color=BLUE,  lw=2.0, label='Before SE (throat mic raw)')
    ax1.plot(freq_khz, ps_after,  color=GREEN, lw=2.0, label='After SE (GTCRN)', ls='--')
    ax1.set_xlabel('Frequency (kHz)', fontsize=11)
    ax1.set_ylabel('Power (dB)', fontsize=11)
    ax1.set_title('(A)  Average Power Spectrum: Before vs After GTCRN  (p00, n=50)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 8); ax1.legend(fontsize=11)
    ax1.axvspan(0, 0.3,  alpha=0.07, color='gray',   label='_')
    ax1.axvspan(2.5, 8.0, alpha=0.12, color=ORANGE)
    ax1.text(0.15, ax1.get_ylim()[0] + 2, 'HPF\ncut', fontsize=8, color=GRAY, ha='center')
    ax1.text(5.0,  ax1.get_ylim()[0] + 2, 'High-freq deficit\n(throat mic characteristic)',
             fontsize=9, color=ORANGE, ha='center')
    ax1.grid(alpha=0.25)

    # ── (B) 差分スペクトル（アーティファクト） ────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axhline(0, color='black', lw=0.8, ls='-')
    ax2.fill_between(freq_khz, ps_diff, 0,
                     where=(ps_diff > 0), color=RED,   alpha=0.6,
                     label='GTCRN adds energy (artifact)')
    ax2.fill_between(freq_khz, ps_diff, 0,
                     where=(ps_diff < 0), color=BLUE,  alpha=0.6,
                     label='GTCRN removes energy (suppression)')
    ax2.plot(freq_khz, ps_diff, color='black', lw=0.8, alpha=0.5)
    ax2.set_xlabel('Frequency (kHz)', fontsize=11)
    ax2.set_ylabel('ΔPower (dB)', fontsize=11)
    ax2.set_title('(B)  Spectral Difference: After − Before  (artifact = unexpected energy change)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 8)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.25)

    # 最大アーティファクト帯域をアノテーション
    peak_idx  = np.argmax(np.abs(ps_diff[freqs < 4000]))
    peak_freq = freq_khz[peak_idx]
    peak_val  = ps_diff[peak_idx]
    ax2.annotate(f'Peak artifact\n{peak_freq:.1f} kHz',
                 xy=(peak_freq, peak_val),
                 xytext=(peak_freq + 0.5, peak_val + (2 if peak_val > 0 else -2)),
                 fontsize=9, color=RED,
                 arrowprops=dict(arrowstyle='->', color=RED))

    # ── (C) 代表サンプルのスペクトログラム Before ─────────
    ax3 = fig.add_subplot(gs[2, 0])
    sg_b = spectrogram(ref_sample['wav'])
    n_frames = sg_b.shape[1]
    dur = len(ref_sample['wav']) / SR
    im3 = ax3.imshow(sg_b, origin='lower', aspect='auto',
                     extent=[0, dur, 0, SR / 2 / 1000],
                     cmap='magma', vmin=-80, vmax=20)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Frequency (kHz)', fontsize=10)
    ax3.set_title(f'(C)  Spectrogram — Before SE\n({ref_sample["sid"]})',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='dB')

    # ── (D) 代表サンプルのスペクトログラム After ──────────
    ax4 = fig.add_subplot(gs[2, 1])
    sg_a = spectrogram(ref_sample['wav_se'])
    im4 = ax4.imshow(sg_a, origin='lower', aspect='auto',
                     extent=[0, dur, 0, SR / 2 / 1000],
                     cmap='magma', vmin=-80, vmax=20)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Frequency (kHz)', fontsize=10)
    ax4.set_title(f'(D)  Spectrogram — After GTCRN SE\n({ref_sample["sid"]})',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='dB')

    fig.suptitle(
        'Spectral Analysis: GTCRN Applied to Throat Mic Audio\n'
        'Throat mic lacks high-freq components → GTCRN generates artifacts in unexpected frequency regions',
        fontsize=13, fontweight='bold', y=1.01
    )

    out = FIG_DIR / 'spectrum_analysis.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n保存: {out}')

    # ── 数値サマリー ────────────────────────────────────────
    print('\n===== 周波数帯別アーティファクト =====')
    bands = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
    for lo, hi in bands:
        mask  = (freq_khz >= lo) & (freq_khz < hi)
        delta = np.mean(ps_diff[mask])
        sign  = '↑追加' if delta > 0 else '↓削除'
        print(f'  {lo:.1f}–{hi:.1f} kHz:  Δ={delta:+.2f} dB  {sign}')


if __name__ == '__main__':
    main()
