"""
PESQ・STOI・CER 統合可視化
出力: results/figures/
  - triple_metrics_white.png  : 白色ノイズ 3指標比較
  - triple_metrics_pink.png   : ピンクノイズ 3指標比較
  - pesq_vs_cer.png           : PESQ vs CER 散布図
  - delta_all_metrics.png     : Δ3指標まとめ（GTCRNのみ）
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
PESQ_CSV = os.path.join(BASE_DIR, 'results', 'pesq_results.csv')
FIG_DIR  = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SNR_ORDER  = ['-5', '+0', '+5', '+10', '+20']
SNR_LABELS = ['-5 dB', '0 dB', '+5 dB', '+10 dB', '+20 dB']
COLORS = {'no_se': '#2196F3', 'dsp_only': '#FF9800', 'gtcrn': '#4CAF50'}
LABELS = {'no_se': 'No SE', 'dsp_only': 'DSP-only', 'gtcrn': 'GTCRN'}


def load():
    def rd(path):
        with open(path, encoding='utf-8') as f:
            return {r['condition']: r for r in csv.DictReader(f)}
    return rd(SUMMARY), rd(STOI_CSV), rd(PESQ_CSV)


def get_val(d, key, field):
    return float(d[key][field]) if key in d else None


# ── 1. 3指標 SNR折れ線（3段）────────────────────────────────
def plot_triple_metrics(cer_d, stoi_d, pesq_d, noise_type, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    fig.suptitle(f'CER / STOI / PESQ vs SNR  —  {noise_type.capitalize()} Noise',
                 fontsize=13, fontweight='bold', y=0.98)

    metrics = [
        (axes[0], cer_d,  'avg_cer',  'CER ↓ (lower is better)',   (0,   1.35), False),
        (axes[1], stoi_d, 'avg_stoi', 'STOI ↑ (higher is better)', (0.3, 1.05), True),
        (axes[2], pesq_d, 'avg_pesq', 'PESQ ↑ (higher is better)', (0.8, 3.5),  True),
    ]

    for ax, data, field, ylabel, ylim, higher_better in metrics:
        for model in ['no_se', 'dsp_only', 'gtcrn']:
            ys = []
            for snr in SNR_ORDER:
                sc  = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
                key = f'no_se/{noise_type}/{sc}' if model == 'no_se' else f'{model}/{noise_type}/{sc}'
                ys.append(get_val(data, key, field))
            ax.plot(SNR_LABELS, ys, marker='o', color=COLORS[model],
                    label=LABELS[model], linewidth=2, markersize=6)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9, loc='lower right' if higher_better else 'upper right')
        ax.grid(axis='y', alpha=0.3)

        # 矢印で「改善方向」を示す
        direction = '↑ better' if higher_better else '↓ better'
        ax.text(0.01, 0.95 if not higher_better else 0.05, direction,
                transform=ax.transAxes, fontsize=8,
                color='green' if higher_better else 'darkred', va='top')

    axes[2].set_xlabel('SNR', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存: {out_path}')


# ── 2. PESQ vs CER 散布図 ────────────────────────────────────
def plot_pesq_vs_cer(cer_d, pesq_d, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for model in ['no_se', 'dsp_only', 'gtcrn']:
        xs, ys, labels = [], [], []
        for key in pesq_d:
            if 'snr_' not in key: continue
            if model == 'no_se' and not key.startswith('no_se/'): continue
            if model != 'no_se' and not key.startswith(model + '/'): continue
            if key not in cer_d: continue
            xs.append(float(pesq_d[key]['avg_pesq']))
            ys.append(float(cer_d[key]['avg_cer']))

        mrk = {'no_se': 'o', 'dsp_only': 's', 'gtcrn': '^'}[model]
        ax.scatter(xs, ys, c=COLORS[model], marker=mrk, s=70,
                   alpha=0.85, label=LABELS[model], zorder=3)

    # 理想ゾーン注釈
    ax.fill_between([2.5, 4.6], [0, 0], [0.3, 0.3],
                    color='green', alpha=0.07, label='Ideal zone')
    ax.annotate('Ideal:\nHigh PESQ\nLow CER',
                xy=(3.5, 0.15), fontsize=8, color='green',
                ha='center', alpha=0.7)

    # GTCRNパラドックスゾーン
    ax.fill_between([1.5, 3.5], [0.6, 0.6], [1.35, 1.35],
                    color='red', alpha=0.05, label='Paradox zone (GTCRN)')
    ax.annotate('Paradox zone:\nGTCRN — PESQ↑ but CER↑',
                xy=(2.2, 1.05), fontsize=8, color='darkred',
                ha='center', alpha=0.7)

    ax.set_xlabel('PESQ ↑ (higher = better perceptual quality)', fontsize=11)
    ax.set_ylabel('CER ↓ (lower = better ASR)', fontsize=11)
    ax.set_title('PESQ vs CER: Perceptual Quality vs ASR Performance Paradox',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0.9, 4.0)
    ax.set_ylim(0.0, 1.35)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存: {out_path}')


# ── 3. GTCRN Δ3指標まとめ（論文メイン図候補） ───────────────
def plot_delta_all(cer_d, stoi_d, pesq_d, out_path):
    """GTCRNのΔCER・ΔSTOI・ΔPESQを白色/ピンク別に表示"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle(
        'GTCRN vs No SE  —  Δ Metrics per SNR\n'
        'Blue/Green = quality improves,  Red = ASR worsens',
        fontsize=12, fontweight='bold'
    )

    for ax, noise_type in zip(axes, ['white', 'pink']):
        delta_cer, delta_stoi, delta_pesq = [], [], []

        for snr in SNR_ORDER:
            sc  = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            key_se  = f'gtcrn/{noise_type}/{sc}'
            key_ref = f'no_se/{noise_type}/{sc}'

            dc = (get_val(cer_d,  key_se, 'avg_cer')  or 0) - (get_val(cer_d,  key_ref, 'avg_cer')  or 0)
            ds = (get_val(stoi_d, key_se, 'avg_stoi') or 0) - (get_val(stoi_d, key_ref, 'avg_stoi') or 0)
            dp = (get_val(pesq_d, key_se, 'avg_pesq') or 0) - (get_val(pesq_d, key_ref, 'avg_pesq') or 0)

            delta_cer.append(dc)
            delta_stoi.append(ds)
            # PESQを0〜1スケールに正規化して並べる（最大変化幅で割る）
            delta_pesq.append(dp)

        x = np.arange(len(SNR_LABELS))
        w = 0.26

        # PESQ差分をスケール表示（右軸）
        ax2 = ax.twinx()

        b1 = ax.bar(x - w,     delta_cer,  w, color='#C62828', alpha=0.85,
                    label='ΔCER (↑=worse ASR)')
        b2 = ax.bar(x,         delta_stoi, w, color='#1565C0', alpha=0.85,
                    label='ΔSTOI (↑=better quality)')
        b3 = ax2.bar(x + w,    delta_pesq, w, color='#2E7D32', alpha=0.85,
                     label='ΔPESQ (↑=better quality)')

        ax.axhline(0, color='black', linewidth=0.8)
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)

        ax.set_xticks(x); ax.set_xticklabels(SNR_LABELS, fontsize=9)
        ax.set_xlabel('SNR', fontsize=10)
        ax.set_ylabel('ΔCER / ΔSTOI', fontsize=10)
        ax2.set_ylabel('ΔPESQ', fontsize=10, color='#2E7D32')
        ax2.tick_params(axis='y', labelcolor='#2E7D32')
        ax.set_title(f'{noise_type.capitalize()} Noise', fontsize=11)
        ax.grid(axis='y', alpha=0.25)

        # 値ラベル（CERのみ、見やすさのため）
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + 0.005 if h >= 0 else h - 0.018,
                    f'{h:+.2f}', ha='center',
                    va='bottom' if h >= 0 else 'top',
                    fontsize=7, color='#C62828')

        # 凡例まとめ
        lines = [b1, b2, b3]
        labs  = ['ΔCER ↑=worse ASR', 'ΔSTOI ↑=better', 'ΔPESQ ↑=better']
        ax.legend(lines, labs, fontsize=8, loc='upper left')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存: {out_path}')


def main():
    cer_d, stoi_d, pesq_d = load()
    print('可視化開始...')
    plot_triple_metrics(cer_d, stoi_d, pesq_d, 'white',
                        os.path.join(FIG_DIR, 'triple_metrics_white.png'))
    plot_triple_metrics(cer_d, stoi_d, pesq_d, 'pink',
                        os.path.join(FIG_DIR, 'triple_metrics_pink.png'))
    plot_pesq_vs_cer(cer_d, pesq_d,
                     os.path.join(FIG_DIR, 'pesq_vs_cer.png'))
    plot_delta_all(cer_d, stoi_d, pesq_d,
                   os.path.join(FIG_DIR, 'delta_all_metrics.png'))
    print('全グラフ完了')


if __name__ == '__main__':
    main()
