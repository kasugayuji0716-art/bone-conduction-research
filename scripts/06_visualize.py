"""
summary.csv から可視化を行う
出力: results/figures/
  - snr_vs_cer_white.png  : SNR vs CER（白色ノイズ）
  - snr_vs_cer_pink.png   : SNR vs CER（ピンクノイズ）
  - heatmap_cer.png       : 全条件ヒートマップ
  - bar_clean.png         : クリーン条件バーグラフ
"""
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
FIG_DIR  = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SNR_ORDER = ['-5', '+0', '+5', '+10', '+20']
SNR_LABELS = ['-5 dB', '0 dB', '+5 dB', '+10 dB', '+20 dB']
COLORS = {'no_se': '#2196F3', 'dsp_only': '#FF9800', 'gtcrn': '#4CAF50'}
LABELS = {'no_se': 'No SE (Noisy)', 'dsp_only': 'DSP-only', 'gtcrn': 'GTCRN'}


def load_summary():
    rows = []
    with open(SUMMARY, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['avg_cer'] = float(r['avg_cer'])
            rows.append(r)
    return rows


# ── 1. SNR vs CER 折れ線グラフ ──────────────────────────────
def plot_snr_vs_cer(rows, noise_type, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    for model in ['no_se', 'dsp_only', 'gtcrn']:
        ys = []
        for snr in SNR_ORDER:
            match = [r for r in rows
                     if r['noise_type'] == noise_type
                     and r['snr_db'] == snr
                     and r['condition'].startswith(model + '/')]
            ys.append(match[0]['avg_cer'] if match else None)
        ax.plot(SNR_LABELS, ys, marker='o', label=LABELS[model],
                color=COLORS[model], linewidth=2)

    # ベースライン横線
    baselines = {r['condition']: r['avg_cer'] for r in rows}
    ax.axhline(baselines.get('baseline_acoustic', None),
               color='black', linestyle='--', linewidth=1, label='Baseline Acoustic')
    ax.axhline(baselines.get('baseline_throat', None),
               color='gray', linestyle='--', linewidth=1, label='Baseline Throat')

    ax.set_xlabel('SNR', fontsize=12)
    ax.set_ylabel('CER', fontsize=12)
    ax.set_title(f'SNR vs CER — {noise_type.capitalize()} Noise', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.4)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  保存: {out_path}')


# ── 2. ヒートマップ ──────────────────────────────────────────
def plot_heatmap(rows, out_path):
    # 行ラベルの定義（表示順）
    row_labels = (
        ['baseline_acoustic', 'baseline_throat',
         'dsp_only/clean', 'gtcrn/clean'] +
        [f'no_se/{nt}/snr_{snr}dB'
         for nt in ['white', 'pink']
         for snr in ['-5', '+0', '+5', '+10', '+20']] +
        [f'{m}/{nt}/snr_{snr}dB'
         for m in ['dsp_only', 'gtcrn']
         for nt in ['white', 'pink']
         for snr in ['-5', '+0', '+5', '+10', '+20']]
    )

    cer_map = {r['condition']: r['avg_cer'] for r in rows}
    values = np.array([cer_map.get(l, np.nan) for l in row_labels])

    fig, ax = plt.subplots(figsize=(4, 12))
    im = ax.imshow(values.reshape(-1, 1), aspect='auto',
                   cmap='RdYlGn_r', vmin=0, vmax=1.2)

    ax.set_xticks([])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i, v in enumerate(values):
        ax.text(0, i, f'{v:.3f}', ha='center', va='center',
                fontsize=7, color='black')

    plt.colorbar(im, ax=ax, label='CER', fraction=0.08)
    ax.set_title('CER Heatmap\n(All Conditions)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存: {out_path}')


# ── 3. クリーン条件バーグラフ ────────────────────────────────
def plot_bar_clean(rows, out_path):
    conditions = ['baseline_acoustic', 'baseline_throat',
                  'dsp_only/clean', 'gtcrn/clean']
    labels     = ['Acoustic\n(Reference)', 'Throat\n(Raw)',
                  'Throat\n+DSP', 'Throat\n+GTCRN']
    colors     = ['#1565C0', '#78909C', '#FF9800', '#4CAF50']

    cer_map = {r['condition']: r['avg_cer'] for r in rows}
    values = [cer_map[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('CER', fontsize=12)
    ax.set_title('CER — Clean Conditions', fontsize=13)
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  保存: {out_path}')


def main():
    rows = load_summary()
    print('可視化開始...')
    plot_snr_vs_cer(rows, 'white', os.path.join(FIG_DIR, 'snr_vs_cer_white.png'))
    plot_snr_vs_cer(rows, 'pink',  os.path.join(FIG_DIR, 'snr_vs_cer_pink.png'))
    plot_heatmap(rows, os.path.join(FIG_DIR, 'heatmap_cer.png'))
    plot_bar_clean(rows, os.path.join(FIG_DIR, 'bar_clean.png'))
    print('全グラフ完了')


if __name__ == '__main__':
    main()
