"""
STOI vs CER 散布図 + 乖離ギャップ図
出力: results/figures/
  - scatter_stoi_vs_cer.png     : STOI vs CER 散布図（全条件）
  - gap_stoi_cer_white.png      : STOI改善量 vs CER悪化量（白色ノイズ）
  - gap_stoi_cer_pink.png       : STOI改善量 vs CER悪化量（ピンクノイズ）
"""
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
SUMMARY    = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV   = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return {r['condition']: r for r in csv.DictReader(f)}


# ── 1. STOI vs CER 散布図 ────────────────────────────────────
def plot_scatter(cer_map, stoi_map, out_path):
    STYLE = {
        'no_se':    dict(marker='o', color='#2196F3', label='No SE'),
        'dsp_only': dict(marker='s', color='#FF9800', label='DSP-only'),
        'gtcrn':    dict(marker='^', color='#4CAF50', label='GTCRN'),
        'baseline': dict(marker='*', color='black',   label='Baseline', s=120),
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    # ノイズ条件をモデル別にプロット
    for cond, stoi_r in stoi_map.items():
        if cond not in cer_map:
            continue
        stoi_val = float(stoi_r['avg_stoi'])
        cer_val  = float(cer_map[cond]['avg_cer'])

        if cond.startswith('no_se/'):
            st = STYLE['no_se']
        elif cond.startswith('dsp_only/') and 'snr_' in cond:
            st = STYLE['dsp_only']
        elif cond.startswith('gtcrn/') and 'snr_' in cond:
            st = STYLE['gtcrn']
        else:
            continue  # clean条件は別扱い

        ax.scatter(stoi_val, cer_val,
                   marker=st['marker'], color=st['color'],
                   s=60, alpha=0.8, zorder=3)

    # クリーン条件・ベースライン
    clean_style = [
        ('throat_clean_ref',  None,   'Throat Raw',      'gray',    'D'),
        ('dsp_only/clean',    SUMMARY,'DSP/clean',        '#FF9800', 's'),
        ('gtcrn/clean',       SUMMARY,'GTCRN/clean',      '#4CAF50', '^'),
    ]
    for cond, src, lbl, col, mrk in clean_style:
        if cond in stoi_map:
            stoi_val = float(stoi_map[cond]['avg_stoi'])
        else:
            continue
        if cond in cer_map:
            cer_val = float(cer_map[cond]['avg_cer'])
        elif cond == 'throat_clean_ref':
            cer_val = float(cer_map.get('baseline_throat', {}).get('avg_cer', 0.269))
        else:
            continue
        ax.scatter(stoi_val, cer_val, marker=mrk, color=col,
                   s=120, zorder=5, edgecolors='black', linewidths=0.8)

    # 参照: baseline_acoustic
    if 'baseline_acoustic' in cer_map:
        ax.axhline(float(cer_map['baseline_acoustic']['avg_cer']),
                   color='black', linestyle=':', linewidth=1,
                   label='Acoustic baseline CER')

    # 凡例
    legend_handles = [
        mpatches.Patch(color='#2196F3', label='No SE (noisy)'),
        mpatches.Patch(color='#FF9800', label='DSP-only'),
        mpatches.Patch(color='#4CAF50', label='GTCRN'),
        mpatches.Patch(color='gray',    label='Clean / Baseline'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')

    # 象限注釈
    ax.text(0.35, 0.20, 'High STOI\nLow CER\n(ideal)', fontsize=8,
            color='green', ha='center', alpha=0.6)
    ax.text(0.35, 1.10, 'High STOI\nHigh CER\n(GTCRN paradox)', fontsize=8,
            color='red', ha='center', alpha=0.6)
    ax.axvline(0.7, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('STOI (↑ better)', fontsize=12)
    ax.set_ylabel('CER  (↓ better)', fontsize=12)
    ax.set_title('STOI vs CER — Perceptual Quality vs ASR Performance', fontsize=12)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.0, 1.35)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  保存: {out_path}')


# ── 2. STOI改善量 vs CER変化量 ギャップ図 ────────────────────
def plot_gap(cer_map, stoi_map, noise_type, out_path):
    SNR_ORDER  = ['-5', '+0', '+5', '+10', '+20']
    SNR_LABELS = ['-5 dB', '0 dB', '+5 dB', '+10 dB', '+20 dB']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, model, color, title in [
        (axes[0], 'dsp_only', '#FF9800', 'DSP-only vs No SE'),
        (axes[1], 'gtcrn',    '#4CAF50', 'GTCRN vs No SE'),
    ]:
        stoi_diff, cer_diff = [], []
        for snr in SNR_ORDER:
            snr_cond = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            cond_se  = f'{model}/{noise_type}/{snr_cond}'
            cond_ref = f'no_se/{noise_type}/{snr_cond}'
            if cond_se not in stoi_map or cond_ref not in stoi_map:
                stoi_diff.append(None); cer_diff.append(None)
                continue
            if cond_se not in cer_map or cond_ref not in cer_map:
                stoi_diff.append(None); cer_diff.append(None)
                continue
            sd = float(stoi_map[cond_se]['avg_stoi']) - float(stoi_map[cond_ref]['avg_stoi'])
            cd = float(cer_map[cond_se]['avg_cer'])   - float(cer_map[cond_ref]['avg_cer'])
            stoi_diff.append(sd)
            cer_diff.append(cd)

        x = np.arange(len(SNR_LABELS))
        w = 0.35
        bars_stoi = ax.bar(x - w/2, stoi_diff, w,
                           label='ΔSTOI (SE − NoSE)', color='#1565C0', alpha=0.8)
        bars_cer  = ax.bar(x + w/2, cer_diff,  w,
                           label='ΔCER  (SE − NoSE)', color='#C62828', alpha=0.8)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(SNR_LABELS, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Delta (SE - No SE)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # 符号注釈
        for bar in bars_stoi:
            h = bar.get_height()
            if h is not None:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005 if h >= 0 else h - 0.015,
                        f'{h:+.2f}', ha='center', va='bottom' if h >= 0 else 'top',
                        fontsize=7, color='#1565C0')
        for bar in bars_cer:
            h = bar.get_height()
            if h is not None:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005 if h >= 0 else h - 0.015,
                        f'{h:+.2f}', ha='center', va='bottom' if h >= 0 else 'top',
                        fontsize=7, color='#C62828')

    fig.suptitle(
        f'SE Effect on STOI vs CER — {noise_type.capitalize()} Noise\n'
        f'Blue↑ = STOI improved,  Red↑ = CER worsened (bad)',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  保存: {out_path}')


def main():
    cer_map  = load_csv(SUMMARY)
    stoi_map = load_csv(STOI_CSV)

    print('STOI vs CER 可視化開始...')
    plot_scatter(cer_map, stoi_map,
                 os.path.join(FIG_DIR, 'scatter_stoi_vs_cer.png'))
    plot_gap(cer_map, stoi_map, 'white',
             os.path.join(FIG_DIR, 'gap_stoi_cer_white.png'))
    plot_gap(cer_map, stoi_map, 'pink',
             os.path.join(FIG_DIR, 'gap_stoi_cer_pink.png'))
    print('完了')


if __name__ == '__main__':
    main()
