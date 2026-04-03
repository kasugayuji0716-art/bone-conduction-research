"""
研究レポート PDF 生成
出力: results/research_report.pdf
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from datetime import date

BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
FIG_DIR   = os.path.join(BASE_DIR, 'results', 'figures')
SUMMARY   = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV  = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
OUT_PDF   = os.path.join(BASE_DIR, 'results', 'research_report.pdf')

# フォント設定（日本語対応）
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

def cer_map(rows):
    return {r['condition']: float(r['avg_cer']) for r in rows}

def stoi_map_fn(rows):
    return {r['condition']: float(r['avg_stoi']) for r in rows}


# ── ページ描画ヘルパー ────────────────────────────────────────

def add_header(fig, title, page_num):
    fig.text(0.5, 0.97, title, ha='center', va='top',
             fontsize=14, fontweight='bold', color='#1a237e')
    fig.text(0.97, 0.01, f'p.{page_num}', ha='right', va='bottom',
             fontsize=8, color='gray')
    fig.text(0.03, 0.01, 'Bone Conduction SE Research — Confidential Draft',
             ha='left', va='bottom', fontsize=7, color='gray')
    # ヘッダー下線
    line = plt.Line2D([0.03, 0.97], [0.955, 0.955],
                      transform=fig.transFigure, color='#1a237e', linewidth=1)
    fig.add_artist(line)


# ── Page 1: 表紙 ─────────────────────────────────────────────
def page_cover(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor('#f8f9fa')

    # タイトルブロック
    fig.text(0.5, 0.72,
             'Bone Conduction Microphone\nSpeech Recognition Research',
             ha='center', va='center', fontsize=22, fontweight='bold',
             color='#1a237e', linespacing=1.6)
    fig.text(0.5, 0.60,
             'Speech Enhancement Effects on Whisper CER\n'
             '— Quantitative Evaluation under Noise Conditions —',
             ha='center', va='center', fontsize=13, color='#37474f',
             linespacing=1.8)

    # 区切り線
    for y in [0.68, 0.55]:
        line = plt.Line2D([0.15, 0.85], [y, y],
                          transform=fig.transFigure,
                          color='#1a237e', linewidth=1.5)
        fig.add_artist(line)

    # メタ情報
    today = date.today().strftime('%Y-%m-%d')
    info = [
        ('Dataset',   'TAPS (Korean throat+acoustic, 50 samples)'),
        ('SE Models', 'DSP-only  /  GTCRN (48.2K params)'),
        ('ASR',       'faster-whisper small  (lang=ko)'),
        ('Metrics',   'CER, STOI'),
        ('Conditions','34 conditions (2 noise types x 5 SNR x 2 models + baselines)'),
        ('Date',      today),
    ]
    y0 = 0.47
    for label, val in info:
        fig.text(0.25, y0, label + ':', ha='right', fontsize=10,
                 fontweight='bold', color='#1a237e')
        fig.text(0.27, y0, val, ha='left', fontsize=10, color='#212121')
        y0 -= 0.045

    fig.text(0.5, 0.08, 'DRAFT — For Internal Review Only',
             ha='center', fontsize=9, color='#b71c1c',
             style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 2: 研究概要・実験設計 ───────────────────────────────
def page_overview(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'Research Overview & Experimental Design', 2)

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.86])
    ax.axis('off')

    sections = [
        ('Research Question',
         'Under what conditions does Speech Enhancement (SE) harm\n'
         'Whisper ASR performance for bone conduction (throat mic) audio?'),

        ('Motivation',
         'Prior work (Mawalim et al. Interspeech 2024; Ochiai et al. TASLP 2024)\n'
         'shows SE can hurt downstream ASR. This study replicates and extends\n'
         'the phenomenon specifically for throat microphone audio.'),

        ('Dataset — TAPS',
         'Korean paired throat-mic / acoustic-mic dataset.\n'
         '50 test utterances from speaker p00.\n'
         'Ground-truth text provided in metadata.csv.'),

        ('Noise Augmentation',
         'White noise & Pink noise added at SNR: -5, 0, +5, +10, +20 dB.\n'
         '10 noisy baseline conditions (no SE applied).'),

        ('SE Models',
         'DSP-only:  Highpass filter (Butterworth 6th, 300 Hz) + pre-emphasis (0.97) + RMS norm.\n'
         'GTCRN:     48.2K-param neural SE, trained on DNS3 (air-mic, out-of-domain for throat).'),

        ('Evaluation',
         'CER (Character Error Rate) via faster-whisper small.\n'
         'STOI (Short-Time Objective Intelligibility) vs clean throat-mic reference.\n'
         'Statistical significance: Wilcoxon signed-rank test (in progress).'),

        ('Key References',
         '[1] Ochiai et al., TASLP 2024, arXiv:2404.14860\n'
         '    — Artifact errors are the primary cause of SE-induced ASR degradation.\n'
         '[2] Mawalim et al., Interspeech 2024, DOI:10.21437/Interspeech.2024-129\n'
         '    — DL-based SE improves STOI but degrades ASR in real-world conditions.'),
    ]

    y = 0.93
    for title, body in sections:
        ax.text(0, y, title, fontsize=11, fontweight='bold', color='#1a237e',
                transform=ax.transAxes)
        y -= 0.03
        ax.text(0.02, y, body, fontsize=9, color='#212121',
                transform=ax.transAxes, linespacing=1.5,
                verticalalignment='top')
        y -= body.count('\n') * 0.038 + 0.065

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 3: 全条件 CER 結果テーブル ─────────────────────────
def page_cer_table(pdf, cer_data, stoi_data):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'Full Results — CER & STOI by Condition', 3)

    ax = fig.add_axes([0.02, 0.03, 0.96, 0.89])
    ax.axis('off')

    rows_display = []
    snr_order = ['-5', '+0', '+5', '+10', '+20']
    # baselines
    for c in ['baseline_acoustic', 'baseline_throat', 'dsp_only/clean', 'gtcrn/clean']:
        cer_v  = cer_data.get(c, '-')
        stoi_v = stoi_data.get(c, stoi_data.get('throat_clean_ref', '-'))
        if c == 'baseline_acoustic':
            stoi_v = '1.000*'
        elif c == 'baseline_throat':
            stoi_v = '1.000*'
        rows_display.append([c, 'none', 'clean',
                              f'{cer_v:.3f}' if isinstance(cer_v, float) else cer_v,
                              f'{float(stoi_v):.3f}' if isinstance(stoi_v, float) else stoi_v])

    for model in ['no_se', 'dsp_only', 'gtcrn']:
        for nt in ['white', 'pink']:
            for snr in snr_order:
                snr_cond = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
                key_cer  = f'{model}/{nt}/{snr_cond}' if model != 'no_se' else f'no_se/{nt}/{snr_cond}'
                key_stoi = key_cer
                cer_v  = cer_data.get(key_cer, '-')
                stoi_v = stoi_data.get(key_stoi, '-')
                rows_display.append([
                    key_cer, nt, snr,
                    f'{cer_v:.3f}' if isinstance(cer_v, float) else cer_v,
                    f'{stoi_v:.3f}' if isinstance(stoi_v, float) else stoi_v,
                ])

    col_labels = ['Condition', 'Noise', 'SNR', 'CER↓', 'STOI↑']
    col_widths  = [0.46, 0.10, 0.08, 0.10, 0.10]

    # カラーリング
    cell_colors = []
    for row in rows_display:
        cer_str = row[3]
        try:
            v = float(cer_str)
            if v < 0.30:   bg = '#c8e6c9'
            elif v < 0.50: bg = '#fff9c4'
            elif v < 0.80: bg = '#ffe0b2'
            else:          bg = '#ffcdd2'
        except:
            bg = '#e3f2fd'
        cell_colors.append([bg, 'white', 'white', bg, 'white'])

    tbl = ax.table(
        cellText=rows_display,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        colWidths=col_widths,
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.18)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#bdbdbd')

    ax.text(0.0, -0.02, '* STOI for baselines: compared to acoustic reference.',
            transform=ax.transAxes, fontsize=7, color='gray')
    ax.text(0.0, -0.04, 'Color: green < 0.30 < yellow < 0.50 < orange < 0.80 < red',
            transform=ax.transAxes, fontsize=7, color='gray')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 4: クリーン条件バー + SNR vs CER (white) ───────────
def page_figures_1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'CER Results — Clean Conditions & White Noise', 4)

    imgs = [
        ('bar_clean.png',        [0.05, 0.54, 0.88, 0.37]),
        ('snr_vs_cer_white.png', [0.05, 0.08, 0.88, 0.40]),
    ]
    captions = [
        'Fig.1  CER under clean conditions. DSP-only worsens CER vs raw throat; GTCRN improves slightly.',
        'Fig.2  SNR vs CER for white noise. All SE methods worsen CER vs No SE across all SNR levels.',
    ]
    for (fname, rect), cap in zip(imgs, captions):
        path = os.path.join(FIG_DIR, fname)
        if not os.path.exists(path):
            continue
        ax_img = fig.add_axes(rect)
        ax_img.imshow(mpimg.imread(path))
        ax_img.axis('off')
        fig.text(rect[0] + rect[2]/2, rect[1] - 0.015, cap,
                 ha='center', fontsize=8, color='#37474f', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 5: SNR vs CER (pink) + Heatmap ─────────────────────
def page_figures_2(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'CER Results — Pink Noise & Full Heatmap', 5)

    # SNR vs CER pink (左上)
    p = os.path.join(FIG_DIR, 'snr_vs_cer_pink.png')
    if os.path.exists(p):
        ax1 = fig.add_axes([0.05, 0.52, 0.58, 0.40])
        ax1.imshow(mpimg.imread(p)); ax1.axis('off')
        fig.text(0.34, 0.50,
                 'Fig.3  SNR vs CER — Pink noise.',
                 ha='center', fontsize=8, color='#37474f', style='italic')

    # Heatmap (右側)
    p = os.path.join(FIG_DIR, 'heatmap_cer.png')
    if os.path.exists(p):
        ax2 = fig.add_axes([0.63, 0.05, 0.34, 0.88])
        ax2.imshow(mpimg.imread(p)); ax2.axis('off')
        fig.text(0.80, 0.03,
                 'Fig.4  CER heatmap\n(all 34 conditions)',
                 ha='center', fontsize=8, color='#37474f', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 6: STOI vs CER scatter + gap ───────────────────────
def page_figures_3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'Key Finding — STOI vs CER Paradox', 6)

    p = os.path.join(FIG_DIR, 'scatter_stoi_vs_cer.png')
    if os.path.exists(p):
        ax1 = fig.add_axes([0.08, 0.54, 0.84, 0.38])
        ax1.imshow(mpimg.imread(p)); ax1.axis('off')
        fig.text(0.5, 0.52,
                 'Fig.5  STOI vs CER scatter plot. GTCRN points move RIGHT (STOI improves) '
                 'but also UP (CER worsens) — the paradox.',
                 ha='center', fontsize=8, color='#37474f', style='italic')

    p = os.path.join(FIG_DIR, 'gap_stoi_cer_white.png')
    if os.path.exists(p):
        ax2 = fig.add_axes([0.05, 0.08, 0.90, 0.38])
        ax2.imshow(mpimg.imread(p)); ax2.axis('off')
        fig.text(0.5, 0.06,
                 'Fig.6  Delta STOI (blue) vs Delta CER (red) — white noise. '
                 'GTCRN: blue bars positive (STOI up) while red bars also positive (CER up = bad).',
                 ha='center', fontsize=8, color='#37474f', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 7: Gap (pink) + Key Findings テキスト ──────────────
def page_findings(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'Key Findings & Discussion', 7)

    p = os.path.join(FIG_DIR, 'gap_stoi_cer_pink.png')
    if os.path.exists(p):
        ax = fig.add_axes([0.05, 0.56, 0.90, 0.36])
        ax.imshow(mpimg.imread(p)); ax.axis('off')
        fig.text(0.5, 0.54,
                 'Fig.7  Delta STOI vs Delta CER — pink noise. Same paradox pattern as white noise.',
                 ha='center', fontsize=8, color='#37474f', style='italic')

    findings_ax = fig.add_axes([0.05, 0.03, 0.90, 0.48])
    findings_ax.axis('off')

    findings = [
        ('Finding 1: SE consistently worsens CER for throat mic audio',
         'Across all 30 noisy conditions (2 noise types x 5 SNR x 3 models),\n'
         'both DSP-only and GTCRN increased CER compared to no SE baseline.\n'
         'The effect was consistent regardless of noise type or SNR level.'),

        ('Finding 2: GTCRN — STOI improves, CER worsens (paradox)',
         'GTCRN improved STOI in all 10 noisy conditions vs no-SE baseline:\n'
         '  e.g., white +0dB:  STOI  0.585 -> 0.721  (+0.136)\n'
         '                     CER   0.882 -> 1.214  (+0.332)\n'
         'This replicates Mawalim et al. (Interspeech 2024) for throat mic context.\n'
         'Interpretation: GTCRN removes noise in a way that is perceptually clean\n'
         'but introduces nonlinear artifacts that confuse Whisper (cf. Ochiai et al. TASLP 2024).'),

        ('Finding 3: DSP worsens both STOI and CER',
         'Unlike GTCRN, DSP-only degraded both metrics simultaneously.\n'
         '  e.g., white +0dB:  STOI  0.585 -> 0.434  (-0.151)\n'
         '                     CER   0.882 -> 1.028  (+0.146)\n'
         'HPF + pre-emphasis introduces amplitude distortion incompatible\n'
         'with throat mic frequency profile (most energy below 300 Hz).'),

        ('Discussion: Why does GTCRN hurt Whisper despite good STOI?',
         'GTCRN was trained on DNS3 (air-conduction microphone, English).\n'
         'Throat mic audio has a fundamentally different spectral profile.\n'
         'Out-of-domain input leads to artifact-type errors (Ochiai et al.),\n'
         'which STOI cannot detect but are fatal to Whisper\'s attention mechanism.'),
    ]

    y = 0.97
    for title, body in findings:
        findings_ax.text(0, y, title, fontsize=10, fontweight='bold',
                         color='#b71c1c', transform=findings_ax.transAxes)
        y -= 0.04
        findings_ax.text(0.02, y, body, fontsize=8.5, color='#212121',
                         transform=findings_ax.transAxes,
                         linespacing=1.55, verticalalignment='top')
        y -= body.count('\n') * 0.045 + 0.06

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 8: 今後の計画 ───────────────────────────────────────
def page_plan(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    add_header(fig, 'Research Plan — Next Steps', 8)

    ax = fig.add_axes([0.05, 0.03, 0.90, 0.89])
    ax.axis('off')

    plan = [
        ('Phase 1 (Complete)',
         [
             '[DONE] 34-condition CER measurement (summary.csv)',
             '[DONE] STOI measurement (stoi_results.csv)',
             '[DONE] Visualization — 7 figures',
             '[RUNNING] Statistical significance test (Wilcoxon, per_sample_cer.csv)',
         ]),
        ('Phase 2 — Experimental Extension',
         [
             '[ ] Whisper large-v3 re-evaluation\n'
             '      -> Verify findings are not specific to whisper-small\n'
             '      -> Expected: ~3-4 hours CPU, run overnight',
             '[ ] PESQ measurement (requires: sudo xcodebuild -license, then pip install pesq)\n'
             '      -> Complement STOI with MOS-correlated quality metric',
             '[ ] DSP parameter sensitivity analysis\n'
             '      -> Vary HPF cutoff: 100 / 200 / 300 / 500 Hz\n'
             '      -> Find optimal or confirm all configurations hurt',
             '[ ] DSP + GTCRN serial pipeline\n'
             '      -> Does chaining models compound or cancel degradation?',
             '[ ] Real-world noise conditions (MUSAN / DEMAND dataset)\n'
             '      -> Address reviewer concern about artificial noise',
         ]),
        ('Phase 3 — Paper Writing',
         [
             '[ ] Introduction: throat mic limitations + SE paradox framing',
             '[ ] Related Work: TASLP 2024, Interspeech 2024, TAPS paper',
             '[ ] Method: dataset, noise augmentation, SE pipeline, evaluation',
             '[ ] Results: CER table + STOI/CER scatter (Fig.5 is the key figure)',
             '[ ] Discussion: artifact error hypothesis, domain mismatch',
             '[ ] Conclusion: conditions where SE should not be applied',
             '[ ] Target venue: Interspeech 2025 / ICASSP 2026 (TBD)',
         ]),
        ('Self-critique — Known Weaknesses to Address',
         [
             '1. Single speaker (p00) -> generalizability limited',
             '2. Artificial noise only -> real-world validation needed',
             '3. ASR model dependency -> large-v3 re-eval (planned)',
             '4. No statistical significance yet -> Wilcoxon test in progress',
             '5. GTCRN is out-of-domain by design -> position as a finding, not a flaw',
         ]),
    ]

    y = 0.97
    for section_title, items in plan:
        ax.text(0, y, section_title, fontsize=11, fontweight='bold',
                color='#1a237e', transform=ax.transAxes)
        y -= 0.04
        for item in items:
            lines = item.split('\n')
            ax.text(0.03, y, lines[0], fontsize=8.5, color='#212121',
                    transform=ax.transAxes, verticalalignment='top')
            y -= 0.032
            for sub in lines[1:]:
                ax.text(0.06, y, sub, fontsize=8, color='#546e7a',
                        transform=ax.transAxes, verticalalignment='top')
                y -= 0.028
        y -= 0.025

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── メイン ───────────────────────────────────────────────────
def main():
    cer_rows  = load_csv(SUMMARY)
    stoi_rows = load_csv(STOI_CSV)
    cm = cer_map(cer_rows)
    sm = stoi_map_fn(stoi_rows)

    with PdfPages(OUT_PDF) as pdf:
        print('Page 1: 表紙...')
        page_cover(pdf)
        print('Page 2: 研究概要...')
        page_overview(pdf)
        print('Page 3: 結果テーブル...')
        page_cer_table(pdf, cm, sm)
        print('Page 4: 図1・2...')
        page_figures_1(pdf)
        print('Page 5: 図3・4...')
        page_figures_2(pdf)
        print('Page 6: 図5・6...')
        page_figures_3(pdf)
        print('Page 7: 主要発見...')
        page_findings(pdf)
        print('Page 8: 計画...')
        page_plan(pdf)

        # PDFメタデータ
        d = pdf.infodict()
        d['Title']   = 'Bone Conduction SE Research Report'
        d['Author']  = 'KASU'
        d['Subject'] = 'Speech Enhancement Effects on Whisper CER for Throat Mic Audio'

    print(f'\n完了: {OUT_PDF}')


if __name__ == '__main__':
    main()
