"""
研究紹介スライド .pptx 生成（全25枚）
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm
from pptx.oxml.ns import qn
from lxml import etree

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
PESQ_CSV = os.path.join(BASE_DIR, 'results', 'pesq_results.csv')
FIG_DIR  = os.path.join(BASE_DIR, 'results', 'figures')
OUT_PPTX = os.path.join(BASE_DIR, 'results', 'slides.pptx')

# ── カラーパレット ────────────────────────────────────────────
C_NAVY   = RGBColor(0x1A, 0x23, 0x7E)
C_BLUE   = RGBColor(0x21, 0x96, 0xF3)
C_ORANGE = RGBColor(0xFF, 0x98, 0x00)
C_GREEN  = RGBColor(0x43, 0xA0, 0x47)
C_RED    = RGBColor(0xC6, 0x28, 0x28)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK   = RGBColor(0x21, 0x21, 0x21)
C_GRAY   = RGBColor(0x78, 0x90, 0x9C)
C_LIGHT  = RGBColor(0xF5, 0xF5, 0xF5)
C_YELLOW = RGBColor(0xFF, 0xF9, 0xC4)

# ── データ ───────────────────────────────────────────────────
def load():
    def rd(path):
        with open(path, encoding='utf-8') as f:
            return {r['condition']: r for r in csv.DictReader(f)}
    return rd(SUMMARY), rd(STOI_CSV), rd(PESQ_CSV)

def v(d, key, field):
    return float(d[key][field]) if key in d else None

# ── pptx ヘルパー ─────────────────────────────────────────────
W = Inches(13.33)   # ワイド16:9
H = Inches(7.5)

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank_slide(prs):
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def rgb_fill(shape, rgb):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb

def add_rect(slide, x, y, w, h, rgb):
    s = slide.shapes.add_shape(1, x, y, w, h)  # MSO_SHAPE_TYPE.RECTANGLE
    rgb_fill(s, rgb)
    s.line.fill.background()
    return s

def txbox(slide, text, x, y, w, h,
          size=20, bold=False, color=C_DARK,
          align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb

def add_title_bar(slide, title, subtitle=None):
    """上部ネイビーバー + タイトル"""
    add_rect(slide, 0, 0, W, Inches(1.4), C_NAVY)
    txbox(slide, title,
          Inches(0.4), Inches(0.18), Inches(12.5), Inches(0.9),
          size=28, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        txbox(slide, subtitle,
              Inches(0.4), Inches(0.95), Inches(12.5), Inches(0.4),
              size=14, color=RGBColor(0xBB, 0xDE, 0xFB), align=PP_ALIGN.LEFT)

def add_footer(slide, page_num, total=28):
    add_rect(slide, 0, H - Inches(0.35), W, Inches(0.35),
             RGBColor(0xE8, 0xEA, 0xF6))
    txbox(slide, f'{page_num} / {total}',
          W - Inches(1.2), H - Inches(0.32), Inches(1.1), Inches(0.3),
          size=10, color=C_NAVY, align=PP_ALIGN.RIGHT)

def bullet_box(slide, items, x, y, w, h,
               size=18, title=None, title_size=16,
               title_color=C_NAVY, bullet='●'):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = title
        run.font.size = Pt(title_size)
        run.font.bold = True
        run.font.color.rgb = title_color
        first = False
    for item in items:
        p = tf.add_paragraph() if not first else tf.paragraphs[0]
        first = False
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = f'{bullet}  {item}'
        run.font.size = Pt(size)
        run.font.color.rgb = C_DARK

def highlight_box(slide, text, x, y, w, h,
                  bg=C_YELLOW, txt_color=C_DARK, size=18, bold=False):
    s = slide.shapes.add_shape(1, x, y, w, h)
    rgb_fill(s, bg)
    s.line.color.rgb = C_NAVY
    s.line.width = Pt(1.5)
    tf = s.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = txt_color

def add_image_from_path(slide, path, x, y, w):
    if os.path.exists(path):
        slide.shapes.add_picture(path, x, y, width=w)

def fig_to_stream(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def add_image_from_buf(slide, buf, x, y, w):
    slide.shapes.add_picture(buf, x, y, width=w)

# ── 図生成 ───────────────────────────────────────────────────
def make_snr_fig(cer, noise_type, figsize=(6, 3.5)):
    SNR = ['-5','+0','+5','+10','+20']
    XL  = ['-5','0','+5','+10','+20']
    fig, ax = plt.subplots(figsize=figsize)
    for model, col, mrk, lbl in [
        ('no_se','#2196F3','o','No SE'),
        ('dsp_only','#FF9800','s','DSP-only'),
        ('gtcrn','#4CAF50','^','GTCRN'),
    ]:
        ys = []
        for snr in SNR:
            sc  = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            key = f'no_se/{noise_type}/{sc}' if model=='no_se' else f'{model}/{noise_type}/{sc}'
            val = v(cer, key, 'avg_cer')
            ys.append(val)
        ax.plot(XL, ys, marker=mrk, color=col, label=lbl, linewidth=2, markersize=8)
    ax.set_xlabel('SNR (dB)', fontsize=12); ax.set_ylabel('CER', fontsize=12)
    ax.set_ylim(0, 1.45); ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=11); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.8)
    return fig_to_stream(fig)

def make_bar_clean(cer, figsize=(5.5, 3.5)):
    conds  = ['baseline_acoustic','baseline_throat','dsp_only/clean','gtcrn/clean']
    labels = ['Acoustic\n(Reference)', 'Throat\n(Raw)', 'Throat\n+ DSP', 'Throat\n+ GTCRN']
    vals   = [float(cer[c]['avg_cer']) for c in conds]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, vals,
                  color=['#1565C0','#78909C','#FF9800','#4CAF50'], width=0.55)
    for b, val in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, val+0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('CER', fontsize=12); ax.set_ylim(0, 0.58)
    ax.tick_params(labelsize=11); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.8)
    return fig_to_stream(fig)

def make_scatter_stoi(cer, stoi, figsize=(5.5, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for model, col, mrk, lbl in [
        ('no_se','#2196F3','o','No SE'),
        ('dsp_only','#FF9800','s','DSP-only'),
        ('gtcrn','#4CAF50','^','GTCRN'),
    ]:
        xs, ys = [], []
        for key in stoi:
            if 'snr_' not in key: continue
            if model == 'no_se' and not key.startswith('no_se/'): continue
            if model != 'no_se' and not key.startswith(model+'/'): continue
            if key not in cer: continue
            xs.append(float(stoi[key]['avg_stoi']))
            ys.append(float(cer[key]['avg_cer']))
        ax.scatter(xs, ys, c=col, marker=mrk, s=80, alpha=0.85, label=lbl)
    ax.set_xlabel('STOI  ↑ better', fontsize=12)
    ax.set_ylabel('CER  ↓ better',  fontsize=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.2)
    ax.tick_params(labelsize=11)
    ax.annotate('GTCRN paradox\nSTOI↑ but CER↑',
                xy=(0.72, 1.08), fontsize=10, color='#C62828',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#C62828'),
                xytext=(0.55, 1.25))
    fig.tight_layout(pad=0.8)
    return fig_to_stream(fig)

def make_delta_fig(cer, stoi, pesq, noise_type, figsize=(7, 3.8)):
    SNR = ['-5','+0','+5','+10','+20']
    XL  = ['-5','0','+5','+10','+20']
    dc_list, ds_list, dp_list = [], [], []
    for snr in SNR:
        sc  = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
        se  = f'gtcrn/{noise_type}/{sc}'
        ref = f'no_se/{noise_type}/{sc}'
        dc_list.append((v(cer,  se,'avg_cer')  or 0) - (v(cer,  ref,'avg_cer')  or 0))
        ds_list.append((v(stoi, se,'avg_stoi') or 0) - (v(stoi, ref,'avg_stoi') or 0))
        dp_list.append((v(pesq, se,'avg_pesq') or 0) - (v(pesq, ref,'avg_pesq') or 0))
    x = np.arange(len(XL)); w = 0.26
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    ax1.bar(x-w, dc_list, w, color='#C62828', alpha=0.85, label='ΔCER (up=worse ASR)')
    ax1.bar(x,   ds_list, w, color='#1565C0', alpha=0.85, label='ΔSTOI (up=better)')
    ax2.bar(x+w, dp_list, w, color='#2E7D32', alpha=0.85, label='ΔPESQ (up=better)')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels(XL, fontsize=11)
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('ΔCER / ΔSTOI', fontsize=11)
    ax2.set_ylabel('ΔPESQ', fontsize=11, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32', labelsize=10)
    ax1.tick_params(labelsize=10); ax1.grid(axis='y', alpha=0.25)
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=10, loc='upper left')
    fig.tight_layout(pad=0.8)
    return fig_to_stream(fig)

# ── スライド定義 ─────────────────────────────────────────────

def s01_title(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, W, H, C_NAVY)
    # アクセント帯
    add_rect(sl, 0, Inches(4.5), W, Inches(0.08), C_BLUE)
    txbox(sl, '喉マイク音声に対する音声強調が\n自動音声認識に与える影響の定量的評価',
          Inches(0.6), Inches(1.2), Inches(12.1), Inches(2.8),
          size=34, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    txbox(sl,
          'Quantitative Evaluation of Speech Enhancement Effects\n'
          'on ASR for Bone Conduction Microphone Audio',
          Inches(0.6), Inches(3.9), Inches(12.1), Inches(0.8),
          size=16, color=RGBColor(0xBB,0xDE,0xFB), align=PP_ALIGN.LEFT)
    txbox(sl, '○ 著者名（所属機関）　|　日本音響学会 2026年秋季研究発表会',
          Inches(0.6), Inches(5.0), Inches(12.1), Inches(0.5),
          size=14, color=RGBColor(0x90,0xCA,0xF9), align=PP_ALIGN.LEFT)
    add_footer(sl, 1)

def s02_agenda(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '目次')
    items = [
        ('1. 研究背景',       '喉マイクとは？  なぜASRが難しいか'),
        ('2. 研究動機・目的', 'SEは本当に役立つのか？'),
        ('3. 関連研究',       'TASLP 2024 / Interspeech 2024'),
        ('4. 実験設計',       'データセット・ノイズ付加・SEモデル'),
        ('5. 評価指標',       'CER・STOI・PESQ'),
        ('6. 実験結果',       '34条件の定量評価'),
        ('7. 考察',           'なぜSEが逆効果になるのか'),
        ('8. まとめ・今後',   '発見のまとめと展望'),
    ]
    y0 = Inches(1.55)
    for i, (num_title, desc) in enumerate(items):
        col = i % 2
        row = i // 2
        x = Inches(0.4) + col * Inches(6.5)
        y = y0 + row * Inches(1.25)
        add_rect(sl, x, y, Inches(6.1), Inches(1.1),
                 RGBColor(0xE8,0xEA,0xF6) if i%2==0 else RGBColor(0xE3,0xF2,0xFD))
        txbox(sl, num_title, x+Inches(0.15), y+Inches(0.05),
              Inches(5.8), Inches(0.45), size=17, bold=True, color=C_NAVY)
        txbox(sl, desc, x+Inches(0.15), y+Inches(0.52),
              Inches(5.8), Inches(0.5), size=13, color=C_GRAY)
    add_footer(sl, 2)

def s03_throat_mic(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '背景①  喉マイク（骨伝導マイク）とは？',
                  subtitle='Throat Microphone / Bone Conduction Microphone')
    # 左：説明
    bullet_box(sl, [
        '声帯の振動を皮膚・骨を通じて直接収音するマイク',
        '気導マイク（通常のマイク）とは収音原理が根本的に異なる',
        'ヘルメット着用中・極端な騒音環境でも安定収音できる',
    ], Inches(0.4), Inches(1.6), Inches(6.0), Inches(2.0), size=18)

    # 用途ボックス
    for i, (label, desc, col) in enumerate([
        ('軍事・防衛', '戦場での通信', RGBColor(0xE3,0xF2,0xFD)),
        ('工業現場',   '騒音下作業', RGBColor(0xE8,0xF5,0xE9)),
        ('スポーツ',   '水中・レース中', RGBColor(0xFF,0xF8,0xE1)),
    ]):
        x = Inches(0.4 + i * 2.05)
        add_rect(sl, x, Inches(3.8), Inches(1.9), Inches(1.1), col)
        txbox(sl, label, x, Inches(3.85), Inches(1.9), Inches(0.45),
              size=14, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
        txbox(sl, desc,  x, Inches(4.25), Inches(1.9), Inches(0.45),
              size=12, color=C_GRAY, align=PP_ALIGN.CENTER)

    # 右：周波数特性の説明
    add_rect(sl, Inches(6.8), Inches(1.55), Inches(6.1), Inches(4.5),
             RGBColor(0xF3,0xE5,0xF5))
    txbox(sl, '喉マイクの音響特性',
          Inches(6.9), Inches(1.65), Inches(5.9), Inches(0.45),
          size=16, bold=True, color=C_NAVY)
    for i, (icon, text) in enumerate([
        ('▼', '低域（〜300Hz）にエネルギーが集中'),
        ('▼', '高周波成分が著しく欠落'),
        ('▼', '音声が「こもった」印象'),
        ('→', 'ASRシステムが誤認識しやすい'),
        ('→', '素の状態でCER ≈ 27%（気導比較: 13%）'),
    ]):
        txbox(sl, f'{icon}  {text}',
              Inches(7.0), Inches(2.2)+i*Inches(0.62), Inches(5.7), Inches(0.55),
              size=15, color=C_DARK if '▼' in icon else C_RED,
              bold='→' in icon)
    add_footer(sl, 3)

def s04_se_intro(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '背景②  音声強調（Speech Enhancement）とは？',
                  subtitle='Speech Enhancement: SE')
    txbox(sl, '音声に含まれるノイズを除去し、音質・了解度を改善する信号処理技術',
          Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.55),
          size=18, bold=True, color=C_NAVY)

    # フロー図
    boxes = [
        ('喉マイク\n音声 + ノイズ', C_GRAY),
        ('SE処理',               C_NAVY),
        ('強調済み\n音声',        C_GREEN),
        ('ASR\n(Whisper)',       C_BLUE),
        ('テキスト',             RGBColor(0x6A,0x1B,0x9A)),
    ]
    bw, bh = Inches(1.8), Inches(1.0)
    y_b = Inches(2.5)
    for i, (label, col) in enumerate(boxes):
        x = Inches(0.5) + i * Inches(2.55)
        add_rect(sl, x, y_b, bw, bh, col)
        txbox(sl, label, x, y_b, bw, bh,
              size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(boxes)-1:
            txbox(sl, '→', x+bw, y_b+Inches(0.25), Inches(0.6), Inches(0.5),
                  size=22, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

    # 2種類のSE
    for i, (title, items, col) in enumerate([
        ('DSP-only（信号処理）', [
            'ハイパスフィルタ（300Hz）',
            'プリエンファシス（高域強調）',
            'RMS 正規化',
            '→ 計算コスト極小，解釈しやすい',
        ], RGBColor(0xFF,0xF3,0xE0)),
        ('GTCRN（ニューラルSE）', [
            '48,200パラメータの超軽量DNN',
            'DNS3（気導マイク）で学習済み',
            'STFT → DNN → iSTFT の処理',
            '→ 高品質ノイズ除去が期待される',
        ], RGBColor(0xE8,0xF5,0xE9)),
    ]):
        x = Inches(0.4) + i * Inches(6.5)
        add_rect(sl, x, Inches(3.8), Inches(6.1), Inches(2.9), col)
        txbox(sl, title, x+Inches(0.1), Inches(3.88), Inches(5.9), Inches(0.45),
              size=16, bold=True, color=C_NAVY)
        for j, item in enumerate(items):
            txbox(sl, f'・ {item}', x+Inches(0.2),
                  Inches(4.4)+j*Inches(0.5), Inches(5.7), Inches(0.45),
                  size=13, color=C_DARK)
    add_footer(sl, 4)

def s05_research_q(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '研究動機と目的')
    txbox(sl, '素朴な疑問：SEをかければASRは改善する？',
          Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.5),
          size=20, bold=True, color=C_NAVY)

    # ✗ 常識
    add_rect(sl, Inches(0.4), Inches(2.2), Inches(5.8), Inches(1.4),
             RGBColor(0xE3,0xF2,0xFD))
    txbox(sl, '一般的なイメージ',
          Inches(0.5), Inches(2.28), Inches(5.6), Inches(0.4),
          size=15, bold=True, color=C_NAVY)
    txbox(sl, 'SE適用 → ノイズ除去 → 音声が綺麗に → ASR改善 ✓',
          Inches(0.5), Inches(2.72), Inches(5.6), Inches(0.6),
          size=15, color=C_DARK)

    txbox(sl, '？', Inches(6.3), Inches(2.5), Inches(0.8), Inches(0.8),
          size=40, bold=True, color=C_ORANGE, align=PP_ALIGN.CENTER)

    # ✓ 実態
    add_rect(sl, Inches(7.1), Inches(2.2), Inches(5.8), Inches(1.4),
             RGBColor(0xFF,0xEB,0xEE))
    txbox(sl, '実際に報告されていること',
          Inches(7.2), Inches(2.28), Inches(5.6), Inches(0.4),
          size=15, bold=True, color=C_RED)
    txbox(sl, 'SE適用 → 音質は改善 → でもASR悪化 ✗\n（Mawalim et al. 2024, Ochiai et al. 2024）',
          Inches(7.2), Inches(2.72), Inches(5.6), Inches(0.6),
          size=14, color=C_DARK)

    # RQ
    highlight_box(sl,
        '本研究のリサーチクエスチョン\n'
        '「喉マイク音声において、どの条件でSEがASRを悪化させるか？\n'
        '　またSTOI・PESQとCERはどのような関係にあるか？」',
        Inches(0.4), Inches(3.85), Inches(12.5), Inches(1.5),
        bg=RGBColor(0xFF,0xF9,0xC4), txt_color=C_DARK, size=17, bold=False)

    txbox(sl, '→ 34条件でCER・STOI・PESQを定量評価（統計検定つき）',
          Inches(0.6), Inches(5.55), Inches(12.1), Inches(0.45),
          size=16, bold=True, color=C_NAVY)
    add_footer(sl, 5)

def s06_related1(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '関連研究①  Ochiai et al. TASLP 2024',
                  subtitle='"Rethinking Processing Distortions" arXiv:2404.14860')
    # 3成分
    txbox(sl, 'SEによる誤差を3成分に直交分解し、ASRへの影響を定量化',
          Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.45),
          size=17, bold=True, color=C_NAVY)
    for i, (name, desc, col, impact) in enumerate([
        ('Interference\nError', '残留ノイズ\n（完全に除去できなかったノイズ）',
         RGBColor(0xE3,0xF2,0xFD), '△  中程度のASR悪化'),
        ('Noise Error', '過剰抑圧\n（音声を削りすぎ）',
         RGBColor(0xFF,0xF8,0xE1), '△  中程度のASR悪化'),
        ('Artifact Error', '非線形歪み\n（SEが生む人工的なノイズ）',
         RGBColor(0xFF,0xEB,0xEE), '✗  最大のASR悪化要因'),
    ]):
        x = Inches(0.4) + i * Inches(4.3)
        add_rect(sl, x, Inches(2.15), Inches(4.0), Inches(3.2), col)
        txbox(sl, name, x+Inches(0.1), Inches(2.22), Inches(3.8), Inches(0.65),
              size=17, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
        txbox(sl, desc, x+Inches(0.1), Inches(2.95), Inches(3.8), Inches(0.85),
              size=14, color=C_DARK, align=PP_ALIGN.CENTER)
        txbox(sl, impact, x+Inches(0.1), Inches(3.9), Inches(3.8), Inches(0.45),
              size=13, bold=True,
              color=C_RED if '✗' in impact else C_ORANGE,
              align=PP_ALIGN.CENTER)

    highlight_box(sl,
        '主要発見：Artifact Errorが最もASRに有害\n'
        '→ SEで音声が「綺麗」に見えても、ASRには致命的な歪みが残る',
        Inches(0.4), Inches(5.6), Inches(12.5), Inches(1.2),
        bg=RGBColor(0xFF,0xEB,0xEE), txt_color=C_DARK, size=16)
    add_footer(sl, 6)

def s07_related2(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '関連研究②  Mawalim et al. Interspeech 2024',
                  subtitle='"Are Recent DL-Based SE Methods Ready?" DOI:10.21437/Interspeech.2024-129')
    txbox(sl, '最新の深層学習SEが「実環境」でASRの役に立つかを実証的に検証',
          Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.45),
          size=17, bold=True, color=C_NAVY)
    # 評価したモデル
    for i, (name, desc) in enumerate([
        ('Denoiser', 'Facebook の音声ノイズ除去モデル'),
        ('DeepFilterNet3', 'リアルタイム対応の高性能SEモデル'),
        ('FullSubNet+', 'フルバンド+サブバンド処理モデル'),
    ]):
        add_rect(sl, Inches(0.4)+i*Inches(4.3), Inches(2.15),
                 Inches(4.0), Inches(0.9), RGBColor(0xE8,0xEA,0xF6))
        txbox(sl, name, Inches(0.5)+i*Inches(4.3), Inches(2.22),
              Inches(3.8), Inches(0.4), size=15, bold=True, color=C_NAVY)
        txbox(sl, desc, Inches(0.5)+i*Inches(4.3), Inches(2.62),
              Inches(3.8), Inches(0.35), size=12, color=C_GRAY)

    bullet_box(sl, [
        '実環境（非定常ノイズ・残響）ではベンチマーク性能から大きく低下',
        'PESQ/STOIが改善してもASR WERが改善しない・悪化するケースが多数',
        '音質指標（知覚品質） ≠ ASR性能　という乖離を実証',
    ], Inches(0.4), Inches(3.3), Inches(12.5), Inches(1.8), size=17)

    highlight_box(sl,
        '本研究との接点：同じ「STOI改善・ASR悪化」の乖離を\n'
        '喉マイク × Whisper × 韓国語 という独自の文脈で再現・拡張する',
        Inches(0.4), Inches(5.35), Inches(12.5), Inches(1.2),
        bg=RGBColor(0xE8,0xF5,0xE9), txt_color=C_DARK, size=16)
    add_footer(sl, 7)

def s08_dataset(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '実験設計①  データセット：TAPS',
                  subtitle='Throat and Acoustic Pairing Speech Dataset')
    # 左
    bullet_box(sl, [
        '韓国語 paired 音声データセット（HuggingFace公開）',
        '喉マイク音声 ＋ 気導マイク音声のペアが揃っている',
        '話者ごとに正解テキスト（metadata.csv）付属',
        '本研究：テストセット 50発話、話者 p00',
        'サンプリングレート：16 kHz',
    ], Inches(0.4), Inches(1.6), Inches(6.2), Inches(3.5), size=17)

    # 右：ベースラインCER表
    add_rect(sl, Inches(7.0), Inches(1.55), Inches(5.9), Inches(3.5),
             RGBColor(0xF3,0xE5,0xF5))
    txbox(sl, 'ベースラインCER（Whisper small）',
          Inches(7.1), Inches(1.65), Inches(5.7), Inches(0.45),
          size=15, bold=True, color=C_NAVY)
    for i, (cond, cer_val, note) in enumerate([
        ('気導マイク（参照）',     '0.131 (13.1%)', '← 達成目標'),
        ('喉マイク生音声',         '0.269 (26.9%)', '← 出発点'),
        ('喉マイク + DSP-only',   '0.416 (41.6%)', '← SEで悪化'),
        ('喉マイク + GTCRN',      '0.296 (29.6%)', '← わずか改善'),
    ]):
        y = Inches(2.2) + i * Inches(0.68)
        bg = [RGBColor(0xE8,0xEA,0xF6),RGBColor(0xFF,0xF9,0xC4),
              RGBColor(0xFF,0xCD,0xD2),RGBColor(0xC8,0xE6,0xC9)][i]
        add_rect(sl, Inches(7.1), y, Inches(5.7), Inches(0.6), bg)
        txbox(sl, cond,    Inches(7.2), y+Inches(0.08), Inches(2.8), Inches(0.45), size=13, color=C_DARK)
        txbox(sl, cer_val, Inches(9.8), y+Inches(0.08), Inches(1.5), Inches(0.45), size=13, bold=True, color=C_NAVY)
        txbox(sl, note,    Inches(11.1),y+Inches(0.08), Inches(1.5), Inches(0.45), size=11, color=C_GRAY)

    txbox(sl, '※ 韓国語の内容確認は不要：CERは文字列比較（意味理解不要）',
          Inches(0.4), Inches(5.4), Inches(12.5), Inches(0.4),
          size=12, color=C_GRAY)
    add_footer(sl, 8)

def s09_noise(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '実験設計②  ノイズ付加と実験条件',
                  subtitle='10 noisy + 4 clean = 34 total conditions')
    # ノイズ種
    for i, (name, desc, col) in enumerate([
        ('白色ノイズ\nWhite Noise', '全周波数帯域に\n均一なエネルギー', RGBColor(0xE3,0xF2,0xFD)),
        ('ピンクノイズ\nPink Noise', '低周波ほど\nエネルギーが大きい', RGBColor(0xFC,0xE4,0xEC)),
    ]):
        x = Inches(0.4) + i * Inches(3.2)
        add_rect(sl, x, Inches(1.6), Inches(3.0), Inches(1.5), col)
        txbox(sl, name, x+Inches(0.1), Inches(1.68), Inches(2.8), Inches(0.65),
              size=15, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
        txbox(sl, desc, x+Inches(0.1), Inches(2.25), Inches(2.8), Inches(0.65),
              size=13, color=C_DARK, align=PP_ALIGN.CENTER)

    # SNR説明
    add_rect(sl, Inches(6.8), Inches(1.6), Inches(6.1), Inches(3.0),
             RGBColor(0xF1,0xF8,0xE9))
    txbox(sl, 'SNR（信号対雑音比）とは？',
          Inches(6.9), Inches(1.7), Inches(5.9), Inches(0.4),
          size=15, bold=True, color=C_NAVY)
    for i, (snr, desc, col) in enumerate([
        ('+20 dB', '音声がノイズの100倍パワー → ほぼクリーン',  C_GREEN),
        ('+5 dB',  '音声がノイズの約3倍       → 少しうるさい', C_ORANGE),
        ('0 dB',   '音声 = ノイズ             → かなり聞こえにくい', C_ORANGE),
        ('-5 dB',  'ノイズが音声より大きい    → ほぼ聞こえない', C_RED),
    ]):
        txbox(sl, f'{snr}：{desc}',
              Inches(7.0), Inches(2.2)+i*Inches(0.55), Inches(5.7), Inches(0.45),
              size=13, color=col, bold=snr in ['+20 dB','-5 dB'])

    # 条件一覧
    add_rect(sl, Inches(0.4), Inches(3.3), Inches(6.1), Inches(3.0),
             RGBColor(0xE8,0xEA,0xF6))
    txbox(sl, '全34実験条件',
          Inches(0.5), Inches(3.38), Inches(5.9), Inches(0.4),
          size=15, bold=True, color=C_NAVY)
    for i, line in enumerate([
        '4条件    ベースライン（acoustic / throat）+ SE×clean',
        '10条件   No SE × 白色/ピンク × 5 SNR',
        '10条件   DSP-only × 白色/ピンク × 5 SNR',
        '10条件   GTCRN × 白色/ピンク × 5 SNR',
    ]):
        txbox(sl, f'・ {line}',
              Inches(0.6), Inches(3.9)+i*Inches(0.55), Inches(5.7), Inches(0.45),
              size=13, color=C_DARK)
    add_footer(sl, 9)

def s10_models(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '実験設計③  音声強調モデルの詳細')
    # DSP
    add_rect(sl, Inches(0.4), Inches(1.6), Inches(6.0), Inches(4.8),
             RGBColor(0xFF,0xF3,0xE0))
    txbox(sl, 'DSP-only',
          Inches(0.5), Inches(1.68), Inches(5.8), Inches(0.45),
          size=19, bold=True, color=C_ORANGE)
    for i, (step, desc) in enumerate([
        ('Step 1', 'ハイパスフィルタ（Butterworth 6次, 300Hz）\n  → 300Hz以下を除去'),
        ('Step 2', 'プリエンファシス（係数 0.97）\n  → 高周波成分を強調'),
        ('Step 3', 'RMS正規化\n  → 音量を一定に揃える'),
    ]):
        add_rect(sl, Inches(0.6), Inches(2.2)+i*Inches(1.2),
                 Inches(5.5), Inches(1.1), RGBColor(0xFF,0xE0,0xB2))
        txbox(sl, step, Inches(0.7), Inches(2.27)+i*Inches(1.2),
              Inches(1.0), Inches(0.4), size=13, bold=True, color=C_ORANGE)
        txbox(sl, desc, Inches(1.6), Inches(2.27)+i*Inches(1.2),
              Inches(4.3), Inches(0.65), size=13, color=C_DARK)
    txbox(sl, '⚠ 300Hz以下が喉マイクの主エネルギー帯域\n→ HPFで大半を削ってしまう可能性',
          Inches(0.5), Inches(5.68), Inches(5.8), Inches(0.65),
          size=12, color=C_RED)

    # GTCRN
    add_rect(sl, Inches(6.9), Inches(1.6), Inches(6.0), Inches(4.8),
             RGBColor(0xE8,0xF5,0xE9))
    txbox(sl, 'GTCRN',
          Inches(7.0), Inches(1.68), Inches(5.8), Inches(0.45),
          size=19, bold=True, color=C_GREEN)
    for i, (label, val) in enumerate([
        ('パラメータ数', '48,200（超軽量ニューラルSE）'),
        ('学習データ',   'DNS3コーパス（気導マイク・主に英語）'),
        ('入力形式',     '16kHz 音声のみ'),
        ('処理方式',     'STFT → 複素領域DNN → iSTFT'),
    ]):
        add_rect(sl, Inches(7.1), Inches(2.2)+i*Inches(0.85),
                 Inches(5.5), Inches(0.75), RGBColor(0xC8,0xE6,0xC9))
        txbox(sl, label+'：', Inches(7.2), Inches(2.27)+i*Inches(0.85),
              Inches(1.9), Inches(0.45), size=13, bold=True, color=C_GREEN)
        txbox(sl, val,        Inches(8.9), Inches(2.27)+i*Inches(0.85),
              Inches(3.5), Inches(0.45), size=13, color=C_DARK)
    txbox(sl, '⚠ 喉マイク音声はドメイン外（学習分布外）入力\n→ 予期しない変換が生じる可能性',
          Inches(7.0), Inches(5.68), Inches(5.8), Inches(0.65),
          size=12, color=C_RED)
    add_footer(sl, 10)

def s11_cer(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '評価指標①  CER（Character Error Rate）',
                  subtitle='文字誤り率 — ASR性能の直接評価')
    txbox(sl, 'ASRが出力したテキストと正解テキストの「文字単位の編集距離」',
          Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.45),
          size=17, bold=True, color=C_NAVY)
    # 数式
    add_rect(sl, Inches(0.4), Inches(2.1), Inches(12.5), Inches(1.2),
             RGBColor(0xE3,0xF2,0xFD))
    txbox(sl, 'CER = （挿入 + 削除 + 置換の文字数）÷ 正解テキストの文字数',
          Inches(0.6), Inches(2.3), Inches(12.1), Inches(0.6),
          size=18, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

    # 例
    for i, (label, ref, hyp, cer_val, col) in enumerate([
        ('良い例（CER低）',
         '오늘 날씨가 맑습니다',
         '오늘 날씨가 맑습니다',
         'CER = 0.00', RGBColor(0xC8,0xE6,0xC9)),
        ('普通（CER中）',
         '오늘 날씨가 맑습니다',
         '오늘 날씨가 말습니다',
         'CER ≈ 0.10', RGBColor(0xFF,0xF9,0xC4)),
        ('悪い例（CER高）',
         '오늘 날씨가 맑습니다',
         '오늘 ????',
         'CER > 0.50', RGBColor(0xFF,0xCD,0xD2)),
    ]):
        x = Inches(0.4) + i * Inches(4.3)
        add_rect(sl, x, Inches(3.5), Inches(4.0), Inches(2.5), col)
        txbox(sl, label,   x+Inches(0.1), Inches(3.58), Inches(3.8), Inches(0.4),
              size=13, bold=True, color=C_NAVY)
        txbox(sl, f'正解: {ref}', x+Inches(0.1), Inches(4.02), Inches(3.8), Inches(0.38),
              size=12, color=C_DARK)
        txbox(sl, f'出力: {hyp}', x+Inches(0.1), Inches(4.42), Inches(3.8), Inches(0.38),
              size=12, color=C_DARK)
        txbox(sl, cer_val, x+Inches(0.1), Inches(4.85), Inches(3.8), Inches(0.38),
              size=14, bold=True, color=C_NAVY)

    txbox(sl, '※ 本研究では言語理解は不要。CERは純粋に文字列の一致度を計算する。',
          Inches(0.4), Inches(6.3), Inches(12.5), Inches(0.4),
          size=12, color=C_GRAY)
    add_footer(sl, 11)

def s12_stoi_pesq(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '評価指標②  STOI と PESQ',
                  subtitle='音声の「知覚品質」を数値化する指標')
    # STOI
    add_rect(sl, Inches(0.4), Inches(1.6), Inches(6.0), Inches(4.7),
             RGBColor(0xE3,0xF2,0xFD))
    txbox(sl, 'STOI（Short-Time Objective Intelligibility）',
          Inches(0.5), Inches(1.68), Inches(5.8), Inches(0.45),
          size=15, bold=True, color=C_NAVY)
    txbox(sl, '「この音声は人間にどれだけ聞き取りやすいか」',
          Inches(0.5), Inches(2.18), Inches(5.8), Inches(0.4),
          size=14, color=C_DARK)
    for i, line in enumerate([
        '値域：0 〜 1（1 = 完全に聞き取れる）',
        '仕組み：短時間フレームごとの周波数エネルギー相関',
        '用途：補聴器・通話システムの評価',
        '本研究：クリーン喉マイク音声を参照に算出',
    ]):
        txbox(sl, f'・ {line}',
              Inches(0.6), Inches(2.7)+i*Inches(0.52), Inches(5.6), Inches(0.45),
              size=13, color=C_DARK)
    add_rect(sl, Inches(0.5), Inches(5.0), Inches(5.7), Inches(0.5),
             RGBColor(0xBB,0xDE,0xFB))
    txbox(sl, '→ 「聞き取りやすさ」の指標（了解度）',
          Inches(0.6), Inches(5.05), Inches(5.5), Inches(0.38),
          size=13, bold=True, color=C_NAVY)

    # PESQ
    add_rect(sl, Inches(6.9), Inches(1.6), Inches(6.0), Inches(4.7),
             RGBColor(0xE8,0xF5,0xE9))
    txbox(sl, 'PESQ（Perceptual Evaluation of Speech Quality）',
          Inches(7.0), Inches(1.68), Inches(5.8), Inches(0.45),
          size=15, bold=True, color=C_GREEN)
    txbox(sl, '「人間が聴いたときの音質スコア」を模倣する指標',
          Inches(7.0), Inches(2.18), Inches(5.8), Inches(0.4),
          size=14, color=C_DARK)
    for i, line in enumerate([
        '値域：−0.5 〜 4.5（ITU-T P.862 国際規格）',
        '仕組み：人間の聴覚モデルに通した知覚的差分を計算',
        '用途：音声コーデック・VoIPの品質評価',
        '本研究：クリーン喉マイク音声を参照に算出（WBモード）',
    ]):
        txbox(sl, f'・ {line}',
              Inches(7.1), Inches(2.7)+i*Inches(0.52), Inches(5.6), Inches(0.45),
              size=13, color=C_DARK)
    add_rect(sl, Inches(7.0), Inches(5.0), Inches(5.7), Inches(0.5),
             RGBColor(0xA5,0xD6,0xA7))
    txbox(sl, '→ 「音の自然さ・快適さ」の指標（音質）',
          Inches(7.1), Inches(5.05), Inches(5.5), Inches(0.38),
          size=13, bold=True, color=C_GREEN)

    highlight_box(sl,
        'CER  ≠  STOI  ≠  PESQ\n'
        'ASR性能と知覚品質は必ずしも一致しない  ←  これが本研究の核心',
        Inches(0.4), Inches(6.4), Inches(12.5), Inches(0.82),
        bg=RGBColor(0xFF,0xF9,0xC4), txt_color=C_DARK, size=16, bold=True)
    add_footer(sl, 12)

def s13_result_clean(prs, cer):
    sl = blank_slide(prs)
    add_title_bar(sl, '結果①  クリーン条件のCER比較')
    buf = make_bar_clean(cer)
    add_image_from_buf(sl, buf, Inches(0.4), Inches(1.55), Inches(6.5))
    # 解説
    for i, (cond, val, comment, col) in enumerate([
        ('気導マイク（参照）', '13.1%', '達成目標ライン', C_BLUE),
        ('喉マイク 生音声',    '26.9%', '基準点（ここから改善したい）', C_GRAY),
        ('喉マイク + DSP',    '41.6%', '▼ SEで15pt悪化！', C_ORANGE),
        ('喉マイク + GTCRN',  '29.6%', '▲ わずかに改善（生音声に近い）', C_GREEN),
    ]):
        y = Inches(1.7) + i * Inches(1.2)
        add_rect(sl, Inches(7.2), y, Inches(5.7), Inches(1.05),
                 RGBColor(0xF5,0xF5,0xF5))
        txbox(sl, cond,    Inches(7.3), y+Inches(0.06), Inches(3.5), Inches(0.4),
              size=14, bold=True, color=col)
        txbox(sl, val,     Inches(10.7),y+Inches(0.06), Inches(1.0), Inches(0.4),
              size=16, bold=True, color=col, align=PP_ALIGN.RIGHT)
        txbox(sl, comment, Inches(7.3), y+Inches(0.52), Inches(4.6), Inches(0.4),
              size=12, color=C_GRAY)

    txbox(sl, 'クリーン条件ではGTCRNが最良。しかしノイズ条件では？',
          Inches(0.4), Inches(6.6), Inches(12.5), Inches(0.45),
          size=16, bold=True, color=C_NAVY)
    add_footer(sl, 13)

def s14_result_noisy(prs, cer):
    sl = blank_slide(prs)
    add_title_bar(sl, '結果②  ノイズ条件：SNR vs CER',
                  subtitle='白色ノイズ / ピンクノイズ — 全モデル比較')
    buf_w = make_snr_fig(cer, 'white', figsize=(5.5, 3.5))
    buf_p = make_snr_fig(cer, 'pink',  figsize=(5.5, 3.5))
    add_image_from_buf(sl, buf_w, Inches(0.3), Inches(1.6), Inches(6.3))
    add_image_from_buf(sl, buf_p, Inches(6.8), Inches(1.6), Inches(6.3))
    txbox(sl, '白色ノイズ', Inches(0.3), Inches(5.25), Inches(6.3), Inches(0.35),
          size=13, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
    txbox(sl, 'ピンクノイズ', Inches(6.8), Inches(5.25), Inches(6.3), Inches(0.35),
          size=13, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
    highlight_box(sl,
        '全SNR帯域・全ノイズ種で  DSP-only & GTCRN  ＞  No SE（CER大 = 悪い）\n'
        '→ ノイズ条件ではSEを適用すると必ずASRが悪化する',
        Inches(0.3), Inches(5.8), Inches(12.7), Inches(1.0),
        bg=RGBColor(0xFF,0xEB,0xEE), txt_color=C_DARK, size=16, bold=True)
    add_footer(sl, 14)

def s15_stat(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '結果③  統計的有意性の検証',
                  subtitle='Wilcoxon符号順位検定（両側）— 50発話のサンプルごとCER')
    txbox(sl, '30ペアの条件比較中 26件（87%）で p < 0.05 を達成',
          Inches(0.4), Inches(1.55), Inches(12.5), Inches(0.5),
          size=19, bold=True, color=C_NAVY)

    rows = [
        ['比較', '白色 −5dB', '白色 0dB', '白色 +5dB', '白色 +10dB', '白色 +20dB'],
        ['DSP vs No SE', 'n.s.',   'p<.001', 'p<.001', 'p<.001', 'p<.001'],
        ['GTCRN vs No SE','p<.01', 'p<.001', 'p<.001', 'p<.001', 'p<.01 '],
        ['GTCRN vs DSP', 'p<.01',  'p<.01 ', 'p<.05 ', 'p<.001', 'p<.001'],
    ]
    col_w = [Inches(2.5)] + [Inches(2.0)]*5
    y0 = Inches(2.25)
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            x = Inches(0.3) + sum(col_w[:ci])
            bg = C_NAVY if ri==0 else (
                RGBColor(0xFF,0xCD,0xD2) if cell.startswith('p') else
                RGBColor(0xEE,0xEE,0xEE))
            add_rect(sl, x, y0+ri*Inches(0.72), col_w[ci]-Inches(0.03),
                     Inches(0.65), bg)
            txbox(sl, cell, x+Inches(0.05), y0+ri*Inches(0.72)+Inches(0.1),
                  col_w[ci]-Inches(0.1), Inches(0.45),
                  size=13 if ri>0 else 12,
                  bold=ri==0,
                  color=C_WHITE if ri==0 else (C_RED if cell.startswith('p') else C_GRAY),
                  align=PP_ALIGN.CENTER)

    bullet_box(sl, [
        '有意差なし（n.s.）の4件 → SNR −5dBの天井効果（CER≥1.0）が原因',
        '天井効果 ＝ 両条件ともASRが完全に破綻しており、SEの有無が測定不能な状態',
        '「SEが逆効果」という結論は統計的に裏付けられている',
    ], Inches(0.4), Inches(5.45), Inches(12.5), Inches(1.6), size=15)
    add_footer(sl, 15)

def s16_paradox(prs, cer, stoi):
    sl = blank_slide(prs)
    add_title_bar(sl, '結果④  STOI vs CER のパラドックス（発見の核心）')
    buf = make_scatter_stoi(cer, stoi, figsize=(5.5, 4.2))
    add_image_from_buf(sl, buf, Inches(0.3), Inches(1.55), Inches(6.5))
    txbox(sl, '具体例（白色ノイズ SNR 0dB）',
          Inches(7.0), Inches(1.6), Inches(5.9), Inches(0.4),
          size=15, bold=True, color=C_NAVY)
    for i, (model, stoi_v, cer_v, col) in enumerate([
        ('No SE',    '0.585', '0.882', C_BLUE),
        ('DSP-only', '0.434', '1.028', C_ORANGE),
        ('GTCRN',    '0.721', '1.214', C_GREEN),
    ]):
        y = Inches(2.15) + i*Inches(1.35)
        add_rect(sl, Inches(7.0), y, Inches(5.9), Inches(1.2),
                 RGBColor(0xF5,0xF5,0xF5))
        txbox(sl, model, Inches(7.1), y+Inches(0.08),
              Inches(5.7), Inches(0.38), size=15, bold=True, color=col)
        txbox(sl, f'STOI: {stoi_v}   CER: {cer_v}',
              Inches(7.1), y+Inches(0.52), Inches(5.7), Inches(0.38),
              size=14, color=C_DARK)

    highlight_box(sl,
        'GTCRNは STOI を改善（0.585 → 0.721）\n'
        'しかし CER は悪化（0.882 → 1.214）\n'
        '→ 「音は綺麗になったのにASRが壊れた」',
        Inches(7.0), Inches(6.05), Inches(5.9), Inches(1.2),
        bg=RGBColor(0xFF,0xEB,0xEE), txt_color=C_DARK, size=15, bold=False)
    add_footer(sl, 16)

def s17_delta(prs, cer, stoi, pesq):
    sl = blank_slide(prs)
    add_title_bar(sl, '結果⑤  PESQ・STOI・CER の三重パラドックス',
                  subtitle='GTCRNとNo SEの差分（Δ）— 白色ノイズ')
    buf = make_delta_fig(cer, stoi, pesq, 'white', figsize=(8, 4.0))
    add_image_from_buf(sl, buf, Inches(0.3), Inches(1.55), Inches(9.5))
    txbox(sl, '読み方：',
          Inches(10.1), Inches(1.6), Inches(3.0), Inches(0.4),
          size=14, bold=True, color=C_NAVY)
    for i, (col_name, meaning, col) in enumerate([
        ('赤棒（ΔCER）',  '上↑ = ASR悪化', C_RED),
        ('青棒（ΔSTOI）', '上↑ = 了解度改善', C_BLUE),
        ('緑棒（ΔPESQ）', '上↑ = 音質改善',  C_GREEN),
    ]):
        txbox(sl, col_name,  Inches(10.1), Inches(2.1)+i*Inches(0.7),
              Inches(2.9), Inches(0.35), size=13, bold=True, color=col)
        txbox(sl, meaning, Inches(10.1), Inches(2.43)+i*Inches(0.7),
              Inches(2.9), Inches(0.28), size=12, color=C_DARK)

    highlight_box(sl,
        '全SNR帯域で\n赤↑（CER悪化）と 青/緑↑（品質改善）が共存\n→ 三重パラドックス確認',
        Inches(10.0), Inches(4.2), Inches(3.1), Inches(2.0),
        bg=RGBColor(0xFF,0xEB,0xEE), txt_color=C_DARK, size=13)
    add_footer(sl, 17)

def s18_discussion1(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '考察①  なぜGTCRNがASRを悪化させるのか',
                  subtitle='Artifact Error仮説（Ochiai et al. TASLP 2024より）')
    # フロー
    boxes = [
        ('喉マイク音声\n（ドメイン外入力）', C_GRAY),
        ('GTCRN\n（DNS3で学習）',           C_GREEN),
        ('Artifact Error\n（非線形歪み）',   C_RED),
        ('Whisperが\n誤認識',               C_ORANGE),
    ]
    for i, (label, col) in enumerate(boxes):
        x = Inches(0.5) + i * Inches(3.2)
        add_rect(sl, x, Inches(2.0), Inches(2.8), Inches(1.2), col)
        txbox(sl, label, x, Inches(2.0), Inches(2.8), Inches(1.2),
              size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(boxes)-1:
            txbox(sl, '→', x+Inches(2.8), Inches(2.4), Inches(0.4), Inches(0.4),
                  size=22, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)

    bullet_box(sl, [
        'GTCRNはDNS3（気導マイク・英語）で学習 → 喉マイクはドメイン外',
        '低域偏重スペクトルをGTCRNが「ノイズ」と誤認識し過剰に操作',
        '人間の耳には「綺麗」に聞こえる音声が生成される（STOI/PESQ改善）',
        'しかしWhisperが学習した音響特徴と乖離 → Artifact Errorとして機能',
        '→ STOI/PESQはArtifact Errorを検出できない指標である',
    ], Inches(0.4), Inches(3.5), Inches(12.5), Inches(2.5), size=16)

    highlight_box(sl,
        '音質指標（STOI/PESQ）がOKでも、ASR指標（CER）がNGになりうる\n'
        '→ ASRを目的とした音声強調では、ASR指標で直接評価すべき',
        Inches(0.4), Inches(6.15), Inches(12.5), Inches(1.0),
        bg=RGBColor(0xFF,0xF9,0xC4), txt_color=C_DARK, size=15, bold=True)
    add_footer(sl, 18)

def s19_discussion2(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '考察②  DSP-onlyはなぜ全指標で失敗したか')
    # スペクトル図（概念）
    add_rect(sl, Inches(0.4), Inches(1.6), Inches(5.8), Inches(4.2),
             RGBColor(0xFF,0xF3,0xE0))
    txbox(sl, '喉マイクの周波数エネルギー分布（概念図）',
          Inches(0.5), Inches(1.68), Inches(5.6), Inches(0.4),
          size=13, bold=True, color=C_ORANGE)
    # 棒グラフ（概念）
    fig2, ax = plt.subplots(figsize=(4.0, 2.5))
    freqs  = ['100', '200', '300', '500', '1k', '2k', '4k', '8k']
    energy = [0.45, 0.35, 0.12, 0.04, 0.02, 0.01, 0.005, 0.002]
    colors = ['#FF9800']*3 + ['#BDBDBD']*5
    ax.bar(freqs, energy, color=colors)
    ax.axvline(2.5, color='red', linestyle='--', linewidth=2, label='HPF 300Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Relative Energy', fontsize=9)
    ax.legend(fontsize=9); ax.tick_params(labelsize=8)
    ax.set_title('Energy below 300Hz = most of throat mic signal', fontsize=9)
    fig2.tight_layout()
    buf2 = fig_to_stream(fig2)
    add_image_from_buf(sl, buf2, Inches(0.5), Inches(2.2), Inches(5.6))

    bullet_box(sl, [
        '300Hz以下に喉マイクの主エネルギーが集中',
        'HPF（300Hzカットオフ）でその帯域を全て除去',
        '→ 信号の大半を捨てているのと同じ状態',
        'STOI・PESQ・CER すべて悪化するのは必然',
        '',
        '【対策案】カットオフ周波数を下げる（50〜100Hz）',
        '【今後】感度分析で最適値を探索予定',
    ], Inches(6.6), Inches(1.6), Inches(6.4), Inches(4.5), size=15)
    add_footer(sl, 19)

def s20_discussion3(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '考察③  SNR -5dBで有意差が出なかった理由',
                  subtitle='天井効果（Ceiling Effect）')
    add_rect(sl, Inches(0.4), Inches(1.6), Inches(12.5), Inches(2.2),
             RGBColor(0xF3,0xE5,0xF5))
    txbox(sl, '天井効果とは？',
          Inches(0.5), Inches(1.68), Inches(12.3), Inches(0.4),
          size=16, bold=True, color=C_NAVY)
    txbox(sl,
          '測定値が上限（天井）に張り付いてしまい、条件間の差を検出できなくなる現象\n'
          '今回の場合：SNR −5dBではSEの有無に関わらず CER ≈ 1.0（100%）に到達',
          Inches(0.5), Inches(2.15), Inches(12.3), Inches(0.8),
          size=15, color=C_DARK)

    for i, (label, val, note) in enumerate([
        ('No SE  / white / −5dB',   'CER = 0.977', ''),
        ('DSP-only / white / −5dB', 'CER = 0.998', 'p = 0.493（n.s.）'),
        ('GTCRN / white / −5dB',    'CER = 1.203', 'p = 0.002（有意）'),
    ]):
        y = Inches(4.05) + i * Inches(0.75)
        col = RGBColor(0xFF,0xF9,0xC4) if i==1 else RGBColor(0xF5,0xF5,0xF5)
        add_rect(sl, Inches(0.5), y, Inches(12.1), Inches(0.65), col)
        txbox(sl, label, Inches(0.6), y+Inches(0.1), Inches(6.0), Inches(0.42),
              size=13, color=C_DARK)
        txbox(sl, val, Inches(7.0), y+Inches(0.1), Inches(2.5), Inches(0.42),
              size=13, bold=True, color=C_NAVY)
        txbox(sl, note, Inches(9.8), y+Inches(0.1), Inches(2.5), Inches(0.42),
              size=13, color=C_ORANGE if 'n.s.' in note else C_GREEN)

    txbox(sl,
          '→ SNR −5dBの天井効果は「SEが逆効果でない」証拠ではない\n'
          '　 単に測定限界に到達しているだけ。逆効果は消えていない。',
          Inches(0.4), Inches(6.5), Inches(12.5), Inches(0.7),
          size=15, bold=True, color=C_RED)
    add_footer(sl, 20)

def s21_summary_results(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '主要発見のまとめ')
    for i, (num, title, body_text, col) in enumerate([
        ('発見 1', 'SE適用で全条件CERが有意に悪化',
         '白色・ピンクノイズ × 5 SNR × 2モデル = 30比較中26件 p<0.05\n'
         'DSP-onlyもGTCRNも、全SNR帯域でNo SEより高いCERを示した',
         RGBColor(0xFF,0xEB,0xEE)),
        ('発見 2', 'GTCRNはSTOI・PESQを改善しながらCERを悪化させる',
         '例）白色SNR 0dB：STOI +0.14、PESQ +0.69、CER +0.33（悪化）\n'
         '「知覚品質改善・ASR性能悪化」の三重パラドックスが全ノイズ条件で一貫',
         RGBColor(0xFF,0xF9,0xC4)),
        ('発見 3', 'DSP-onlyは3指標すべてを悪化させる',
         'HPF（300Hz）が喉マイクの主エネルギー帯域を除去するため\n'
         'STOI・PESQ・CERいずれも No SE より悪化',
         RGBColor(0xE8,0xF5,0xE9)),
    ]):
        y = Inches(1.6) + i * Inches(1.7)
        add_rect(sl, Inches(0.4), y, Inches(12.5), Inches(1.55), col)
        txbox(sl, num, Inches(0.55), y+Inches(0.06),
              Inches(1.5), Inches(0.45), size=14, bold=True, color=C_NAVY)
        txbox(sl, title, Inches(1.9), y+Inches(0.06),
              Inches(10.8), Inches(0.45), size=16, bold=True, color=C_DARK)
        txbox(sl, body_text, Inches(0.55), y+Inches(0.6),
              Inches(12.1), Inches(0.75), size=13, color=C_DARK)
    add_footer(sl, 21)

def s22_phase2_approach(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, 'フェーズ2：アプローチの転換')
    txbox(sl, 'なぜSEではなくASR再学習か？',
          Inches(0.4), Inches(1.5), Inches(12.5), Inches(0.5),
          size=20, bold=True, color=C_NAVY)

    for i, (label, detail, col, bad) in enumerate([
        ('SEによる音声変換（フェーズ2-試行）',
         'GTCRNを「喉マイク→気導マイク変換」にFT\n→ 存在しない高周波を生成することはできない\n→ Loss収束せず、CER≈1.0で失敗',
         RGBColor(0xFF,0xEB,0xEE), True),
        ('ASRモデルのドメイン適応（採用）',
         'Whisper smallをTAPS喉マイク音声でFT（4,000発話・10時間）\n→ 「喉マイクの音響特性」をASR側が直接学習\n→ CER 0.269 → 0.095（64.6%改善）',
         RGBColor(0xE8,0xF5,0xE9), False),
    ]):
        y = Inches(2.2) + i * Inches(2.1)
        add_rect(sl, Inches(0.4), y, Inches(12.5), Inches(1.85), col)
        mark = '✗  ' if bad else '✓  '
        mc = C_RED if bad else C_GREEN
        txbox(sl, mark + label, Inches(0.55), y+Inches(0.1), Inches(12.0), Inches(0.45),
              size=16, bold=True, color=mc)
        txbox(sl, detail, Inches(0.7), y+Inches(0.6), Inches(12.0), Inches(1.1),
              size=13, color=C_DARK)
    add_footer(sl, 22)


def s23_phase2_result(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, 'フェーズ2：Whisperファインチューニング結果')

    # 大きな数値表示
    add_rect(sl, Inches(0.4), Inches(1.55), Inches(5.9), Inches(2.5), RGBColor(0xE3,0xF2,0xFD))
    txbox(sl, '未学習 Whisper small', Inches(0.5), Inches(1.65), Inches(5.7), Inches(0.45),
          size=14, color=C_NAVY, bold=True, align=PP_ALIGN.CENTER)
    txbox(sl, 'CER  0.269', Inches(0.5), Inches(2.1), Inches(5.7), Inches(0.8),
          size=32, bold=True, color=C_RED, align=PP_ALIGN.CENTER)
    txbox(sl, '（フェーズ1 ベースライン）', Inches(0.5), Inches(2.9), Inches(5.7), Inches(0.4),
          size=12, color=C_GRAY, align=PP_ALIGN.CENTER)

    txbox(sl, '→', Inches(6.4), Inches(2.3), Inches(0.9), Inches(0.8),
          size=36, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

    add_rect(sl, Inches(7.4), Inches(1.55), Inches(5.5), Inches(2.5), RGBColor(0xE8,0xF5,0xE9))
    txbox(sl, 'FT済み Whisper small', Inches(7.5), Inches(1.65), Inches(5.3), Inches(0.45),
          size=14, color=C_GREEN, bold=True, align=PP_ALIGN.CENTER)
    txbox(sl, 'CER  0.095', Inches(7.5), Inches(2.1), Inches(5.3), Inches(0.8),
          size=32, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)
    txbox(sl, '64.6% 改善', Inches(7.5), Inches(2.9), Inches(5.3), Inches(0.4),
          size=16, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)

    # 学習設定
    add_rect(sl, Inches(0.4), Inches(4.25), Inches(12.5), Inches(1.0), C_LIGHT)
    txbox(sl, '学習設定：train 4,000発話 / epochs=20（early stop, best=epoch3）/ batch=16 / lr=1e-5 / fp16',
          Inches(0.55), Inches(4.4), Inches(12.2), Inches(0.5),
          size=13, color=C_DARK)

    # メッセージ
    add_rect(sl, Inches(0.4), Inches(5.45), Inches(12.5), Inches(0.8), RGBColor(0xFF,0xF9,0xC4))
    txbox(sl, '「音声を直す」より「ASRを慣れさせる」方が喉マイクには効果的',
          Inches(0.55), Inches(5.55), Inches(12.2), Inches(0.5),
          size=17, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
    add_footer(sl, 23)


def s23b_2x2(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '補足：2×2 比較（ASRモデル × SE）', subtitle='FT後もSEは有害—ただし悪影響は縮小')

    # 2x2 テーブル
    cols = ['', 'No SE', 'GTCRN SE']
    rows = [
        ['未学習 Whisper', '0.269  [A]', '0.296  [B]  (+0.027)'],
        ['FT済み Whisper', '0.095  [C]', '0.1015 [D]  (+0.007)'],
    ]
    col_x = [Inches(0.4), Inches(3.5), Inches(8.0)]
    col_w = [Inches(3.0), Inches(4.3), Inches(4.5)]
    row_y = [Inches(1.65), Inches(2.6), Inches(3.55)]
    row_h = Inches(0.85)

    header_bg = RGBColor(0x1A, 0x23, 0x7E)
    cell_bgs = [
        [RGBColor(0xE3,0xF2,0xFD), RGBColor(0xFF,0xEB,0xEE)],
        [RGBColor(0xC8,0xE6,0xC9), RGBColor(0xFF,0xF9,0xC4)],
    ]

    # ヘッダ行
    for ci, (cx, cw, label) in enumerate(zip(col_x, col_w, cols)):
        add_rect(sl, cx, row_y[0], cw, row_h, header_bg)
        txbox(sl, label, cx+Inches(0.05), row_y[0]+Inches(0.22),
              cw-Inches(0.1), Inches(0.4),
              size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # データ行
    for ri, (row, ry) in enumerate(zip(rows, row_y[1:])):
        for ci, (cx, cw, cell) in enumerate(zip(col_x, col_w, row)):
            bg = header_bg if ci == 0 else cell_bgs[ri][ci-1]
            txt_col = C_WHITE if ci == 0 else C_DARK
            sz = 15 if ci == 0 else 20
            add_rect(sl, cx, ry, cw, row_h, bg)
            txbox(sl, cell, cx+Inches(0.05), ry+Inches(0.2),
                  cw-Inches(0.1), Inches(0.5),
                  size=sz, bold=(ci==0), color=txt_col, align=PP_ALIGN.CENTER)

    # 矢印と解説
    txbox(sl, '▼ SE悪化幅', Inches(8.15), Inches(2.55), Inches(4.3), Inches(0.35),
          size=12, color=C_GRAY, align=PP_ALIGN.CENTER)
    txbox(sl, '+0.027 → +0.007 に縮小',
          Inches(8.0), Inches(4.5), Inches(5.0), Inches(0.5),
          size=16, bold=True, color=C_GREEN, align=PP_ALIGN.LEFT)

    add_rect(sl, Inches(0.4), Inches(5.3), Inches(12.5), Inches(0.8), RGBColor(0xFF,0xF9,0xC4))
    txbox(sl, 'FTによりSEの悪影響は縮小するが消えない—SEの適用はFT後も推奨されない',
          Inches(0.55), Inches(5.42), Inches(12.2), Inches(0.5),
          size=16, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

    add_footer(sl, 24)


def s24_conclusion(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, 'まとめ')
    txbox(sl,
          '喉マイク音声に対するSEの効果と、ASRモデルのドメイン適応効果を定量評価した。',
          Inches(0.4), Inches(1.6), Inches(12.5), Inches(0.55),
          size=17, color=C_DARK)
    for i, (mark, text, col) in enumerate([
        ('（1）', 'DSP-onlyおよびGTCRNは全ノイズ条件でCERを有意に悪化させる（30検定中26件 p<0.05）', C_RED),
        ('（2）', 'GTCRNはSTOI・PESQを改善しながらCERを悪化させる三重パラドックスが全ノイズ条件で観測された', C_ORANGE),
        ('（3）', 'DSP-onlyは3指標すべてを悪化させ、喉マイクには不適切な前処理である', C_ORANGE),
        ('（4）', 'Whisper smallのFTによりCERが0.269→0.095（64.6%改善）—「ASR側の適応」が音声処理より有効', C_GREEN),
        ('（5）', 'FT後もSEの逆効果は残存するが悪化幅は縮小（+0.027→+0.007）—FTがSEアーティファクトへの感受性を低減', C_GREEN),
    ]):
        y = Inches(1.65) + i * Inches(0.77)
        add_rect(sl, Inches(0.4), y, Inches(12.5), Inches(0.68), C_LIGHT)
        txbox(sl, mark, Inches(0.55), y+Inches(0.1),
              Inches(0.8), Inches(0.45), size=14, bold=True, color=col)
        txbox(sl, text, Inches(1.25), y+Inches(0.1),
              Inches(11.4), Inches(0.45), size=14, color=C_DARK)
    add_footer(sl, 25)

def s25_future(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '今後の展望')
    for i, (pri, title, items, col) in enumerate([
        ('優先度：高', 'Whisper large-v3 でのFTと比較',
         ['smallとlargeでFT効果の差を比較',
          'モデルサイズとドメイン適応効果の関係を明らかにする'],
         RGBColor(0xFF,0xEB,0xEE)),
        ('優先度：中', 'ノイズ条件下でのWhisper FT評価',
         ['現在のFT評価はクリーン音声のみ',
          'フェーズ1と同じSNR条件（白色・ピンクノイズ）でのCER比較'],
         RGBColor(0xFF,0xF8,0xE1)),
        ('優先度：中', '実環境ノイズでの検証',
         ['現在は人工ノイズ（白色・ピンク）のみ',
          'MUSAN・DEMANDなどの実録音ノイズで再現実験'],
         RGBColor(0xE8,0xF5,0xE9)),
    ]):
        y = Inches(1.6) + i * Inches(1.75)
        add_rect(sl, Inches(0.4), y, Inches(12.5), Inches(1.6), col)
        txbox(sl, pri,   Inches(0.55), y+Inches(0.05), Inches(2.8), Inches(0.35),
              size=11, bold=True, color=C_GRAY)
        txbox(sl, title, Inches(0.55), y+Inches(0.38), Inches(12.1), Inches(0.42),
              size=16, bold=True, color=C_NAVY)
        for j, item in enumerate(items):
            txbox(sl, f'・ {item}',
                  Inches(0.7), y+Inches(0.88)+j*Inches(0.35), Inches(12.0), Inches(0.32),
                  size=13, color=C_DARK)
    add_footer(sl, 26)

def s26_refs(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '参考文献')
    refs = [
        ('[1]  C.O. Mawalim, S. Okada, M. Unoki,\n'
         '     "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront\n'
         '     Real-World Noisy Environments?"\n'
         '     Interspeech 2024, pp. 1735–1739, DOI: 10.21437/Interspeech.2024-129'),
        ('[2]  T. Ochiai, K. Iwamoto, M. Delcroix et al.,\n'
         '     "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?"\n'
         '     IEEE/ACM Transactions on Audio, Speech, and Language Processing, TASLP 2024,\n'
         '     arXiv:2404.14860'),
        ('[3]  X. Rong, C. Li, J. Xiao,\n'
         '     "GTCRN: A Speech Enhancement Model Requiring Ultra-Tiny Resources"\n'
         '     arXiv:2404.11567, 2024'),
        ('[4]  Y. Kim et al.,\n'
         '     "TAPS: Throat and Acoustic Pairing Speech Dataset"\n'
         '     HuggingFace: yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset'),
    ]
    for i, ref in enumerate(refs):
        y = Inches(1.65) + i * Inches(1.35)
        add_rect(sl, Inches(0.4), y, Inches(12.5), Inches(1.25),
                 RGBColor(0xF5,0xF5,0xF5))
        txbox(sl, ref, Inches(0.55), y+Inches(0.08), Inches(12.2), Inches(1.1),
              size=12, color=C_DARK)
    add_footer(sl, 27)

def s27_end(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, W, H, C_NAVY)
    add_rect(sl, 0, Inches(3.3), W, Inches(0.08), C_BLUE)
    txbox(sl, 'ご清聴ありがとうございました',
          Inches(0.6), Inches(2.2), Inches(12.1), Inches(1.0),
          size=36, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txbox(sl, '質問・ご意見をお待ちしております',
          Inches(0.6), Inches(3.6), Inches(12.1), Inches(0.6),
          size=20, color=RGBColor(0xBB,0xDE,0xFB), align=PP_ALIGN.CENTER)
    txbox(sl,
          '連絡先：著者名（所属機関）　author@example.ac.jp\n'
          'コード・データ：実験スクリプト一式（問い合わせ可）',
          Inches(0.6), Inches(5.0), Inches(12.1), Inches(0.9),
          size=14, color=RGBColor(0x90,0xCA,0xF9), align=PP_ALIGN.CENTER)
    add_footer(sl, 28)


# ── メイン ───────────────────────────────────────────────────
def main():
    cer, stoi, pesq = load()
    prs = new_prs()

    print('スライド生成中...')
    s01_title(prs)
    s02_agenda(prs)
    s03_throat_mic(prs)
    s04_se_intro(prs)
    s05_research_q(prs)
    s06_related1(prs)
    s07_related2(prs)
    s08_dataset(prs)
    s09_noise(prs)
    s10_models(prs)
    s11_cer(prs)
    s12_stoi_pesq(prs)
    s13_result_clean(prs, cer)
    s14_result_noisy(prs, cer)
    s15_stat(prs)
    s16_paradox(prs, cer, stoi)
    s17_delta(prs, cer, stoi, pesq)
    s18_discussion1(prs)
    s19_discussion2(prs)
    s20_discussion3(prs)
    s21_summary_results(prs)
    s22_phase2_approach(prs)
    s23_phase2_result(prs)
    s23b_2x2(prs)
    s24_conclusion(prs)
    s25_future(prs)
    s26_refs(prs)
    s27_end(prs)

    prs.save(OUT_PPTX)
    print(f'完了: {OUT_PPTX}  ({len(prs.slides)}枚)')

if __name__ == '__main__':
    main()
