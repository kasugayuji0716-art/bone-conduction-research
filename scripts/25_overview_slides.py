"""
研究概要スライド（5枚）
対象：この分野を知らない人
生成: results/overview_slides.pptx
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUT_PPTX = os.path.join(BASE_DIR, 'results', 'overview_slides.pptx')

# ── カラーパレット ────────────────────────────────────────────
C_NAVY   = RGBColor(0x1A, 0x23, 0x7E)
C_BLUE   = RGBColor(0x21, 0x96, 0xF3)
C_ORANGE = RGBColor(0xFF, 0x6F, 0x00)
C_GREEN  = RGBColor(0x2E, 0x7D, 0x32)
C_RED    = RGBColor(0xC6, 0x28, 0x28)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK   = RGBColor(0x21, 0x21, 0x21)
C_GRAY   = RGBColor(0x78, 0x90, 0x9C)
C_LIGHT  = RGBColor(0xF5, 0xF5, 0xF5)
C_YELLOW = RGBColor(0xFF, 0xF9, 0xC4)
C_LBLUE  = RGBColor(0xE3, 0xF2, 0xFD)
C_LGREEN = RGBColor(0xE8, 0xF5, 0xE9)
C_LRED   = RGBColor(0xFF, 0xEB, 0xEE)

W = Inches(13.33)
H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rgb_fill(shape, rgb):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb


def add_rect(slide, x, y, w, h, rgb, line=False):
    s = slide.shapes.add_shape(1, x, y, w, h)
    rgb_fill(s, rgb)
    if line:
        s.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
        s.line.width = Pt(0.5)
    else:
        s.line.fill.background()
    return s


def txbox(slide, text, x, y, w, h,
          size=18, bold=False, color=C_DARK,
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


def add_title_bar(slide, title, subtitle=None, n=None, total=5):
    add_rect(slide, 0, 0, W, Inches(1.3), C_NAVY)
    txbox(slide, title,
          Inches(0.45), Inches(0.15), Inches(11.8), Inches(0.85),
          size=26, bold=True, color=C_WHITE)
    if subtitle:
        txbox(slide, subtitle,
              Inches(0.45), Inches(0.88), Inches(11.0), Inches(0.38),
              size=13, color=RGBColor(0xBB, 0xDE, 0xFB))
    if n:
        txbox(slide, f'{n} / {total}',
              Inches(12.3), Inches(0.45), Inches(0.9), Inches(0.4),
              size=13, color=C_GRAY, align=PP_ALIGN.RIGHT)
    add_rect(slide, 0, H - Inches(0.3), W, Inches(0.3), RGBColor(0xE8, 0xEA, 0xF6))
    txbox(slide, '骨伝導マイク音声認識研究 — 概要',
          Inches(0.3), H - Inches(0.28), Inches(12.7), Inches(0.25),
          size=9, color=C_GRAY)


def add_img(slide, buf, x, y, w):
    p = slide.shapes.add_picture(buf, x, y, width=Inches(w))
    return p


# ── 図生成 ───────────────────────────────────────────────────

def fig_spectrum():
    """気導マイク vs 喉マイクのスペクトルイメージ"""
    freqs = np.linspace(0, 8000, 500)
    air   = np.exp(-((freqs - 2000) / 2500) ** 2) * 0.9 + 0.05
    throat = np.exp(-((freqs - 500) / 800) ** 2) * 0.85 + 0.02
    throat[freqs > 2500] *= np.exp(-(freqs[freqs > 2500] - 2500) / 800)

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.plot(freqs / 1000, air,    color='#2196F3', lw=2.2, label='気導マイク（通常）')
    ax.plot(freqs / 1000, throat, color='#C62828', lw=2.2, label='喉マイク', ls='--')
    ax.fill_between(freqs / 1000, throat, alpha=0.15, color='#C62828')
    ax.set_xlabel('Frequency (kHz)', fontsize=10)
    ax.set_ylabel('Energy (relative)', fontsize=10)
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.1)
    ax.legend(['Air mic (normal)', 'Throat mic'], fontsize=10, loc='upper right')
    ax.annotate('High-freq\nloss', xy=(5, 0.08), fontsize=10,
                color='#C62828', ha='center',
                arrowprops=dict(arrowstyle='->', color='#C62828'),
                xytext=(5, 0.35))
    ax.grid(alpha=0.25)
    fig.tight_layout(pad=0.4)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf


def fig_paradox():
    """三重パラドックス：GTCRN適用前後のSTOI・PESQ・CER"""
    labels  = ['STOI\n(intelligibility)', 'PESQ\n(quality)', 'CER\n(ASR error rate)']
    before  = [0.585, 1.018, 0.882]
    after   = [0.721, 1.706, 1.214]
    colors_b = ['#90CAF9', '#A5D6A7', '#EF9A9A']
    colors_a = ['#1565C0', '#2E7D32', '#C62828']
    arrows  = ['+', '+', '+']
    good    = [True, True, False]  # 上がって良いか

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.array([0, 1.8, 3.6])
    w = 0.65
    bars_b = ax.bar(x - w/2, before, w, color=colors_b, label='SE前', zorder=3)
    bars_a = ax.bar(x + w/2, after,  w, color=colors_a, label='SE後（GTCRN）', zorder=3)

    for i, (xp, bv, av, g) in enumerate(zip(x, before, after, good)):
        delta = av - bv
        sign  = '+' if delta > 0 else ''
        col   = '#2E7D32' if g and delta > 0 else ('#C62828' if not g and delta > 0 else '#2E7D32')
        ax.annotate(f'{sign}{delta:+.2f}',
                    xy=(xp + w/2, av + 0.03),
                    fontsize=11, fontweight='bold', color=col,
                    ha='center')
        arrow_col = '#2E7D32' if g else '#C62828'
        ax.annotate('', xy=(xp + w/2, av + 0.25),
                    xytext=(xp + w/2, av + 0.12),
                    arrowprops=dict(arrowstyle='->', color=arrow_col, lw=2.0))

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=10)
    ax.legend(['Before SE', 'After SE (GTCRN)'], fontsize=10, loc='upper left')
    ax.set_ylim(0, 1.65)
    ax.grid(axis='y', alpha=0.25, zorder=0)

    # 吹き出し
    ax.text(3.6 + w/2 + 0.25, 1.45,
            'Worse!\n(paradox)',
            fontsize=11, color='#C62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#C62828'))
    fig.tight_layout(pad=0.4)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf


def fig_ft_result():
    """Whisper FTの結果棒グラフ"""
    labels = ['Pretrained\nWhisper', 'Fine-tuned\nWhisper', 'Air mic\n(reference)']
    values = [0.544, 0.150, 0.131]
    colors = ['#EF9A9A', '#A5D6A7', '#90CAF9']
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(1, 0.105), xytext=(0, 0.259),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))
    ax.text(0.5, 0.58, '−72.4%', fontsize=12, color='#2E7D32',
            fontweight='bold', ha='center')
    ax.set_ylabel('CER (lower is better)', fontsize=10)
    ax.set_ylim(0, 0.65); ax.grid(axis='y', alpha=0.25, zorder=0)
    fig.tight_layout(pad=0.4)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf


def fig_2x2():
    """2×2比較"""
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    ax.axis('off')
    data = [['', 'No SE', 'GTCRN SE'],
            ['Pretrained', '0.544', '0.635\n(+0.091)'],
            ['Fine-tuned', '0.150', '0.186\n(+0.036)']]
    tbl = ax.table(cellText=[r[1:] for r in data[1:]],
                   rowLabels=[r[0] for r in data[1:]],
                   colLabels=data[0][1:],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.4, 1.8)
    cell_colors = {
        (1,0): '#E3F2FD', (1,1): '#FFEBEE',
        (2,0): '#C8E6C9', (2,1): '#FFF9C4',
    }
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#1A237E')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == -1:
            cell.set_facecolor('#1A237E')
            cell.set_text_props(color='white', fontweight='bold')
        elif (r, c) in cell_colors:
            cell.set_facecolor(cell_colors[(r, c)])
    fig.tight_layout(pad=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf


# ── スライド定義 ─────────────────────────────────────────────

def s1_title(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, W, H, C_NAVY)
    add_rect(sl, 0, Inches(2.8), W, Inches(0.06), C_BLUE)

    txbox(sl, '喉マイク音声認識の改善に\n音声強調は役立つか？',
          Inches(0.6), Inches(0.7), Inches(12.1), Inches(2.0),
          size=36, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    txbox(sl,
          '知覚品質（STOI・PESQ）とASR性能（CER）の逆行現象を喉マイクドメインで実証',
          Inches(0.6), Inches(2.95), Inches(12.1), Inches(0.6),
          size=17, color=RGBColor(0xBB, 0xDE, 0xFB), align=PP_ALIGN.LEFT)

    # キーワード箱
    for i, kw in enumerate(['骨伝導マイク', '音声強調（SE）', 'Whisper ASR', 'CER評価']):
        add_rect(sl, Inches(0.6 + i * 3.0), Inches(3.75), Inches(2.7), Inches(0.55),
                 RGBColor(0x28, 0x3A, 0x8C))
        txbox(sl, kw,
              Inches(0.6 + i * 3.0), Inches(3.78), Inches(2.7), Inches(0.48),
              size=14, color=RGBColor(0x90, 0xCA, 0xF9), align=PP_ALIGN.CENTER)

    txbox(sl, '使用データ: TAPS Dataset（韓国語 60話者・6,000発話）',
          Inches(0.6), Inches(4.55), Inches(12.1), Inches(0.4),
          size=13, color=C_GRAY, align=PP_ALIGN.LEFT)

    add_rect(sl, 0, H - Inches(0.3), W, Inches(0.3), RGBColor(0x0D, 0x15, 0x5C))
    txbox(sl, '1 / 5',
          Inches(12.3), H - Inches(0.28), Inches(0.9), Inches(0.25),
          size=9, color=C_GRAY, align=PP_ALIGN.RIGHT)


def s2_background(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '喉マイクとは？　なぜASRが難しいか', n=2)

    # 左：説明テキスト
    add_rect(sl, Inches(0.4), Inches(1.45), Inches(6.0), Inches(4.8), C_LBLUE)
    txbox(sl, '喉マイク（骨伝導マイク）とは',
          Inches(0.55), Inches(1.55), Inches(5.7), Inches(0.45),
          size=15, bold=True, color=C_NAVY)
    for i, line in enumerate([
        '・ 声帯の振動を皮膚・骨から直接収音',
        '・ 周囲の騒音を拾わない → 高騒音環境で有用',
        '・ 軍・工業・スポーツ・ヘルメット着用場面',
        '',
        '問題点',
        '・ 低域エネルギーが大半を占める',
        '・ 子音など高周波成分が大きく欠落',
        '→ 通常のASRには「未知の音」として届く',
        '',
        '結果：一般的な音声認識では',
        '　　　CER（文字誤り率）が非常に高くなる',
    ]):
        col  = C_NAVY if line.startswith('問題') else C_DARK
        bold = line.startswith('問題')
        txbox(sl, line,
              Inches(0.6), Inches(2.1) + i * Inches(0.31),
              Inches(5.6), Inches(0.3),
              size=13, bold=bold, color=col)

    # 右：スペクトル図
    buf = fig_spectrum()
    add_img(sl, buf, Inches(6.7), Inches(1.55), 6.1)
    txbox(sl, '図：気導マイクと喉マイクの周波数特性の違い（模式図）',
          Inches(6.7), Inches(5.3), Inches(6.3), Inches(0.35),
          size=10, color=C_GRAY, align=PP_ALIGN.CENTER)

    # CER数値
    add_rect(sl, Inches(0.4), Inches(6.35), Inches(12.5), Inches(0.75),
             RGBColor(0xFF, 0xEB, 0xEE))
    txbox(sl, '喉マイク生音声のCER: 0.269　vs　気導マイク: 0.131　→ 約2倍の誤り率',
          Inches(0.6), Inches(6.48), Inches(12.1), Inches(0.45),
          size=15, bold=True, color=C_RED, align=PP_ALIGN.CENTER)


def s3_prior_work(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '先行研究：「音声強調でASRを改善できる」は本当か？', n=3)

    # 左上：SEとは
    add_rect(sl, Inches(0.4), Inches(1.45), Inches(5.9), Inches(2.3), C_LBLUE)
    txbox(sl, '音声強調（SE）とは',
          Inches(0.55), Inches(1.55), Inches(5.6), Inches(0.4),
          size=15, bold=True, color=C_NAVY)
    for i, t in enumerate([
        'ノイズを除去し，音声を聴きやすくする技術',
        'STOI（了解度）・PESQ（音質）で評価される',
        '「ASRの前処理として使えば精度が上がる」',
        'という期待のもとに広く使われてきた',
    ]):
        txbox(sl, f'・ {t}',
              Inches(0.6), Inches(2.05) + i * Inches(0.41),
              Inches(5.6), Inches(0.38), size=13, color=C_DARK)

    # 右上：先行研究2件
    add_rect(sl, Inches(6.6), Inches(1.45), Inches(6.3), Inches(2.3), C_LRED)
    txbox(sl, '通説を覆した先行研究',
          Inches(6.75), Inches(1.55), Inches(6.0), Inches(0.4),
          size=15, bold=True, color=C_RED)
    for i, (ref, detail) in enumerate([
        ('Ochiai et al. (TASLP 2024)',
         'SEが生成する「アーティファクト誤差」が\n残留ノイズよりASRに有害と実証'),
        ('Mawalim et al. (Interspeech 2024)',
         '最新DL-SEは実環境でASRを\n必ずしも改善しないことを確認'),
    ]):
        y = Inches(2.0) + i * Inches(0.95)
        add_rect(sl, Inches(6.75), y, Inches(5.9), Inches(0.82),
                 RGBColor(0xFF, 0xCD, 0xCD))
        txbox(sl, ref,  Inches(6.9), y + Inches(0.04),
              Inches(5.6), Inches(0.32), size=12, bold=True, color=C_RED)
        txbox(sl, detail, Inches(6.9), y + Inches(0.38),
              Inches(5.6), Inches(0.38), size=12, color=C_DARK)

    # 下：未解決の問い
    add_rect(sl, Inches(0.4), Inches(3.95), Inches(12.5), Inches(1.05),
             RGBColor(0xFF, 0xF9, 0xC4))
    txbox(sl, '未解決の問い',
          Inches(0.6), Inches(4.0), Inches(2.5), Inches(0.35),
          size=14, bold=True, color=C_ORANGE)
    txbox(sl, '上記現象は「一般的なマイク・一般ノイズ」での研究。'
              '高周波成分が欠落した喉マイクというドメイン外入力に対して'
              'SEを適用したとき，知覚品質とASR性能はどう変化するか？',
          Inches(0.6), Inches(4.4), Inches(12.1), Inches(0.55),
          size=14, color=C_DARK)

    # 下：本研究のSEモデル
    add_rect(sl, Inches(0.4), Inches(5.2), Inches(12.5), Inches(1.9), C_LIGHT)
    txbox(sl, '本研究で使用したSEモデル',
          Inches(0.6), Inches(5.3), Inches(5.0), Inches(0.35),
          size=13, bold=True, color=C_NAVY)
    for i, (name, detail, col) in enumerate([
        ('DSP-only',
         'ハイパスフィルタ（300Hz）＋プリエンファシス＋正規化',
         RGBColor(0xFF, 0x98, 0x00)),
        ('GTCRN（48K パラメータの軽量DNN）',
         'DNS3（気導マイク・英語）で学習済み → 喉マイクはドメイン外',
         RGBColor(0x43, 0xA0, 0x47)),
    ]):
        y = Inches(5.7) + i * Inches(0.6)
        txbox(sl, f'● {name}', Inches(0.6), y, Inches(4.5), Inches(0.3),
              size=13, bold=True, color=col)
        txbox(sl, detail, Inches(5.3), y, Inches(7.5), Inches(0.3),
              size=13, color=C_DARK)


def s4_paradox(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '主要発見：知覚品質↑なのにASR性能↓（三重パラドックス）', n=4)

    # 左：図
    buf = fig_paradox()
    add_img(sl, buf, Inches(0.4), Inches(1.5), 6.0)
    txbox(sl, '図：SNR 0 dB 白色ノイズ条件での SE前後の指標変化',
          Inches(0.4), Inches(5.3), Inches(6.2), Inches(0.35),
          size=10, color=C_GRAY, align=PP_ALIGN.CENTER)

    # 右：解説
    add_rect(sl, Inches(6.8), Inches(1.5), Inches(6.1), Inches(3.8), C_LRED)
    txbox(sl, '何が起きているか',
          Inches(6.95), Inches(1.6), Inches(5.8), Inches(0.42),
          size=16, bold=True, color=C_RED)
    for i, t in enumerate([
        'GTCRN（SE）を適用すると…',
        '  STOI（了解度）: 0.585 → 0.721 ↑ 改善',
        '  PESQ（音質）:   1.018 → 1.706 ↑ 改善',
        '  CER（ASR誤り率）: 0.882 → 1.214 ↑ 悪化',
        '',
        '「聴こえやすくなったのに認識できない」',
    ]):
        col  = C_RED   if '悪化' in t else \
               C_GREEN if '改善' in t else \
               C_NAVY  if '聴こえ' in t else C_DARK
        bold = '聴こえ' in t
        txbox(sl, t, Inches(6.95), Inches(2.15) + i * Inches(0.42),
              Inches(5.8), Inches(0.38), size=13, bold=bold, color=col)

    # 右下：原因
    add_rect(sl, Inches(6.8), Inches(5.45), Inches(6.1), Inches(1.15),
             RGBColor(0xFF, 0xF9, 0xC4))
    txbox(sl, '原因（Ochiai et al. TASLP 2024 の枠組みより）',
          Inches(6.95), Inches(5.5), Inches(5.8), Inches(0.38),
          size=12, bold=True, color=C_ORANGE)
    txbox(sl, 'SEの非線形処理が「アーティファクト誤差」を生成。\n'
              '自然界に存在しない音響パターンがASRを混乱させる。\n'
              'STOI・PESQはこの誤差を検出できない。',
          Inches(6.95), Inches(5.9), Inches(5.8), Inches(0.65),
          size=12, color=C_DARK)

    # 下：統計
    add_rect(sl, Inches(0.4), Inches(6.35), Inches(12.5), Inches(0.75),
             RGBColor(0xFF, 0xEB, 0xEE))
    txbox(sl, '統計検定：34ノイズ条件のうち30条件でWilcoxon検定を実施 → 26/30条件で有意差 (p < 0.05)',
          Inches(0.6), Inches(6.48), Inches(12.1), Inches(0.45),
          size=14, bold=True, color=C_RED, align=PP_ALIGN.CENTER)


def s5_mechanism(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, 'なぜASRが悪化するのか：スペクトル分析', n=5)

    spec_fig_path = os.path.join(BASE_DIR, 'results', 'figures', 'spectrum_analysis.png')
    if os.path.exists(spec_fig_path):
        with open(spec_fig_path, 'rb') as _f:
            spec_buf = BytesIO(_f.read())
        add_img(sl, spec_buf, Inches(0.3), Inches(1.45), 8.0)

    # 右：解説（グラフの読み方）
    txbox(sl, 'パネル(B)の差分グラフを読む',
          Inches(8.55), Inches(1.5), Inches(4.55), Inches(0.38),
          size=13, bold=True, color=C_NAVY)
    txbox(sl, '赤塗り＝GTCRN がエネルギー追加\n青塗り＝削除',
          Inches(8.55), Inches(1.92), Inches(4.55), Inches(0.45),
          size=12, color=C_GRAY)

    # 帯域カード（修正版）
    items = [
        ('0–50 Hz', '+15 dB', '極低域にアーティファクト', C_RED,
         'ASR の音韻識別には無関係な帯域'),
        ('1–4 kHz', '−1〜2 dB', 'フォルマント帯域を削減', C_ORANGE,
         '母音・子音の聞き分けに必要な成分'),
        ('4–8 kHz', '−12.7 dB', '高域を壊滅的に削除', C_RED,
         '喉マイクで既に欠落→さらに除去'),
    ]
    for i, (band, delta, title, col, sub) in enumerate(items):
        y = Inches(2.5) + i * Inches(1.1)
        add_rect(sl, Inches(8.55), y, Inches(4.55), Inches(1.0),
                 RGBColor(0xF5, 0xF5, 0xF5))
        txbox(sl, band,  Inches(8.65), y+Inches(0.05), Inches(1.1), Inches(0.32),
              size=11, bold=True, color=C_DARK)
        txbox(sl, delta, Inches(9.75), y+Inches(0.05), Inches(1.2), Inches(0.32),
              size=13, bold=True, color=col)
        txbox(sl, title, Inches(8.65), y+Inches(0.38), Inches(4.3), Inches(0.28),
              size=11, bold=True, color=col)
        txbox(sl, sub,   Inches(8.65), y+Inches(0.66), Inches(4.3), Inches(0.28),
              size=10, color=C_GRAY)

    # 結論ボックス
    add_rect(sl, Inches(8.55), Inches(5.9), Inches(4.55), Inches(1.3),
             RGBColor(0xFF, 0xEB, 0xEE))
    txbox(sl, '本質的な問題',
          Inches(8.7), Inches(5.95), Inches(4.2), Inches(0.32),
          size=12, bold=True, color=C_RED)
    txbox(sl,
          '喉マイクは元々 2 kHz 以上が欠落。\n'
          'GTCRN はそれを「ノイズ」と誤判定し\n'
          'さらに削除 → 逆方向の動作。\n'
          'STOI・PESQ はこの歪みを検出できない。',
          Inches(8.7), Inches(6.3), Inches(4.2), Inches(0.85),
          size=11, color=C_NAVY)


def s6_solution(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '解決策と結論：「音声を直す」より「ASRを慣れさせる」', n=6)

    # 左上：Whisper FT結果
    add_rect(sl, Inches(0.4), Inches(1.45), Inches(5.9), Inches(3.5), C_LGREEN)

    txbox(sl, '解決策：Whisper ファインチューニング（FT）',
          Inches(0.55), Inches(1.55), Inches(5.6), Inches(0.4),
          size=14, bold=True, color=C_GREEN)
    txbox(sl, '喉マイク音声 4,000発話でWhisper smallを再学習\n'
              '→ ASRモデルが喉マイクの音響特性を直接学習',
          Inches(0.6), Inches(2.05), Inches(5.6), Inches(0.55),
          size=13, color=C_DARK)
    buf = fig_ft_result()
    add_img(sl, buf, Inches(0.5), Inches(2.65), 5.3)

    # 右上：2×2表
    add_rect(sl, Inches(6.7), Inches(1.45), Inches(6.2), Inches(3.5), C_LIGHT)
    txbox(sl, '補足：SE × FT の組み合わせ比較',
          Inches(6.85), Inches(1.55), Inches(5.9), Inches(0.4),
          size=14, bold=True, color=C_NAVY)
    buf2 = fig_2x2()
    add_img(sl, buf2, Inches(6.8), Inches(2.05), 5.9)
    txbox(sl, 'FT後もSEの逆効果は残るが悪影響は縮小（+0.091→+0.036）',
          Inches(6.85), Inches(4.6), Inches(5.9), Inches(0.3),
          size=12, color=C_GRAY)

    # 下：まとめ
    add_rect(sl, Inches(0.4), Inches(5.1), Inches(12.5), Inches(2.05),
             RGBColor(0xFF, 0xF9, 0xC4))
    txbox(sl, 'まとめ',
          Inches(0.6), Inches(5.18), Inches(2.0), Inches(0.38),
          size=15, bold=True, color=C_NAVY)
    for i, (mark, text, col) in enumerate([
        ('①', 'STOI↑PESQ↑でもCERが悪化する逆行現象が喉マイクでも全ノイズ条件で一貫して観測された', C_RED),
        ('②', '知覚品質指標（STOI・PESQ）は喉マイクASR性能の指標として不適切', C_ORANGE),
        ('③', 'Whisper FTでCER 72.4%改善（0.544→0.150，10話者）—音声処理よりASR適応が根本解決', C_GREEN),
    ]):
        y = Inches(5.6) + i * Inches(0.5)
        txbox(sl, mark, Inches(0.6), y, Inches(0.5), Inches(0.42),
              size=14, bold=True, color=col)
        txbox(sl, text, Inches(1.1), y, Inches(11.6), Inches(0.42),
              size=14, color=C_DARK)


def s7_future(prs):
    sl = blank_slide(prs)
    add_title_bar(sl, '本研究の新規性・限界・今後の展望', n=7)

    # ── 左列：新規性 ──────────────────────────────────────────
    add_rect(sl, Inches(0.3), Inches(1.45), Inches(4.1), Inches(5.7),
             RGBColor(0xE8, 0xF5, 0xE9))
    txbox(sl, '本研究の新規性',
          Inches(0.45), Inches(1.55), Inches(3.8), Inches(0.38),
          size=14, bold=True, color=C_GREEN)
    novelty = [
        ('喉マイク特有のドメインで\nSE逆効果を系統的に実証',
         'Ochiai・Mawalim は気導マイク対象。\n高周波が構造的に欠落したドメインでの\n検証は本研究が初。'),
        ('三重パラドックスの定量化',
         'STOI↑PESQ↑CER↑が全ノイズ条件で\n一貫することを30条件・統計検定で示した。'),
        ('スペクトル分析による\nメカニズムの可視化',
         '4–8 kHz での −12.7 dB 削除という\n具体的な原因を実測で特定。'),
    ]
    for i, (title, detail) in enumerate(novelty):
        y = Inches(2.05) + i * Inches(1.65)
        add_rect(sl, Inches(0.45), y, Inches(3.8), Inches(1.5),
                 RGBColor(0xC8, 0xE6, 0xC9))
        txbox(sl, f'● {title}', Inches(0.55), y+Inches(0.05),
              Inches(3.6), Inches(0.55), size=12, bold=True, color=C_GREEN)
        txbox(sl, detail, Inches(0.55), y+Inches(0.62),
              Inches(3.6), Inches(0.8), size=11, color=C_DARK)

    # ── 中列：限界 ────────────────────────────────────────────
    add_rect(sl, Inches(4.6), Inches(1.45), Inches(4.1), Inches(5.7),
             RGBColor(0xFF, 0xF9, 0xC4))
    txbox(sl, '現状の限界',
          Inches(4.75), Inches(1.55), Inches(3.8), Inches(0.38),
          size=14, bold=True, color=C_ORANGE)
    limits = [
        ('既存モデルの組み合わせ',
         'GTCRN も Whisper も既存モデルの\n適用のみ。SE の喉マイク向け\n再設計はしていない。'),
        ('韓国語・単一データセット',
         'TAPS のみ使用。他言語・他データへの\n汎化性は未検証。'),
        ('クリーン条件でのFT',
         'FT はクリーン音声のみ。\nノイズ下での頑健性評価が不足。'),
        ('音素レベルの分析なし',
         'どの音素が特に誤認識されるかの\n詳細分析は未実施。'),
    ]
    for i, (title, detail) in enumerate(limits):
        y = Inches(2.05) + i * Inches(1.22)
        add_rect(sl, Inches(4.75), y, Inches(3.8), Inches(1.1),
                 RGBColor(0xFF, 0xF0, 0xB0))
        txbox(sl, f'▲ {title}', Inches(4.85), y+Inches(0.05),
              Inches(3.6), Inches(0.32), size=12, bold=True, color=C_ORANGE)
        txbox(sl, detail, Inches(4.85), y+Inches(0.4),
              Inches(3.6), Inches(0.65), size=11, color=C_DARK)

    # ── 右列：展望 ────────────────────────────────────────────
    add_rect(sl, Inches(8.9), Inches(1.45), Inches(4.1), Inches(5.7),
             C_LBLUE)
    txbox(sl, '今後の展望',
          Inches(9.05), Inches(1.55), Inches(3.8), Inches(0.38),
          size=14, bold=True, color=C_NAVY)
    futures = [
        ('ドメイン適合型 SE の開発',
         'TAPS で学習した SE-conformer\n（Kim et al. 2025）との組み合わせ。\nFT × 適合型SE で CER がどこまで下がるか。'),
        ('ノイズ下での FT 評価',
         '実環境ノイズ（MUSAN・DEMAND）を\n使った頑健な学習と評価。'),
        ('音素誤認識の分析',
         '喉マイクで特に苦手な音素を特定し\nデータ収集・学習戦略に活かす。'),
        ('Whisper large-v3 との比較',
         'モデルサイズとドメイン適応効果\nのトレードオフを定量化。'),
    ]
    for i, (title, detail) in enumerate(futures):
        y = Inches(2.05) + i * Inches(1.22)
        add_rect(sl, Inches(9.05), y, Inches(3.8), Inches(1.1),
                 RGBColor(0xBB, 0xDE, 0xFB))
        txbox(sl, f'→ {title}', Inches(9.15), y+Inches(0.05),
              Inches(3.6), Inches(0.32), size=12, bold=True, color=C_NAVY)
        txbox(sl, detail, Inches(9.15), y+Inches(0.4),
              Inches(3.6), Inches(0.65), size=11, color=C_DARK)


# ── メイン ───────────────────────────────────────────────────
def main():
    prs = new_prs()
    print('概要スライド生成中...')
    s1_title(prs)
    s2_background(prs)
    s3_prior_work(prs)
    s4_paradox(prs)
    s5_mechanism(prs)
    s6_solution(prs)
    s7_future(prs)
    prs.save(OUT_PPTX)
    print(f'完了: {OUT_PPTX}  (7枚)')


if __name__ == '__main__':
    main()
