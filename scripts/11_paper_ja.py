"""
日本語論文フォーマット PDF 生成
情報処理学会・日本音響学会スタイル（2段組）
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether, HRFlowable,
    NextPageTemplate, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ── フォント登録 ─────────────────────────────────────────────
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))    # 明朝体
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5')) # ゴシック体

MINCHO = 'HeiseiMin-W3'
GOTHIC = 'HeiseiKakuGo-W5'

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FIG_DIR  = os.path.join(BASE_DIR, 'results', 'figures')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
OUT_PDF  = os.path.join(BASE_DIR, 'results', 'paper_ja.pdf')

# ── スタイル定義 ─────────────────────────────────────────────
W, H = A4  # 210 x 297 mm
MARGIN_TOP    = 20*mm
MARGIN_BOTTOM = 20*mm
MARGIN_LEFT   = 20*mm
MARGIN_RIGHT  = 20*mm
COL_GAP       = 5*mm
COL_W = (W - MARGIN_LEFT - MARGIN_RIGHT - COL_GAP) / 2

def style(name, font=MINCHO, size=9, leading=14, align=TA_JUSTIFY, **kw):
    return ParagraphStyle(name, fontName=font, fontSize=size,
                          leading=leading, alignment=align, **kw)

S_BODY      = style('body')
S_BODY_G    = style('body_g', font=GOTHIC)
S_HEADING1  = style('h1', font=GOTHIC, size=10, leading=16,
                    spaceBefore=8, spaceAfter=4)
S_HEADING2  = style('h2', font=GOTHIC, size=9, leading=14,
                    spaceBefore=6, spaceAfter=2)
S_CAPTION   = style('cap', font=GOTHIC, size=8, leading=11,
                    align=TA_CENTER, spaceBefore=3)
S_ABSTRACT  = style('abs', size=8, leading=12,
                    leftIndent=4*mm, rightIndent=4*mm)
S_REF       = style('ref', size=7.5, leading=11, spaceBefore=1)
S_TITLE_JA  = style('tja', font=GOTHIC, size=14, leading=20, align=TA_CENTER)
S_TITLE_EN  = style('ten', font=GOTHIC, size=9,  leading=13, align=TA_CENTER,
                    textColor=colors.HexColor('#37474f'))
S_AUTHOR    = style('aut', font=MINCHO, size=9, leading=13, align=TA_CENTER)
S_AFFIL     = style('aff', font=MINCHO, size=8, leading=12, align=TA_CENTER,
                    textColor=colors.HexColor('#555555'))

# ── データ読み込み ────────────────────────────────────────────
def load_data():
    with open(SUMMARY, encoding='utf-8') as f:
        cer = {r['condition']: float(r['avg_cer']) for r in csv.DictReader(f)}
    with open(STOI_CSV, encoding='utf-8') as f:
        stoi = {r['condition']: float(r['avg_stoi']) for r in csv.DictReader(f)}
    return cer, stoi

# ── matplotlib 図を reportlab Image に変換 ───────────────────
def fig_to_image(fig, width_mm):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    scale = (width_mm * mm) / img.imageWidth
    img.drawWidth  = width_mm * mm
    img.drawHeight = img.imageHeight * scale
    return img

def load_png(fname, width_mm):
    path = os.path.join(FIG_DIR, fname)
    img = Image(path)
    scale = (width_mm * mm) / img.imageWidth
    img.drawWidth  = width_mm * mm
    img.drawHeight = img.imageHeight * scale
    return img

# ── 図の生成（論文用・小サイズ・英語ラベル） ─────────────────
def make_snr_fig(cer, noise_type, width_mm=82):
    SNR = ['-5', '+0', '+5', '+10', '+20']
    XLABELS = ['-5', '0', '+5', '+10', '+20']
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    styles = {
        'no_se':    ('#2196F3', 'o', 'No SE'),
        'dsp_only': ('#FF9800', 's', 'DSP-only'),
        'gtcrn':    ('#4CAF50', '^', 'GTCRN'),
    }
    for model, (col, mrk, lbl) in styles.items():
        ys = []
        for snr in SNR:
            sc = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            key = f'{model}/{noise_type}/{sc}' if model != 'no_se' else f'no_se/{noise_type}/{sc}'
            ys.append(cer.get(key))
        ax.plot(XLABELS, ys, marker=mrk, color=col, label=lbl, linewidth=1.5, markersize=5)
    ax.set_xlabel('SNR (dB)', fontsize=8)
    ax.set_ylabel('CER', fontsize=8)
    ax.set_ylim(0, 1.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig_to_image(fig, width_mm)

def make_scatter_fig(cer, stoi, width_mm=82):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    configs = [
        ('no_se',    '#2196F3', 'o', 'No SE'),
        ('dsp_only', '#FF9800', 's', 'DSP-only'),
        ('gtcrn',    '#4CAF50', '^', 'GTCRN'),
    ]
    for model, col, mrk, lbl in configs:
        xs, ys = [], []
        for key in stoi:
            if not key.startswith(model + '/') or 'snr_' not in key:
                continue
            if key not in cer:
                continue
            xs.append(stoi[key])
            ys.append(cer[key])
        ax.scatter(xs, ys, c=col, marker=mrk, s=40, alpha=0.8, label=lbl)
    ax.set_xlabel('STOI ↑', fontsize=8)
    ax.set_ylabel('CER ↓', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.5)
    return fig_to_image(fig, width_mm)

def make_bar_clean(cer, width_mm=82):
    conds  = ['baseline_acoustic','baseline_throat','dsp_only/clean','gtcrn/clean']
    labels = ['Acoustic\n(Ref)', 'Throat\n(Raw)', 'Throat\n+DSP', 'Throat\n+GTCRN']
    cols   = ['#1565C0','#78909C','#FF9800','#4CAF50']
    vals   = [cer[c] for c in conds]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bars = ax.bar(labels, vals, color=cols, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('CER', fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig_to_image(fig, width_mm)

# ── 結果テーブル ─────────────────────────────────────────────
def make_result_table(cer, stoi):
    header = ['条件', 'ノイズ', 'SNR', 'CER↓', 'STOI↑']
    SNR = ['-5', '+0', '+5', '+10', '+20']
    rows = [header]

    def row(cond, nt, snr):
        cv = cer.get(cond, '-')
        sv = stoi.get(cond, '-')
        return [cond.replace('snr_+','').replace('dB',''),
                nt, snr,
                f'{cv:.3f}' if isinstance(cv, float) else cv,
                f'{sv:.3f}' if isinstance(sv, float) else sv]

    for c, nt, snr in [
        ('baseline_acoustic','—','clean'),
        ('baseline_throat','—','clean'),
        ('dsp_only/clean','—','clean'),
        ('gtcrn/clean','—','clean'),
    ]:
        rows.append(row(c, nt, snr))

    for model in ['no_se','dsp_only','gtcrn']:
        for nt in ['white','pink']:
            for snr in SNR:
                sc = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
                key = f'{model}/{nt}/{sc}' if model != 'no_se' else f'no_se/{nt}/{sc}'
                rows.append(row(key, nt, snr))

    col_widths = [COL_W*0.44, COL_W*0.14, COL_W*0.12, COL_W*0.15, COL_W*0.15]

    def cell_bg(val_str):
        try:
            v = float(val_str)
            if v < 0.30:   return colors.HexColor('#c8e6c9')
            elif v < 0.50: return colors.HexColor('#fff9c4')
            elif v < 0.80: return colors.HexColor('#ffe0b2')
            else:           return colors.HexColor('#ffcdd2')
        except:
            return colors.white

    style_cmds = [
        ('FONTNAME',  (0,0), (-1,0), GOTHIC),
        ('FONTSIZE',  (0,0), (-1,-1), 6.5),
        ('LEADING',   (0,0), (-1,-1), 9),
        ('BACKGROUND',(0,0), (-1,0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN',     (0,0), (-1,-1), 'CENTER'),
        ('ALIGN',     (0,1), (0,-1), 'LEFT'),
        ('GRID',      (0,0), (-1,-1), 0.3, colors.HexColor('#bdbdbd')),
        ('TOPPADDING',(0,0), (-1,-1), 1),
        ('BOTTOMPADDING',(0,0),(-1,-1), 1),
    ]
    for i, row_data in enumerate(rows[1:], start=1):
        bg = cell_bg(row_data[3])
        style_cmds.append(('BACKGROUND', (3,i), (3,i), bg))

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl

# ── ページレイアウト ─────────────────────────────────────────
def build_pdf(cer, stoi):
    doc = BaseDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
    )

    # タイトルページ: 1段（全幅）→ 本文: 2段
    frame_full = Frame(MARGIN_LEFT, MARGIN_BOTTOM,
                       W - MARGIN_LEFT - MARGIN_RIGHT,
                       H - MARGIN_TOP - MARGIN_BOTTOM,
                       id='full')
    frame_left  = Frame(MARGIN_LEFT, MARGIN_BOTTOM,
                        COL_W, H - MARGIN_TOP - MARGIN_BOTTOM, id='left')
    frame_right = Frame(MARGIN_LEFT + COL_W + COL_GAP, MARGIN_BOTTOM,
                        COL_W, H - MARGIN_TOP - MARGIN_BOTTOM, id='right')

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(GOTHIC, 7)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(MARGIN_LEFT, H - 13*mm,
                          '骨伝導マイク音声認識研究　―　音声強調がWhisper CERに与える影響')
        canvas.drawRightString(W - MARGIN_RIGHT, H - 13*mm,
                               f'ドラフト版　2026年4月')
        canvas.setLineWidth(0.3)
        canvas.setStrokeColor(colors.HexColor('#1a237e'))
        canvas.line(MARGIN_LEFT, H - 14.5*mm, W - MARGIN_RIGHT, H - 14.5*mm)
        canvas.line(MARGIN_LEFT, MARGIN_BOTTOM - 5*mm,
                    W - MARGIN_RIGHT, MARGIN_BOTTOM - 5*mm)
        canvas.drawCentredString(W/2, MARGIN_BOTTOM - 9*mm, f'— {doc.page} —')
        canvas.restoreState()

    doc.addPageTemplates([
        PageTemplate(id='twocol',
                     frames=[frame_left, frame_right],
                     onPage=header_footer),
    ])

    story = []
    FULL = W - MARGIN_LEFT - MARGIN_RIGHT  # 全幅

    # ── タイトルブロック（2段にまたがるテーブルで実現） ────────
    title_tbl = Table([[
        Paragraph('喉マイク音声に対する音声強調が<br/>自動音声認識に与える影響の定量的評価', S_TITLE_JA),
    ]], colWidths=[FULL])
    title_tbl.setStyle(TableStyle([('BOTTOMPADDING',(0,0),(-1,-1),4)]))

    en_title_tbl = Table([[
        Paragraph('Quantitative Evaluation of Speech Enhancement Effects<br/>'
                  'on Automatic Speech Recognition for Bone Conduction Microphone Audio', S_TITLE_EN),
    ]], colWidths=[FULL])

    author_tbl = Table([[
        Paragraph('○ 著者名　（所属機関）', S_AUTHOR),
    ]], colWidths=[FULL])

    story += [
        Spacer(1, 3*mm),
        title_tbl,
        Spacer(1, 2*mm),
        en_title_tbl,
        Spacer(1, 3*mm),
        author_tbl,
        Spacer(1, 4*mm),
        HRFlowable(width=FULL, thickness=0.5, color=colors.HexColor('#1a237e')),
        Spacer(1, 2*mm),
    ]

    # ── あらまし ────────────────────────────────────────────
    abstract_text = (
        '骨伝導マイク（喉マイク）は高騒音環境下での音声通信に有用であるが，'
        'その音声は低域偏重かつ高周波成分が欠落するという独特の周波数特性を持つ．'
        '本研究では，DSPベースおよびニューラルネットワークベースの音声強調（SE）を喉マイク音声に適用し，'
        'Whisper を用いた文字誤り率（CER）への影響を定量的に評価した．'
        'TAPSデータセット（韓国語，50発話）を用い，白色・ピンクノイズを SNR −5〜+20 dB で付加した'
        '計34条件で実験を行った．'
        'その結果，全ノイズ条件においてSE適用後のCERがSE未適用より悪化した．'
        '特にGTCRNはSTOI（短時間客観了解度）を改善しながらCERを悪化させるという'
        '知覚品質とASR性能の乖離を示した．'
        'この現象はOchiai ら（TASLP 2024）が指摘するアーティファクト誤差に起因すると考えられ，'
        '喉マイク音声に対して気導マイク学習済みSEモデルを適用することの危険性を示す．'
    )
    abs_tbl = Table([[Paragraph(abstract_text, S_ABSTRACT)]],
                    colWidths=[FULL])
    abs_tbl.setStyle(TableStyle([
        ('BOX', (0,0),(-1,-1), 0.5, colors.HexColor('#1a237e')),
        ('BACKGROUND',(0,0),(-1,-1), colors.HexColor('#f5f5f5')),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))

    story += [
        Paragraph('【あらまし】', style('abs_h', font=GOTHIC, size=8, leading=11)),
        abs_tbl,
        Spacer(1, 2*mm),
        HRFlowable(width=FULL, thickness=0.5, color=colors.HexColor('#1a237e')),
        Spacer(1, 3*mm),
    ]

    # ── 本文（2段組開始） ────────────────────────────────────
    def h1(text): return Paragraph(f'{text}', S_HEADING1)
    def h2(text): return Paragraph(f'{text}', S_HEADING2)
    def p(text):  return Paragraph(text, S_BODY)
    def sp(h=2):  return Spacer(1, h*mm)

    # 1. はじめに
    story += [
        h1('1. はじめに'),
        p('骨伝導マイク（喉マイク・振動ピックアップ）は，ヘルメット着用や騒音環境下でも'
          '安定した音声収音が可能なため，軍事・工業・スポーツ用途で利用されている．'
          'しかし，その出力音声は低域エネルギーが支配的であり，'
          '高周波成分が著しく欠落するという特性を持つ．'
          'この特性により，一般的な音声認識（ASR）システムでは高い文字誤り率（CER）が生じる．'),
        sp(),
        p('音声強調（Speech Enhancement: SE）は，ノイズ除去や音質改善を目的とした前処理として'
          '広く用いられる．しかし，Mawalim ら [1] は深層学習型SEが実環境では'
          'ASR性能を必ずしも改善しないことを示した．'
          'また Ochiai ら [2] は，SEが導入するアーティファクト誤差（artifact error）が'
          '残留ノイズよりもASRに有害であることを定量的に示している．'),
        sp(),
        p('本稿では，喉マイク音声という特殊なドメインにおいて，'
          'DSPベースおよびGTCRN [3] を用いたニューラルSEが'
          'Whisperの韓国語CERに与える影響を34条件にわたって評価し，'
          'SE逆効果が生じる条件を特定することを目的とする．'),
        sp(3),
    ]

    # 2. 実験設計
    story += [
        h1('2. 実験設計'),
        h2('2.1 データセット'),
        p('TAPSデータセット（Throat and Acoustic Pairing Speech Dataset）[4] の'
          'テストセットから50発話を使用した．'
          '各発話に対して喉マイク収録音声と気導マイク収録音声のペアが存在し，'
          '韓国語の正解テキスト（metadata.csv）が付属する．'
          '話者は1名（p00），サンプリングレートは16 kHz である．'),
        sp(),
        h2('2.2 ノイズ付加'),
        p('喉マイク音声に対し，白色ノイズおよびピンクノイズを SNR '
          '−5, 0, +5, +10, +20 dB で混合し，計10条件のノイズ付き音声を生成した．'
          'ノイズはターゲット音声の RMS を基準に振幅を調整した．'),
        sp(),
        h2('2.3 音声強調モデル'),
        p('<b>DSP-only</b>：Butterworth 6次ハイパスフィルタ（カットオフ300 Hz）+'
          'プリエンファシス（係数0.97）+RMS正規化の組み合わせ．'),
        p('<b>GTCRN</b>：48.2K パラメータの超軽量ニューラルSEモデル [3]．'
          'DNS3コーパス（気導マイク・英語）で学習済み．'
          'STFT（窓長512，シフト256，Hann窓）を用いた複素領域処理．'),
        sp(),
        h2('2.4 評価指標'),
        p('<b>CER</b>（Character Error Rate）：faster-whisper small モデルにより'
          'transcribe し，jiwer ライブラリで算出した（言語指定：ko）．'),
        p('<b>STOI</b>（Short-Time Objective Intelligibility）：'
          'クリーン喉マイク音声を参照としてpystoi で算出した（値域0〜1，高いほど良好）．'),
        sp(3),
    ]

    # 3. 実験結果
    story += [
        h1('3. 実験結果'),
        h2('3.1 クリーン条件'),
        p('図1にクリーン条件のCER比較を示す．'
          '気導マイクベースライン（CER=0.131）に対し，喉マイク生音声は0.269と大幅に悪化する．'
          'DSP適用（0.416）はさらに悪化するが，GTCRN（0.296）は喉マイク生音声に近い性能を維持した．'
          'この結果はGTCRNがクリーン条件では有用であることを示す．'),
        sp(),
        KeepTogether([
            make_bar_clean(cer, width_mm=COL_W/mm),
            Paragraph('図1　クリーン条件における各手法のCER比較', S_CAPTION),
        ]),
        sp(),
        h2('3.2 ノイズ条件'),
        p('図2および図3に白色・ピンクノイズ下でのSNRとCERの関係を示す．'
          '全SEモデルが全SNR条件でSE未適用（No SE）よりも高いCERを示した．'
          'GTCRNはSNR +20 dBでは No SE に近い値を示すものの（white: 0.378 vs 0.330），'
          'SNRが低下するほど乖離が拡大し，SNR 0 dBでは1.214（No SE: 0.882）に達した．'),
        sp(),
        KeepTogether([
            make_snr_fig(cer, 'white', width_mm=COL_W/mm),
            Paragraph('図2　SNR対CER（白色ノイズ）', S_CAPTION),
        ]),
        sp(),
        KeepTogether([
            make_snr_fig(cer, 'pink', width_mm=COL_W/mm),
            Paragraph('図3　SNR対CER（ピンクノイズ）', S_CAPTION),
        ]),
        sp(),
        h2('3.3 STOIとCERの乖離'),
        p('図4に全ノイズ条件のSTOI対CER散布図を示す．'
          'GTCRNの点群は No SE と比較してSTOI軸方向（右方向）に移動する一方，'
          'CER軸方向（上方向）にも移動するという逆説的なパターンを示した．'
          '例えば白色ノイズ SNR 0 dB では，'
          'STOI：0.585→0.721（+0.136），CER：0.882→1.214（+0.332）となった．'
          '一方 DSP-only は STOI・CER ともに No SE より悪化した．'),
        sp(),
        KeepTogether([
            make_scatter_fig(cer, stoi, width_mm=COL_W/mm),
            Paragraph('図4　STOI対CER散布図．GTCRNはSTOI改善・CER悪化の乖離を示す', S_CAPTION),
        ]),
        sp(3),
    ]

    # 4. 考察
    story += [
        h1('4. 考察'),
        h2('4.1 アーティファクト誤差仮説'),
        p('Ochiai ら [2] はSEによる誤差をアーティファクト誤差・干渉誤差・ノイズ誤差の3成分に分解し，'
          'アーティファクト誤差がASRに最も有害であることを示した．'
          '本実験でGTCRNがSTOIを改善しながらCERを悪化させる現象は，'
          'GTCRNが知覚的に自然な音声を生成しつつ，'
          'Whisperが学習した音響特徴と乖離したアーティファクトを導入していると解釈できる．'),
        sp(),
        h2('4.2 ドメインミスマッチの影響'),
        p('GTCRNはDNS3（気導マイク，主に英語）で学習されており，'
          '喉マイク音声は学習分布外である．'
          '低域偏重かつ高周波欠落という喉マイクの周波数特性は，'
          'GTCRNに「ノイズ」として誤認識され，過剰な抑圧や変形を引き起こす可能性がある．'
          '喉マイク専用学習済みSEモデルを用いた場合の効果検証が今後の課題である．'),
        sp(),
        h2('4.3 DSP-onlyの問題点'),
        p('DSP-only はSTOI・CER の両方を悪化させた．'
          '300 Hz ハイパスフィルタは喉マイクのエネルギーの大半（300 Hz 以下）を除去するため，'
          'むしろ有害な前処理となっていると考えられる．'
          'カットオフ周波数のパラメータ感度分析が必要である．'),
        sp(3),
    ]

    # 5. おわりに
    story += [
        h1('5. おわりに'),
        p('本稿では喉マイク音声に対するSEの効果をCERとSTOIの両面から評価した．'
          '主要な知見は以下の通りである．'),
        p('（1）DSP-onlyおよびGTCRNは全ノイズ条件でCERを悪化させる．'),
        p('（2）GTCRNはSTOIを改善しながらCERを悪化させる知覚品質とASR性能の乖離を示す．'),
        p('（3）本現象は気導マイク学習済みSEモデルの喉マイクへのドメイン外適用に起因する'
          'アーティファクト誤差によって説明される．'),
        sp(),
        p('今後は，Whisper large-v3 による再評価，PESQ計測，'
          'DSPパラメータ感度分析，および実環境ノイズを用いた検証を行う予定である．'),
        sp(3),
    ]

    # 結果テーブル（2段またがり → 1段の幅で掲載）
    story += [
        h1('付表　全条件のCER・STOI計測結果'),
        make_result_table(cer, stoi),
        sp(),
        Paragraph(
            '色：緑(CER<0.30)・黄(0.30–0.50)・橙(0.50–0.80)・赤(≥0.80)',
            style('note', font=GOTHIC, size=6.5, leading=9,
                  textColor=colors.HexColor('#555555'))
        ),
        sp(3),
    ]

    # 謝辞
    story += [
        h1('謝辞'),
        p('本研究に使用したTAPSデータセットを公開してくださった研究者の方々に感謝する．'),
        sp(3),
    ]

    # 参考文献
    refs = [
        '[1] C.O. Mawalim, S. Okada, M. Unoki, "Are Recent Deep Learning-Based Speech '
        'Enhancement Methods Ready to Confront Real-World Noisy Environments?" '
        'Proc. Interspeech 2024, pp. 1735–1739, DOI: 10.21437/Interspeech.2024-129.',

        '[2] T. Ochiai et al., "Rethinking Processing Distortions: How Do They Affect '
        'the Downstream ASR Performance?" IEEE/ACM Trans. Audio, Speech, Lang. Process., '
        'TASLP 2024, arXiv:2404.14860.',

        '[3] X. Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultra-Tiny Resources," '
        'arXiv:2404.11567, 2024.',

        '[4] Y. Kim et al., "TAPS: Throat and Acoustic Pairing Speech Dataset," '
        'HuggingFace: yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset.',
    ]
    story.append(h1('参考文献'))
    for r in refs:
        story.append(Paragraph(r, S_REF))
        story.append(sp(1))

    doc.build(story)
    print(f'完了: {OUT_PDF}')


if __name__ == '__main__':
    cer, stoi = load_data()
    build_pdf(cer, stoi)
