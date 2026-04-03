"""
日本音響学会 研究発表会 フォーマット .docx 生成
仕様: A4, 上17mm/下19mm/左右23mm, 2段組, 2〜4ページ
"""
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from copy import deepcopy

from docx import Document
from docx.shared import Pt, Mm, RGBColor, Cm, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
OUT_DOCX = os.path.join(BASE_DIR, 'results', 'paper_asj.docx')

# ── データ ───────────────────────────────────────────────────
def load_data():
    with open(SUMMARY,  encoding='utf-8') as f:
        cer  = {r['condition']: float(r['avg_cer'])  for r in csv.DictReader(f)}
    with open(STOI_CSV, encoding='utf-8') as f:
        stoi = {r['condition']: float(r['avg_stoi']) for r in csv.DictReader(f)}
    return cer, stoi

# ── 2段組XML設定 ─────────────────────────────────────────────
def set_two_columns(section, num=2, space_twips=425):
    """セクションに2段組を設定する"""
    sectPr = section._sectPr
    # 既存の cols を削除
    for old in sectPr.findall(qn('w:cols')):
        sectPr.remove(old)
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'),   str(num))
    cols.set(qn('w:space'), str(space_twips))  # 約7.5mm
    cols.set(qn('w:equalWidth'), '1')
    sectPr.append(cols)

def set_one_column(section):
    sectPr = section._sectPr
    for old in sectPr.findall(qn('w:cols')):
        sectPr.remove(old)
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '1')
    sectPr.append(cols)

# ── ヘッダー設定 ─────────────────────────────────────────────
def set_header(section, left_text, right_text):
    header = section.header
    header.is_linked_to_previous = False
    # 既存パラグラフをクリア
    for p in header.paragraphs:
        p.clear()
    p = header.paragraphs[0] if header.paragraphs else header.add_paragraph()

    # TAB で左右配置
    p.clear()
    run_left = p.add_run(left_text)
    run_left.font.size = Pt(8)

    # タブストップ（右端）
    pPr = p._p.get_or_add_pPr()
    tabs = OxmlElement('w:tabs')
    tab = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'right')
    tab.set(qn('w:pos'), '9360')  # 約165mm
    tabs.append(tab)
    pPr.append(tabs)

    p.add_run('\t')
    run_right = p.add_run(right_text)
    run_right.font.size = Pt(8)

    # ヘッダー下線
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '000000')
    pBdr.append(bottom)
    pPr.append(pBdr)

# ── フッター（ページ番号） ────────────────────────────────────
def set_footer(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in footer.paragraphs:
        p.clear()
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.size = Pt(9)
    # ページ番号フィールド
    fldChar1 = OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText'); instrText.text = ' PAGE '
    fldChar2 = OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1); run._r.append(instrText); run._r.append(fldChar2)

# ── 図生成 ───────────────────────────────────────────────────
def fig_bar_clean(cer, w_cm=7.5):
    conds  = ['baseline_acoustic','baseline_throat','dsp_only/clean','gtcrn/clean']
    labels = ['Acoustic\n(Ref)', 'Throat\n(Raw)', 'Throat\n+DSP', 'Throat\n+GTCRN']
    vals   = [cer[c] for c in conds]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    bars = ax.bar(labels, vals, color=['#1565C0','#78909C','#FF9800','#4CAF50'], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('CER', fontsize=9); ax.set_ylim(0, 0.58)
    ax.tick_params(labelsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.6)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf

def fig_snr(cer, noise_type):
    SNR = ['-5', '+0', '+5', '+10', '+20']
    XLABELS = ['-5','0','+5','+10','+20']
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for model, col, mrk, lbl in [
        ('no_se','#2196F3','o','No SE'),
        ('dsp_only','#FF9800','s','DSP-only'),
        ('gtcrn','#4CAF50','^','GTCRN'),
    ]:
        ys = []
        for snr in SNR:
            sc  = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            key = f'no_se/{noise_type}/{sc}' if model=='no_se' else f'{model}/{noise_type}/{sc}'
            ys.append(cer.get(key))
        ax.plot(XLABELS, ys, marker=mrk, color=col, label=lbl, linewidth=1.5, markersize=5)
    ax.set_xlabel('SNR (dB)', fontsize=9); ax.set_ylabel('CER', fontsize=9)
    ax.set_ylim(0, 1.4); ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.6)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf

def fig_scatter(cer, stoi):
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for model, col, mrk, lbl in [
        ('no_se','#2196F3','o','No SE'),
        ('dsp_only','#FF9800','s','DSP-only'),
        ('gtcrn','#4CAF50','^','GTCRN'),
    ]:
        xs, ys = [], []
        for key in stoi:
            if not key.startswith(model+'/') or 'snr_' not in key: continue
            if key not in cer: continue
            xs.append(stoi[key]); ys.append(cer[key])
        ax.scatter(xs, ys, c=col, marker=mrk, s=40, alpha=0.8, label=lbl)
    ax.set_xlabel('STOI ↑', fontsize=9); ax.set_ylabel('CER ↓', fontsize=9)
    ax.legend(fontsize=7); ax.tick_params(labelsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.6)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf

# ── Wordヘルパー ─────────────────────────────────────────────
def para(doc, text, size=10, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_before=0, space_after=3, first_indent_mm=0, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    if first_indent_mm:
        p.paragraph_format.first_line_indent = Mm(first_indent_mm)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    return p

def heading(doc, text, num, size=10):
    p = para(doc, f'{num}　{text}', size=size, bold=True,
             align=WD_ALIGN_PARAGRAPH.LEFT,
             space_before=6, space_after=2)
    return p

def subheading(doc, text, num, size=10):
    p = para(doc, f'{num}　{text}', size=size, bold=True,
             align=WD_ALIGN_PARAGRAPH.LEFT,
             space_before=4, space_after=1)
    return p

def body(doc, text, size=10):
    return para(doc, text, size=size, first_indent_mm=5, space_after=3)

def caption(doc, text, size=9):
    return para(doc, text, size=size, align=WD_ALIGN_PARAGRAPH.CENTER,
                space_before=1, space_after=6, bold=True)

def add_fig(doc, buf, width_cm, cap_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(0)
    p.add_run().add_picture(buf, width=Cm(width_cm))
    caption(doc, cap_text)

def add_hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    top = OxmlElement('w:top')
    top.set(qn('w:val'), 'single'); top.set(qn('w:sz'),'6')
    top.set(qn('w:space'),'1'); top.set(qn('w:color'),'000000')
    pBdr.append(top); pPr.append(pBdr)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)

# ── 結果テーブル（コンパクト版） ─────────────────────────────
def add_result_table(doc, cer, stoi):
    SNR = ['-5', '+0', '+5', '+10', '+20']
    rows = [['条件', 'ノイズ', 'SNR', 'CER↓', 'STOI↑']]

    def r(cond, nt, snr):
        short = cond.replace('snr_+','').replace('snr_','').replace('dB','')
        cv = cer.get(cond,'-'); sv = stoi.get(cond,'-')
        return [short, nt, snr,
                f'{cv:.3f}' if isinstance(cv,float) else cv,
                f'{sv:.3f}' if isinstance(sv,float) else sv]

    for c,nt,snr in [('baseline_acoustic','—','—'),('baseline_throat','—','—'),
                     ('dsp_only/clean','—','—'),('gtcrn/clean','—','—')]:
        rows.append(r(c,nt,snr))

    for model in ['no_se','dsp_only','gtcrn']:
        for nt in ['white','pink']:
            for snr in SNR:
                sc = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
                key = f'no_se/{nt}/{sc}' if model=='no_se' else f'{model}/{nt}/{sc}'
                rows.append(r(key,nt,snr))

    tbl = doc.add_table(rows=len(rows), cols=5)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 列幅（2段組の1段幅に収める）
    col_w = [Cm(4.5), Cm(1.2), Cm(0.9), Cm(1.1), Cm(1.1)]
    for ci, w in enumerate(col_w):
        for cell in tbl.columns[ci].cells:
            cell.width = w

    def bg(cell, hex_color):
        tc = cell._tc; tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto')
        shd.set(qn('w:fill'), hex_color); tcPr.append(shd)

    def cer_color(s):
        try:
            v = float(s)
            if v < 0.30: return 'c8e6c9'
            elif v < 0.50: return 'fff9c4'
            elif v < 0.80: return 'ffe0b2'
            else: return 'ffcdd2'
        except: return 'ffffff'

    for ri, row_data in enumerate(rows):
        row = tbl.rows[ri]
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = val
            run = cell.paragraphs[0].runs
            fs = Pt(7) if ri > 0 else Pt(7.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r_ in cell.paragraphs[0].runs:
                r_.font.size = fs
                if ri == 0: r_.font.bold = True
            if ri == 0:
                bg(cell, '1a237e')
                for r_ in cell.paragraphs[0].runs:
                    r_.font.color.rgb = RGBColor(255,255,255)
            elif ci == 3:
                bg(cell, cer_color(val))

    # TP/BP 余白を小さく
    for row in tbl.rows:
        for cell in row.cells:
            for p_ in cell.paragraphs:
                p_.paragraph_format.space_before = Pt(0)
                p_.paragraph_format.space_after  = Pt(0)

# ── メイン ───────────────────────────────────────────────────
def main():
    cer, stoi = load_data()
    doc = Document()

    # ── ページ設定 ──
    sec = doc.sections[0]
    sec.page_width   = Mm(210); sec.page_height  = Mm(297)
    sec.top_margin   = Mm(17);  sec.bottom_margin = Mm(19)
    sec.left_margin  = Mm(23);  sec.right_margin  = Mm(23)
    sec.header_distance = Mm(9)
    sec.footer_distance = Mm(9)

    # ヘッダー・フッター
    set_header(sec,
               '日本音響学会　2026年春季研究発表会',
               '論文番号：X-X-XX')
    set_footer(sec)

    # ── タイトルブロック（1段） ──
    set_one_column(sec)

    # 日本語タイトル
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after  = Pt(4)
    run = p_title.add_run('喉マイク音声に対する音声強調が自動音声認識に与える影響の定量的評価')
    run.font.size = Pt(14); run.font.bold = True

    # 英語タイトル
    p_en = doc.add_paragraph()
    p_en.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_en.paragraph_format.space_after = Pt(4)
    run = p_en.add_run(
        'Quantitative Evaluation of Speech Enhancement Effects on '
        'Automatic Speech Recognition for Bone Conduction Microphone Audio'
    )
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x37,0x47,0x4f)

    # 著者
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(2)
    run = p_auth.add_run('○ 著者名（所属機関）')
    run.font.size = Pt(10)

    add_hr(doc)

    # あらまし
    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p_abs_h.add_run('【あらまし】')
    r.font.size = Pt(9); r.font.bold = True
    p_abs_h.paragraph_format.space_after = Pt(1)

    p_abs = doc.add_paragraph(
        '骨伝導マイク（喉マイク）は高騒音環境での音声収音に有用であるが，'
        'その音声は低域偏重かつ高周波欠落という独特の周波数特性を持つ．'
        '本研究では，DSPベースおよびGTCRNニューラルSEを喉マイク音声に適用し，'
        'Whisperを用いた文字誤り率（CER）への影響を34条件で定量評価した．'
        'TAPSデータセット（韓国語，50発話）を使用し，白色・ピンクノイズを SNR −5〜+20 dB で付加した．'
        '全ノイズ条件でSE適用後のCERが悪化し，GTCRNはSTOIを改善しながら'
        'CERを悪化させる知覚品質とASR性能の乖離を示した．'
    )
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_after = Pt(2)
    p_abs.runs[0].font.size = Pt(9)

    # キーワード
    p_kw = doc.add_paragraph('【キーワード】骨伝導マイク，音声強調，自動音声認識，Whisper，CER，STOI')
    p_kw.runs[0].font.size = Pt(9)
    p_kw.paragraph_format.space_after = Pt(4)

    add_hr(doc)

    # ── 2段組本文 ──
    set_two_columns(sec, num=2, space_twips=425)

    # 1. はじめに
    heading(doc, 'はじめに', '1.')
    body(doc,
        '骨伝導マイク（喉マイク）は，ヘルメット着用や騒音環境下でも安定した音声収音が可能なため，'
        '軍事・工業・スポーツ用途で利用されている．'
        'しかし，その出力音声は低域エネルギーが支配的であり，'
        '高周波成分が著しく欠落するという特性を持つ．'
        'この特性により，一般的なASRシステムでは高いCERが生じる．')
    body(doc,
        '音声強調（SE）は前処理として広く用いられるが，Mawalimら [1] は'
        '深層学習型SEが実環境ではASR性能を必ずしも改善しないことを示した．'
        'またOchiaiら [2] はSEが導入するアーティファクト誤差がASRに'
        '最も有害であることを定量的に示している．')
    body(doc,
        '本稿では喉マイクという特殊なドメインにおいて，DSP-onlyおよびGTCRN [3] による'
        'SEがWhisperのCERに与える影響を34条件で評価し，'
        'SE逆効果が生じる条件の特定を目的とする．')

    # 2. 実験設計
    heading(doc, '実験設計', '2.')
    subheading(doc, 'データセット', '2.1')
    body(doc,
        'TAPSデータセット [4] のテストセットから50発話を使用した．'
        '喉マイク・気導マイクのペア音声と韓国語正解テキストが付属し，'
        '話者1名（p00），サンプリングレート16 kHz である．')

    subheading(doc, 'ノイズ付加', '2.2')
    body(doc,
        '白色・ピンクノイズをSNR −5, 0, +5, +10, +20 dBで混合し，'
        '計10条件のノイズ付き音声を生成した．')

    subheading(doc, '音声強調モデル', '2.3')
    body(doc,
        'DSP-only：Butterworth 6次ハイパスフィルタ（300 Hz）+'
        'プリエンファシス（0.97）+RMS正規化．')
    body(doc,
        'GTCRN：48.2K パラメータのニューラルSE [3]．'
        'DNS3（気導マイク，英語）で学習済み（喉マイクはドメイン外）．'
        'STFT（窓長512，シフト256）による複素領域処理．')

    subheading(doc, '評価指標', '2.4')
    body(doc,
        'CER：faster-whisper small（言語：ko）でtranscribeしjiwer で算出．'
        'STOI：クリーン喉マイク音声を参照にpystoi で算出（0〜1，高いほど良好）．')

    # 3. 実験結果
    heading(doc, '実験結果', '3.')
    subheading(doc, 'クリーン条件', '3.1')
    body(doc,
        '図1にクリーン条件のCERを示す．'
        '気導マイクベースライン（0.131）に対し，喉マイク生音声は0.269と悪化する．'
        'DSP適用（0.416）はさらに悪化するが，GTCRN（0.296）は生音声に近い値を維持した．')

    add_fig(doc, fig_bar_clean(cer), width_cm=7.5,
            cap_text='図1　クリーン条件のCER比較')

    subheading(doc, 'ノイズ条件', '3.2')
    body(doc,
        '図2・図3に白色・ピンクノイズ下でのSNRとCERの関係を示す．'
        '全SEモデルが全SNR条件でNo SEよりも高いCERを示した．'
        'GTCRNはSNR +20 dBでは No SE に近い値（white: 0.378 vs 0.330）だが，'
        'SNR低下とともに乖離が拡大し，SNR 0 dBでは1.214（No SE: 0.882）に達した．')

    add_fig(doc, fig_snr(cer,'white'), width_cm=7.5,
            cap_text='図2　SNR対CER（白色ノイズ）')
    add_fig(doc, fig_snr(cer,'pink'), width_cm=7.5,
            cap_text='図3　SNR対CER（ピンクノイズ）')

    subheading(doc, 'STOIとCERの乖離', '3.3')
    body(doc,
        '図4にSTOI対CER散布図を示す．GTCRNはNo SEと比較して'
        'STOI（右方向）とCER（上方向）が同時に増加するという逆説的パターンを示した．'
        '例えば白色SNR 0 dBでは，STOI: 0.585→0.721（+0.136），'
        'CER: 0.882→1.214（+0.332）となった．')

    add_fig(doc, fig_scatter(cer,stoi), width_cm=7.5,
            cap_text='図4　STOI対CER散布図')

    # 4. 考察
    heading(doc, '考察', '4.')
    body(doc,
        'Ochiaiら [2] が示したアーティファクト誤差の枠組みで解釈すると，'
        'GTCRNは知覚的に自然な音声を生成しながらも，'
        'Whisperの学習分布と乖離した非線形歪みを導入していると考えられる．'
        'GTCRNがDNS3（気導マイク）で学習されており，'
        '喉マイクの低域偏重スペクトルはドメイン外入力であることも一因である．')
    body(doc,
        'DSP-onlyはSTOI・CERの両方を悪化させた．'
        '300 Hzハイパスフィルタが喉マイクの主要エネルギー帯域を除去するためであり，'
        'カットオフ周波数の最適化が今後の課題である．')

    # 5. おわりに
    heading(doc, 'おわりに', '5.')
    body(doc,
        '喉マイク音声に対するSEの効果をCERとSTOIの両面から評価した結果，'
        '（1）DSP-onlyおよびGTCRNは全ノイズ条件でCERを悪化させる，'
        '（2）GTCRNはSTOIを改善しながらCERを悪化させる乖離を示す，'
        'という2点が明らかになった．'
        '今後はWhisper large-v3による再評価，PESQ計測，'
        'DSPパラメータ感度分析を行う予定である．')

    # 参考文献
    heading(doc, '参考文献', '')
    refs = [
        '[1] C.O. Mawalim et al., Interspeech 2024, pp.1735–1739.',
        '[2] T. Ochiai et al., TASLP 2024, arXiv:2404.14860.',
        '[3] X. Rong et al., arXiv:2404.11567, 2024.',
        '[4] Y. Kim et al., HuggingFace: yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset.',
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.runs[0].font.size = Pt(8.5)
        p.paragraph_format.space_after = Pt(2)

    # ── 付表（改ページ後・1段） ──
    doc.add_page_break()

    # 改ページ後のセクションを1段に戻す
    new_sec_elm = OxmlElement('w:sectPr')
    # 1段設定
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '1')
    new_sec_elm.append(cols)
    # 改ページ前のパラグラフに挿入
    last_p = doc.paragraphs[-1]._p
    last_p.append(new_sec_elm)

    heading(doc, '全条件のCER・STOI計測結果', '付表')
    add_result_table(doc, cer, stoi)
    p_note = doc.add_paragraph('色凡例：緑(CER<0.30)・黄(0.30–0.50)・橙(0.50–0.80)・赤(≥0.80)')
    p_note.runs[0].font.size = Pt(8)

    # ── 脚注（1ページ目の英語翻訳） ──
    # Word の脚注機能ではなく，末尾に英語情報として追記
    add_hr(doc)
    p_fn = doc.add_paragraph(
        'Title (EN): Quantitative Evaluation of Speech Enhancement Effects on ASR '
        'for Bone Conduction Microphone Audio.  '
        'Author (EN): Author Name (Affiliation)'
    )
    p_fn.runs[0].font.size = Pt(8)
    p_fn.runs[0].font.color.rgb = RGBColor(0x55,0x55,0x55)

    doc.save(OUT_DOCX)
    print(f'完了: {OUT_DOCX}')


if __name__ == '__main__':
    main()
