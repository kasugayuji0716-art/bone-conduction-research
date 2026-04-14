"""
日本音響学会 研究発表会 最終版 .docx
2ページ構成、PESQ/STOI/CER/統計検定すべて反映
"""
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from docx import Document
from docx.shared import Pt, Mm, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
SUMMARY    = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV   = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
PESQ_CSV   = os.path.join(BASE_DIR, 'results', 'pesq_results.csv')
WFT_CSV    = os.path.join(BASE_DIR, 'results', 'phase2_whisper_summary.csv')
OUT_DOCX   = os.path.join(BASE_DIR, 'results', 'paper_asj_final.docx')

# ── データ ───────────────────────────────────────────────────
def load():
    def rd(path, key='condition'):
        with open(path, encoding='utf-8') as f:
            return {r[key]: r for r in csv.DictReader(f)}
    return rd(SUMMARY), rd(STOI_CSV), rd(PESQ_CSV), rd(WFT_CSV)

def v(d, key, field):
    return float(d[key][field]) if key in d else None

# ── 図生成 ───────────────────────────────────────────────────
def fig_snr_cer(cer, noise_type):
    SNR = ['-5','+0','+5','+10','+20']
    XL  = ['-5','0','+5','+10','+20']
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for model, col, mrk, lbl in [
        ('no_se','#2196F3','o','No SE'),
        ('dsp_only','#FF9800','s','DSP-only'),
        ('gtcrn','#4CAF50','^','GTCRN'),
    ]:
        ys = []
        for snr in SNR:
            sc  = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            key = f'no_se/{noise_type}/{sc}' if model=='no_se' else f'{model}/{noise_type}/{sc}'
            ys.append(cer.get(key, {}).get('avg_cer') and float(cer[key]['avg_cer']))
        ax.plot(XL, ys, marker=mrk, color=col, label=lbl, linewidth=1.8, markersize=5)
    ax.set_xlabel('SNR (dB)', fontsize=9); ax.set_ylabel('CER', fontsize=9)
    ax.set_ylim(0, 1.4); ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(labelsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.5)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf

def fig_delta_all(cer, stoi, pesq, noise_type):
    """GTCRNのΔCER・ΔSTOI・ΔPESQを1図に"""
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

    fig, ax1 = plt.subplots(figsize=(4.2, 2.8))
    ax2 = ax1.twinx()
    x = np.arange(len(XL)); w = 0.28
    ax1.bar(x-w,   dc_list, w, color='#C62828', alpha=0.85, label='ΔCER (up=worse)')
    ax1.bar(x,     ds_list, w, color='#1565C0', alpha=0.85, label='ΔSTOI (up=better)')
    ax2.bar(x+w,   dp_list, w, color='#2E7D32', alpha=0.85, label='ΔPESQ (up=better)')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(XL, fontsize=8)
    ax1.set_xlabel('SNR (dB)', fontsize=9)
    ax1.set_ylabel('ΔCER / ΔSTOI', fontsize=8)
    ax2.set_ylabel('ΔPESQ', fontsize=8, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32', labelsize=7)
    ax1.tick_params(labelsize=7)
    ax1.grid(axis='y', alpha=0.25)
    # 凡例を合体
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=7, loc='upper left',
               ncol=1, framealpha=0.8)
    fig.tight_layout(pad=0.5)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf

def fig_metrics_table_img(cer, stoi, pesq):
    """代表4条件の3指標を小さな表イメージで出力"""
    rows = [
        ['Condition',         'CER↓', 'STOI↑', 'PESQ↑'],
        ['No SE (white +0)',  '0.882', '0.585', '1.018'],
        ['DSP   (white +0)',  '1.028', '0.434', '1.020'],
        ['GTCRN (white +0)',  '1.214', '0.721', '1.706'],
        ['No SE (white+20)',  '0.330', '0.854', '1.144'],
        ['DSP   (white+20)',  '0.521', '0.699', '1.019'],
        ['GTCRN (white+20)',  '0.378', '0.900', '2.982'],
    ]
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    ax.axis('off')
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 1:  # CER列
            try:
                val = float(rows[r+1][c])
                if val < 0.40:   cell.set_facecolor('#c8e6c9')
                elif val < 0.60: cell.set_facecolor('#fff9c4')
                elif val < 0.90: cell.set_facecolor('#ffe0b2')
                else:            cell.set_facecolor('#ffcdd2')
            except: pass
    fig.tight_layout(pad=0.3)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig); buf.seek(0); return buf

# ── XML/Wordヘルパー ─────────────────────────────────────────
def set_two_columns(section, space_twips=425):
    sectPr = section._sectPr
    for old in sectPr.findall(qn('w:cols')): sectPr.remove(old)
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), str(space_twips))
    cols.set(qn('w:equalWidth'), '1')
    sectPr.append(cols)

def set_one_column(section):
    sectPr = section._sectPr
    for old in sectPr.findall(qn('w:cols')): sectPr.remove(old)
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '1')
    sectPr.append(cols)

def set_header(section):
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    p.clear()
    pPr = p._p.get_or_add_pPr()
    tabs = OxmlElement('w:tabs')
    tab  = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'right'); tab.set(qn('w:pos'), '9360')
    tabs.append(tab); pPr.append(tabs)
    r1 = p.add_run('日本音響学会　2026年秋季研究発表会')
    r1.font.size = Pt(8)
    p.add_run('\t')
    r2 = p.add_run('X-X-XX')
    r2.font.size = Pt(8)
    pBdr = OxmlElement('w:pBdr')
    bot  = OxmlElement('w:bottom')
    bot.set(qn('w:val'),'single'); bot.set(qn('w:sz'),'4')
    bot.set(qn('w:space'),'1');   bot.set(qn('w:color'),'000000')
    pBdr.append(bot); pPr.append(pBdr)

def set_footer(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(); run.font.size = Pt(9)
    for tag, text in [('begin', None), (None, ' PAGE '), ('end', None)]:
        if tag:
            fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), tag)
            run._r.append(fc)
        else:
            it = OxmlElement('w:instrText'); it.text = text
            run._r.append(it)

def add_hr(doc, color='000000'):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    for side in ['top']:
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'6')
        el.set(qn('w:space'),'1'); el.set(qn('w:color'), color)
        pBdr.append(el)
    pPr.append(pBdr)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)

def para(doc, text='', size=10, bold=False,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         sb=0, sa=3, indent=0, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(sb)
    p.paragraph_format.space_after  = Pt(sa)
    if indent: p.paragraph_format.first_line_indent = Mm(indent)
    if text:
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.font.bold = bold
        if color: run.font.color.rgb = color
    return p

def h1(doc, num, text):
    return para(doc, f'{num}．{text}', size=10, bold=True,
                align=WD_ALIGN_PARAGRAPH.LEFT, sb=5, sa=2)

def h2(doc, num, text):
    return para(doc, f'{num}　{text}', size=10, bold=True,
                align=WD_ALIGN_PARAGRAPH.LEFT, sb=4, sa=1)

def body(doc, text):
    return para(doc, text, size=10, indent=5, sa=3)

def caption(doc, text):
    return para(doc, text, size=9, bold=True,
                align=WD_ALIGN_PARAGRAPH.CENTER, sb=1, sa=5)

def add_fig(doc, buf, w_cm, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(0)
    p.add_run().add_picture(buf, width=Cm(w_cm))
    caption(doc, cap)

# ── メイン ───────────────────────────────────────────────────
def main():
    cer, stoi, pesq, wft = load()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width   = Mm(210); sec.page_height  = Mm(297)
    sec.top_margin   = Mm(17);  sec.bottom_margin = Mm(19)
    sec.left_margin  = Mm(23);  sec.right_margin  = Mm(23)
    sec.header_distance = Mm(9); sec.footer_distance = Mm(9)
    set_header(sec); set_footer(sec)
    set_one_column(sec)

    # ── タイトルブロック ──────────────────────────────────────
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(3)
    r = p.add_run('喉マイク音声に対する音声強調が\n自動音声認識に与える影響の定量的評価')
    r.font.size = Pt(14); r.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(
        'Quantitative Evaluation of Speech Enhancement Effects '
        'on ASR for Bone Conduction Microphone Audio')
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x37, 0x47, 0x4f)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('○ 著者名（所属機関）　author@example.ac.jp')
    r.font.size = Pt(10)

    add_hr(doc)

    # ── あらまし ──────────────────────────────────────────────
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run('【あらまし】')
    r.font.size = Pt(9); r.font.bold = True

    p = doc.add_paragraph(
        '骨伝導マイク（喉マイク）音声では，音声強調（SE）を適用すると'
        '知覚品質指標（STOI・PESQ）が改善するにもかかわらず'
        'ASR文字誤り率（CER）が悪化するという逆行現象が生じうる．'
        '本研究はこの現象を喉マイクドメインで系統的に実証した．'
        'DNS3学習済みGTCRN（ドメイン外SE）を適用した結果，'
        '全ノイズ条件でCERが有意に悪化し（30検定中26件 p<0.05），'
        'STOI・PESQとCERの一貫した逆行が観測された．'
        'この現象はOchiaiら [2] のアーティファクト誤差フレームワークで説明できる．'
        '一方，ASRモデル自体のドメイン適応（Whisper small FT）により'
        'CERは0.269→0.095（64.6%改善）となり，FT後のSE悪化幅も縮小（+0.027→+0.007）した．'
        '知覚品質指標によるSE評価は喉マイクASR性能を予測しないことを定量的に示す．'
    )
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(1)
    p.runs[0].font.size = Pt(9)

    p = doc.add_paragraph(
        '【キーワード】骨伝導マイク，音声強調，自動音声認識，Whisper，CER，STOI，PESQ')
    p.runs[0].font.size = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    add_hr(doc)

    # ── 2段組本文 ────────────────────────────────────────────
    set_two_columns(sec, space_twips=400)

    # 1. はじめに
    h1(doc, '1', 'はじめに')
    body(doc,
        '骨伝導マイク（喉マイク）は，騒音環境下でも安定した音声収音が可能なため'
        '軍事・工業・スポーツ用途で利用される．'
        'しかしその音声は低域エネルギーが支配的で高周波成分が欠落するため，'
        '一般的なASRシステムでは高いCERが生じる．'
        'この問題に対してSEの適用が自然な解決策と見なされるが，'
        '「SEで音質指標が改善してもASR性能が下がる」という逆行現象の存在が問題となる．')
    body(doc,
        'Ochiaiら [2] はSEが生成するアーティファクト誤差をOPDにより定量化し，'
        'これが残留ノイズよりもASRに有害であることを示した．'
        'Mawalimら [1] も最新DL-SEが実環境でASRを改善しないことを確認しており，'
        'この現象は一般騒音・遠距離収音・無線通信など複数ドメインで報告されている．'
        'しかし喉マイクは高周波成分が原理的に欠落しており，'
        'DNS3（気導マイク）学習済みのSEを適用した場合のドメインシフトは特に大きい．'
        'TAPSデータセット [4] ではドメイン適合型SE-conformerがCERを改善する [5] が，'
        'ドメイン外SEをノイズ条件下で適用した際の知覚品質とASR性能の乖離は未検証である．')
    body(doc,
        '本稿の問い：「喉マイクへのドメイン外SE適用時，STOI・PESQの改善はCER改善を保証するか？」'
        'さらに「ASRドメイン適応（Whisper FT）はこの乖離を解消するか？」を34条件で検証する．')

    # 2. 実験設計
    h1(doc, '2', '実験設計')
    h2(doc, '2.1', 'データセットおよびノイズ付加')
    body(doc,
        'TAPSデータセット [4] のテストセットから50発話（話者p00，16 kHz）を使用した．'
        '喉マイク音声に白色・ピンクノイズを'
        'SNR −5, 0, +5, +10, +20 dBで混合し計10条件を生成した．')
    h2(doc, '2.2', '音声強調モデル')
    body(doc,
        'DSP-only：Butterworth 6次ハイパスフィルタ（300 Hz）+'
        'プリエンファシス（0.97）+RMS正規化．')
    body(doc,
        'GTCRN：48.2Kパラメータのニューラル SE [3]．'
        'DNS3（気導マイク・英語）で学習済みであり，喉マイクはドメイン外入力となる．')
    h2(doc, '2.3', '評価指標')
    body(doc,
        'CER：faster-whisper small（言語：ko）によりtranscribeしjiwer で算出．'
        'STOI：クリーン喉マイク音声を参照にpystoi で算出（0〜1）．'
        'PESQ：同参照でpesq（WBモード，16 kHz）により算出（−0.5〜4.5）．'
        '統計検定：50発話のサンプルごとCERにWilcoxon符号順位検定（両側）を適用した．')

    # 3. 実験結果
    h1(doc, '3', '実験結果')
    h2(doc, '3.1', 'クリーン条件')
    body(doc,
        '気導マイクベースライン（CER=0.131）に対し，喉マイク生音声は0.269と悪化する．'
        'クリーン条件ではDSPが0.416（悪化），GTCRNが0.296（微改善）であった．'
        'STOIはDSP=0.847，GTCRN=0.990，PESQはDSP=2.10，GTCRN=4.09 であり，'
        'クリーン条件ではGTCRNが音質・了解度ともに優れる．')
    h2(doc, '3.2', 'ノイズ条件：CER')
    body(doc,
        '図1に白色ノイズ下のSNRとCERを示す．'
        '全SEモデルが全SNR条件でNo SEよりも高いCERを示した．'
        'GTCRNはSNR低下とともに差が拡大し，SNR 0 dBでは1.214（No SE: 0.882）に達した．'
        'Wilcoxon検定では26/30条件で有意差を確認した（p<0.05，'
        '例外はSNR −5 dBピンクノイズの天井効果による4条件）．')

    add_fig(doc, fig_snr_cer(cer, 'white'), w_cm=7.5,
            cap='図1　SNR対CER（白色ノイズ）')

    h2(doc, '3.3', '知覚品質とASR性能の逆行（主要発見）')
    body(doc,
        '図2に白色ノイズにおけるGTCRNとNo SEの差分（Δ）を示す．'
        'ΔCER（赤棒）は全SNR帯域で正（ASR悪化）であるにもかかわらず，'
        'ΔSTOI（青棒）・ΔPESQ（緑棒）も全帯域で正（知覚品質改善）である．'
        'すなわち「SEにより音声は聴こえやすくなり音質も向上するが，ASRは悪化する」'
        'という三指標の逆行が全ノイズ条件で一貫して観測された．'
        'SNR 0 dBでの具体値はΔSTOI=+0.14，ΔPESQ=+0.69，ΔCER=+0.29 であった．'
        'これはSTOI・PESQによるSE評価が喉マイクASR性能を予測しないことを示す．'
        'DSP-onlyはSTOI・PESQも悪化し，3指標すべてで逆効果を示した．')

    add_fig(doc, fig_delta_all(cer, stoi, pesq, 'white'), w_cm=7.5,
            cap='図2　GTCRNのΔCER（赤）・ΔSTOI（青）・ΔPESQ（緑）\n（白色ノイズ，vs No SE）')

    body(doc,
        '表1に代表条件の3指標を示す．'
        'ピンクノイズでも同様の傾向が観測された（省略）．')

    add_fig(doc, fig_metrics_table_img(cer, stoi, pesq), w_cm=7.5,
            cap='表1　代表条件のCER・STOI・PESQ（白色ノイズ）')

    # 3.4 フェーズ2：Whisperファインチューニング
    h2(doc, '3.4', 'Whisperファインチューニングの効果')
    body(doc,
        'SEによる音声変換ではなく，ASRモデル自体を喉マイクに適応させる'
        'アプローチとして，Whisper smallをTAPSのtrain split'
        '（40話者・4,000発話・10.2時間）を用いてファインチューニングした．'
        '学習設定はepochs=20（early stopping，best at epoch 3），'
        'batch_size=16，lr=1e-5，warmup_steps=500，fp16=Trueである．')
    body(doc,
        '表2にテストセット（話者p00，50発話）における結果を示す．'
        'ファインチューニング後のCERは0.095であり，'
        '未学習時（0.269）から64.6%の改善が得られた．'
        'これはSE適用時のCER悪化とは対照的な結果であり，'
        '喉マイクASRの改善には音声処理よりも'
        'ドメイン適応型ASR学習が有効であることを示す．')
    body(doc,
        '補足として，FT済みWhisperにGTCRN SEを組み合わせた場合（D条件）のCERは0.1015であった．'
        'FT前の悪化幅（0.296−0.269=+0.027）と比較するとFT後は+0.007と縮小しており，'
        'ドメイン適応によりSEアーティファクトへの感受性が低減することが示唆される．'
        'ただしSEの逆効果は依然として残存するため，FT後もSEの適用は推奨されない．')

    # フェーズ2結果表
    p2_rows = [
        ['Condition',                    'CER (↓)',  'Improvement'],
        ['Whisper small (pretrained)',   '0.269',    '—'],
        ['Whisper small (fine-tuned)',   '0.095',    '-64.6%'],
        ['Acoustic mic (reference)',     '0.131',    '—'],
    ]
    fig2, ax2 = plt.subplots(figsize=(4.2, 1.8))
    ax2.axis('off')
    tbl2 = ax2.table(cellText=p2_rows[1:], colLabels=p2_rows[0],
                     loc='center', cellLoc='center')
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(8.5); tbl2.scale(1, 1.4)
    for (r, c), cell in tbl2.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == 2 and c == 1:
            cell.set_facecolor('#c8e6c9')
        elif r == 2 and c == 2:
            cell.set_facecolor('#c8e6c9')
    fig2.tight_layout(pad=0.3)
    buf2 = BytesIO(); fig2.savefig(buf2, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig2); buf2.seek(0)
    add_fig(doc, buf2, w_cm=7.5, cap='表2　Whisperファインチューニングの効果（テストセット話者p00）')

    # 4. 考察
    h1(doc, '4', '考察')
    body(doc,
        'GTCRNがSTOI・PESQを改善しながらCERを悪化させる現象は，'
        'Ochiaiら [2] のOPD枠組みで解釈できる．'
        'GTCRNはDNS3（気導マイク・英語）で学習されており，'
        '喉マイクの低域偏重スペクトルはドメイン外入力である．'
        'このため非線形変換が大きなアーティファクト誤差を生じさせ，'
        '多条件訓練されたASRモデルでも対処できない特徴空間の歪みを引き起こす．'
        'なおTAPSのSE-conformer [5] はTAPSデータで学習したドメイン適合型SEであり'
        'CER 84.4%→24.4%と改善する．'
        '本結果との比較は「ドメイン適合型SEは有効，ドメイン外SEは逆効果」という'
        '一貫した解釈を支持する．')
    body(doc,
        'DSP-onlyは300 Hzハイパスフィルタにより喉マイクの主要エネルギー帯域を除去し，'
        'STOI・PESQ・CERの三指標すべてを悪化させた．'
        'この結果はDSP設計が喉マイクの周波数特性を考慮していないことを示し，'
        'カットオフ周波数の再検討が必要である．')
    body(doc,
        'SNR −5 dBでWilcoxon検定が有意差を示さなかった条件（ピンクノイズ）は，'
        '両条件ともCER≥1.0という天井効果によるものであり，'
        'SE逆効果が消失したわけではない．'
        'Whisper FT後にSEの悪影響が縮小した（+0.027→+0.007）ことは，'
        'ドメイン適応がアーティファクト感受性を低減するというOchiaiらの'
        'Observation Addingアプローチ [2] と方向性が一致する．')

    # 5. おわりに
    h1(doc, '5', 'おわりに')
    body(doc,
        '喉マイク音声へのドメイン外SE適用において，'
        'STOI・PESQとCERが全ノイズ条件で逆行することを定量的に実証した．'
        '主要な発見は以下のとおりである．'
        '（1）STOI・PESQを改善するGTCRNが全ノイズ条件でCERを有意に悪化させる'
        '（26/30条件，p<0.05）—知覚品質指標はASR性能を予測しない．'
        '（2）この逆行はOchiaiら [2] のアーティファクト誤差フレームワークで説明され，'
        'DNS3学習済みGTCRNの喉マイクへのドメインシフトが大きなアーティファクトを生成する．'
        '（3）一方，ASRモデルのドメイン適応（Whisper small FT）は'
        'CERを0.269→0.095（64.6%改善）と根本的に解決し，'
        'FT後はSEの悪影響も縮小（+0.027→+0.007）した．'
        'これらの結果は，喉マイクASRの改善においてSEの知覚品質最適化よりも'
        'ASRドメイン適応が本質的に有効であることを示す．'
        '今後は複数話者評価による汎化性の検証およびWhisper large-v3でのFTを予定する．')

    # 参考文献
    h1(doc, '', '参考文献')
    refs = [
        '[1] C.O. Mawalim, S. Okada, M. Unoki, "Are Recent DL-Based SE Methods Ready to Confront Real-World Noisy Environments?", Interspeech 2024, pp.1735–1739.',
        '[2] T. Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?", IEEE/ACM TASLP 2024, arXiv:2404.14860.',
        '[3] X. Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultra-Tiny Resources", arXiv:2404.11567, 2024.',
        '[4] Y. Kim, Y. Song, Y. Chung, "TAPS: Throat and Acoustic Paired Speech Dataset for DL-Based SE", arXiv:2502.11478, 2025.',
        '[5] Y. Kim and Y. Chung, "Modality-Specific SE and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework", Interspeech 2025.',
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.runs[0].font.size = Pt(8.5)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(0)

    # 英語情報（脚注）
    add_hr(doc)
    p = doc.add_paragraph(
        'Title: Quantitative Evaluation of SE Effects on ASR for Bone Conduction Mic Audio. '
        'Author: Author Name (Affiliation)')
    p.runs[0].font.size = Pt(7.5)
    p.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.save(OUT_DOCX)
    print(f'完了: {OUT_DOCX}')

if __name__ == '__main__':
    main()
