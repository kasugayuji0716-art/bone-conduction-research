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

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
PESQ_CSV = os.path.join(BASE_DIR, 'results', 'pesq_results.csv')
OUT_DOCX = os.path.join(BASE_DIR, 'results', 'paper_asj_final.docx')

# ── データ ───────────────────────────────────────────────────
def load():
    def rd(path, key='condition'):
        with open(path, encoding='utf-8') as f:
            return {r[key]: r for r in csv.DictReader(f)}
    return rd(SUMMARY), rd(STOI_CSV), rd(PESQ_CSV)

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
    cer, stoi, pesq = load()

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
        '骨伝導マイク（喉マイク）は高騒音環境での音声収音に有用だが，'
        '低域偏重・高周波欠落という独特の周波数特性を持ち，'
        '一般的なASRシステムでは高い文字誤り率（CER）が生じる．'
        '本研究では，DSPベースおよびGTCRNニューラルSEを喉マイク音声に適用し，'
        'Whisperを用いたCERへの影響をSTOI・PESQとあわせて34条件で定量評価した．'
        '全ノイズ条件においてSE適用後のCERが有意に悪化し（Wilcoxon検定，30検定中26件 p<0.05），'
        'GTCRNはSTOIおよびPESQを改善しながらCERを悪化させるという'
        '知覚品質とASR性能の乖離を示した．'
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
        '骨伝導マイク（喉マイク）は，ヘルメット着用や騒音環境下でも'
        '安定した音声収音が可能なため，軍事・工業・スポーツ用途で利用される．'
        'しかしその音声は低域エネルギーが支配的で高周波成分が欠落するため，'
        '一般的なASRシステムでは高いCERが生じる．')
    body(doc,
        '音声強調（SE）はASR前処理として広く用いられるが，'
        'Mawalimら [1] は深層学習型SEが実環境ではASR性能を必ずしも改善しないことを示した．'
        'またOchiaiら [2] はSEが導入するアーティファクト誤差が'
        '残留ノイズよりもASRに有害であることを定量的に示している．'
        'しかし喉マイクという特殊なドメインでの検証はこれまでなされていない．')
    body(doc,
        '本稿では，DSP-onlyおよびGTCRN [3] によるSEが'
        'Whisperの韓国語CERに与える影響を，STOI・PESQとあわせて'
        '34条件で評価し，SE逆効果が生じる条件を特定することを目的とする．')

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
        '図1に白色ノイズ下のSNRとCERの関係を示す．'
        '全SEモデルが全SNR条件でNo SEよりも高いCERを示した．'
        'GTCRNはSNR +20 dBではNo SEに近い値（0.378 vs 0.330）だが，'
        'SNR低下とともに差が拡大し，SNR 0 dBでは1.214（No SE: 0.882）に達した．'
        'Wilcoxon検定では，SNR −5 dBのピンクノイズ（天井効果）を除く'
        '26/30条件で有意差を確認した（p<0.05）．')

    add_fig(doc, fig_snr_cer(cer, 'white'), w_cm=7.5,
            cap='図1　SNR対CER（白色ノイズ）')

    h2(doc, '3.3', 'STOI・PESQとCERの乖離')
    body(doc,
        '図2に白色ノイズにおけるGTCRNとNo SEの差分（Δ）を示す．'
        '赤棒（ΔCER）は全SNR帯域で正（ASR悪化），'
        '青棒（ΔSTOI）・緑棒（ΔPESQ）も全帯域で正（品質改善）であり，'
        '知覚品質とASR性能の逆行が一貫して観測された．'
        '例えばSNR 0 dBでは，ΔSTOI=+0.14，ΔPESQ=+0.69，ΔCER=+0.29 であった．'
        'DSP-onlyはSTOI・PESQともにNo SEより悪化し，3指標すべてで逆効果を示した．')

    add_fig(doc, fig_delta_all(cer, stoi, pesq, 'white'), w_cm=7.5,
            cap='図2　GTCRNのΔCER（赤）・ΔSTOI（青）・ΔPESQ（緑）\n（白色ノイズ，vs No SE）')

    body(doc,
        '表1に代表条件の3指標を示す．'
        'ピンクノイズでも同様の傾向が観測された（省略）．')

    add_fig(doc, fig_metrics_table_img(cer, stoi, pesq), w_cm=7.5,
            cap='表1　代表条件のCER・STOI・PESQ（白色ノイズ）')

    # 4. 考察
    h1(doc, '4', '考察')
    body(doc,
        'GTCRNがSTOI・PESQを改善しながらCERを悪化させる現象は，'
        'Ochiaiら [2] の枠組みで解釈すると，'
        'SEが知覚的に自然な音声を生成しつつ'
        'Whisperの学習分布と乖離したアーティファクト誤差を導入しているためと考えられる．'
        'GTCRNはDNS3（気導マイク）で学習されており，'
        '喉マイクの低域偏重スペクトルはドメイン外入力であるため'
        '過剰な周波数操作が生じやすい．')
    body(doc,
        'DSP-onlyは300 Hzハイパスフィルタにより喉マイクの主要エネルギー帯域を除去し，'
        'STOI・PESQ・CERの三指標すべてを悪化させた．'
        'この結果はDSP設計が喉マイクの周波数特性を考慮していないことを示し，'
        'カットオフ周波数の再検討が必要である．')
    body(doc,
        'SNR −5 dBでWilcoxon検定が有意差を示さなかった条件（ピンクノイズ）は，'
        '両条件ともCER≥1.0という天井効果によるものであり，'
        'SE逆効果が消失したわけではない．')

    # 5. おわりに
    h1(doc, '5', 'おわりに')
    body(doc,
        '喉マイク音声に対するSEの影響をCER・STOI・PESQの3指標で評価した結果，'
        '以下を明らかにした．'
        '（1）DSP-onlyおよびGTCRNは全ノイズ条件でCERを有意に悪化させる'
        '（30検定中26件，p<0.05）．'
        '（2）GTCRNはSTOI・PESQを改善しながらCERを悪化させるという，'
        '知覚品質とASR性能の乖離が全ノイズ条件で一貫して観測される．'
        '（3）DSP-onlyは3指標すべてを悪化させ，喉マイクには不適切な前処理である．'
        '今後はWhisper large-v3による再評価および喉マイク専用学習済みSEモデルとの比較を行う予定である．')

    # 参考文献
    h1(doc, '', '参考文献')
    refs = [
        '[1] C.O. Mawalim, S. Okada, M. Unoki, Interspeech 2024, pp.1735–1739.',
        '[2] T. Ochiai et al., IEEE/ACM TASLP 2024, arXiv:2404.14860.',
        '[3] X. Rong et al., arXiv:2404.11567, 2024.',
        '[4] Y. Kim et al., HuggingFace: yskim3271/TAPS, 2023.',
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
