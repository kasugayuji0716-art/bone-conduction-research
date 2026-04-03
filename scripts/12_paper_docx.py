"""
日本語論文 .docx 生成
Word / Google Docs で直接編集可能
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from docx import Document
from docx.shared import Pt, Mm, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import docx.opc.constants

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SUMMARY  = os.path.join(BASE_DIR, 'results', 'summary.csv')
STOI_CSV = os.path.join(BASE_DIR, 'results', 'stoi_results.csv')
OUT_DOCX = os.path.join(BASE_DIR, 'results', 'paper_ja.docx')

# ── データ読み込み ────────────────────────────────────────────
def load_data():
    with open(SUMMARY, encoding='utf-8') as f:
        cer = {r['condition']: float(r['avg_cer']) for r in csv.DictReader(f)}
    with open(STOI_CSV, encoding='utf-8') as f:
        stoi = {r['condition']: float(r['avg_stoi']) for r in csv.DictReader(f)}
    return cer, stoi

# ── matplotlib 図を BytesIO に保存 ───────────────────────────
def make_snr_fig(cer, noise_type):
    SNR = ['-5', '+0', '+5', '+10', '+20']
    XLABELS = ['-5', '0', '+5', '+10', '+20']
    fig, ax = plt.subplots(figsize=(5, 3.2))
    styles = {
        'no_se':    ('#2196F3', 'o', 'No SE'),
        'dsp_only': ('#FF9800', 's', 'DSP-only'),
        'gtcrn':    ('#4CAF50', '^', 'GTCRN'),
    }
    for model, (col, mrk, lbl) in styles.items():
        ys = []
        for snr in SNR:
            sc = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            key = f'no_se/{noise_type}/{sc}' if model == 'no_se' else f'{model}/{noise_type}/{sc}'
            ys.append(cer.get(key))
        ax.plot(XLABELS, ys, marker=mrk, color=col, label=lbl, linewidth=1.8, markersize=6)
    ax.set_xlabel('SNR (dB)', fontsize=9); ax.set_ylabel('CER', fontsize=9)
    ax.set_ylim(0, 1.4); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.8)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf

def make_bar_clean(cer):
    conds  = ['baseline_acoustic','baseline_throat','dsp_only/clean','gtcrn/clean']
    labels = ['Acoustic\n(Ref)', 'Throat\n(Raw)', 'Throat\n+DSP', 'Throat\n+GTCRN']
    vals   = [cer[c] for c in conds]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(labels, vals, color=['#1565C0','#78909C','#FF9800','#4CAF50'], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('CER', fontsize=9); ax.set_ylim(0, 0.55); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.8)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf

def make_scatter(cer, stoi):
    fig, ax = plt.subplots(figsize=(5, 4))
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
        ax.scatter(xs, ys, c=col, marker=mrk, s=50, alpha=0.8, label=lbl)
    ax.set_xlabel('STOI ↑', fontsize=9); ax.set_ylabel('CER ↓', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.8)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf

# ── Wordスタイルヘルパー ─────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def add_para(doc, text, indent=False):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Mm(5)
    fmt = p.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after  = Pt(4)
    for run in p.runs:
        run.font.size = Pt(10)
    return p

def add_caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(8)
    for run in p.runs:
        run.font.size = Pt(9)
        run.font.bold = True
    return p

def add_figure(doc, buf, width_cm=12, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(buf, width=Cm(width_cm))
    if caption:
        add_caption(doc, caption)

# ── 結果テーブル ─────────────────────────────────────────────
def add_result_table(doc, cer, stoi):
    SNR = ['-5', '+0', '+5', '+10', '+20']
    headers = ['条件', 'ノイズ', 'SNR', 'CER↓', 'STOI↑']
    rows_data = []

    for cond, nt, snr in [
        ('baseline_acoustic','—','clean'),
        ('baseline_throat','—','clean'),
        ('dsp_only/clean','—','clean'),
        ('gtcrn/clean','—','clean'),
    ]:
        cv = cer.get(cond, '-')
        sv = stoi.get(cond, '-')
        rows_data.append([
            cond, nt, snr,
            f'{cv:.3f}' if isinstance(cv, float) else cv,
            f'{sv:.3f}' if isinstance(sv, float) else sv,
        ])

    for model in ['no_se', 'dsp_only', 'gtcrn']:
        for nt in ['white', 'pink']:
            for snr in SNR:
                sc = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
                key = f'no_se/{nt}/{sc}' if model == 'no_se' else f'{model}/{nt}/{sc}'
                cv = cer.get(key, '-')
                sv = stoi.get(key, '-')
                rows_data.append([
                    key, nt, snr,
                    f'{cv:.3f}' if isinstance(cv, float) else cv,
                    f'{sv:.3f}' if isinstance(sv, float) else sv,
                ])

    tbl = doc.add_table(rows=len(rows_data)+1, cols=5)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    col_widths = [Cm(6.5), Cm(1.8), Cm(1.5), Cm(1.8), Cm(1.8)]
    for i, w in enumerate(col_widths):
        for cell in tbl.columns[i].cells:
            cell.width = w

    # ヘッダー行
    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        cell.text = h
        set_cell_bg(cell, '1a237e')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # データ行
    def cer_bg(val_str):
        try:
            v = float(val_str)
            if v < 0.30:   return 'c8e6c9'
            elif v < 0.50: return 'fff9c4'
            elif v < 0.80: return 'ffe0b2'
            else:           return 'ffcdd2'
        except:
            return 'ffffff'

    for i, row_data in enumerate(rows_data):
        row = tbl.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if j == 3:
                set_cell_bg(cell, cer_bg(val))

    doc.add_paragraph('※ 色凡例：緑(CER<0.30)、黄(0.30–0.50)、橙(0.50–0.80)、赤(≥0.80)').runs[0].font.size = Pt(8)


# ── メイン ───────────────────────────────────────────────────
def main():
    cer, stoi = load_data()
    doc = Document()

    # ページ設定（A4）
    section = doc.sections[0]
    section.page_width  = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = section.right_margin   = Mm(25)
    section.top_margin  = section.bottom_margin  = Mm(25)

    # ── タイトル ──
    title = doc.add_heading('喉マイク音声に対する音声強調が\n自動音声認識に与える影響の定量的評価', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    en_title = doc.add_paragraph(
        'Quantitative Evaluation of Speech Enhancement Effects\n'
        'on Automatic Speech Recognition for Bone Conduction Microphone Audio'
    )
    en_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    en_title.runs[0].font.size = Pt(10)
    en_title.runs[0].font.color.rgb = RGBColor(0x37, 0x47, 0x4f)

    author = doc.add_paragraph('○ 著者名　（所属機関）　連絡先メールアドレス')
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.runs[0].font.size = Pt(10)

    doc.add_paragraph()

    # ── あらまし ──
    add_heading(doc, 'あらまし', level=2)
    add_para(doc,
        '骨伝導マイク（喉マイク）は高騒音環境での音声収音に有用だが，'
        'その出力音声は低域偏重かつ高周波欠落という独特の周波数特性を持つ．'
        '本研究では，DSPベースおよびGTCRNニューラルSEを喉マイク音声に適用し，'
        'Whisperを用いた文字誤り率（CER）への影響を34条件にわたり定量評価した．'
        'TAPSデータセット（韓国語，50発話）を用い，白色・ピンクノイズをSNR −5〜+20 dBで付加した．'
        '全ノイズ条件でSE適用後のCERが悪化し，特にGTCRNはSTOIを改善しながら'
        'CERを悪化させる知覚品質とASR性能の乖離を示した．'
    )
    doc.add_paragraph()

    # ── 1. はじめに ──
    add_heading(doc, '1. はじめに', level=1)
    add_para(doc,
        '骨伝導マイク（喉マイク・振動ピックアップ）は，ヘルメット着用や騒音環境下でも'
        '安定した音声収音が可能なため，軍事・工業・スポーツ用途で利用されている．'
        'しかし，その出力音声は低域エネルギーが支配的であり，'
        '高周波成分が著しく欠落するという特性を持つ．'
        'この特性により，一般的なASRシステムでは高いCERが生じる．', indent=True)
    add_para(doc,
        '音声強調（Speech Enhancement: SE）は，ノイズ除去・音質改善を目的とした前処理として'
        '広く用いられる．しかし，Mawalimら [1] は深層学習型SEが実環境では'
        'ASR性能を必ずしも改善しないことを示した．'
        'またOchiaiら [2] は，SEが導入するアーティファクト誤差が'
        '残留ノイズよりもASRに有害であることを定量的に示している．', indent=True)
    add_para(doc,
        '本稿では，喉マイク音声という特殊なドメインにおいて，'
        'DSP-onlyおよびGTCRN [3] によるSEがWhisperのCERに与える影響を'
        '34条件で評価し，SE逆効果が生じる条件を特定することを目的とする．', indent=True)

    # ── 2. 実験設計 ──
    add_heading(doc, '2. 実験設計', level=1)

    add_heading(doc, '2.1 データセット', level=2)
    add_para(doc,
        'TAPSデータセット [4] のテストセットから50発話を使用した．'
        '各発話に対して喉マイク収録音声と気導マイク収録音声のペアが存在し，'
        '韓国語の正解テキスト（metadata.csv）が付属する．'
        '話者は1名（p00），サンプリングレートは16 kHz である．', indent=True)

    add_heading(doc, '2.2 ノイズ付加', level=2)
    add_para(doc,
        '喉マイク音声に対し，白色ノイズおよびピンクノイズを'
        'SNR −5, 0, +5, +10, +20 dBで混合し，計10条件のノイズ付き音声を生成した．'
        'ノイズはターゲット音声のRMSを基準に振幅を調整した．', indent=True)

    add_heading(doc, '2.3 音声強調モデル', level=2)
    add_para(doc,
        '【DSP-only】Butterworth 6次ハイパスフィルタ（カットオフ300 Hz）+'
        'プリエンファシス（係数0.97）+RMS正規化の組み合わせ．', indent=True)
    add_para(doc,
        '【GTCRN】48.2Kパラメータの超軽量ニューラルSEモデル [3]．'
        'DNS3コーパス（気導マイク，主に英語）で学習済み．'
        'STFT（窓長512，シフト256，Hann窓）を用いた複素領域処理．', indent=True)

    add_heading(doc, '2.4 評価指標', level=2)
    add_para(doc,
        'CER：faster-whisper smallモデルでtranscribeし，jiwer ライブラリで算出（言語指定：ko）．'
        'STOI：クリーン喉マイク音声を参照としてpystoiで算出（値域0〜1，高いほど良好）．', indent=True)

    # ── 3. 実験結果 ──
    add_heading(doc, '3. 実験結果', level=1)

    add_heading(doc, '3.1 クリーン条件', level=2)
    add_para(doc,
        '図1にクリーン条件のCER比較を示す．'
        '気導マイクベースライン（CER=0.131）に対し，喉マイク生音声は0.269と大幅に悪化する．'
        'DSP適用（0.416）はさらに悪化するが，GTCRN（0.296）は喉マイク生音声に近い性能を維持した．', indent=True)
    add_figure(doc, make_bar_clean(cer), width_cm=11,
               caption='図1　クリーン条件における各手法のCER比較')

    add_heading(doc, '3.2 ノイズ条件', level=2)
    add_para(doc,
        '図2・図3に白色・ピンクノイズ下でのSNRとCERの関係を示す．'
        '全SEモデルが全SNR条件でNo SEより高いCERを示した．'
        'GTCRNはSNR +20 dBでは No SE に近い値（white: 0.378 vs 0.330）だが，'
        'SNRが低下するほど乖離が拡大し，SNR 0 dBでは1.214（No SE: 0.882）に達した．', indent=True)
    add_figure(doc, make_snr_fig(cer, 'white'), width_cm=11,
               caption='図2　SNR対CER（白色ノイズ）')
    add_figure(doc, make_snr_fig(cer, 'pink'), width_cm=11,
               caption='図3　SNR対CER（ピンクノイズ）')

    add_heading(doc, '3.3 STOIとCERの乖離', level=2)
    add_para(doc,
        '図4に全ノイズ条件のSTOI対CER散布図を示す．'
        'GTCRNの点群はNo SEと比較してSTOI軸方向（右）に移動しながら'
        'CER軸方向（上）にも移動するという逆説的パターンを示した．'
        '例えば白色ノイズSNR 0 dBでは，'
        'STOI：0.585→0.721（+0.136），CER：0.882→1.214（+0.332）となった．'
        '一方DSP-onlyはSTOI・CERともにNo SEより悪化した．', indent=True)
    add_figure(doc, make_scatter(cer, stoi), width_cm=11,
               caption='図4　STOI対CER散布図．GTCRNはSTOI改善・CER悪化の乖離を示す')

    # ── 4. 考察 ──
    add_heading(doc, '4. 考察', level=1)

    add_heading(doc, '4.1 アーティファクト誤差仮説', level=2)
    add_para(doc,
        'Ochiaiら [2] はSEによる誤差をアーティファクト・干渉・ノイズの3成分に分解し，'
        'アーティファクト誤差がASRに最も有害であることを示した．'
        '本実験でGTCRNがSTOIを改善しながらCERを悪化させる現象は，'
        'GTCRNが知覚的に自然な音声を生成しつつ，'
        'Whisperが学習した音響特徴と乖離したアーティファクトを導入していると解釈できる．', indent=True)

    add_heading(doc, '4.2 ドメインミスマッチの影響', level=2)
    add_para(doc,
        'GTCRNはDNS3（気導マイク，主に英語）で学習されており，'
        '喉マイク音声は学習分布外である．'
        '低域偏重・高周波欠落という喉マイクの周波数特性は，'
        'GTCRNに「ノイズ」として誤認識される可能性がある．'
        '喉マイク専用学習済みSEモデルを用いた場合の効果検証が今後の課題である．', indent=True)

    add_heading(doc, '4.3 DSP-onlyの問題点', level=2)
    add_para(doc,
        '300 HzハイパスフィルタはΔ喉マイクのエネルギーの大半（300 Hz以下）を除去するため，'
        'むしろ有害な前処理となっていると考えられる．'
        'カットオフ周波数のパラメータ感度分析が今後必要である．', indent=True)

    # ── 5. おわりに ──
    add_heading(doc, '5. おわりに', level=1)
    add_para(doc,
        '本稿では喉マイク音声に対するSEの効果をCERとSTOIの両面から評価した．'
        '主要な知見は以下の通りである．', indent=True)
    for item in [
        '（1）DSP-onlyおよびGTCRNは全ノイズ条件でCERを悪化させる．',
        '（2）GTCRNはSTOIを改善しながらCERを悪化させる知覚品質とASR性能の乖離を示す．',
        '（3）本現象は気導マイク学習済みSEモデルのドメイン外適用によるアーティファクト誤差によって説明される．',
    ]:
        add_para(doc, item, indent=True)
    add_para(doc,
        '今後は，Whisper large-v3による再評価，PESQ計測，'
        'DSPパラメータ感度分析，および実環境ノイズを用いた検証を行う予定である．', indent=True)

    # ── 付表 ──
    doc.add_page_break()
    add_heading(doc, '付表　全条件のCER・STOI計測結果', level=1)
    add_result_table(doc, cer, stoi)
    doc.add_paragraph()

    # ── 参考文献 ──
    add_heading(doc, '参考文献', level=1)
    refs = [
        '[1] C.O. Mawalim, S. Okada, M. Unoki, "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?" Proc. Interspeech 2024, pp.1735–1739, DOI: 10.21437/Interspeech.2024-129.',
        '[2] T. Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?" IEEE/ACM Trans. Audio Speech Lang. Process., TASLP 2024, arXiv:2404.14860.',
        '[3] X. Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultra-Tiny Resources," arXiv:2404.11567, 2024.',
        '[4] Y. Kim et al., "TAPS: Throat and Acoustic Pairing Speech Dataset," HuggingFace: yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset.',
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.space_after = Pt(4)
        p.runs[0].font.size = Pt(9)

    doc.save(OUT_DOCX)
    print(f'完了: {OUT_DOCX}')

if __name__ == '__main__':
    main()
