const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, SectionType, ImageRun } = require("docx");

// ── ASJ: A4, 22mm LR, 18mm TB, 2-column ──
const mm = v => Math.round(v * 56.69);
const DXA_A4_W = 11906, DXA_A4_H = 16838;
const M_LR = mm(22), M_TB = mm(18), GAP = mm(6);

// ── Text helpers (10pt body) ──
const S = 20; // 10pt
const txt = (t, o = {}) => new TextRun({ text: t, font: "MS Mincho", size: S, ...o });
const txtB = (t, o = {}) => txt(t, { bold: true, ...o });
const sup = (t) => new TextRun({ text: t, font: "MS Mincho", size: S, superScript: true });
const sub = (t) => new TextRun({ text: t, font: "Cambria Math", size: 17, subScript: true });
const mi = (t, o = {}) => new TextRun({ text: t, font: "Cambria Math", size: S, italics: true, ...o });
const mo = (t) => new TextRun({ text: t, font: "Cambria Math", size: S });
const msub = (base, s) => [mi(base), sub(s)];
const eq = (t) => new TextRun({ text: t, font: "Cambria Math", size: 19, italics: true });
const p = (c, o = {}) => new Paragraph({ spacing: { after: 60, line: 300 }, ...o, children: Array.isArray(c) ? c : [c] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 200, after: 100 }, children: [txtB(t, { size: 24, font: "MS Gothic" })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 80 }, children: [txtB(t, { size: 22, font: "MS Gothic" })] });

// ── Table helpers ──
const bdr = { style: BorderStyle.SINGLE, size: 1, color: "888888" };
const bdrs = { top: bdr, bottom: bdr, left: bdr, right: bdr };
const cp = { top: 20, bottom: 20, left: 60, right: 60 };

function cell(text, { width, shade, bold, align } = {}) {
  return new TableCell({
    borders: bdrs, margins: cp,
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: align || AlignmentType.LEFT,
      spacing: { after: 0, line: 240 },
      children: [bold ? txtB(text, { size: 17 }) : txt(text, { size: 17 })]
    })]
  });
}

function tbl(hdrs, rows, ws) {
  const tw = ws.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tw, type: WidthType.DXA }, columnWidths: ws,
    rows: [
      new TableRow({ children: hdrs.map((h, i) => cell(h, { width: ws[i], shade: "D9E2F3", bold: true, align: AlignmentType.CENTER })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, { width: ws[i], align: i > 0 ? AlignmentType.CENTER : AlignmentType.LEFT })) }))
    ]
  });
}

const cap = (t) => p([txt(t, { size: 17, italics: true })], { alignment: AlignmentType.CENTER, spacing: { before: 30, after: 140 } });

// ══════════════════════════════════════════════════════════════
// CONTENT
// ══════════════════════════════════════════════════════════════

// 案A「効果と汎化範囲」版。数値は results/RESULTS_SUMMARY.md「確定値」節（scripts 62-64）
const PENDING = (t) => txt(t, { color: "C00000", bold: true });
const ref = (t) => p([txt(t, { size: 17 })], { spacing: { after: 20, line: 260 } });

const content = [
  // ────────────────────────────────────────────────
  h1("1. はじめに"),

  p([txt("喉マイクは頸部表面の振動を接触型センサで検出するため、周囲の環境騒音の影響を受けにくい。一方、頸部の軟組織を介した伝播は物理的なローパスフィルタとして働き、高域成分が大きく失われる。韓国語喉マイクデータセットTAPS[4]では喉マイクが8 kHzで収録されており、4 kHz以上は収録時点から存在しない。このためWhisper-small[11]で喉マイク音声を認識した場合の文字誤り率（CER）は0.47と高い。")]),

  p([txt("前処理としての音声強調（SE）は有力な対策であるが、知覚品質（STOI・PESQ）の改善がASR精度の改善を保証しないことが知られている[1][2]。この乖離に対し、ASRの損失を直接用いてSEを学習する研究が行われてきた。Bagchiら[5]は凍結した音響モデルの出力を模倣する損失でSEを学習し、Dissenら[6]は凍結したWhisperの損失で前段ネットワークを学習した。")]),

  p([txt("しかし、SEを前処理として用いる最大の利点は、後段のASRを選ばずに接続できる点にある。特定のASRの損失で学習したSEの改善が、学習に使用していないASR、特に異なる系列のASRにも持ち越されるかは十分に検証されていない。")]),

  p([txt("本研究では、凍結したWhisperの交差エントロピー（CE）損失でSEを追加学習し、その効果と汎化範囲をTAPSで検証する。本研究の貢献は以下の3点である。")]),
  p([txt("(1) 学習に用いたWhisper-smallでは、句読点を正規化した評価でもCERがTAPS SE比で15.0%低下し、10話者全員で改善することを示す。")]),
  p([txt("(2) その改善はWhisper-mediumには持ち越されるが、baseでは有意でなく、CTC系のXLS-Rではむしろ25.8%悪化することを示す。")]),
  p([txt("(3) 正規化しない評価では改善の一部が句読点出力の抑制によるものであることを示し、ASR損失で学習したSEの評価における注意点を明らかにする。")]),

  // ────────────────────────────────────────────────
  h1("2. 手法"),

  p([txt("ベースラインはTAPS論文[4]のSE-Conformer（12.1Mパラメータ）の事前学習済みモデル（以下TAPS SE）である。このモデルはL1波形損失と多解像度STFT損失の和（再構成損失"), ...msub("L", "recon"), txt("）で学習されている。提案手法では、TAPS SEの重みを初期値とし、次の損失で追加学習する。")]),
  p([
    ...msub("L", "CE"), mo(" = "), ...msub("L", "recon"), mo("("), mi("ŝ"), mo(", "), ...msub("s", "air"), mo(") + "), mi("λ"), mo(" × CE(Whisper("), mi("ŝ"), mo("), "), mi("y"), mo(")")
  ], { alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 } }),
  p([txt("ここで"), mi("ŝ"), txt("はSE出力、"), mi("s"), sub("air"), txt("は同時収録の気導マイク音声、"), mi("y"), txt("は正解テキストである。Whisper-smallの全パラメータは凍結し、勾配はdecoder・encoder・対数メルスペクトログラム変換を経由してSEに逆伝播する。")]),
  p([txt("比較手法として、SE出力と気導音声のWhisper encoder出力間のL1距離を"), mi("λ"), txt("倍して加える手法（Enc L1、Closeら[3]と同系統）も学習した。また、"), mi("λ"), txt(" = 0（再構成損失のみで同一条件の追加学習）を対照とした。")]),

  // ────────────────────────────────────────────────
  h1("3. 実験条件"),

  h2("3.1 データと学習"),
  p([
    txt("TAPS[4]（60話者・6,000発話）の話者独立な分割（train 40話者・dev 10話者・test 10話者、各100発話/話者）を用いた。学習では、音声とテキストの対応を保つため15秒を超える発話を除外した（train 78件、dev 9件）。評価は全test 1,000発話で行った。最適化はAdam（lr = 3×10"), sup("-4"),
    txt("）、batch size 4、最大50エポック、early stoppingの基準はdevの学習損失とした。"), mi("λ"), txt("は{0, 0.1, 0.5, 1, 2, 5, 10}を探索し、devのCER（3.2節の正規化後）で選択した。各条件の学習は1回である。"),
  ]),

  h2("3.2 評価"),
  p([txtB("テキスト正規化．"), txt("TAPSの正解テキストは句読点を含まない。一方Whisperは文末に句点などを出力するため、未正規化のCERには句読点の有無が混入する。そこで正解と認識結果の双方からUnicodeの句読点を除去し、連続する空白を1つにまとめてからCERを計算した（以下CER）。参考として未正規化の値（raw）も示す。CERは発話ごとに計算して1.0で打ち切り、発話平均をとった。")]),
  p([txtB("認識器．"), txt("学習に用いたWhisper-smallに加え、Whisper系列内の汎化としてWhisper-base・medium、および喉マイク音声でファインチューニングしたWhisper-small（FT Whisper、dev CERが最良のエポックを採用）を用いた。系列外の汎化として、CTC型のMMS-1B[9]（韓国語アダプタ）とXLS-R[10]を韓国語で追加学習したモデル（kresnik/wav2vec2-large-xlsr-korean）を用いた。Whisper系列はfaster-whisper（beam size 5）、CTC系はgreedy復号で認識した。")]),
  p([txtB("統計．"), txt("同一話者の発話は独立でないため、話者ごとの平均CERを単位としたWilcoxon符号順位検定（"), mi("n"), txt(" = 10）を用い、改善した話者数を併記した。")]),

  // ────────────────────────────────────────────────
  h1("4. 実験結果"),

  h2("4.1 学習に用いた認識器での効果"),
  p([txt("表1に"), mi("λ"), txt("ごとのCERを示す。devのCERは"), mi("λ"), txt(" = 10で最小（0.227）となったが、"), mi("λ"), txt(" = 2〜10ではほぼ横ばいであった（0.231・0.232・0.227）。選択した"), mi("λ"), txt(" = 10のtest CERは0.196であり、TAPS SE（0.230）から15.0%低下した。10話者全員でCERが低下し、話者単位の検定でも有意であった（"), mi("p"), txt(" = 0.002）。一方、"), mi("λ"), txt(" = 0（再構成損失のみの追加学習）のtest CERは0.235とTAPS SEから改善しなかったことから、改善は追加学習そのものではなくCE損失によるものである。再構成損失を除いたCE onlyは0.203であり、再構成損失の併用が有効であった。")]),
  p([txt("未正規化のCERでは、TAPS SEの0.252から提案手法の0.200へと20.8%の低下に見える。しかし文末に句読点が付く発話の割合は、TAPS SEの94%に対し提案手法では1%であった。正解テキストに句読点がないため、CE損失はWhisperが句読点を出力しないようにSE出力を変化させており、未正規化の改善幅（0.052）のうち34%（0.018）はこの効果による。句点率は"), mi("λ"), txt(" = 2以上で急激に低下した（表1）。")]),
  tbl(
    ["条件", "dev", "test", "test (raw)", "句点率"],
    [
      ["TAPS SE", "0.265", "0.230", "0.252", "94%"],
      ["λ=0（再構成のみ）", "0.265", "0.235", "0.257", "95%"],
      ["CE λ=1", "0.244", "0.208", "0.228", "90%"],
      ["CE λ=2", "0.231", "0.200", "0.216", "67%"],
      ["CE λ=5", "0.232", "0.199", "0.203", "0%"],
      ["CE λ=10（選択）", "0.227", "0.196", "0.200", "1%"],
      ["CE only λ=10", "0.238", "0.203", "0.205", "0%"],
    ],
    [1300, 700, 700, 800, 700]
  ),
  cap("表1: λごとのCER（Whisper-small、正規化後。rawは未正規化、句点率はtest）"),

  h2("4.2 他の認識器への汎化"),
  p([txt("表2に、SE条件と認識器の組み合わせごとのCERを示す。Whisper-mediumでは提案手法がTAPS SEからCERを14.2%下げ（10/10話者、"), mi("p"), txt(" = 0.002）、改善が持ち越された。一方Whisper-baseでは1.8%の低下にとどまり、有意ではなかった（6/10話者、"), mi("p"), txt(" = 0.23）。")]),
  p([txt("系列外では、XLS-Rで提案手法がTAPS SEよりCERを25.8%上げた（0/10話者、"), mi("p"), txt(" = 0.002）。"), mi("λ"), txt(" = 0と比べても20.9%悪化しており（0/10話者）、この悪化はCE損失によるものである。MMS-1Bでは差は有意でなかった（+1.0%、"), mi("p"), txt(" = 0.38）。Enc L1はいずれの認識器でもTAPS SEとほぼ同等であった。")]),
  p([txt("FT WhisperはSEなしで0.138と最も低く、TAPS SEを接続すると0.146に悪化した。提案手法を接続すると平均は0.121に下がったが、改善は10話者中5話者にとどまり有意ではなかった（"), mi("p"), txt(" = 0.38）。提案手法単体（Whisper-smallで0.196）はFT Whisperに及ばない。")]),
  tbl(
    ["SE", "W-b", "W-s*", "W-m", "FT", "MMS", "XLS-R"],
    [
      ["No SE", "0.655", "0.446", "0.337", "0.138", "0.497", "0.584"],
      ["TAPS SE", "0.273", "0.230", "0.197", "0.146", "0.350", "0.222"],
      ["Enc L1", "0.277", "0.230", "0.197", "—", "0.353", "0.220"],
      ["λ=0", "—", "0.235", "—", "—", "0.358", "0.231"],
      ["CE（提案）", "0.268", "0.196", "0.169", "0.121", "0.354", "0.279"],
    ],
    [1000, 580, 580, 580, 580, 580, 580]
  ),
  cap("表2: 認識器ごとのCER（正規化後、*は学習に使用、—は未評価）"),

  h2("4.3 提案手法は何を変えたか"),

  p([txt("CE損失がSE出力をどう変えたかを調べるため、test 100発話（10話者×10発話）の平均パワースペクトルを求め、移動中央値で包絡を除いた残差を比較した（図1）。提案手法の出力には、3.0–7.75 kHzの250 Hz間隔の位置に10–20 dBの鋭いピークが並ぶ櫛状の成分が現れた。4–8 kHzにおける250 Hz格子上の残差の平均は、TAPS SEの2.6 dB、"), mi("λ"), txt(" = 0の2.8 dBに対し、"), mi("λ"), txt(" = 1、5、10でそれぞれ6.7、10.1、12.5 dB、再構成損失を除いたCE onlyで15.4 dBと"), mi("λ"), txt("とともに単調に増加した。250 Hzは本SE-Conformerの最深層の時間解像度（64 kHz / 4"), sup("4"), txt("）と一致しており、転置畳み込みに由来する可能性がある。")]),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 0 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(__dirname + "/figures/peak_periodicity_top.png"),
      transformation: { width: 300, height: 99 } })] }),
  cap("図1: 包絡除去後の平均スペクトル（灰線: 250 Hz間隔）"),
  p([txt("この櫛状成分と帯域ごとの寄与を調べるため、提案手法の出力の250 Hz格子位置にノッチフィルタを適用し、また4 kHzを境に低域と高域をTAPS SEと提案手法で入れ替えた（表3）。ノッチの適用でCERは0.196から0.204へ上昇し（10/10話者で悪化）、櫛状成分は改善幅（0.035）の約1/4を担っていた。一方、提案手法の低域とTAPS SEの高域を組み合わせると0.201となり、改善幅の約83%が保たれた。したがって改善の大部分は4 kHz以下の変化によるものである。なお、TAPS SEの4 kHz以上を除去するとCERは0.221に下がり、TAPS SEが補間した高域はWhisperにとってむしろ不利であった。")]),
  tbl(
    ["条件", "CER"],
    [
      ["TAPS SE", "0.230"],
      ["TAPS SE（4 kHz以上を除去）", "0.221"],
      ["提案手法", "0.196"],
      ["提案手法＋櫛ノッチ", "0.204"],
      ["提案低域＋TAPS高域", "0.201"],
      ["TAPS低域＋提案高域", "0.217"],
    ],
    [2400, 900]
  ),
  cap("表3: 帯域入れ替え・ノッチの効果（Whisper-small、正規化後）"),
  p([txt("なお提案手法はTAPS SEに比べSTOI（0.892→0.792）・PESQ（1.975→1.215）がともに低く、気導音声とのWhisper encoder出力の距離も大きい（0.200→0.326）。提案手法は気導音声に近づくのではなく、Whisperにとって認識しやすい別の信号を作っている。Iwamotoら[7]は気導マイク音声においてASRの学習がSEのartifact errorを減らすことを報告しているが、本研究ではCE損失によって気導音声に存在しない成分が増え、その一部はCERの改善にも寄与していた。")]),

  // ────────────────────────────────────────────────
  h1("5. 考察"),
  p([txt("以上の結果は、CE損失で学習したSEが、学習に用いた認識器に向けて入力を調整する「波形領域のアダプタ」として働くことを示唆する。その効果はWhisper系列内の大きいモデルには持ち越されるが、系列外の認識器では失われ、むしろ悪化する。一方、気導音声の表現に近づけるEnc L1は認識器を選ばない代わりに改善も小さい。ASR損失によるSEの学習には、対象の認識器への特化と汎用性のトレードオフが存在する。自己教師あり表現空間の距離を損失として学習したSEが複数の下流タスクで有効であるという報告[8]と合わせると、汎化の範囲は損失の種類に依存すると考えられる。")]),
  p([txt("また、未正規化のCERでは句読点出力の抑制が改善として計上された。de Oliveiraら[12]も、ASRによるSEの評価結果が認識器の選択とテキスト正規化の方法に大きく依存することを指摘している。ただし同研究は通常の損失で学習したSEの評価を扱っているのに対し、本研究の結果は、ASR損失で学習するとSE自体が正解テキストの表記（句読点の有無）に適応し、評価の偏りが学習の段階で生じうることを示している。ASR損失で学習したSEの評価では、テキストの正規化と、学習に用いていない系列の認識器による評価が特に重要である。")]),
  p([txt("本研究の限界として、各条件の学習が1回であること、"), mi("λ"), txt("の探索範囲が10までであること、韓国語のみの評価であることが挙げられる。")]),

  // ────────────────────────────────────────────────
  h1("6. まとめ"),
  p([txt("凍結したWhisperのCE損失で喉マイク用SEを学習し、その効果と汎化範囲を検証した。学習に用いたWhisper-smallでは正規化後もCERが15.0%低下したが、改善はWhisper系列内に限られ、CTC系のXLS-Rでは悪化した。また未正規化の改善幅の34%は句読点出力の抑制によるものであった。今後は複数系列の認識器の損失を組み合わせ、学習に用いていない認識器で評価する枠組みにより、認識器に依存しないASR指向のSEを検討する。")]),

  // ────────────────────────────────────────────────
  h1("参考文献"),
  ref("[1] T. Ochiai, et al., “Rethinking Processing Distortions: Disentangling the Impact of Speech Enhancement Errors on Speech Recognition Performance,” IEEE/ACM TASLP, vol. 32, 2024."),
  ref("[2] C.O. Mawalim, S. Okada, and M. Unoki, “Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?” Proc. Interspeech, 2024."),
  ref("[3] G. Close, W. Ravenscroft, T. Hain, and S. Goetze, “Perceive and Predict: Self-Supervised Speech Representation Based Loss Functions for Speech Enhancement,” Proc. ICASSP, 2023."),
  ref("[4] Y. Kim, et al., “Throat and Acoustic Paired Speech Dataset for Deep Learning-Based Speech Enhancement,” Scientific Data, 2026."),
  ref("[5] D. Bagchi, P. Plantinga, A. Stiff, and E. Fosler-Lussier, “Spectral Feature Mapping with Mimic Loss for Robust Speech Recognition,” Proc. ICASSP, 2018."),
  ref("[6] Y. Dissen, S. Yonash, I. Cohen, and J. Keshet, “Enhanced ASR Robustness to Packet Loss with a Front-End Adaptation Network,” Proc. Interspeech, 2024."),
  ref("[7] K. Iwamoto, T. Ochiai, M. Delcroix, et al., “How Does End-to-End Speech Recognition Training Impact Speech Enhancement Artifacts?” Proc. ICASSP, pp. 11031–11035, 2024."),
  ref("[8] H. Sato, T. Ochiai, M. Delcroix, et al., “Generic Speech Enhancement with Self-Supervised Representation Space Loss,” Frontiers in Signal Processing, vol. 5, 1587969, 2025."),
  ref("[9] V. Pratap, et al., “Scaling Speech Technology to 1,000+ Languages,” JMLR, vol. 25, 2024."),
  ref("[10] A. Babu, et al., “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” Proc. Interspeech, 2022."),
  ref("[11] A. Radford, et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” Proc. ICML, 2023."),
  ref("[12] D. de Oliveira, T. Peer, and T. Gerkmann, “Too Good to Be True: A Study on Modern Automatic Speech Recognition for the Evaluation of Speech Enhancement,” arXiv:2605.12107, 2026."),
];

// ══════════════════════════════════════════════════════════════
// Build
// ══════════════════════════════════════════════════════════════
const doc = new Document({
  styles: {
    default: { document: { run: { font: "MS Mincho", size: S } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "MS Gothic" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "MS Gothic" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ]
  },
  sections: [
    // Title (1-col)
    {
      properties: {
        page: { size: { width: DXA_A4_W, height: DXA_A4_H }, margin: { top: M_TB, right: M_LR, bottom: M_TB, left: M_LR } },
      },
      children: [
        p([txtB("ASR損失で学習した喉マイク音声強調の効果と汎化範囲", { size: 28, font: "MS Gothic" })], { alignment: AlignmentType.CENTER, spacing: { after: 100 } }),
        p([txt("春日 裕次（明治大学 総合数理学部 先端メディアサイエンス学科）", { size: 19 })], { alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
      ]
    },
    // Body (2-col, continuous)
    {
      properties: {
        page: { size: { width: DXA_A4_W, height: DXA_A4_H }, margin: { top: M_TB, right: M_LR, bottom: M_TB, left: M_LR } },
        type: SectionType.CONTINUOUS,
        column: { count: 2, space: GAP, equalWidth: true },
      },
      children: content,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = __dirname + "/thesis_v3.docx";
  fs.writeFileSync(out, buf);
  console.log("Created:", out);
});
