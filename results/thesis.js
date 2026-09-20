const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, SectionType } = require("docx");

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

const content = [
  // ────────────────────────────────────────────────
  // 1. はじめに
  // ────────────────────────────────────────────────
  h1("1. はじめに"),

  p([txt("喉マイク（咽喉マイク）は、頸部表面の振動を接触型加速度センサで検出するマイクロフォンである。通常の気導マイクが空気中の音波を収音するのに対し、喉マイクは体内を伝播する振動を直接検出するため、周囲の環境騒音の影響を受けにくいという大きな利点がある。この特性から、高騒音環境下での通信（軍事・産業）や、発話障害者の音声補助など、多様な応用が期待されている。")]),

  p([txt("しかし、頸部の軟組織を介した信号伝播は物理的なローパスフィルタとして機能し、3〜4 kHz以上の高域成分が大幅に減衰する。この帯域制限により、喉マイク音声は気導マイク音声と比較して音声品質が低下し、自動音声認識（Automatic Speech Recognition; ASR）の精度が著しく劣化する。例えば、TAPSデータセット[4]においてWhisper-smallで韓国語喉マイク音声を認識した場合のCER（Character Error Rate）は0.47であり、同一発話の気導マイク音声のCER 0.13と比較して約3.6倍の誤りが生じる。")]),

  p([txt("この課題に対し、音声強調（Speech Enhancement; SE）を前処理として適用する手法が研究されてきた。しかし近年、Ochiaiら[1]は気導マイクの文脈において、SEがASR精度を改善するどころか悪化させる場合があることを報告した。彼らは直交射影に基づく分解（Orthogonal Projection-based Decomposition; OPD）を提案し、SEの処理歪みをtarget成分・interference error・noise error・artifact errorに分解した。その結果、artifact errorが逆効果の主因であることを特定した。Mawalimら[2]も、STOI（Short-Time Objective Intelligibility）やPESQ（Perceptual Evaluation of Speech Quality）が改善しているにも関わらずASR精度が悪化する現象を報告している。")]),

  p([txt("この「知覚品質の改善がASR精度の改善を保証しない」という問題の根本原因は、従来のSEが波形品質の復元（L1損失やSTFT損失）を最適化目標としており、ASR精度の最適化が間接的にしか行われない点にある。Closeら[3]は自己教師あり音声表現の特徴距離をSE損失に利用するPerceive & Predictを提案し、知覚品質指標との相関を示したが、実際のASR性能（CER）への効果は評価されていない。")]),

  p([
    txt("本研究では、この問題に対して"),
    txtB("Whisperの交差エントロピー（Cross-Entropy; CE）損失をSEの学習損失に直接組み込む"),
    txt("ASR-aware SEを提案する。提案手法は、SEモデルの出力をWhisperに入力し、その書き起こし誤りを最小化するようにSEを学習させる。韓国語喉マイクデータセットTAPS[4]を用いた実験により、以下の知見を得た。"),
  ]),

  p([txt("(1) 提案手法はベースラインSEのCERを9.3%改善した。")]),
  p([txt("(2) 学習に使用していないASRモデルでも改善が確認された。")]),
  p([txt("(3) 知覚品質とASR精度は異なる最適化目標であることが確認された。")]),

  // ────────────────────────────────────────────────
  // 2. 提案手法
  // ────────────────────────────────────────────────
  h1("2. ベースラインと提案手法"),

  h2("2.1 ベースライン：TAPS SE-Conformer"),
  p([txt("ベースラインとして、TAPS論文[4]で提供されたSE-Conformer（12.1Mパラメータ）の事前学習済みモデルを追加学習なしでそのまま使用する。このモデルはL1波形損失と多解像度STFT損失の和（以下、再構成損失）により200エポック学習されたものである。")]),

  h2("2.2 提案手法：ASR損失による追加学習"),
  p([txt("提案手法では、上記ベースラインの重みを初期化点とし、再構成損失にASR損失を加えた損失関数で追加学習（fine-tuning）を行う。再構成損失はTAPS論文と同一であり、波形品質を維持する正則化として機能する。")]),
  p([
    ...msub("L", "recon"), mo(" = "), mi("L"), mo("1("), mi("s\u0302"), mo(", "), ...msub("s", "air"), mo(") + STFT("), mi("s\u0302"), mo(", "), ...msub("s", "air"), mo(")")
  ], { alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 } }),
  p([
    txt("ここで"), mi("s\u0302"), txt("はSE出力、"), mi("s"), sub("air"), txt("は気導マイク音声（参照信号）である。ASR損失として以下の2種を比較する。"),
  ]),

  p([txtB("(a) Encoder距離損失")]),
  p([txt("SE出力と気導音声のWhisper encoder出力間のL1距離を損失に加える。")]),
  p([
    ...msub("L", "enc"), mo(" = "), ...msub("L", "recon"), mo(" + "), mi("\u03BB"), mo(" \u00D7 "), mi("L"), mo("1(Enc("), mi("s\u0302"), mo("), Enc("), ...msub("s", "air"), mo("))")
  ], { alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 } }),
  p([txt("Whisper encoderは凍結し、勾配はencoder出力のL1距離からメルスペクトログラム変換を経由してSEモデルに逆伝播する。Closeら[3]のPerceive & Predictも同様にencoder表現の距離をSE損失に利用するが、ASR精度（CER）への効果は評価されていない。")]),

  p([txtB("(b) CE損失（提案手法）")]),
  p([txt("Whisper全体（encoder＋decoder）の交差エントロピー損失を使用する。")]),
  p([
    ...msub("L", "CE"), mo(" = "), ...msub("L", "recon"), mo(" + "), mi("\u03BB"), mo(" \u00D7 CE(Whisper("), mi("s\u0302"), mo("), "), mi("y"), mo(")")
  ], { alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 } }),
  p([txt("ここで"), mi("y"), txt("は正解テキスト、CEはWhisper decoder出力と正解テキスト間の交差エントロピーである。Whisperの全パラメータは凍結し、勾配はCE損失からdecoder、encoder、メルスペクトログラム変換を経由してSEモデルに逆伝播する。")]),
  p([txt("encoder距離損失が中間表現の類似度という間接的な目標を最適化するのに対し、CE損失は最終的な書き起こし精度を直接最適化する点が本質的に異なる。")]),

  // ────────────────────────────────────────────────
  // 3. 実験条件
  // ────────────────────────────────────────────────
  h1("3. 実験条件"),

  h2("3.1 データセット"),
  p([txt("韓国語喉マイクデータセットTAPS（Throat and Acoustic Pairing Speech Dataset）[4]を使用した。TAPSは喉マイクと気導マイクのペア同時収録データセットであり、60話者・6,000発話から構成される。話者独立な分割設計（train: 40話者・4,000発話、dev: 10話者・1,000発話、test: 10話者・1,000発話）が採用されている。喉マイクは8 kHzで収録され16 kHzにアップサンプリングされている。")]),

  h2("3.2 SEモデルと学習設定"),
  p([
    txt("TAPS論文で提供されたSE-Conformer（12.1Mパラメータ）の事前学習済み重みから初期化し追加学習を行った。最適化にはAdam（lr=3\u00D710"),
    sup("-4"),
    txt("、\u03B2=(0.9, 0.99)）を使用し、batch size=4、最大50エポック、early stopping（patience=5）で学習した。CE損失のWhisper forward部分にはAMP（混合精度演算）を適用した。"),
  ]),

  h2("3.3 評価設定"),
  p([txt("ASR評価にはfaster-whisper（CTranslate2形式）を使用し、言語=韓国語、beam size=5でCERを計測した。CERは1.0でキャップした。汎用性検証としてWhisper-base（74M）・small（244M）・medium（769M）の3モデルで評価した。知覚品質はpySTOIによるSTOIとITU-T P.862に基づくPESQ（広帯域モード）で計測し、同一発話の気導マイク音声を参照信号とした。")]),

  // ────────────────────────────────────────────────
  // 4. 実験結果
  // ────────────────────────────────────────────────
  h1("4. 実験結果"),

  h2("4.1 CE損失の\u03BB探索"),
  p([
    txt("表1にCE損失の重み"), mi("\u03BB"), txt("とCERの関係を示す。"), mi("\u03BB"), txt(" = 2.0で最良のCER 0.2285を達成し、TAPS pretrainedベースラインと比較して9.3%の改善となった。Wilcoxon符号順位検定の結果、"),
    mi("p"), txt(" = 2.75\u00D710"),
    sup("-25"),
    txt("であり統計的に極めて有意な差が確認された。また、testセットの全10話者において提案手法のCERがベースラインを下回り、改善は話者に依存しない一貫したものである。"), mi("\u03BB"), txt("が大きすぎると（"), mi("\u03BB"), txt(" = 5.0以降）CERが悪化に転じる。これはASR損失への偏重により波形品質が崩壊するためと考えられる。"),
  ]),
  tbl(
    ["\u03BB", "CER", "\u0394 vs TAPS"],
    [
      ["TAPS pretrained", "0.2518", "\u2014"],
      ["CE \u03BB=0.5", "0.2360", "-6.3%"],
      ["CE \u03BB=1.0", "0.2294", "-8.9%"],
      ["CE \u03BB=2.0（最良）", "0.2285", "-9.3%"],
      ["CE \u03BB=5.0", "0.2404", "-4.5%"],
    ],
    [1600, 1300, 1300]
  ),
  cap("表1: CE損失の\u03BB値とCER（Whisper-small、test 1,000発話）"),

  p([txt("また、再構成損失（L1+STFT）の必要性を検証するablation実験を行った。CE損失のみ（再構成損失なし、"), mi("\u03BB"), txt(" = 2.0）で学習した場合のCERは0.237であり、TAPS pretrained（0.252）を上回るものの、再構成損失を加えた場合（0.229）と比較すると劣る。この結果は、再構成損失による波形品質の維持がASR精度にも寄与しており、両損失が相補的に機能していることを示している。")]),

  h2("4.2 別ASRモデルでの評価"),
  p([txt("表2に、Whisper-smallで学習したCE-aware SEを異なるASRモデルで評価した結果を示す。提案手法（CE "), mi("\u03BB"), txt(" = 2.0）は全3モデルでTAPS pretrainedを上回っており、学習に使用した特定のASRモデルへの過学習は確認されなかった。一方、encoder距離損失（Enc L1）はWhisper-baseおよびWhisper-smallでTAPSよりCERが悪化している。")]),
  tbl(
    ["SE条件", "W-base", "W-small", "W-medium"],
    [
      ["No SE", "0.676", "0.471", "0.360"],
      ["TAPS pretrained", "0.294", "0.252", "0.219"],
      ["Enc L1（\u03BB=5.0）", "0.308", "0.257", "0.225"],
      ["CE（\u03BB=2.0）", "0.284", "0.229", "0.206"],
    ],
    [1200, 1000, 1000, 1000]
  ),
  cap("表2: 異なるASRモデルでのCER比較"),

  h2("4.3 Encoder距離分析"),
  p([txt("CE損失がなぜ汎用的に有効かを分析するため、各SE条件について3つのASRモデルのencoder表現間のL1距離を計測した（表3）。encoder距離は、SE出力と気導音声を各Whisperのencoderに入力し、出力表現のL1距離を計算したものである。")]),
  tbl(
    ["SE条件", "base L1", "small L1", "medium L1"],
    [
      ["No SE", "0.367", "0.386", "0.408"],
      ["TAPS pretrained", "0.157", "0.201", "0.226"],
      ["Enc L1（\u03BB=5.0）", "0.163", "0.199", "0.230"],
      ["CE（\u03BB=2.0）", "0.215", "0.290", "0.343"],
    ],
    [1200, 1000, 1000, 1000]
  ),
  cap("表3: 各ASRモデルのencoder表現間L1距離"),

  p([txt("注目すべきは、CE損失（CER最良）がTAPS pretrainedよりもencoder距離が"),
    txtB("大きい"),
    txt("点である。すなわち、CE損失で学習したSEの出力は気導音声にencoder空間で近づいておらず、むしろ離れている。それにも関わらずCERは改善している（表2）。これは「気導音声にencoder表現を近づける＝ASR精度が向上する」という仮説が成立しないことを意味する。")]),
  p([txt("一方、Enc L1はWhisper-smallのencoder距離をわずかに縮小（0.201→0.199）しているが、CERはむしろ悪化している（0.252→0.257）。また、学習に使用していないWhisper-base・mediumではencoder距離が逆に増加しており、特定のencoderへの過学習が確認された。")]),
  p([txt("以上の結果は、CE損失が気導音声への復元とは異なるメカニズムでASR精度を改善していることを示唆する。CE損失は「気導音声に似た音声」ではなく、「ASRが正しく書き起こせる音声」を生成しており、これは気導音声とは異なる表現である可能性がある。")]),

  h2("4.4 知覚品質評価"),
  p([txt("表4にSTOI・PESQ・CERの比較を示す。TAPS pretrainedはSTOI 0.892・PESQ 1.975と最も高い知覚品質を達成するが、CERは0.252にとどまる。一方、提案手法（CE "), mi("\u03BB"), txt(" = 2.0）はSTOI 0.825・PESQ 1.256と知覚品質が低下するにも関わらず、CERは0.229と最良である。")]),
  tbl(
    ["条件", "STOI", "PESQ", "CER"],
    [
      ["No SE", "0.697", "1.224", "0.471"],
      ["TAPS pretrained", "0.892", "1.975", "0.252"],
      ["Enc L1（\u03BB=5.0）", "0.871", "1.750", "0.257"],
      ["CE（\u03BB=2.0）", "0.825", "1.256", "0.229"],
    ],
    [1200, 1000, 1000, 1000]
  ),
  cap("表4: STOI・PESQ・CERの比較"),

  p([txt("この結果は、従来報告されてきた現象 — SEにより知覚品質（STOI・PESQ）が改善するにも関わらずASR精度（CER）が悪化する[1][2] — の逆パターンを示している。すなわち、知覚品質が低下するにも関わらずCERが改善する現象が観測された。知覚品質の最適化とASR精度の最適化は双方向に乖離しうることを示しており、「気導音声への復元」を目標とする従来SEの前提そのものが、ASR応用には最適でない可能性を示唆する。")]),

  // ────────────────────────────────────────────────
  // 5. 考察とまとめ
  // ────────────────────────────────────────────────
  h1("5. 考察とまとめ"),

  p([txt("本研究では、Whisperの交差エントロピー損失をSE学習に直接組み込むASR-aware SEを提案し、韓国語喉マイクデータセットTAPSで評価した。主要な知見を以下にまとめる。")]),

  p([txt("第一に、提案手法（CE "), mi("\u03BB"), txt(" = 2.0）はTAPS pretrainedベースラインと比較してCERを9.3%改善し（"), mi("p"), txt(" < 0.001）、全10話者で一貫した改善を達成した。第二に、この改善は学習に使用していないWhisper-baseおよびWhisper-mediumでも確認され、提案手法の汎用性が実証された。第三に、encoder距離分析（表3）により、CE損失が気導音声への復元とは異なるメカニズムでASR精度を改善していることが明らかになった。CE損失で学習したSEはencoder空間において気導音声からむしろ離れるが、CERは改善する。これは「気導音声の復元」を目標とする従来SEの前提が、ASR応用には最適でないことを示す。encoder距離損失は特定モデルの中間表現に過学習するのに対し、CE損失は最終出力を最適化するため、モデル内部表現に依存しない汎用的な改善を達成する。")]),

  p([txt("一方、提案手法のCER 0.229はWhisperファインチューニング（CER 0.136）には及ばない。しかし、SEはASRモデルに依存しない前処理であり、モデル更新のたびに再学習が不要という汎用性がある。今後は、CE損失とencoder距離の組み合わせ、他言語・他データセットでの検証、より大規模なSEモデルでの評価が課題である。")]),

  // ────────────────────────────────────────────────
  // 参考文献
  // ────────────────────────────────────────────────
  h1("参考文献"),
  p([txt("[1] T. Ochiai, et al., \u201CRethinking Processing Distortions: Disentangling the Impact of Speech Enhancement Errors on Speech Recognition Performance,\u201D IEEE/ACM Trans. Audio, Speech, and Language Processing, vol. 32, 2024.", { size: 17 })], { spacing: { after: 20, line: 260 } }),
  p([txt("[2] C.O. Mawalim, S. Okada, and M. Unoki, \u201CAre Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?\u201D Proc. Interspeech, DOI: 10.21437/Interspeech.2024-129, 2024.", { size: 17 })], { spacing: { after: 20, line: 260 } }),
  p([txt("[3] G. Close, W. Ravenscroft, T. Hain, and S. Goetze, \u201CPerceive and Predict: Self-Supervised Speech Representation Based Loss Functions for Speech Enhancement,\u201D Proc. ICASSP, 2023.", { size: 17 })], { spacing: { after: 20, line: 260 } }),
  p([txt("[4] Y. Kim, et al., \u201CThroat and Acoustic Paired Speech Dataset for Deep Learning-Based Speech Enhancement,\u201D Scientific Data (Nature), DOI: 10.1038/s41597-026-07268-2, 2026.", { size: 17 })], { spacing: { after: 20, line: 260 } }),
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
        p([txtB("喉マイク音声認識のためのASR損失に基づく音声強調の最適化", { size: 28, font: "MS Gothic" })], { alignment: AlignmentType.CENTER, spacing: { after: 100 } }),
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
  const out = __dirname + "/thesis.docx";
  fs.writeFileSync(out, buf);
  console.log("Created:", out);
});
