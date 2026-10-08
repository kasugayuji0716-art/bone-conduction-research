// 日本音響学会 講演論文形式（A4・2段組）の原稿 → paper_asj_2026-10.docx
// 書式は thesis_v3.js と同じ。数値の出典: results/GENERATIVE_RESULTS_2026-09-30.md §1・§6・§7、図は scripts/83
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
        BorderStyle, WidthType, ShadingType, SectionType, ImageRun, FootnoteReferenceRun } = require("docx");

// ── ASJ: A4, 22mm LR, 18mm TB, 2-column ──
const mm = v => Math.round(v * 56.69);
const DXA_A4_W = 11906, DXA_A4_H = 16838;
const M_LR = mm(22), M_TB = mm(18), GAP = mm(6);

const S = 20; // 10pt
const txt = (t, o = {}) => new TextRun({ text: t, font: "MS Mincho", size: S, ...o });
const txtB = (t, o = {}) => txt(t, { bold: true, ...o });
const sub = (t) => new TextRun({ text: t, font: "Cambria Math", size: 17, subScript: true });
const sup = (t) => new TextRun({ text: t, font: "MS Mincho", size: S, superScript: true });
const mi = (t, o = {}) => new TextRun({ text: t, font: "Cambria Math", size: S, italics: true, ...o });
const mo = (t) => new TextRun({ text: t, font: "Cambria Math", size: S });
const p = (c, o = {}) => new Paragraph({ spacing: { after: 60, line: 300 }, ...o, children: Array.isArray(c) ? c : [c] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 200, after: 100 }, children: [txtB(t, { size: 24, font: "MS Gothic" })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 160, after: 80 }, children: [txtB(t, { size: 22, font: "MS Gothic" })] });
const eqp = (runs) => p(runs, { alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 } });

const bdr = { style: BorderStyle.SINGLE, size: 1, color: "888888" };
const bdrs = { top: bdr, bottom: bdr, left: bdr, right: bdr };
const cp = { top: 20, bottom: 20, left: 50, right: 50 };
function cell(text, { width, shade, bold, align } = {}) {
  return new TableCell({
    borders: bdrs, margins: cp, width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT, spacing: { after: 0, line: 240 }, keepNext: true, keepLines: true,
      children: [bold ? txtB(text, { size: 16 }) : txt(text, { size: 16 })] })],
  });
}
// 太字にしたいセルは "**0.163" のように先頭に ** を付ける
function tbl(hdrs, rows, ws) {
  const tw = ws.reduce((a, b) => a + b, 0);
  const c = (v, i) => {
    const b = v.startsWith("**");
    return cell(b ? v.slice(2) : v, { width: ws[i], bold: b, align: i > 0 ? AlignmentType.CENTER : AlignmentType.LEFT });
  };
  return new Table({
    width: { size: tw, type: WidthType.DXA }, columnWidths: ws,
    rows: [new TableRow({ children: hdrs.map((h, i) => cell(h, { width: ws[i], shade: "D9E2F3", bold: true, align: AlignmentType.CENTER })) }),
           ...rows.map(r => new TableRow({ children: r.map(c) }))],
  });
}
const cap = (t) => p([txt(t, { size: 17 })], { alignment: AlignmentType.CENTER, keepLines: true, spacing: { before: 30, after: 140 } });
const capTop = (t) => p([txt(t, { size: 17 })], { alignment: AlignmentType.CENTER, keepNext: true, keepLines: true, spacing: { before: 100, after: 40 } });
const ref = (t) => p([txt(t, { size: 16 })], { spacing: { after: 0, line: 230 } });

const content = [
  // ────────────────────────────────────────────────
  h1("1. はじめに"),
  p([txt("喉マイクは頸部の振動を接触型センサで捉えるため環境騒音に強いが、軟組織の伝搬特性により高域が大きく失われる。喉マイクと気導マイクの同時収録コーパスTAPS[1]では喉マイクが8 kHzで収録されており、4 kHz以上は収録時点から存在しない。Whisper-small[2]で喉マイク音声をそのまま認識すると文字誤り率（CER）は0.446に達する。")]),
  p([txt("前処理としての音声強調（SE）は有力な対策であるが、知覚品質の改善が音声認識（ASR）の改善を保証しないことが知られている[3]。この乖離に対し、凍結したASRの損失でSEを学習する方法が提案されてきた[4,5]。しかし、損失に用いた認識器以外への効果は十分に調べられていない。本稿の実験でも、Whisper-smallの交差エントロピー（CE）損失で学習した喉マイクSEは、WhisperではCERを下げる一方、入力特徴量の異なる認識器では悪化した（4.1節）。ASR損失で学習したSEは、損失に用いた認識器の入力特徴量に特化しうる。")]),
  p([txt("一方、喉マイクの問題は雑音の混入ではなく帯域や子音の欠落であり、欠けた成分を補う必要がある。大規模音声で事前学習した生成モデル[6]は、入力が持たない成分を自然な音声として生成できる。体内伝導マイクではこの種の生成モデルの適応が有効との報告がある[16]が、評価は単一の認識器に限られる。また、生成的な帯域拡張や声質変換が喉マイク音声のCERを悪化させたという報告もある[8]。")]),
  p([txt("本稿では対象をWhisperと同じ対数メルスペクトログラムを入力とする認識器（以下Whisper系）に定め、損失と設定の選択にはWhisper-smallのみを用い、他のWhisper系5種を評価専用とする。そのうえで次の3点を示す。(1) 事前学習済みのflow matching生成モデルをTAPSで適応すると、入力特徴量の異なる認識器を含む9種すべてでCERが下がり、効果は事前学習に由来する。(2) 生成モデルの複数サンプルの振幅平均に、Whisper用SEの出力を振幅で融合すると、学習に用いていないWhisper系にもさらに上乗せが得られる。(3) 生成モデル自体をWhisperの損失で学習する方法は失敗し、融合の上乗せは帯域の両側から生じ、他系統の認識器への害は元々存在しない4 kHz以上の成分から生じる。")]),

  // ────────────────────────────────────────────────
  h1("2. 方法"),
  h2("2.1 生成モデルによる復元"),
  p([txt("NVIDIAが公開しているflow matching[9]型の音声復元モデル（4.3億パラメータ、複素STFT領域）[6]を用いる。このモデルは大規模な音声データで、マスクした区間を復元するよう事前学習されている。本研究ではTAPSの喉マイク音声を条件、同時収録の気導マイク音声を目標として追加学習する（ASRの損失は用いない）。推論では標準正規分布の雑音から出発し、喉マイク音声を条件として常微分方程式をEuler法20ステップで解く。")]),
  p([txt("生成は出発点の雑音に依存して発話ごとにばらつく。そこで乱数の種を変えた"), mi("N"), txt("個のサンプルのSTFT振幅を平均し、位相は1つ目のサンプルのものを用いる。")]),
  eqp([mo("|"), mi("G"), mo("| = (1/"), mi("N"), mo(") Σ"), sub("n"), mo(" |"), mi("G"), sub("n"), mo("|,   ∠"), mi("G"), mo(" = ∠"), mi("G"), sub("1")]),
  h2("2.2 Whisper用SE"),
  p([txt("TAPS論文[1]のSE-Conformer（1210万パラメータ、以下TAPS SE）を初期値とし、次の損失で追加学習したモデル（以下CE-SE）を用いる。")]),
  eqp([mi("L"), mo(" = "), mi("L"), sub("recon"), mo("("), mi("ŝ"), mo(", "), mi("s"), sub("air"), mo(") + "), mi("λ"), mo(" CE(Whisper("), mi("ŝ"), mo("), "), mi("y"), mo(")")]),
  p([mi("ŝ"), txt("はSE出力、"), mi("s"), sub("air"), txt("は気導音声、"), mi("y"), txt("は正解テキスト、"), mi("L"), sub("recon"), txt("はL1波形損失と多重解像度STFT損失の和である。Whisper-smallは凍結し、"), mi("λ"), txt("はdevのCERで{0, 0.1, 0.5, 1, 2, 5, 10}から選んで10とした。")]),
  h2("2.3 振幅融合"),
  p([txt("生成モデルの出力"), mi("G"), txt("（"), mi("N"), txt(" = 4の平均）とCE-SEの出力"), mi("C"), txt("を、STFT（512点、シフト128点）の振幅で重み付き平均し、位相は生成モデルのものを用いる。")]),
  eqp([mi("Y"), mo(" = {(1 − "), mi("w"), mo(")|"), mi("G"), mo("| + "), mi("w"), mo("|"), mi("C"), mo("|} exp(j∠"), mi("G"), mo(")")]),
  p([mi("w"), txt("はdevでWhisper-smallのCERのみを見て{0.25, 0.5, 0.75}から選び、"), mi("w"), txt(" = 0.5とした（devのCERはそれぞれ0.202・0.196・0.198）。融合に新たな学習は要らない。")]),

  // ────────────────────────────────────────────────
  h1("3. 実験条件"),
  h2("3.1 データと学習"),
  p([txt("TAPS[1]の話者独立な分割（train 40話者・dev 10話者・test 10話者、各話者100発話）を用いた。生成モデルはtrainの4,000対で20k step追加学習した（batch 8、6.14秒の切り出し、学習率10"), sup("−4"), txt("、EMA、32ビット浮動小数点）。モデルの選択には最終ステップを用い、testを見て選んでいない。CE-SEはAdam（学習率3×10"), sup("−4"), txt("、batch 4）で最大50エポック学習し、devの損失でearly stoppingを行った。15秒を超える発話は学習から除外した。")]),
  h2("3.2 比較手法"),
  p([txt("TAPS SE、CE-SEに加え、2つの既存手法を比較した。SSL-MSEは、SE出力と気導音声のWavLM-Large[10]全層の表現の二乗誤差を再構成損失に加えてTAPS SEを追加学習したもので、Satoら[5]の方式に従う。Dissen型は、凍結したASRの損失で前段を学習する方法[4]に倣い、喉マイクの対数メルスペクトログラムを入力・出力とするU-Net（約700万パラメータ）を、Whisper-smallのCEと気導音声の対数メルへのL1損失で学習したものである。")]),
  h2("3.3 認識器と評価"),
  p([txtB("Whisper系．"), txt("Whisper-small（損失と選択に使用）、Whisper-base、Whisper-medium、Whisper-large-v3-turbo、TAPSの喉マイク音声でファインチューニングしたWhisper-small（FT Whisper）、Qwen3-ASR-1.7Bの6種である。Qwen3-ASRのエンコーダはWhisperと異なるが、入力は同形式の対数メル（128次元）である。")]),
  p([txtB("入力特徴量の異なる認識器．"), txt("Whisper系への特化を確認する対照として、波形入力のCTC型であるMMS-1B[11]とXLS-R[12]（韓国語で追加学習したもの）、Kaldi型のフィルタバンクを入力とするZipformer[13]を用いた。")]),
  p([txtB("指標．"), txt("test 1,000発話で、正解と認識結果の双方から句読点を除いたCERを発話ごとに求め（1.0で打ち切り）、平均した。Zipformerは空白を含めたCERである。同一話者の発話は独立でないため、話者ごとの平均CERを単位とするWilcoxon符号順位検定（"), mi("n"), txt(" = 10、最小の"), mi("p"), txt(" = 0.002）を用いた。")]),

  // ────────────────────────────────────────────────
  h1("4. 実験結果"),
  h2("4.1 主な結果"),
  p([txt("表1に各手法のCERを、図1にTAPS SE比の相対変化を示す。生成モデル（1回の生成）は9種の認識器すべてでTAPS SEよりCERを15〜24%下げ、いずれも10話者全員で改善した（"), mi("p"), txt(" = 0.002）。Whisper-smallの認識結果で見ると、CERが0.99以上に崩れた発話はなく、出力長の中央値も変わらず（正解長比0.976）、置換と挿入の誤りが減っていた。生成による作り話の兆候は見られなかった。")]),
  p([txt("同じ構造・同じ学習量で初期値を乱数としたモデルは、Whisper-smallで0.551、XLS-Rで0.524とTAPS SEより大幅に悪化した。生成モデルの効果はモデルの規模ではなく事前学習に由来する。")]),
  capTop("表1: 各手法のtest CER（*は損失・選択に使用）"),
  tbl(
    ["認識器", "TAPS", "CE-SE", "SSL-MSE", "生成×1", "生成×4", "融合"],
    [
      ["W-small*", "0.230", "0.196", "0.204", "0.180", "0.169", "**0.163"],
      ["W-base", "0.273", "0.267", "0.241", "0.218", "0.205", "**0.201"],
      ["W-medium", "0.196", "0.169", "0.177", "0.159", "0.151", "**0.144"],
      ["W-turbo", "0.170", "0.142", "0.153", "0.137", "0.131", "**0.125"],
      ["FT Whisper", "0.146", "0.121", "0.124", "0.110", "0.100", "**0.092"],
      ["Qwen3-ASR", "0.136", "0.114", "0.114", "0.104", "0.097", "**0.094"],
      ["XLS-R", "0.222", "0.279", "0.190", "0.169", "**0.161", "0.163"],
      ["MMS-1B", "0.350", "0.354", "0.321", "0.297", "**0.284", "0.285"],
      ["Zipformer", "0.542", "0.791", "0.486", "0.451", "**0.433", "0.483"],
    ],
    [1150, 560, 560, 620, 560, 560, 530]
  ),
  p([], { spacing: { after: 60 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 0 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(__dirname + "/figures/fusion_relative.png"), transformation: { width: 250, height: 228 } })] }),
  cap("図1: TAPS SE比のCER変化（破線より上がWhisper系）"),

  h2("4.2 サンプル平均と融合"),
  p([txt("4サンプルの振幅平均は、1回の生成よりさらにCERを下げた（Whisper-small 0.180→0.169、10/10話者、"), mi("p"), txt(" = 0.002）。平均CERは乱数の種によらずほぼ一定（0.180・0.182・0.181）であったが、発話単位では3回のCERの差が0.1を超える発話が82/1,000あり、平均によりこのばらつきが抑えられたと考えられる。この改善は入力特徴量の異なる認識器でも同様であった。")]),
  p([txt("CE-SEとの融合は、Whisper系の6種すべてで4サンプル平均よりCERを下げた（−1.8〜−8.0%）。Whisper-medium・large-v3-turbo・FT Whisperでは10/10話者（"), mi("p"), txt(" = 0.002）、Whisper-smallで9/10話者（"), mi("p"), txt(" = 0.004）で改善し、Whisper-baseとQwen3-ASRでは有意でなかった。TAPS SE比ではWhisper系で−26〜−37%となり、SSL-MSE・CE-SE単体のいずれよりも良い。一方、入力特徴量の異なる認識器では融合の上乗せはなく、Zipformerでは4サンプル平均より11.7%悪化した（0/10話者）。融合はWhisper系に特化した上乗せである。")]),
  p([txt("Dissen型は80次元メルのWhisperにのみ適用できるため、全条件をtransformers版Whisperの貪欲復号で認識し直して比較した。Whisper-smallでTAPS SE 0.237、Dissen型 0.239、融合 0.169であり、Dissen型はWhisper-base・medium・FT WhisperでもTAPS SEより悪かった。喉マイクのメルのみを入力とするため、欠けた帯域を補えなかったと考えられる。")]),

  h2("4.3 生成モデル自体をWhisperに合わせる試み"),
  p([txt("融合の代わりに、生成モデル自体をWhisper-smallの損失で学習する方法を試した（表2）。途中時刻"), mi("t"), txt("から1段で得られる完成音の見積もり"), mi("x̂"), sub("1"), mo(" = "), mi("x"), sub("t"), mo(" + (1 − "), mi("t"), mo(")"), mi("v̂"), txt("にCEをかけて追加学習すると、見積もりのCEは下がるが、20段で生成した出力のCERは悪化した。CEなしで同じ手順をとると悪化しないため、原因はCE損失にある。最終段のみに損失をかけるDRaFT-K[14]型の学習、生成出力に小さな補正モジュールを加える方法も効果がなかった。CE-SEを融合後の音に対するCEで再学習すると、Whisper系では一貫して改善したが、幅は0.2〜1.9%にとどまった。")]),
  capTop("表2: 生成モデルをWhisperに合わせる試み（W-small）"),
  tbl(
    ["方法", "結果"],
    [
      ["生成×1（基準）", "0.180"],
      ["x̂1にCE（λ=0.1／0.01）", "0.193／0.191"],
      ["同じ手順でCEなし（λ=0）", "0.179"],
      ["DRaFT-K（最終段にCE）", "dev損失が悪化"],
      ["補正モジュール（0.03M）", "変化なし"],
      ["融合後の音でCE-SEを再学習", "0.162（融合0.163）"],
    ],
    [2700, 1650]
  ),
  p([], { spacing: { after: 60 } }),

  h2("4.4 融合の分解"),
  p([txt("CE-SEの何がWhisper系に効くのかを、融合の作り方を変えて調べた（表3）。CE-SEの出力には、3〜7.75 kHzの250 Hz間隔の位置に鋭いピークが並ぶ櫛状の成分が現れる。このピークをノッチで除いてから融合しても結果はほぼ変わらず、上乗せの原因ではなかった。4 kHz未満のみ、または4 kHz以上のみを融合すると、Whisper系の上乗せはそれぞれ約半分になり、両帯域から生じていた。一方、Zipformerは4 kHz未満のみの融合では悪化せず、悪化の原因はCE-SEが作る4 kHz以上の成分であった。喉マイクに元々存在しない帯域に、Whisper向けに作られた成分が、Kaldi型の特徴量を用いる認識器には害となっている。帯域ごとに重みを変えた12通りをdevで比較しても差は1%前後で、"), mi("w"), txt(" = 0.5の一様な重みで十分であった。")]),
  capTop("表3: 融合の分解（test CER）"),
  tbl(
    ["認識器", "生成×4", "融合", "櫛除去", "<4 kHz", "≥4 kHz"],
    [
      ["W-small*", "0.169", "0.163", "0.163", "0.166", "0.167"],
      ["W-medium", "0.151", "0.144", "0.145", "0.145", "0.148"],
      ["W-turbo", "0.131", "0.125", "0.126", "0.126", "0.128"],
      ["FT Whisper", "0.100", "0.092", "0.093", "0.094", "0.099"],
      ["Zipformer", "0.433", "0.483", "0.491", "**0.431", "0.479"],
    ],
    [1150, 640, 640, 640, 640, 640]
  ),
  p([], { spacing: { after: 60 } }),

  h2("4.5 気導音声との距離"),
  p([txt("ASRの学習に用いていない表現であるHuBERT-Large[15]の全層について、同一発話の気導音声とのフレームごとのコサイン距離を求めた。これまでに評価した17条件をまたぐと、この距離は9認識器のうち最も悪化した認識器のCER変化と強く相関した（Spearman "), mi("ρ"), txt(" = 0.92）。一方、Whisper系のCERとの相関は弱く（Whisper-small "), mi("ρ"), txt(" = 0.37）、CE-SEのように気導音声から離れる変化でもWhisper系のCERは下がる。融合はこの両者を組み合わせており、距離は生成×4の0.092から0.111へ増える。")]),

  // ────────────────────────────────────────────────
  h1("5. 考察"),
  p([txt("結果は、喉マイク音声のWhisper系向けSEにおいて2つの成分が役割を分けていることを示す。事前学習済み生成モデルは欠けた成分を自然な音声として補い、認識器の系統によらずCERを下げる。CE-SEは特定の入力特徴量に合わせた変化を加え、その効果は損失に用いたWhisper-smallだけでなく、学習に用いていない他サイズのWhisperやQwen3-ASRにも及ぶ。ただしCE-SE単体ではWhisper-baseの改善は小さく（−2.2%）、生成モデルの出力と融合して初めて系列全体に安定して効いた。")]),
  p([txt("生成モデル自体をCE損失で学習する試みが失敗したことは、Whisperに合わせる成分を生成過程に組み込むよりも、生成の外側で足すほうが容易であることを示唆する。1段の見積もりに対する損失が多段の生成に持ち越されない点は、flow matchingの学習目標がサンプルの分布を合わせるものであり、個々の出力を特定の損失で最適化する構造になっていないことと整合する。")]),
  p([txt("実用上は2つの選択肢がある。Whisper系のみを用いるなら一様な融合が最良であり、他系統の認識器も併用するなら4 kHz未満のみの融合でZipformerへの害を避けつつWhisper系の上乗せの約半分が得られる。")]),
  p([txt("本研究の限界として、計算量が大きい（4.3億パラメータのモデルを20段・4サンプル実行）こと、評価がTAPS（韓国語・1機種）のみであること、各モデルの学習が1回であることが挙げられる。融合自体は既存の部品の組み合わせであり、今後は融合の出力を教師とした軽量なSEへの蒸留と、別言語・別機種の体内伝導マイクコーパス[7]での検証を行う。")]),

  // ────────────────────────────────────────────────
  h1("6. まとめ"),
  p([txt("Whisper系の認識器に向けた喉マイクSEについて、事前学習済み生成モデルとASR損失で学習したSEの役割を調べた。生成モデルの適応は9種の認識器すべてでCERを下げ、その効果は事前学習に由来した。生成モデルの4サンプル平均とWhisper用SEの振幅融合は、学習に用いていないWhisper系にも上乗せをもたらし、TAPS SE比で26〜37%CERを下げた。一方、生成モデル自体をWhisperの損失で学習する方法は効果がなく、Whisper用の成分のうち4 kHz以上のものは他系統の認識器を悪化させた。")]),

  // ────────────────────────────────────────────────
  h1("参考文献"),
  ref("[1] Y. Kim et al., “Throat and acoustic paired speech dataset for deep learning-based speech enhancement,” Scientific Data, 2026."),
  ref("[2] A. Radford et al., “Robust speech recognition via large-scale weak supervision,” Proc. ICML, 2023."),
  ref("[3] T. Ochiai et al., “Rethinking processing distortions: Disentangling the impact of speech enhancement errors on speech recognition performance,” IEEE/ACM TASLP, vol. 32, 2024."),
  ref("[4] Y. Dissen, S. Yonash, I. Cohen, and J. Keshet, “Enhanced ASR robustness to packet loss with a front-end adaptation network,” Proc. Interspeech, 2024."),
  ref("[5] H. Sato et al., “Generic speech enhancement with self-supervised representation space loss,” Frontiers in Signal Processing, vol. 5, 2025."),
  ref("[6] P.-J. Ku et al., “Generative speech foundation model pretraining for high-quality speech extraction and restoration,” arXiv:2409.16117, 2024."),
  ref("[7] J. Hauret et al., “Vibravox: A dataset of French speech captured with body-conduction audio sensors,” Speech Communication, 2025."),
  ref("[8] 山中 涼雅 ほか, “Zero-shot音声変換を用いた咽喉マイク音声の気導音復元に関する初期検討,” 音講論集, 1-Q-42, 2026（秋）."),
  ref("[9] Y. Lipman et al., “Flow matching for generative modeling,” Proc. ICLR, 2023."),
  ref("[10] S. Chen et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE JSTSP, vol. 16, 2022."),
  ref("[11] V. Pratap et al., “Scaling speech technology to 1,000+ languages,” JMLR, vol. 25, 2024."),
  ref("[12] A. Babu et al., “XLS-R: Self-supervised cross-lingual speech representation learning at scale,” Proc. Interspeech, 2022."),
  ref("[13] Z. Yao et al., “Zipformer: A faster and better encoder for automatic speech recognition,” Proc. ICLR, 2024."),
  ref("[14] K. Clark et al., “Directly fine-tuning diffusion models on differentiable rewards,” Proc. ICLR, 2024."),
  ref("[15] W.-N. Hsu et al., “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM TASLP, vol. 29, 2021."),
  ref("[16] J. Hauret et al., arXiv:2508.02974, 2025."),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "MS Mincho", size: S } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "MS Gothic" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "MS Gothic" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  footnotes: {
    1: { children: [new Paragraph({ children: [txt("Roles of a pretrained generative model and ASR-loss enhancement in throat-microphone speech enhancement for Whisper-family recognizers, by KASUGA, Yuji (Meiji University).", { size: 16 })] })] },
  },
  sections: [
    {
      properties: { page: { size: { width: DXA_A4_W, height: DXA_A4_H }, margin: { top: M_TB, right: M_LR, bottom: M_TB, left: M_LR } } },
      children: [
        p([txtB("Whisper系音声認識に向けた喉マイク音声強調における", { size: 28, font: "MS Gothic" })], { alignment: AlignmentType.CENTER, spacing: { after: 0 } }),
        p([txtB("事前学習済み生成モデルとASR損失の役割", { size: 28, font: "MS Gothic" }), new FootnoteReferenceRun(1)], { alignment: AlignmentType.CENTER, spacing: { after: 100 } }),
        p([txt("春日 裕次（明治大学 総合数理学部 先端メディアサイエンス学科）", { size: 19 })], { alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
      ],
    },
    {
      properties: {
        page: { size: { width: DXA_A4_W, height: DXA_A4_H }, margin: { top: M_TB, right: M_LR, bottom: M_TB, left: M_LR } },
        type: SectionType.CONTINUOUS, column: { count: 2, space: GAP, equalWidth: true },
      },
      children: content,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = __dirname + "/paper_asj_2026-10.docx";
  fs.writeFileSync(out, buf);
  console.log("Created:", out);
});
