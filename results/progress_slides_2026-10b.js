// ゼミ進捗報告（2026-10-08、約5分）→ progress_slides_2026-10b.pptx
// DESIGN.md と academic-pptx スキルの規則で作成（白背景・1書体・3色・見出しは結論の1文・1枚1図・カード/色帯なし）
// 数値の出典: results/GENERATIVE_RESULTS_2026-09-30.md §6–7
const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in

const FONT = "Yu Gothic";
const C = { bg: "FFFFFF", primary: "1F4E79", accent: "2E75B6", alert: "C55A11", body: "2D2D2D", muted: "777777", rule: "CCCCCC", grey: "A6A6A6", highlight: "FFF2CC" };
const M = 0.6, W = 13.333 - 2 * M;

const T = (s, t, o) => s.addText(t, { fontFace: FONT, color: C.body, valign: "top", margin: 0, ...o });
const newSlide = () => { const s = pptx.addSlide(); s.background = { color: C.bg }; return s; };
const title = (s, t) => T(s, t, { x: M, y: 0.4, w: W, h: 1.0, fontSize: 28, bold: true, color: C.primary, valign: "middle" });
const cite = (s, t) => T(s, t, { x: M, y: 6.9, w: W, h: 0.35, fontSize: 12, color: C.muted });
const bullets = (items, base = {}) => items.map((it, i) => {
  const parts = Array.isArray(it) ? it : [it];
  return parts.map((p, j) => ({
    text: typeof p === "string" ? p : p.b,
    options: { ...base, ...(typeof p === "string" ? {} : { bold: true, ...(p.c ? { color: p.c } : {}) }),
               ...(j === 0 ? { bullet: { indent: 18 } } : {}), ...(j === parts.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) },
  }));
}).flat();
const line = (s, x, y, w, h, color = C.rule, width = 1) => s.addShape(pptx.ShapeType.line, { x, y, w, h, line: { color, width } });
const box = (s, x, y, w, h, strong) => s.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: C.bg }, line: { color: strong ? C.primary : C.rule, width: strong ? 2 : 1 } });
const arrow = (s, x, y) => s.addShape(pptx.ShapeType.rightArrow, { x, y, w: 0.3, h: 0.3, fill: { color: C.grey }, line: { color: C.grey } });
const hdr = (t, o = {}) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 16, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }], ...o } });
const cell = (t, o = {}) => ({ text: t, options: { fontSize: 16, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
const pct = v => (100 * v).toFixed(1);

// 1. 表紙
{
  const s = newSlide();
  T(s, "喉マイクの音声を\nWhisper 系の音声認識向けに直す", { x: M, y: 2.2, w: W, h: 1.9, fontSize: 38, bold: true, color: C.primary, lineSpacingMultiple: 1.15 });
  T(s, "生成モデルと Whisper 用の音声強調を混ぜる方法と、その分析", { x: M, y: 4.25, w: W, h: 0.5, fontSize: 20 });
  T(s, "春日 裕次　ゼミ進捗報告　2026年10月8日", { x: M, y: 5.0, w: W, h: 0.4, fontSize: 16, color: C.muted });
  s.addNotes("【台本】（約10秒）\n進捗を報告します。前回から目標を Whisper 系の認識器に絞りました。今日はその結果と、手法としての評価、今後を話します。");
}

// 2. 目標を絞った
{
  const s = newSlide();
  title(s, "前回から、目標を「Whisper 系の誤りを下げる音声強調」に絞った");
  T(s, "前回まで", { x: M, y: 1.65, w: 5.6, h: 0.45, fontSize: 20, bold: true, color: C.muted });
  T(s, "どの認識器でも誤りが減る音声強調\n（入力の違う認識器も含めて 9つ）", { x: M, y: 2.15, w: 5.6, h: 1.0, fontSize: 18, color: C.muted, lineSpacingMultiple: 1.2 });
  T(s, "→", { x: M + 5.65, y: 1.95, w: 0.6, h: 0.8, fontSize: 32, bold: true, color: C.grey, align: "center" });
  T(s, "今回から", { x: M + 6.4, y: 1.65, w: W - 6.4, h: 0.45, fontSize: 20, bold: true, color: C.primary });
  T(s, "Whisper 系の認識器の誤りを下げる音声強調\n（Whisper と同じ音の特徴を入力にする認識器）", { x: M + 6.4, y: 2.15, w: W - 6.4, h: 1.0, fontSize: 18, lineSpacingMultiple: 1.2 });
  line(s, M, 3.55, W, 0);
  T(s, bullets([
    ["学習と設定選びに使うのは ", { b: "Whisper-small だけ" }],
    ["ほかの Whisper 系 5つ（base・medium・large-v3-turbo・喉マイクで再学習した Whisper・Qwen3-ASR）は", { b: "評価専用" }],
    ["入力の違う認識器（XLS-R・Zipformer）は、", { b: "Whisper 系専用になっているか" }, "を見る対照"],
  ], { fontSize: 20 }), { x: M, y: 3.8, w: W, h: 2.8, paraSpaceAfter: 14, lineSpacingMultiple: 1.15 });
  cite(s, "評価: TAPS テスト 1000発話（10話者）、句読点を除いた文字誤り率、話者単位の符号順位検定");
  s.addNotes("【台本】（約30秒）\n前回は、どの認識器でも効く音声強調を目標にしていましたが、手法として形にしにくいので、Whisper 系の認識器に絞りました。\n\n学習と設定選びには Whisper-small だけを使い、ほかの Whisper 系5つは評価専用にしています。入力の違う認識器は、Whisper 系専用になっているかを確かめる対照として残しています。\n\n【補足】Qwen3-ASR はエンコーダは独自（AuT）だが、入力は Whisper と同じ形式の log-mel。");
}

// 3. 方法
{
  const s = newSlide();
  title(s, "方法：生成モデルの4回平均に、Whisper 用の音声強調を半分混ぜる");
  const y1 = 1.85, y2 = 3.75, bh = 1.25;
  T(s, "喉マイクの音", { x: M, y: 2.65, w: 1.9, h: 0.9, fontSize: 18, bold: true, valign: "middle" });
  line(s, M + 1.95, 2.45, 0, 1.75, C.grey, 1.5);
  [[y1, "生成モデル（喉マイク用に調整）", "出発点を変えて 4回生成し、振幅を平均", false],
   [y2, "Whisper 用の音声強調（卒論）", "Whisper-small が正しく書けるよう学習", false]].forEach(([y, a, b]) => {
    arrow(s, M + 2.05, y + bh / 2 - 0.15);
    box(s, M + 2.5, y, 4.6, bh, false);
    T(s, a, { x: M + 2.7, y: y + 0.15, w: 4.3, h: 0.45, fontSize: 17, bold: true, color: C.primary });
    T(s, b, { x: M + 2.7, y: y + 0.62, w: 4.3, h: 0.55, fontSize: 15 });
    arrow(s, M + 7.25, y + bh / 2 - 0.15);
  });
  box(s, M + 7.75, 2.4, 2.6, 1.9, true);
  T(s, [{ text: "振幅を半々で平均", options: { bold: true, color: C.primary, breakLine: true } }, { text: "位相は生成モデル", options: { fontSize: 15 } }],
    { x: M + 7.9, y: 2.4, w: 2.3, h: 1.9, fontSize: 17, valign: "middle", align: "center", lineSpacingMultiple: 1.3 });
  arrow(s, M + 10.5, 3.2);
  T(s, "Whisper 系の\n認識器", { x: M + 10.95, y: 2.4, w: 1.2, h: 1.9, fontSize: 17, bold: true, valign: "middle" });
  T(s, bullets([
    ["混ぜる割合 0.5 は、開発データで Whisper-small だけを見て選んだ（0.25〜0.75 で結果はほぼ同じ）"],
    ["新たな学習は不要。部品はどちらも学習済み"],
  ], { fontSize: 18 }), { x: M, y: 5.35, w: W, h: 1.3, paraSpaceAfter: 10 });
  cite(s, "生成モデル: NVIDIA の flow matching（4.3億パラメータ、Ku et al., arXiv:2409.16117）を TAPS で追加学習");
  s.addNotes("【台本】（約40秒）\n方法です。喉マイクの音を2つの処理に通します。\n\n上は前回の生成モデルで、ランダムな出発点を変えて4回生成し、振幅を平均します。これで生成のばらつきが打ち消されます。\n\n下は卒論で作った、Whisper が正しく書き起こせるように学習した音声強調です。\n\nこの2つの出力の振幅を半々で平均し、位相は生成モデルのものを使います。混ぜる割合は開発データで Whisper-small だけを見て選びましたが、0.25 から 0.75 でほとんど変わりません。\n\n【補足】振幅は STFT（512点、ずらし128点）。帯域ごとに割合を変えても差は1%前後。");
}

// 4. 結果
{
  const s = newSlide();
  title(s, "Whisper 系6つすべてで、生成モデル単体よりさらに誤りが減った");
  const R = [["Whisper-small", 0.2297, 0.1685, 0.1633], ["Whisper-base", 0.2726, 0.2048, 0.2011], ["Whisper-medium", 0.1962, 0.1505, 0.1436],
             ["large-v3-turbo", 0.1697, 0.1312, 0.1248], ["Whisper（喉マイクで再学習）", 0.1462, 0.1000, 0.0920], ["Qwen3-ASR", 0.1359, 0.0968, 0.0936]];
  const labels = R.map(r => r[0]);
  s.addChart(pptx.charts.BAR, [
    { name: "元の音声強調（TAPS）", labels, values: R.map(r => 100 * r[1]) },
    { name: "生成モデル（4回平均）", labels, values: R.map(r => 100 * r[2]) },
    { name: "混ぜたもの", labels, values: R.map(r => 100 * r[3]) },
  ], {
    x: M, y: 1.5, w: 7.9, h: 5.15, barDir: "bar", barGapWidthPct: 55, chartColors: [C.grey, C.accent, C.primary],
    catAxisOrientation: "maxMin", catAxisLabelFontFace: FONT, catAxisLabelFontSize: 14, catAxisLabelColor: C.body,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 30,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 11, dataLabelColor: C.body,
    showLegend: true, legendPos: "t", legendFontFace: FONT, legendFontSize: 13, legendColor: C.body,
  });
  const rx = M + 8.3, rw = W - 8.3;
  T(s, bullets([
    ["元の音声強調より", { b: "26〜37% 減" }],
    ["生成モデル単体より", { b: "さらに減少" }],
    ["Zipformer では", { b: "12% 増", c: C.alert }, "→ Whisper 系専用の上乗せ"],
  ], { fontSize: 18 }), { x: rx, y: 1.8, w: rw, h: 4.8, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "数値は文字誤り率（%）。生成モデル単体との差: medium・turbo・再学習 Whisper は 10/10 話者（p=0.002）、small 9/10、base・Qwen3 は有意差なし");
  s.addNotes("【台本】（約45秒）\n結果です。灰色が元の音声強調、青が生成モデルの4回平均、紺が混ぜたものです。\n\n混ぜたものは Whisper 系の6つすべてで一番誤りが少なく、元の音声強調と比べて 26〜37% 減りました。生成モデル単体と比べても6つすべてで減っています。学習に使っていない medium や large-v3-turbo でも減っているので、Whisper-small だけに合わせ込んだのではなく、Whisper 系全体に効いています。\n\n一方、入力の違う Zipformer では 12% 増えていて、Whisper 系専用の上乗せになっています。\n\n【補足】XLS-R は +1.2%（n.s.）。混ぜたものの元の音声強調比: small −28.9%、base −26.2%、medium −26.8%、turbo −26.5%、再学習 −37.1%、Qwen3 −31.1%。");
}

// 5. 比較対象
{
  const s = newSlide();
  title(s, "既存の方法と同じ条件で比べても、混ぜたものが最も良い");
  const rows = [
    ["Whisper-small", 0.2297, 0.1956, 0.2035, 0.1685, 0.1633], ["Whisper-base", 0.2726, 0.2667, 0.2413, 0.2048, 0.2011],
    ["Whisper-medium", 0.1962, 0.1686, 0.1772, 0.1505, 0.1436], ["large-v3-turbo", 0.1697, 0.1415, 0.1529, 0.1312, 0.1248],
    ["Whisper（再学習）", 0.1462, 0.1211, 0.1236, 0.1000, 0.0920], ["Qwen3-ASR", 0.1359, 0.1140, 0.1138, 0.0968, 0.0936],
  ];
  const best = { bold: true, color: C.primary };
  s.addTable([
    [hdr(""), hdr("元の音声強調", { align: "center" }), hdr("Whisper 用\n音声強調", { align: "center" }), hdr("自己教師あり\n表現の損失", { align: "center" }), hdr("生成モデル\n（4回平均）", { align: "center" }), hdr("混ぜたもの", { align: "center" })],
    ...rows.map(r => [cell(r[0]), ...r.slice(1).map((v, j) => cell(pct(v), { align: "center", ...(j === 4 ? best : {}) }))]),
  ], { x: M, y: 1.6, w: W, colW: [2.9, 1.86, 1.86, 1.86, 1.86, W - 2.9 - 4 * 1.86], fontFace: FONT, rowH: [0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], valign: "middle", margin: [3, 6, 3, 6] });
  T(s, bullets([
    ["Whisper の損失で mel を直す方法（Dissen ら型）は元の音声強調と同程度"],
  ], { fontSize: 17 }), { x: M, y: 5.6, w: W, h: 0.9 });
  cite(s, "文字誤り率（%）。自己教師あり表現の損失: Sato et al. 2025 型。Dissen ら型（Whisper-small）: 23.9%、同じ認識経路で元 23.7%・混ぜたもの 16.9%");
  s.addNotes("【台本】（約35秒）\n既存の方法とも同じ条件で比べました。卒論の Whisper 用音声強調、自己教師あり表現を損失に使う Sato らの方法、生成モデル単体のどれと比べても、混ぜたものが6つすべてで一番良い結果です。\n\nまた、Whisper の損失で mel を直接直す Dissen らの方法も試しましたが、喉マイクでは欠けた高域を補えず、元の音声強調と同じ程度でした。\n\n【補足】Dissen ら型: U-Net 約700万パラメータ、mel→mel、Whisper-small の CE＋気導 mel の L1。80本 mel の Whisper でしか使えない。");
}

// 6. 失敗
{
  const s = newSlide();
  title(s, "生成モデル自体を Whisper の損失で学習する方法は全滅した");
  s.addTable([
    [hdr("試したこと"), hdr("結果")],
    [cell("生成の途中の見積もりに Whisper の損失をかけて追加学習"), cell("Whisper-small 18.0% → 19.3%（悪化）", { color: C.alert })],
    [cell("生成の最後の段にだけ損失をかける（DRaFT）"), cell("開発データの損失が悪化、打ち切り", { color: C.alert })],
    [cell("生成の出力に小さな補正モジュールを足す"), cell("効果なし")],
    [cell("Whisper 用音声強調を「混ぜた後の音」で学習し直す"), cell("Whisper 系で 0.2〜1.9% だけ改善")],
  ], { x: M, y: 1.7, w: W, colW: [7.2, W - 7.2], fontFace: FONT, rowH: 0.72, valign: "middle", margin: [4, 8, 4, 8] });
  T(s, bullets([
    ["Whisper に合わせる成分は、", { b: "生成モデルの外で足す（混ぜる）方が素直に効く" }],
  ], { fontSize: 20 }), { x: M, y: 5.6, w: W, h: 0.9 });
  cite(s, "途中の見積もり = 生成の途中から1段で出した仮の完成音。この見積もりの損失は下がるが、20段で生成した音には持ち越されない");
  s.addNotes("【台本】（約35秒）\n本当は、生成モデル自体を Whisper に合わせて学習するのが「提案手法」らしい形でした。ですが4通り試して、どれもうまくいきませんでした。\n\n生成の途中の見積もりに損失をかけると、見積もりは良くなっても、20段で生成した結果は悪くなります。最後の段だけに損失をかける方法も悪化し、補正モジュールは効果がありませんでした。\n\n結局、Whisper に合わせる成分は、生成モデルの外で混ぜる方が素直に効く、というのが今の結論です。");
}

// 7. 分析
{
  const s = newSlide();
  title(s, "上乗せは両方の帯域から、他の認識器の悪化は高域から来る");
  const rows = [
    ["Whisper-small", 0.1685, 0.1633, 0.1629, 0.1658, 0.1673], ["Whisper-medium", 0.1505, 0.1436, 0.1445, 0.1453, 0.1476],
    ["Whisper（再学習）", 0.1000, 0.0920, 0.0925, 0.0943, 0.0986], ["Zipformer（対照）", 0.4328, 0.4833, 0.4907, 0.4309, 0.4787],
  ];
  s.addTable([
    [hdr(""), hdr("生成モデル\n（4回平均）", { align: "center" }), hdr("混ぜたもの", { align: "center" }), hdr("櫛状の成分を\n除いて混ぜる", { align: "center" }), hdr("4 kHz 未満\nだけ混ぜる", { align: "center" }), hdr("4 kHz 以上\nだけ混ぜる", { align: "center" })],
    ...rows.map((r, i) => [cell(r[0], i === 3 ? { color: C.accent } : {}), ...r.slice(1).map((v, j) => cell(pct(v), {
      align: "center", ...(i === 3 && (j === 1 || j === 2 || j === 4) ? { color: C.alert, bold: true } : {}), ...(i === 3 && j === 3 ? { bold: true, color: C.primary } : {}) }))]),
  ], { x: M, y: 1.6, w: W, colW: [2.9, 1.86, 1.86, 1.86, 1.86, W - 2.9 - 4 * 1.86], fontFace: FONT, rowH: [0.8, 0.52, 0.52, 0.52, 0.52], valign: "middle", margin: [3, 6, 3, 6] });
  T(s, bullets([
    ["Whisper 用音声強調が作る", { b: "櫛状の成分は原因ではない" }, "（除いても同じ）"],
    ["Whisper 系の上乗せは", { b: "低域・高域のどちらか一方だと約半分" }],
    ["Zipformer の悪化は、", { b: "元々無い 4 kHz 以上を Whisper 向けに作った成分", c: C.alert }, "が原因"],
  ], { fontSize: 18 }), { x: M, y: 4.55, w: W, h: 2.2, paraSpaceAfter: 10, lineSpacingMultiple: 1.15 });
  cite(s, "文字誤り率（%）。櫛状の成分 = Whisper 用音声強調の出力に現れる 250 Hz 間隔のピーク。喉マイクは 8 kHz 収録のため 4 kHz 以上は元々無い");
  s.addNotes("【台本】（約40秒）\nなぜ Whisper 系にだけ効くのかを、混ぜ方を変えて分析しました。\n\nWhisper 用音声強調の出力には、250 Hz おきの櫛のようなピークが出ていて、これが怪しいと思っていましたが、取り除いても結果は同じで、原因ではありませんでした。\n\nWhisper 系の上乗せは低域と高域の両方から来ていて、どちらか一方だけ混ぜると効果は約半分です。\n\n一方、Zipformer の悪化は高域から来ていて、低域だけ混ぜれば悪化しません。喉マイクに元々無い 4 kHz 以上の音を、Whisper 向けに作った成分が、ほかの認識器には害になっています。\n\n【補足】Qwen3-ASR（128本 mel）の上乗せはほぼ高域から。");
}

// 8. 手法としての評価
{
  const s = newSlide();
  title(s, "今の形は「既存の部品を混ぜただけ」で、提案手法としては弱い");
  T(s, "言えること", { x: M, y: 1.6, w: 5.7, h: 0.45, fontSize: 20, bold: true, color: C.primary });
  T(s, bullets([
    "Whisper 系6つで既存の方法より明確に良い",
    "事前学習が効果の源（事前学習なしでは悪化）",
    "Whisper 用の成分の効く帯域・害になる帯域",
  ], { fontSize: 18 }), { x: M, y: 2.15, w: 5.7, h: 3.0, paraSpaceAfter: 12, lineSpacingMultiple: 1.15 });
  line(s, M + 6.1, 1.7, 0, 2.9);
  const rx = M + 6.45, rw = W - 6.45;
  T(s, "弱い点", { x: rx, y: 1.6, w: rw, h: 0.45, fontSize: 20, bold: true, color: C.alert });
  T(s, bullets([
    "新しい部品がない",
    "重い（大きな生成モデルを 4回）",
    "データは TAPS（韓国語・1機種）だけ",
  ], { fontSize: 18 }), { x: rx, y: 2.15, w: rw, h: 3.0, paraSpaceAfter: 12, lineSpacingMultiple: 1.15 });
  T(s, "→ 音響学会は「分析」として出し、手法は別に作る", { x: M, y: 4.9, w: W, h: 0.6, fontSize: 21, bold: true, color: C.primary });
  cite(s, "事前学習なしの対照: 同じ構造・同じ学習量で Whisper-small 55.1%（元の音声強調 23.0%、事前学習あり 18.0%）");
  s.addNotes("【台本】（約35秒）\n手法として正直に評価すると、今の形は既存の生成モデルと音声強調を混ぜただけで、提案手法としては弱いです。重く、データも1つだけです。\n\n一方で、Whisper 系で既存の方法より明確に良いこと、効果の源が事前学習であること、Whisper 用の成分がどこに効いてどこで害になるかは、分析としてしっかり言えます。\n\nなので、音響学会には分析として出し、手法は別に作ることにします。原稿の下書きはできています。");
}

// 9. 今後
{
  const s = newSlide();
  title(s, "次は、提案手法になる形を3つの方向で小さく試す");
  const rows = [
    ["1. 軽くする", "混ぜた出力を教師にして、軽い音声強調1つに移す", "効果が軽いモデルに残るか"],
    ["2. 別のデータ", "フランス語の喉マイク（VibraVox）で試す", "TAPS 以外でも同じ結論か"],
    ["3. 他分野の発想", "敵対的サンプルの「転移」の技法で Whisper 用音声強調を学習し直す", "学習に使わない Whisper にも効くか"],
  ];
  s.addTable([
    [hdr("方向"), hdr("最初の小さな実験"), hdr("確かめること")],
    ...rows.map(r => [cell(r[0], { bold: true, color: C.primary, fontSize: 18 }), cell(r[1], { fontSize: 17 }), cell(r[2], { fontSize: 17 })]),
  ], { x: M, y: 1.7, w: W, colW: [2.4, 5.6, W - 8.0], fontFace: FONT, rowH: [0.55, 0.95, 0.95, 0.95], valign: "middle", margin: [4, 8, 4, 8] });
  T(s, "確認はいつも同じ：Whisper 系6つ・入力の違う認識器・話者単位の検定", { x: M, y: 5.6, w: W, h: 0.5, fontSize: 18, color: C.body });
  s.addNotes("【台本】（約35秒）\n今後は、提案手法になる形を3つの方向で小さく試します。\n\n1つ目は軽くすることで、混ぜた出力を教師にして、軽い音声強調1つに移します。2つ目はフランス語の喉マイクのデータで、同じ結論になるかを確かめます。3つ目は他分野の発想です。画像の敵対的サンプルの研究では、1つのモデルの中をランダムに揺らしながら作った摂動ほど、別のモデルにも効くことが知られています。これを Whisper 用音声強調の学習に持ち込み、Whisper-small だけで学習しても、base など学習に使わない Whisper にも効くようにします。卒論の Whisper 用音声強調は base にはほとんど効かなかったので、ここが直接の課題です。\n\n【補足】Ghost Networks（Li et al., AAAI 2020）、Input Diversity（Xie et al., CVPR 2019）。音声強調の学習に使った例は調べた範囲では見つからない。ほかの候補（生成サンプルの区間ごとの選択、時間周波数ごとの不確かさで重みづけ、ケプストラム領域での融合）は別資料。\n\n以上です。");
}

// 付録: 参考文献
{
  const s = newSlide();
  title(s, "参考文献");
  T(s, bullets([
    "TAPS: Y. Kim et al., “Throat and acoustic paired speech dataset for deep learning-based speech enhancement,” Scientific Data, 2026.",
    "Ku et al., “Generative speech foundation model pretraining for high-quality speech extraction and restoration,” arXiv:2409.16117.",
    "Sato et al., “Generic speech enhancement with self-supervised representation space loss,” arXiv:2507.07631, 2025.",
    "Dissen et al., Interspeech 2024.（凍結した Whisper の損失で前処理を学習）",
    "Ochiai et al., “Rethinking processing distortions: How do they affect the downstream ASR performance?” IEEE/ACM TASLP, 2024.",
    "J. Hauret et al., “Vibravox: A dataset of French speech captured with body-conduction audio sensors,” Speech Communication, 2025.",
  ], { fontSize: 14 }), { x: M, y: 1.6, w: W, h: 5.0, paraSpaceAfter: 10 });
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-10b.pptx" }).then(f => console.log("wrote", f));
