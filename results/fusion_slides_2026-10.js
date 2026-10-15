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


// 融合手法の結果（1〜3枚の短い報告）。数値の出典: results/GENERATIVE_RESULTS_2026-09-30.md §7.2・§7.7

// 1. 方法
{
  const s = newSlide();
  title(s, "生成モデルの出力に、Whisper 用の音声強調を半分混ぜる");
  const y1 = 1.85, y2 = 3.75, bh = 1.25;
  T(s, "喉マイクの音", { x: M, y: 2.65, w: 1.9, h: 0.9, fontSize: 18, bold: true, valign: "middle" });
  line(s, M + 1.95, 2.45, 0, 1.75, C.grey, 1.5);
  [[y1, "生成モデル（喉マイク用に調整）", "欠けた音を描き足す。4回生成して振幅を平均"],
   [y2, "Whisper 用の音声強調", "Whisper-small が正しく書けるよう学習"]].forEach(([y, a, b]) => {
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
    ["混ぜる割合 0.5 は、開発データで Whisper-small だけを見て決めた"],
    ["新たな学習は不要（部品はどちらも学習済み）"],
  ], { fontSize: 18 }), { x: M, y: 5.35, w: W, h: 1.3, paraSpaceAfter: 10 });
  cite(s, "生成モデル: NVIDIA の flow matching（4.3億パラメータ、Ku et al., arXiv:2409.16117）を TAPS で追加学習");
  s.addNotes("【台本】（約30秒）\n喉マイクの音を2つの処理に通します。上は生成モデルで、欠けた高い音を自然な声として描き足します。4回生成して平均します。下は、Whisper が正しく書き起こせるように学習した音声強調です。この2つの出力の振幅を半々で混ぜて、Whisper 系の認識器に入れます。混ぜるだけなので、新しい学習はいりません。");
}

// 2. 結果
{
  const s = newSlide();
  title(s, "混ぜると、Whisper 系6つすべてで誤りがさらに減った");
  const R = [["Whisper-small", 0.2297, 0.1685, 0.1633], ["Whisper-base", 0.2726, 0.2048, 0.2011], ["Whisper-medium", 0.1962, 0.1505, 0.1436],
             ["large-v3-turbo", 0.1697, 0.1312, 0.1248], ["Whisper（喉マイクで再学習）", 0.1462, 0.1000, 0.0920], ["Qwen3-ASR", 0.1359, 0.0968, 0.0936]];
  const labels = R.map(r => r[0]);
  s.addChart(pptx.charts.BAR, [
    { name: "元の音声強調（TAPS）", labels, values: R.map(r => 100 * r[1]) },
    { name: "生成モデルのみ", labels, values: R.map(r => 100 * r[2]) },
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
    ["生成モデルのみより", { b: "2〜8% 減" }],
    ["学習に使っていない Whisper にも効いた"],
  ], { fontSize: 18 }), { x: rx, y: 1.8, w: rw, h: 4.8, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "文字誤り率（%）、TAPS テスト 1000発話。生成モデルのみとの差は 4つで有意（medium・turbo・再学習 Whisper・small）");
  s.addNotes("【台本】（約40秒）\n結果です。灰色が元の音声強調、青が生成モデルのみ、紺が混ぜたものです。混ぜたものは Whisper 系の6つすべてで一番誤りが少なく、元の音声強調より 26〜37% 減りました。生成モデルのみと比べても、さらに 2〜8% 減っています。学習に使ったのは Whisper-small だけですが、学習に使っていない medium や large-v3-turbo にも効いています。");
}

// 3. 副作用と対策
{
  const s = newSlide();
  title(s, "他の認識器への悪影響は、混ぜ方を工夫すればほぼ消せる");
  const rows = [
    ["Whisper-small", 0.1685, 0.1633, 0.1624], ["Whisper-base", 0.2048, 0.2011, 0.1987], ["Whisper（再学習）", 0.1000, 0.0920, 0.0920],
    ["Qwen3-ASR", 0.0968, 0.0936, 0.0927], ["XLS-R（対照）", 0.1613, 0.1632, 0.1610], ["Zipformer（対照）", 0.4328, 0.4833, 0.4419],
  ];
  const ctl = new Set([4, 5]);
  s.addTable([
    [hdr(""), hdr("生成モデルのみ", { align: "center" }), hdr("混ぜたもの", { align: "center" }), hdr("混ぜ方を工夫", { align: "center" })],
    ...rows.map((r, i) => [cell(r[0], ctl.has(i) ? { color: C.accent } : {}), ...r.slice(1).map((v, j) => cell(pct(v), {
      align: "center", ...(i === 5 && j === 1 ? { color: C.alert, bold: true } : {}), ...(i === 5 && j === 2 ? { color: C.primary, bold: true } : {}) }))]),
  ], { x: M, y: 1.6, w: 8.2, colW: [2.9, 1.77, 1.77, 1.76], fontFace: FONT, rowH: 0.52, valign: "middle", margin: [3, 6, 3, 6] });
  const rx = M + 8.6, rw = W - 8.6;
  T(s, bullets([
    ["混ぜると、Whisper 以外の Zipformer は", { b: "12% 悪化", c: C.alert }],
    ["工夫：音声強調の重みを", { b: "3割元に戻し" }, "、", { b: "対数で" }, "混ぜる"],
    ["Whisper 系はそのままで、", { b: "悪化は 2% に縮小" }],
  ], { fontSize: 17 }), { x: rx, y: 1.7, w: rw, h: 4.8, paraSpaceAfter: 16, lineSpacingMultiple: 1.2 });
  cite(s, "文字誤り率（%）。工夫 = Whisper 用音声強調と元の音声強調の重みを 7:3 で内挿し、生成モデルの出力と対数振幅で平均");
  s.addNotes("【台本】（約30秒）\nただし副作用もありました。Whisper と作りの違う Zipformer では、混ぜると逆に 12% 悪くなりました。そこで、Whisper 用の音声強調の重みを3割だけ元に戻し、混ぜ方も対数での平均に変えると、Whisper 系の効果はそのままで、Zipformer の悪化は 2% まで小さくなりました。");
}

// 4. 発見と価値
{
  const s = newSlide();
  title(s, "欠けた音の補完と Whisper への合わせ込みは、分けて設計できる");
  const lw = 6.6;
  T(s, "分かったこと", { x: M, y: 1.55, w: lw, h: 0.4, fontSize: 19, bold: true, color: C.primary });
  const finds = [
    ["Whisper 用に学習すると、他の認識器では悪化", "作りの違う XLS-R・Zipformer では誤りが 26〜46% 増える"],
    ["生成モデルは、どの認識器にも効く", "9つすべてで誤りが 15〜24% 減る。効果の源は事前学習"],
    ["Whisper に効くのは「声の大まかな形」", "母音などを決める周波数の山の形（スペクトル包絡）。\n細かい倍音だけ混ぜても効かない"],
    ["他の認識器への害は「元々無い高い音」から", "喉マイクに無い 4 kHz 以上を Whisper 向けに作った成分。\n4 kHz 未満だけ混ぜれば害は出ない"],
  ];
  let y = 2.05;
  finds.forEach(([h, d], k) => {
    const n = d.split("\n").length;
    T(s, [{ text: `${k + 1}. `, options: { color: C.primary } }, { text: h, options: {} }], { x: M, y, w: lw, h: 0.4, fontSize: 17, bold: true });
    T(s, d, { x: M + 0.3, y: y + 0.42, w: lw - 0.3, h: 0.3 * n, fontSize: 14, color: C.muted, lineSpacingMultiple: 1.15 });
    y += 0.42 + 0.3 * n + 0.32;
  });
  line(s, M + 6.95, 1.65, 0, 4.7);
  const rx = M + 7.3, rw = W - 7.3;
  T(s, "この研究の価値", { x: rx, y: 1.55, w: rw, h: 0.4, fontSize: 19, bold: true, color: C.primary });
  const vals = [
    ["前処理だけで誤り 26〜37% 減", "クラウドの認識器など、中身を変えられない\nWhisper 系にも使える"],
    ["設計の指針になる", "欠けた音は生成モデルで補い、\nWhisper 向けの調整は外側で控えめに足す"],
  ];
  vals.forEach(([h, d], k) => {
    const y = 2.05 + k * 1.4;
    T(s, h, { x: rx, y, w: rw, h: 0.4, fontSize: 17, bold: true });
    T(s, d, { x: rx, y: y + 0.42, w: rw, h: 0.8, fontSize: 14, color: C.muted, lineSpacingMultiple: 1.15 });
  });
  cite(s, "限界：TAPS で学習した部品は、別機種の喉マイク（VibraVox、仏語）にはそのままでは効かない（機種ごとの適応が必要）");
  s.addNotes("【台本】（約50秒）\nこの研究で分かったことは4つです。\n1つ目、Whisper の損失で学習した音声強調は、作りの違う認識器では逆に誤りが 26〜46% 増えます。\n2つ目、事前学習済みの生成モデルは、9つの認識器すべてで誤りを 15〜24% 減らします。効いているのは事前学習の知識です。\n3つ目、混ぜたときに Whisper に効いているのは、声の大まかな形、つまり母音などを決める周波数の山の形です。細かい倍音の部分だけ混ぜても効きませんでした。\n4つ目、他の認識器への悪影響は、喉マイクに元々無い 4 kHz 以上の音を、Whisper 向けに作ってしまった成分から来ています。4 kHz 未満だけを混ぜれば悪影響は出ません。\n\n価値としては、クラウドのように中身を変えられない Whisper 系の認識器でも、前処理だけで誤りを 26〜37% 減らせること、そして、欠けた音は生成で補い、Whisper 向けの調整は外側で控えめに足す、という設計の指針になることです。ただし、別の機種の喉マイクにはそのままでは効かないという限界もあります。");
}

pptx.writeFile({ fileName: __dirname + "/fusion_slides_2026-10.pptx" }).then(f => console.log("wrote", f));
