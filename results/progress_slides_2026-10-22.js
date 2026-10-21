// 進捗発表（2026-10-22、約6分）→ progress_slides_2026-10-22.pptx
// DESIGN.md と academic-pptx スキルの規則で作成（白背景・1書体・3色・見出しは結論の1文・1枚1図・カード/色帯なし）
// 数値の出典: results/paper_asj/main.tex（表1・表5）、results/GENERATIVE_RESULTS_2026-09-30.md §7.12・§7.17
const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in

const FONT = "Yu Gothic";
const C = { bg: "FFFFFF", primary: "1F4E79", accent: "2E75B6", alert: "C55A11", body: "2D2D2D", muted: "777777", rule: "CCCCCC", grey: "A6A6A6" };
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
const hdr = (t, o = {}) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 16, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }], ...o } });
const cell = (t, o = {}) => ({ text: t, options: { fontSize: 16, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
const f3 = v => v.toFixed(3);

// 1. 問題と方法
{
  const s = newSlide();
  title(s, "欠けた高域を生成モデルで補い、Whisper 用の強調を混ぜる");
  s.addImage({ path: __dirname + "/figures/method_overview.png", x: M, y: 1.5, w: 7.6, h: 7.6 * 0.606 });
  const rx = M + 8.0, rw = W - 8.0;
  T(s, bullets([
    ["喉マイクは騒音に強いが、", { b: "4 kHz 以上が無い" }],
    ["生成モデル（NVIDIA、4.3億パラメータ）を喉マイク→気導マイクの対で適応"],
    ["Whisper 用の音声強調と", { b: "振幅を半々で混ぜる" }, "（融合A、学習不要）"],
    ["9種の認識器で評価（損失・選択は Whisper-small のみ）"],
  ], { fontSize: 18 }), { x: rx, y: 1.6, w: rw, h: 5.0, paraSpaceAfter: 16, lineSpacingMultiple: 1.15 });
  cite(s, "データ: TAPS（韓国語、喉マイクと気導マイクの同時収録。test 10話者・1,000発話、学習と話者は重ならない）");
  s.addNotes("【台本】（約50秒）\n喉マイクは首の振動を拾うので騒音に強いのですが、TAPS では 8 kHz で収録されていて、4 kHz 以上の成分が最初からありません。そのまま Whisper に入れると誤り率は 0.45 です。\nそこで、NVIDIA が公開している事前学習済みの生成モデルを、喉マイクから気導マイクへの対で追加学習し、欠けた成分を補います。さらに、Whisper の損失で学習した音声強調の出力と、振幅を半分ずつ混ぜます。これを融合Aと呼んでいます。混ぜるだけなので新しい学習はいりません。\n評価は9種の認識器で行い、損失と設定の選択には Whisper-small だけを使っています。");
}

// 2. 主結果
{
  const s = newSlide();
  title(s, "生成モデルは9種すべてで誤りを減らし、喉マイク用に学習した Whisper にも効いた");
  const rows = [
    ["Whisper-small*", 0.447, 0.230, 0.169, 0.163], ["Whisper-medium", 0.337, 0.196, 0.151, 0.144], ["Whisper-large-v3-turbo", 0.257, 0.170, 0.131, 0.125],
    ["FT Whisper（喉マイク）", 0.138, 0.146, 0.100, 0.092], ["Qwen3-ASR", 0.243, 0.136, 0.097, 0.094],
    ["XLSR（系統が違う）", 0.584, 0.222, 0.161, 0.163], ["Zipformer（系統が違う）", 0.997, 0.542, 0.433, 0.483],
  ];
  s.addTable([
    [hdr(""), hdr("処理なし", { align: "center" }), hdr("従来の強調", { align: "center" }), hdr("生成モデル", { align: "center" }), hdr("融合A", { align: "center" })],
    ...rows.map((r, i) => [cell(r[0], i >= 5 ? { color: C.accent } : i === 3 ? { bold: true } : {}),
      ...r.slice(1).map((v, j) => cell(f3(v), { align: "center",
        ...(i === 3 && (j === 0 || j === 3) ? { bold: true, color: C.primary } : {}), ...(i === 6 && j === 3 ? { color: C.alert, bold: true } : {}) }))]),
  ], { x: M, y: 1.55, w: 8.4, colW: [3.2, 1.3, 1.3, 1.3, 1.3], fontFace: FONT, rowH: 0.5, valign: "middle", margin: [3, 6, 3, 6] });
  const rx = M + 8.8, rw = W - 8.8;
  T(s, bullets([
    ["生成モデルは従来の強調より", { b: "19〜32% 減" }, "（9種すべて）"],
    ["喉マイクで学習した Whisper も", { b: "0.138 → 0.092" }],
    ["融合A は Whisper 系にさらに 2〜8%"],
    ["Zipformer は融合で", { b: "悪化", c: C.alert }],
  ], { fontSize: 17 }), { x: rx, y: 1.65, w: rw, h: 5.0, paraSpaceAfter: 16, lineSpacingMultiple: 1.15 });
  cite(s, "文字誤り率（句読点除去、TAPS test）。従来の強調 = TAPS 公開の SE-Conformer。*: 損失・選択に使用");
  s.addNotes("【台本】（約60秒）\n主な結果です。左から、処理なし、TAPS が公開している従来の音声強調、生成モデル、融合Aです。\n生成モデルは、系統の違う認識器も含めた9種すべてで、従来の強調より誤りを 19〜32% 減らしました。\n特に見てほしいのは太字の行で、喉マイクの音で追加学習した Whisper です。以前の実験では、この Whisper に音声強調を足すと逆に悪くなっていましたが、生成モデルなら 0.138 から 0.092 まで下がりました。認識器を学習し直した後でも、前処理で上乗せできます。\n融合A は Whisper 系にさらに 2〜8% 効きますが、作りの違う Zipformer では悪化します。融合の上乗せは、学習を3回やり直しても再現しました。");
}

// 3. 別データ
{
  const s = newSlide();
  title(s, "別の喉マイク（VibraVox）でも、適応し直せば改善が再現した");
  const rows = [["Whisper-small", 0.422, 0.711, 0.087, 0.079], ["Whisper-large-v3-turbo", 0.270, 0.654, 0.053, 0.045], ["Qwen3-ASR", 0.242, 0.594, 0.047, 0.041], ["XLSR（仏語）", 0.275, 0.501, 0.063, 0.058]];
  s.addTable([
    [hdr(""), hdr("処理なし", { align: "center" }), hdr("TAPS の\n部品のまま", { align: "center" }), hdr("適応後\n生成モデル", { align: "center" }), hdr("適応後\n融合A", { align: "center" })],
    ...rows.map(r => [cell(r[0]), ...r.slice(1).map((v, j) => cell(f3(v), { align: "center", ...(j === 1 ? { color: C.alert } : {}), ...(j === 3 ? { bold: true, color: C.primary } : {}) }))]),
  ], { x: M, y: 1.55, w: 8.6, colW: [2.6, 1.3, 1.6, 1.55, 1.55], fontFace: FONT, rowH: 0.55, valign: "middle", margin: [3, 6, 3, 6] });
  const rx = M + 9.0, rw = W - 9.0;
  T(s, bullets([
    ["そのまま使うと", { b: "どれも悪化", c: C.alert }],
    ["適応し直すと", { b: "70〜82% 減" }],
    ["1,000対（約1.3時間）で全データの約86%"],
  ], { fontSize: 17 }), { x: rx, y: 1.65, w: rw, h: 3.6, paraSpaceAfter: 16, lineSpacingMultiple: 1.15 });
  cite(s, "VibraVox speech_clean test（21話者・3,064発話、学習と話者は重ならない）。少量適応は1試行・10k step");
  s.addNotes("【台本】（約45秒）\n評価が TAPS の中で閉じているので、別のデータセットでも確かめました。フランス語の VibraVox です。\nTAPS で作った部品をそのまま使うと、どの処理も処理なしより悪くなりました。音量や帯域をそろえても変わらなかったので、機器の違いが効いていると考えています。\n一方、VibraVox の対データで同じ手順の適応をやり直すと、生成モデルで誤りが 70〜82% 減り、融合A でさらに減りました。手法の作り方はデータセットを越えて成り立ちますが、部品は機器ごとに適応が必要です。ただ、1,000 対、1.3 時間ほどのデータで、全データの改善量の 86% が得られています。");
}

// 4. 新しい分析
{
  const s = newSlide();
  title(s, "融合の効果は平均的な音色ではなく、時間変化する包絡から来る");
  const labels = ["Whisper-small", "Whisper-medium", "large-v3-turbo", "FT Whisper"];
  const g = [0.1685, 0.1505, 0.1312, 0.1000];
  const rel = a => a.map((v, i) => +(100 * (v / g[i] - 1)).toFixed(1));
  s.addChart(pptx.charts.BAR, [
    { name: "発話ごとのイコライザ（上限）", labels, values: rel([0.1701, 0.1508, 0.1315, 0.1001]) },
    { name: "包絡だけ移す", labels, values: rel([0.1655, 0.1445, 0.1248, 0.0937]) },
    { name: "融合A", labels, values: rel([0.1633, 0.1436, 0.1248, 0.0920]) },
  ], {
    x: M, y: 1.5, w: 8.0, h: 5.1, barDir: "col", barGapWidthPct: 60, chartColors: [C.grey, C.accent, C.primary],
    catAxisLabelFontFace: FONT, catAxisLabelFontSize: 14, catAxisLabelColor: C.body, catAxisLabelPos: "low",
    valAxisMinVal: -9, valAxisMaxVal: 2, valAxisMajorUnit: 3, valAxisLabelFontSize: 12, valAxisLabelColor: C.muted, valAxisLabelFormatCode: "0",
    valGridLine: { color: "E6E6E6", size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 11, dataLabelColor: C.body,
    showLegend: true, legendPos: "t", legendFontFace: FONT, legendFontSize: 13, legendColor: C.body,
    showValAxisTitle: true, valAxisTitle: "生成モデル比の誤り率の変化 [%]", valAxisTitleFontSize: 12, valAxisTitleColor: C.muted,
  });
  const rx = M + 8.4, rw = W - 8.4;
  T(s, bullets([
    ["固定のイコライザは", { b: "どれも効かない" }, "（±1.5% 以内）"],
    ["時間変化する包絡を移すと", { b: "効果の 58〜100%" }, "が戻る"],
    ["傾き・高域の量・フォルマントの変化は、誤りの減り方と", { b: "相関しない" }],
  ], { fontSize: 17 }), { x: rx, y: 1.65, w: rw, h: 5.0, paraSpaceAfter: 16, lineSpacingMultiple: 1.15 });
  cite(s, "発話ごとのイコライザ = その発話の融合A の平均スペクトルに合わせる（現実には使えない上限）。dev の固定・区間別のイコライザも同様に効果なし");
  s.addNotes("【台本】（約60秒）\n今週の新しい分析です。融合A がなぜ Whisper に効くのかを、音声処理の観点から調べました。\nまず、融合A が平均的な音色、つまりスペクトルの傾きや高域の量を変えているだけなのかを確かめました。灰色は、生成モデルの出力にイコライザをかけて、その発話の融合A と同じ平均スペクトルにしたものです。融合A の出力を見て作るので本来は使えない上限ですが、それでも誤りはまったく減りませんでした。\n一方、青の、時間とともに変わるスペクトル包絡だけを移したものは、融合A の効果の 6割から10割を取り戻しました。\nまた、フォルマントや傾きなど発話ごとの音響特徴の変化は、誤りの減り方と相関しませんでした。効いているのは平均的な音色ではなく、フレームごとに動く包絡の変化だと言えます。\nちなみに Zipformer では逆に、固定のイコライザだけなら良くなり、時間変化する部分が悪化の原因でした。");
}

// 5. まとめ
{
  const s = newSlide();
  title(s, "まとめ: 欠けた成分の補完が土台、Whisper 向けの調整は上乗せ");
  const lw = 6.6;
  T(s, "分かったこと", { x: M, y: 1.55, w: lw, h: 0.4, fontSize: 19, bold: true, color: C.primary });
  T(s, bullets([
    ["生成モデルの適応は9種すべてに効き、喉マイク用に学習した Whisper にも上乗せできる"],
    ["Whisper 用の音声強調を混ぜると Whisper 系にさらに 2〜8%。効くのは時間変化する包絡"],
    ["別の喉マイクでは部品の再適応が必要だが、作り方は再現する"],
  ], { fontSize: 17 }), { x: M, y: 2.05, w: lw, h: 4.4, paraSpaceAfter: 16, lineSpacingMultiple: 1.15 });
  s.addShape(pptx.ShapeType.line, { x: M + 6.95, y: 1.65, w: 0, h: 4.7, line: { color: C.rule, width: 1 } });
  const rx = M + 7.3, rw = W - 7.3;
  T(s, "次の予定", { x: rx, y: 1.55, w: rw, h: 0.4, fontSize: 19, bold: true, color: C.primary });
  T(s, bullets([
    ["日本音響学会 2027年春季に4ページで申し込む（申込 11/11〜11/24）"],
    ["融合を一部のフレームだけにかけ、どこで効くかを確かめる（評価中）"],
    ["計算量の削減（生成のステップ数・回数を減らす）"],
  ], { fontSize: 17 }), { x: rx, y: 2.05, w: rw, h: 4.4, paraSpaceAfter: 16, lineSpacingMultiple: 1.15 });
  cite(s, "限界: 計算量が大きい（4.3億パラメータを20段×4回）。蒸留して1つの強調器にする試みは融合A に届かなかった");
  s.addNotes("【台本】（約40秒）\nまとめです。事前学習済みの生成モデルで欠けた成分を補うことが土台で、9種すべての認識器に効き、喉マイクで学習した Whisper にも上乗せできました。Whisper 用の音声強調を混ぜると Whisper 系にさらに数%効き、その効果は時間とともに変わる包絡から来ています。別の喉マイクでは部品の適応し直しが必要ですが、作り方は再現しました。\n次は、日本音響学会の春季研究発表会に4ページで申し込みます。分析では、融合をどのフレームにかけると効くのかを確かめている途中です。計算量が大きいのが課題で、生成の回数を減らす方向を試します。");
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-10-22.pptx" }).then(f => console.log("wrote", f));
