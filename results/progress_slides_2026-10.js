// ゼミ進捗報告（2026-10-01、5分）→ progress_slides_2026-10.pptx
// academic-pptx スキル（~/.claude/skills/academic-pptx）の規則で作成:
//   白背景・1書体・3色まで・見出しは結論の1文・1枚1図・カード/色帯/暗背景なし（まとめスライドは発表者の希望で省略）
// 数値の出典: results/GENERATIVE_RESULTS_2026-09-30.md
const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in

const FONT = "Yu Gothic";
const C = { bg: "FFFFFF", primary: "1F4E79", accent: "2E75B6", alert: "C55A11", body: "2D2D2D", muted: "777777", rule: "CCCCCC", grid: "E6E6E6", highlight: "FFF2CC" };
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

// 1. 表紙
{
  const s = newSlide();
  T(s, "喉マイクの音声を\n「どの音声認識でも読める音」に直す", { x: M, y: 2.2, w: W, h: 1.9, fontSize: 38, bold: true, color: C.primary, lineSpacingMultiple: 1.15 });
  T(s, "音声強調から生成モデルへ：ここまでの振り返りと今後", { x: M, y: 4.25, w: W, h: 0.5, fontSize: 20, color: C.body });
  T(s, "春日 裕次　ゼミ進捗報告　2026年10月1日", { x: M, y: 5.0, w: W, h: 0.4, fontSize: 16, color: C.muted });
  s.addNotes("【台本】（約10秒）\n進捗を報告します。今日は、音声強調から生成モデルにたどり着くまでの流れと、その結果、今後の方針を話します。");
}

// 2. 背景
{
  const s = newSlide();
  title(s, "喉マイクは騒音に強いが、音が欠けていて認識の誤りが多い");
  const boxes = [["喉マイクの音", "喉の振動を拾う。4 kHz 以上の音や子音の細部が入らない"], ["音声強調（前処理）", "認識の前に音を直す"], ["音声認識", "どの認識器にもつなげられる"]];
  const bw = 3.45, gap = 0.6; let x = M;
  boxes.forEach(([a, b], i) => {
    s.addShape(pptx.ShapeType.rect, { x, y: 1.75, w: bw, h: 1.7, fill: { color: C.bg }, line: { color: i === 1 ? C.primary : C.rule, width: i === 1 ? 2 : 1 } });
    T(s, a, { x: x + 0.2, y: 1.95, w: bw - 0.4, h: 0.45, fontSize: 19, bold: true, color: C.primary });
    T(s, b, { x: x + 0.2, y: 2.5, w: bw - 0.4, h: 0.85, fontSize: 16, lineSpacingMultiple: 1.2 });
    if (i < 2) s.addShape(pptx.ShapeType.rightArrow, { x: x + bw + 0.15, y: 2.45, w: 0.3, h: 0.32, fill: { color: "A6A6A6" }, line: { color: "A6A6A6" } });
    x += bw + gap;
  });
  T(s, bullets([
    ["喉マイクの音をそのまま Whisper-small にかけると、", { b: "文字の誤り率 45%" }],
    ["普通のマイク向けの雑音除去を通すと、", { b: "かえって誤りが増えた", c: C.alert }, "（卒論前半）"],
    ["目標：", { b: "どの音声認識でも誤りが減る" }, "喉マイク用の音声強調（クラウドなど作り直せない認識器にも使える）"],
  ], { fontSize: 20 }), { x: M, y: 3.85, w: W, h: 2.6, paraSpaceAfter: 14, lineSpacingMultiple: 1.15 });
  cite(s, "データ: TAPS（喉マイクと普通のマイクの同時録音、韓国語、Kim et al. 2026）");
  s.addNotes("【台本】（約40秒）\nまず背景です。喉マイクは喉の振動を拾うので周りの騒音に強い一方、4キロヘルツ以上の音や子音の細部が入らず、こもった音になります。そのまま音声認識にかけると、文字の誤り率が45%にもなります。\n\nそこで、認識の前に音を直す音声強調を研究しています。ただ、普通のマイク向けの雑音除去をそのまま使うと、かえって誤りが増えました。\n\n目標は、クラウドのような作り直せない認識器にも使える、どの音声認識でも誤りが減る喉マイク用の音声強調です。\n\n【補足】45% は Whisper-small・処理なし・句読点除去後の CER（test 0.446）。卒論前半の雑音除去は GTCRN（DNS3で学習）。");
}

// 3. Whisperに合わせた音声強調
{
  const s = newSlide();
  title(s, "Whisperに合わせた音声強調は、入力の違う認識器では逆効果だった");
  const labels = ["Whisper-small", "Qwen3-ASR", "XLS-R", "Zipformer"];
  s.addChart(pptx.charts.BAR, [
    { name: "誤りが減った", labels, values: [-15, -16, 0, 0] },
    { name: "誤りが増えた", labels, values: [0, 0, 26, 46] },
  ], {
    x: M, y: 1.6, w: 7.0, h: 4.6, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 60,
    chartColors: [C.accent, C.alert], catAxisOrientation: "maxMin", catAxisLabelPos: "low",
    catAxisLabelFontFace: FONT, catAxisLabelFontSize: 16, catAxisLabelColor: C.body,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    valAxisMinVal: -25, valAxisMaxVal: 55,
    showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: '+0"%";-0"%";;', dataLabelColor: "FFFFFF", dataLabelFontSize: 15, dataLabelFontBold: true,
    showLegend: false,
  });
  T(s, "文字誤り率の相対的な増減（元の音声強調と比べて）", { x: M, y: 6.25, w: 7.0, h: 0.35, fontSize: 14, color: C.muted, align: "center" });
  T(s, bullets([
    ["Whisperと", { b: "同じ入力" }, "（音の特徴の作り方）の認識器では誤りが減る"],
    ["入力が違う認識器（波形をそのまま入れる等）では", { b: "誤りが増える", c: C.alert }],
    [{ b: "特定の認識器に合わせると、その認識器専用になる" }],
  ], { fontSize: 20, color: C.body }), { x: M + 7.4, y: 1.8, w: W - 7.4, h: 4.4, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "卒論: TAPS 公開の音声強調（SE-Conformer）を、Whisper-small が正しく書き起こせるように追加学習。比べる相手は追加学習前");
  s.addNotes("【台本】（約45秒）\n卒論では、Whisperが正しく書き起こせるように音声強調を学習しました。Whisperや、Whisperと同じ入力を使う認識器では誤りが15%ほど減りましたが、入力の作りが違う認識器では、逆に26〜46%増えました。\n\nつまり、特定の認識器に合わせて学習すると、その認識器専用の加工になってしまい、どの認識器にも使えるという音声強調の利点が消えてしまいます。\n\n【補足】数値は CE-SE（λ=10）の TAPS SE-Conformer 比（test、句読点除去CER）。MMS はほぼ±0。");
}

// 4. 手がかり：認識器に頼らない音声強調の平均
{
  const s = newSlide();
  title(s, "認識器に頼らない音声強調を4つ平均すると、全9つで少し改善");
  const labels = ["Zipformer", "XLS-R", "MMS-1B", "Whisper-base", "Qwen3-ASR", "Whisper-small", "Whisper-medium", "Whisper-large-v3-turbo", "Whisper（喉マイクで再学習）"];
  s.addChart(pptx.charts.BAR, [{ name: "4つの平均", labels, values: [-3.0, -2.9, -4.1, -6.4, -7.4, -9.8, -9.8, -10.5, -14.2] }], {
    x: M, y: 1.55, w: 7.4, h: 4.9, barDir: "bar", barGapWidthPct: 45, chartColors: [C.accent],
    catAxisOrientation: "maxMin", catAxisLabelPos: "high",
    catAxisLabelFontFace: FONT, catAxisLabelFontSize: 14, catAxisLabelColor: C.body,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisMinVal: -16, valAxisMaxVal: 0,
    showValue: true, dataLabelPosition: "inBase", dataLabelFormatCode: '-0.0"%";-0.0"%"', dataLabelColor: "FFFFFF", dataLabelFontSize: 13, dataLabelFontBold: true,
    showLegend: false,
  });
  T(s, "文字誤り率の相対変化（元の音声強調と比べて）", { x: M, y: 6.45, w: 7.4, h: 0.35, fontSize: 14, color: C.muted, align: "center" });
  cite(s, "4つ = TAPS 公開の SE-Conformer・Demucs・TSTNN と、再構成のみで再学習した SE-Conformer");
  T(s, bullets([
    ["4つの音声強調の出力を、振幅で平均しただけ（認識器は使わない）"],
    [{ b: "入力が違う認識器も含め、全部で改善" }, " → 認識器に頼らない方向が正しい"],
    ["ただし改善は小さく、4つ動かすので重い"],
  ], { fontSize: 20 }), { x: M + 7.9, y: 1.8, w: W - 7.9, h: 4.6, paraSpaceAfter: 16, lineSpacingMultiple: 1.2 });
  s.addNotes("【台本】（約45秒）\nそこで、認識器を一切使わない方法を探しました。手がかりになったのが、認識器を使わずに作られた音声強調4つの出力を平均する方法です。\n\n（図を指して）これだけで、入力の違う認識器も含めて9つすべてで誤りが減りました。認識器に頼らない方向が正しいことが分かりました。\n\nただ、改善は3〜14%と小さく、4つのモデルを動かすので重いという問題がありました。\n\n【補足】4つ = TAPS SE-Conformer・Demucs・TSTNN・再構成のみで再学習した SE。STFT 振幅を平均し位相は TAPS。イコライザや平滑化では同じ効果が出ない（効果は各手法の癖の打ち消しと解釈）。推論は約1秒/発話（Demucs が大半）。1つのモデルへの蒸留は失敗。");
}

// 5. 着想：欠けた音は描き足す
{
  const s = newSlide();
  title(s, "問題は音の「汚れ」ではなく「欠け」なので、生成モデルに着目");
  const hdr = (t, c) => ({ text: t, options: { bold: true, color: c, fontSize: 19, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }] } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 18, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
  s.addTable([
    [hdr("", C.primary), hdr("これまでの音声強調", C.body), hdr("生成モデル", C.primary)],
    [cell("仕組み", { bold: true }), cell("入力の音を直接加工する"), cell("「自然な声」を学んでおき、描き直す")],
    [cell("欠けた音", { bold: true }), cell("平均的な音で埋める → こもる"), cell("自然な声として描き足せる", { bold: true, color: C.primary })],
    [cell("認識器", { bold: true }), cell("使わなくてよい"), cell("使わなくてよい")],
  ], { x: M, y: 1.75, w: W, colW: [1.8, 4.6, W - 6.4], fontFace: FONT, rowH: [0.6, 0.95, 0.95, 0.7], valign: "middle", margin: [4, 10, 4, 10] });
  T(s, bullets([
    ["先行例：仏語の喉マイクでは、生成モデルの調整が最良（ただし評価は認識器1つ）"],
    [{ b: "→ 9つの認識器で、どれでも効くかを確かめる" }],
  ], { fontSize: 20 }), { x: M, y: 5.25, w: W, h: 1.3, paraSpaceAfter: 10 });
  cite(s, "Hauret et al., arXiv:2508.02974（VibraVox の喉マイク: 処理なし 50.8% → 生成モデル 7.6%、音素誤り率）");
  s.addNotes("【台本】（約45秒）\nでは、もっと大きく改善するにはどうすればいいか。喉マイクの問題は、音が汚れていることではなく、高い音や子音が欠けていることです。\n\nこれまでの音声強調は入力を直接加工するので、欠けた部分は、ありえる答えの平均しか出せず、こもった音になります。一方、生成モデルは大量の音声で自然な声の形を学んでいるので、欠けた部分を自然な声として描き足せます。\n\nフランス語の喉マイクで生成モデルが最良という報告もあり、9つの認識器で確かめることにしました。\n\n【補足】Hauret ら: EBEN 18.6%、Mimi 追加学習 15.1%、NeMo flow matching 追加学習 7.6%。この案は 9/29 の方針議論（SE 分野の観点）で出た。");
}

// 6. 方法（生成モデルの調整）
{
  const s = newSlide();
  title(s, "NVIDIAの事前学習済み生成モデルを、喉マイク用に調整した");
  const steps = [
    ["① 事前学習済みモデル", "NVIDIA 公開。大量の英語音声（約6万時間）で、隠した部分を描き直す練習をしたモデル"],
    ["② 喉マイク用に調整", "本研究。TAPS の同時録音 4,000組で、喉マイクの音から普通のマイクの音を描くよう追加学習"],
    ["③ 使うとき", "ランダムな状態から、喉マイクの音を手がかりに20段階で描き直す → 各認識器へ"],
  ];
  const bw = 3.55, gap = 0.5; let x = M;
  steps.forEach(([h, b], i) => {
    s.addShape(pptx.ShapeType.rect, { x, y: 1.75, w: bw, h: 2.2, fill: { color: C.bg }, line: { color: i === 1 ? C.primary : C.rule, width: i === 1 ? 2 : 1 } });
    T(s, h, { x: x + 0.2, y: 1.95, w: bw - 0.4, h: 0.45, fontSize: 18, bold: true, color: C.primary });
    T(s, b, { x: x + 0.2, y: 2.5, w: bw - 0.4, h: 1.5, fontSize: 16, lineSpacingMultiple: 1.25 });
    if (i < steps.length - 1) s.addShape(pptx.ShapeType.rightArrow, { x: x + bw + 0.1, y: 2.75, w: 0.3, h: 0.32, fill: { color: "A6A6A6" }, line: { color: "A6A6A6" } });
    x += bw + gap;
  });
  T(s, bullets([
    [{ b: "音声認識を学習に一切使わない" }, " → 特定の認識器に合わせ込まない"],
    [{ b: "欠けた高い音や子音" }, "も、事前学習で覚えた「自然な声」から補える"],
  ], { fontSize: 20 }), { x: M, y: 4.55, w: W, h: 1.4, paraSpaceAfter: 14 });
  cite(s, "生成モデル: NVIDIA NeMo flow matching（4.3億パラメータ、Ku et al., arXiv:2409.16117）。元の音声強調は1200万パラメータ");
  s.addNotes("【台本】（約40秒）\n使ったのは、（左）NVIDIAが公開している、大量の音声で事前学習されたモデルです。音の一部を隠して描き直す練習をしていて、自然な声がどういう形かを知っています。\n\n（中央）これを、TAPSの同時録音4,000組で、喉マイクの音から普通のマイクの音を描くように追加学習しました。\n\n（右）使うときは、ランダムな状態から出発して、喉マイクの音を手がかりに20段階で少しずつ描き直します。\n\nここでも音声認識は一切使っていません。\n\n【補足】画像生成AIと同じ仕組み（flow matching）。ランダムな出発点を変えると毎回少し違う音になる。学習の損失は生成モデル本来のもの（ASR損失なし）。fp32、2万ステップ、約8.5時間。");
}

// 7. 結果（生成モデル）
{
  const s = newSlide();
  title(s, "生成モデルの調整で、9つの音声認識すべての誤りが減った");
  const rows = [["Zipformer", 54.2, 45.1], ["MMS-1B", 35.0, 29.7], ["Whisper-base", 27.3, 21.8], ["Whisper-small", 23.0, 18.0], ["XLS-R", 22.2, 16.9],
                ["Whisper-medium", 19.6, 15.9], ["Whisper-large-v3-turbo", 17.0, 13.7], ["Whisper（喉マイクで再学習）", 14.6, 11.0], ["Qwen3-ASR", 13.6, 10.4]];
  const diffInput = new Set(["Zipformer", "MMS-1B", "XLS-R"]);
  const lw = 3.0, cx = M + lw + 0.15, cw = 4.4, sc = cw / 60, y0 = 1.95, rh = 0.46;
  [0, 20, 40, 60].forEach(v => {
    s.addShape(pptx.ShapeType.line, { x: cx + v * sc, y: y0 - 0.1, w: 0, h: rows.length * rh + 0.05, line: { color: C.grid, width: 1 } });
    T(s, `${v}%`, { x: cx + v * sc - 0.35, y: y0 + rows.length * rh + 0.02, w: 0.7, h: 0.3, fontSize: 14, color: C.muted, align: "center" });
  });
  rows.forEach(([n, a, b], i) => {
    const y = y0 + i * rh;
    T(s, n, { x: M, y, w: lw, h: 0.3, fontSize: 15, align: "right", valign: "middle", color: diffInput.has(n) ? C.accent : C.body, bold: diffInput.has(n) });
    s.addShape(pptx.ShapeType.line, { x: cx + b * sc, y: y + 0.15, w: (a - b) * sc, h: 0, line: { color: C.rule, width: 2.5 } });
    s.addShape(pptx.ShapeType.ellipse, { x: cx + a * sc - 0.09, y: y + 0.06, w: 0.18, h: 0.18, fill: { color: "A6A6A6" }, line: { color: "A6A6A6" } });
    s.addShape(pptx.ShapeType.ellipse, { x: cx + b * sc - 0.1, y: y + 0.05, w: 0.2, h: 0.2, fill: { color: C.primary }, line: { color: C.primary } });
  });
  T(s, [{ text: "●", options: { color: "A6A6A6" } }, { text: " 元の音声強調　", options: {} }, { text: "●", options: { color: C.primary } }, { text: " 生成モデル　（文字の誤り率）", options: {} }],
    { x: cx - 0.3, y: 1.45, w: 5.5, h: 0.35, fontSize: 14, color: C.muted });
  s.addShape(pptx.ShapeType.roundRect, { x: cx + 2.3, y: y0 + 3 * rh, w: 1.95, h: 0.85, fill: { color: C.highlight }, line: { color: "E6C800", width: 1 }, rectRadius: 0.06 });
  T(s, "全9つで\n相対15〜24%減", { x: cx + 2.3, y: y0 + 3 * rh, w: 1.95, h: 0.85, fontSize: 16, bold: true, color: "7A5200", align: "center", valign: "middle" });
  const rx = M + 7.95;
  T(s, bullets([
    ["Whisper系でも、", { b: "Whisperに合わせた音声強調より良い" }],
    [{ b: "入力が違う認識器", c: C.accent }, "でも改善"],
    ["テスト話者", { b: "10人全員" }, "で改善"],
  ], { fontSize: 20 }), { x: rx, y: 1.9, w: W - 7.95, h: 4.6, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "生成は毎回少し変わるが、今はテストデータで1回生成した分のみ（再確認中）。生成モデル: NVIDIA NeMo（Ku et al., arXiv:2409.16117）");
  s.addNotes("【台本】（約60秒）\n結果です。（図を指して）これは9つの音声認識の文字の誤り率で、灰色が元の音声強調、紺色が生成モデルです。すべての認識器で誤りが15〜24%減り、テストの10人全員で改善しました。入力の違う認識器でも減っていて、Whisper系でも、卒論のWhisperに合わせた音声強調より良い結果です。ただし、この結果はテストデータで1回生成した分だけで、再現性は確認中です。\n\n【想定質問】\nQ. 作り話（ありもしない内容）をしていないか？\nA. Whisper-smallで確認した範囲では、崩れた発話は0、出力の長さも変わらず、置き換え・挿入の誤りはむしろ減っている。\nQ. Qwen3＋生成モデル（10.4%）は、学習し直したWhisper（SEなし13.8%）より良いのでは？\nA. Qwen3は元の音声強調でも13.6%で、差の大半は認識器の強さ。公平に言えるのは「学習し直したWhisper自身も13.8→11.0%に下がる」こと。\nQ. 元の音声強調とは？\nA. TAPSデータセットの論文で公開されている SE-Conformer（1200万パラメータ）。\nQ. ほかの方法は？\nA. 複数の認識器の損失で学習 → 学習外の Zipformer で悪化。Sato ら型（WavLM 表現の損失）→ 開発データの4認識器で −7〜−13%（同じ条件での比較はこれから）。");
}

// 8. 最終ゴールと作業仮説
{
  const s = newSlide();
  title(s, "最終ゴール：どの認識器でも効く前処理と、その設計原理");
  T(s, "最終ゴール（修士）", { x: M, y: 1.6, w: 6.0, h: 0.45, fontSize: 20, bold: true, color: C.primary });
  T(s, "どの認識器でも誤りが減る喉マイク用の前処理を作る\nなぜ効くのかを示す\n（認識器の重みは変えない）",
    { x: M, y: 2.1, w: 6.0, h: 1.2, fontSize: 18, lineSpacingMultiple: 1.25 });
  T(s, "達成の基準", { x: M, y: 3.45, w: 6.0, h: 0.4, fontSize: 18, bold: true, color: C.primary });
  T(s, bullets([
    "入力の違う認識器を含め、どれも改善・悪化なし",
    "既存の喉マイク用音声強調（TAPS）より明確に良い",
    "実際に使える計算量",
    "別のデータでも同じ結論",
  ], { fontSize: 17 }), { x: M, y: 3.9, w: 6.0, h: 2.6, paraSpaceAfter: 8 });
  s.addShape(pptx.ShapeType.line, { x: M + 6.35, y: 1.7, w: 0, h: 4.8, line: { color: C.rule, width: 1 } });
  const rx = M + 6.7, rw = W - 6.7;
  T(s, "作業仮説（これまでの結果から）", { x: rx, y: 1.6, w: rw, h: 0.45, fontSize: 20, bold: true, color: C.primary });
  s.addShape(pptx.ShapeType.rect, { x: rx, y: 2.15, w: rw, h: 1.25, fill: { color: C.highlight }, line: { color: "E6C800", width: 1 } });
  T(s, [{ text: "どの認識器にも効く ＝ 自然な声に近い音", options: { bold: true, breakLine: true } }, { text: "特定の認識器だけ ＝ その癖に合わせた音" }],
    { x: rx + 0.2, y: 2.15, w: rw - 0.4, h: 1.25, fontSize: 18, color: "5A4300", valign: "middle", lineSpacingMultiple: 1.3 });
  T(s, bullets([
    ["Whisper用の学習：人工的な成分 → ", { b: "他で悪化", c: C.alert }],
    ["4つの平均：癖が打ち消し合う → 全9つで改善"],
    ["生成モデル：自然な声に描き直す → 全9つ改善"],
  ], { fontSize: 16 }), { x: rx, y: 3.6, w: rw, h: 2.9, paraSpaceAfter: 10, lineSpacingMultiple: 1.15 });
  s.addNotes("【台本】（約45秒）\nここから今後の話です。生成モデルを提案手法にするかは、まだ決めていません。先に最終ゴールをはっきりさせました。\n\n（左）ゴールは、重みを変えられない、どの音声認識器でも誤りが減る喉マイク用の前処理を作り、なぜ効くのかを示すことです。基準はこの4つです。\n\n（右）これまでの結果は、1つの作業仮説でまとめられます。どの認識器にも効くのは自然な声に近い音で、特定の認識器にだけ効くのはその癖に合わせた音、という仮説です。これまでの結果はどれもこの仮説と合います。\n\n【補足】人工的な成分＝Whisper損失SEの出力に現れた 250 Hz 間隔の櫛状ピーク。SSL-MSE（自然な声の内部表現に近づける損失）も4認識器で改善しており仮説と整合。");
}

// 9. 可能性の地図とPDCA
{
  const s = newSlide();
  title(s, "生成モデルは候補の一つ。仮説ごとに小さく試して絞り込む");
  // 行ごとに帯（1行 = 1つの方向）。行の中は「仮説 → 実験 → 続ける条件」と左から右へ読む
  const rows = [
    ["A. 分析", "最初に回す", "自然な声に近いほど、どの認識器でも誤りが減る", "既存の全出力で、自然な声との距離と9認識器の誤りの関係を見る", "関係が明確なら柱に"],
    ["B. 事前学習", "", "効いているのは生成でなく、事前学習の知識", "事前学習なしの対照・1段で出す版", "事前学習ありが大差"],
    ["C. 生成のさせ方", "", "生成のばらつきが誤りの原因", "シード違いのばらつきと誤りの関係", "両者が連動"],
    ["D. 損失の設計", "", "自然な声との距離を損失にすれば軽くても効く", "SSL表現の損失などを9認識器で評価", "生成モデルに近づく"],
    ["E. 軽量化", "", "良い出力を教師にして、軽いモデルへ移せる", "未学習のデータで教師を作り、蒸留", "効果の大半が残る"],
  ];
  const cx = [M + 0.2, M + 2.55, M + 6.35, M + 10.05], cw = [2.2, 3.4, 3.3, W - 10.05 - 0.15], aw = 0.35;
  ["方向", "確かめる仮説", "最初の小さな実験", "続ける条件"].forEach((h, i) => T(s, h, { x: cx[i], y: 1.55, w: cw[i], h: 0.3, fontSize: 13, color: C.muted, bold: true }));
  const y0 = 1.9, rh = 0.74, gap = 0.12;
  rows.forEach(([lab, tag, hyp, exp, cond], i) => {
    const y = y0 + i * (rh + gap);
    s.addShape(pptx.ShapeType.rect, { x: M, y, w: W, h: rh, fill: { color: i === 0 ? C.highlight : "F2F4F7" }, line: { color: i === 0 ? "E6C800" : "F2F4F7", width: 1 } });
    T(s, tag ? [{ text: lab, options: { breakLine: true } }, { text: tag, options: { fontSize: 12, bold: false, color: "7A5200" } }] : lab,
      { x: cx[0], y, w: cw[0], h: rh, fontSize: 16, bold: true, color: C.primary, valign: "middle" });
    [hyp, exp, cond].forEach((t, j) => T(s, t, { x: cx[j + 1], y, w: cw[j + 1], h: rh, fontSize: 14, valign: "middle", lineSpacingMultiple: 1.1 }));
    [cx[2] - aw - 0.02, cx[3] - aw - 0.02].forEach(ax => T(s, "→", { x: ax, y, w: aw, h: rh, fontSize: 20, bold: true, color: "8C8C8C", align: "center", valign: "middle" }));
  });
  T(s, bullets([
    ["1週間ごとに「仮説 → 小さな実験 → 9つの認識器で確認 → 続ける／捨てる」を回す"],
  ], { fontSize: 17 }), { x: M, y: 6.33, w: W, h: 0.35 });
  cite(s, "確認はいつも同じ条件: 入力の違う認識器を含む9つ・話者単位の検定・悪化する認識器がないか");
  s.addNotes("【台本】（約50秒）\nそのうえで、生成モデルは候補の一つと位置づけて、仮説ごとに小さく試して絞り込んでいきます。\n\n最初に回すのはAの分析です。すでにある全ての出力で、自然な声との距離と、9つの認識器の誤りの関係を見ます。新しい学習がいらないので数日で結論が出ます。関係がはっきりすれば、これが研究の柱、設計原理になります。\n\nBからEも、並行して小さく試します。\n\n1週間ごとに、仮説、小さな実験、9つの認識器での確認、続けるか捨てるか、を回していきます。以上です。\n\n【補足】自然な声との距離の候補: 自己教師あり音声モデル（WavLM など）の内部表現での、同じ発話の気導音との距離、または気導音全体の分布との距離（FAD 型）。「続ける条件」は目安で、ゴールの基準に照らして判断する。");
}

// 10. 参考文献（付録）
{
  const s = newSlide();
  title(s, "参考文献");
  T(s, bullets([
    "TAPS: Y. Kim et al., “Throat and acoustic paired speech dataset for deep learning-based speech enhancement,” Scientific Data, 2026.",
    "Ku et al., “Generative speech foundation model pretraining for high-quality speech extraction and restoration,” arXiv:2409.16117.",
    "山中 涼雅 ほか, “Zero-shot音声変換を用いた咽喉マイク音声の気導音復元に関する初期検討,” 日本音響学会 2026年秋季, 1-Q-42.",
    "和田 航次郎 ほか, “咽喉マイク自由発話音声における感情認識の検討,” 日本音響学会 2026年秋季, 1-Q-56.",
    "Sato et al., “Generic speech enhancement with self-supervised representation space loss,” arXiv:2507.07631, 2025.",
    "J. Hauret et al., arXiv:2508.02974, 2025.（喉マイクのリアルタイム変換、生成モデルとの比較を含む）",
    "J. Hauret et al., “Vibravox: A dataset of French speech captured with body-conduction audio sensors,” Speech Communication, 2025.",
  ], { fontSize: 14 }), { x: M, y: 1.6, w: W, h: 5.0, paraSpaceAfter: 10 });
}

// 付録A. 試したアプローチの一覧
{
  const s = newSlide();
  title(s, "付録：試した6つの方法と結果");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 16, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }] } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 16, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
  const ng = (t) => cell(t, { color: C.alert });
  s.addTable([
    [hdr("方法"), hdr("考え方"), hdr("どの認識器でも効いたか")],
    [cell("普通のマイク向けの音声強調"), cell("既存の雑音除去をそのまま使う"), ng("✕ 誤りが増えた（卒論前半）")],
    [cell("Whisperに合わせて学習"), cell("Whisperが読みやすい音にする"), ng("△ Whisper系だけ改善、他は悪化")],
    [cell("複数の認識器に合わせて学習"), cell("入力の違う認識器にも合わせる"), ng("✕ 学習に使っていない認識器で悪化")],
    [cell("4つの音声強調の出力を平均"), cell("各手法の癖を打ち消す（重い）"), cell("○ 全9つで改善（−3〜−14%）")],
    [cell("特徴を普通のマイクに近づける"), cell("認識器に頼らない（Sato ら型）"), cell("○ 調べた4つで改善（−7〜−13%）")],
    [cell("生成モデルを調整", { bold: true, color: C.primary }), cell("音そのものを作り直す", { bold: true, color: C.primary }), cell("◎ 全9つで −15〜−24%", { bold: true, color: C.primary })],
  ], { x: M, y: 1.65, w: W, colW: [3.5, 4.1, W - 7.6], fontFace: FONT, rowH: 0.66, valign: "middle", margin: [4, 8, 4, 8] });
  cite(s, "％は元の音声強調（TAPS SE-Conformer）と比べた文字誤り率の相対変化。Sato ら型は開発データ、その他はテストデータの結果で、条件はまだそろっていない");
  s.addNotes("【台本】（約40秒）\nここまでに試した方法をまとめます。普通のマイク向けの音声強調をそのまま使うと誤りが増え、Whisperに合わせて学習するとWhisper系だけ、複数の認識器に合わせても、学習に使っていない認識器ではやはり悪化しました。\n\n認識器に頼らない方法として、4つの音声強調の出力を平均する方法と、先行研究の方法は、どの認識器でも改善しましたが、改善は小さめでした。\n\nその中で、生成モデルを調整した方法だけが、全部の認識器で大きく改善しました。次で詳しく見ます。\n\n【補足】4つの平均：TAPS SE-Conformer・Demucs・TSTNN・再構成のみで再学習したSEの振幅を平均（推論は約1秒/発話と重い）。複数の認識器に合わせる：Whisper の損失＋XLS-R の損失で学習し、Zipformer で +9〜15%。Sato ら型：WavLM の内部表現を普通のマイクの音に近づける損失。");
}

// 付録B. 関連研究
{
  const s = newSlide();
  title(s, "付録：関連研究");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 16, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }] } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 15, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
  s.addTable([
    [hdr("研究"), hdr("対象"), hdr("目的"), hdr("音声認識の誤り")],
    [cell("山中ら（東大 2026）"), cell("喉マイク（韓国語・日本語）"), cell("普通のマイクらしい音に変換"), cell("どの方法でも増加", { color: C.alert, bold: true })],
    [cell("Sato ら（2025）"), cell("普通のマイクの雑音"), cell("認識器に頼らない音声強調"), cell("複数の認識器で減少")],
    [cell("本研究", { bold: true, color: C.primary }), cell("喉マイク（韓国語）"), cell("どの認識器でも誤りを減らす"), cell("9つすべてで減少\n（入力の違う認識器も）", { bold: true, color: C.primary })],
  ], { x: M, y: 1.75, w: W, colW: [2.6, 3.3, 3.4, W - 9.3], fontFace: FONT, rowH: 0.75, valign: "middle", margin: [4, 8, 4, 8] });
  T(s, bullets([
    ["山中らの方が進んでいる点：", { b: "別の喉マイク・日本語でも検証" }, "、同時録音のデータが不要"],
  ], { fontSize: 20 }), { x: M, y: 5.2, w: W, h: 0.9 });
  cite(s, "山中ら: 日本音響学会 2026年秋 1-Q-42（誤り率の基準が本研究と異なる）／Sato et al., arXiv:2507.07631");
  s.addNotes("【台本】（約60秒）\n関連研究です。一番近いのは、今回の音響学会で東大の山中さんたちが発表した研究で、同じTAPSのデータを使っています。声質変換などで喉マイクの音を普通のマイクの音に近づけていて、聞こえは良くなりますが、音声認識の誤りはどの方法でも増えていました。一方、向こうは別の喉マイクや日本語でも確かめていて、同時録音のデータも要りません。ここは私に足りない点です。\n\n喉マイクで、入力の違う複数の音声認識で誤りが減るかを確かめた例は、調べた範囲では見当たりません。\n\n【補足】山中ら：無処理0.233、Sidon 0.308、Kanade(ft) 0.550。和田ら（神戸大・三菱電機）：喉マイクの感情認識でも汎用の雑音除去で悪化。");
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-10.pptx" }).then(f => console.log("wrote", f));
