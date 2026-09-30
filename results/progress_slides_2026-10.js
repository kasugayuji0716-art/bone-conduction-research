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
  T(s, "生成モデルの予備実験・関連研究・課題・今後の方針", { x: M, y: 4.25, w: W, h: 0.5, fontSize: 20, color: C.body });
  T(s, "春日 裕次　ゼミ進捗報告　2026年10月1日", { x: M, y: 5.0, w: W, h: 0.4, fontSize: 16, color: C.muted });
  s.addNotes("【台本】（約10秒）\n進捗を報告します。今日は、予備実験の結果と課題、これからの方針を中心に話します。");
}

// 2. 前提（Whisperに合わせた音声強調の限界）
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
    ["目標：", { b: "どの音声認識でも誤りが減る前処理" }, "（クラウドなど作り直せない認識器にも使える）"],
  ], { fontSize: 20, color: C.body }), { x: M + 7.4, y: 1.8, w: W - 7.4, h: 4.4, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "データ: TAPS（喉マイクと普通のマイクの同時録音、韓国語）。元の音声強調 = TAPS 公開の SE-Conformer");
  s.addNotes("【台本】（約50秒）\n喉マイクは騒音に強い一方、こもった音になるので、音声認識の誤りが多くなります。そこで、認識の前に音を直す「音声強調」を研究しています。\n\n卒論では、Whisperが正しく書き起こせるように音声強調を学習しました。Whisperや、Whisperと同じ入力を使う認識器では誤りが15%ほど減りましたが、入力の作りが違う認識器では、逆に26〜46%増えました。\n\nそこで目標を、クラウドのような作り直せない認識器にも使える、どの音声認識でも誤りが減る前処理にしました。\n\n【補足】数値は CE-SE（λ=10）の TAPS SE-Conformer 比（test、句読点除去CER）。MMS はほぼ±0。");
}

// 2b. 試したアプローチの一覧
{
  const s = newSlide();
  title(s, "6つの方法のうち、どの認識器でも大きく効いたのは生成モデルだけ");
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

// 3. 結果（生成モデル）
{
  const s = newSlide();
  title(s, "生成モデルの調整で、9つの音声認識すべての誤りが減った");
  const rows = [["Zipformer", 54.2, 45.1], ["MMS-1B", 35.0, 29.7], ["Whisper-base", 27.3, 21.8], ["Whisper-small", 23.0, 18.0], ["XLS-R", 22.2, 16.9],
                ["Whisper-medium", 19.6, 15.9], ["Whisper-large-v3-turbo", 17.0, 13.7], ["Whisper（喉マイクで再学習）", 14.6, 11.0], ["Qwen3-ASR", 13.6, 10.4]];
  T(s, "狙い：特定の認識器に合わせず、音そのものを直す（音声認識は学習に使わない）", { x: M, y: 1.3, w: W, h: 0.4, fontSize: 18, color: C.muted });
  const diffInput = new Set(["Zipformer", "MMS-1B", "XLS-R"]);
  const lw = 3.0, cx = M + lw + 0.15, cw = 4.4, sc = cw / 60, y0 = 2.3, rh = 0.46;
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
    { x: cx - 0.3, y: 1.85, w: 5.5, h: 0.35, fontSize: 14, color: C.muted });
  s.addShape(pptx.ShapeType.roundRect, { x: cx + 2.3, y: y0 + 3 * rh, w: 1.95, h: 0.85, fill: { color: C.highlight }, line: { color: "E6C800", width: 1 }, rectRadius: 0.06 });
  T(s, "全9つで\n相対15〜24%減", { x: cx + 2.3, y: y0 + 3 * rh, w: 1.95, h: 0.85, fontSize: 16, bold: true, color: "7A5200", align: "center", valign: "middle" });
  const rx = M + 7.95;
  T(s, bullets([
    [{ b: "音声を作るAI" }, "（大量の音声で事前学習済み）を、喉マイク用に同時録音で調整"],
    [{ b: "入力が違う認識器", c: C.accent }, "でも改善"],
    ["テスト話者", { b: "10人全員" }, "で改善"],
  ], { fontSize: 20 }), { x: rx, y: 2.2, w: W - 7.95, h: 4.6, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "生成は毎回少し変わるが、今はテストデータで1回生成した分のみ（再確認中）。生成モデル: NVIDIA NeMo（Ku et al., arXiv:2409.16117）");
  s.addNotes("【台本】（約60秒）\n今回一番よかったのが、大量の音声で事前に学習された「音声を作るAI」、生成モデルを、喉マイクと普通のマイクの同時録音で調整したものです。音声認識は学習に一切使っていません。\n\n（図を指して）これは9つの音声認識の文字の誤り率で、灰色が元の音声強調、紺色が生成モデルです。すべての認識器で誤りが15〜24%減り、テストの10人全員で改善しました。ただし、この結果はテストデータで1回生成した分だけで、再現性は確認中です。\n\n【想定質問】\nQ. 作り話（ありもしない内容）をしていないか？\nA. Whisper-smallで確認した範囲では、崩れた発話は0、出力の長さも変わらず、置き換え・挿入の誤りはむしろ減っている。\nQ. Qwen3＋生成モデル（10.4%）は、学習し直したWhisper（SEなし13.8%）より良いのでは？\nA. Qwen3は元の音声強調でも13.6%で、差の大半は認識器の強さ。公平に言えるのは「学習し直したWhisper自身も13.8→11.0%に下がる」こと。\nQ. 元の音声強調とは？\nA. TAPSデータセットの論文で公開されている SE-Conformer（1200万パラメータ）。\nQ. ほかの方法は？\nA. 複数の認識器の損失で学習 → 学習外の Zipformer で悪化。Sato ら型（WavLM 表現の損失）→ 開発データの4認識器で −7〜−13%（同じ条件での比較はこれから）。");
}

// 4. 関連研究
{
  const s = newSlide();
  title(s, "聞こえを良くする研究はあるが、どの認識器でも効くかは未確認");
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

// 5. 課題
{
  const s = newSlide();
  title(s, "効果は大きいが、理由・確かさ・他のデータでの効果・新しさが課題");
  const items = [["① なぜ効いたか", "事前学習のおかげか、単にモデルが大きい（約35倍）からか"],
                 ["② 結果は確かか", "1回生成しただけ。Sato らの方法との同条件比較もまだ"],
                 ["③ 他のデータでも効くか", "1つのデータセット（韓国語 TAPS）でしか確かめていない"],
                 ["④ 自分の工夫はどこか", "公開モデルを調整しただけ（計算も重い）"]];
  items.forEach(([h, b], i) => {
    const y = 1.8 + i * 1.3;
    T(s, h, { x: M, y, w: 3.6, h: 0.9, fontSize: 22, bold: true, color: C.primary, valign: "middle" });
    T(s, b, { x: M + 3.8, y, w: W - 3.8, h: 0.9, fontSize: 20, valign: "middle" });
    if (i < items.length - 1) s.addShape(pptx.ShapeType.line, { x: M, y: y + 1.1, w: W, h: 0, line: { color: C.grid, width: 1 } });
  });
  s.addNotes("【台本】（約50秒）\n課題は4つです。\n\n①なぜ効いたのかが、まだ分かっていません。大量の音声での事前学習が効いたのか、単にモデルが大きいからなのかを切り分ける必要があります。\n\n②結果が確かかどうか。今の結果はテストデータで1回生成しただけなので、開発データや生成し直した場合でも同じになるか、先行研究の方法と同じ条件で比べる必要があります。\n\n③他のデータでも効くか。今は韓国語のTAPSという1つのデータセットでしか確かめていません。\n\n④一番大きいのがこれで、今は公開されているモデルを調整しただけなので、手法としての新しさがありません。\n\n【補足】生成モデル：4.3億パラメータ、非商用ライセンス（CC-BY-NC-SA）。元の音声強調は1200万パラメータ。1発話の変換に20段階の計算。事前学習なしの対照実験は研究室PCの停止で中断中。");
}

// 6. 方針（今後の展開）
{
  const s = newSlide();
  title(s, "次は、認識に向いた生成のさせ方を探し、最後に軽くする");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 16, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }] } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 17, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
  s.addTable([
    [hdr("段階"), hdr("ねらい"), hdr("やること")],
    [cell("1. 結果を固める", { bold: true }), cell("課題①②"), cell("事前学習なしの同じモデルと比較\n生成し直し・開発データ・別のデータセット（VibraVox）で再確認")],
    [cell("2. 生成のさせ方", { bold: true, color: C.primary }), cell("本研究の中心", { bold: true, color: C.primary }), cell("何回か生成して平均する／互いに一番似た候補を選ぶ\nランダムな状態ではなく、元の音声強調の出力から生成を始める")],
    [cell("3. 軽くする", { bold: true }), cell("実用"), cell("2 の出力を正解にして、軽い音声強調へ蒸留")],
  ], { x: M, y: 1.75, w: W, colW: [2.7, 2.1, W - 4.8], fontFace: FONT, rowH: [0.6, 1.15, 1.15, 0.85], valign: "middle", margin: [4, 8, 4, 8] });
  T(s, bullets([
    ["2 は今の学習済みモデルのまま試せる（認識器は使わない）"],
  ], { fontSize: 18, color: C.muted }), { x: M, y: 5.85, w: W, h: 0.45 });
  s.addNotes("【台本】（約55秒）\nこれからの方針です。\n\nまず、課題①②の確認として、事前学習なしの同じモデルと比べ、生成し直しや開発データ、別のデータセットでも同じ結果になるかを確かめます。\n\nそのうえで中心にしたいのが、認識に向いた生成のさせ方です。今は、砂嵐のようなランダムな状態から1回だけ描き直していて、描くたびに少しずつ違う音になります。誤りが増えた発話も、音の取り違えでした。そこで、何回か生成して平均したり、互いに一番似た候補を選んだり、砂嵐ではなく元の音声強調の出力から描き始めたりして、内容を取り違えにくくします。どれも今のモデルのまま、認識器を使わずに試せます。\n\n最後に、見つけた出し方の出力を正解にして、軽い音声強調へ蒸留し、実用的な速さにしたいと考えています。以上です。\n\n【補足】互いに一番似た候補を選ぶ＝最小ベイズリスク選択（類似度は認識器に依存しない音声特徴で測る）。蒸留の正解は、生成モデルが学習で見ていない喉マイク音声（VibraVox のラベルなし約9時間）で作る（以前、学習データの丸暗記で蒸留が失敗したため）。VibraVox は結果の確認に使うだけで、マイクへの適応はしない。");
}

// 8. 参考文献（付録）
{
  const s = newSlide();
  title(s, "参考文献");
  T(s, bullets([
    "TAPS: Y. Kim et al., “Throat and acoustic paired speech dataset for deep learning-based speech enhancement,” Scientific Data, 2026.",
    "Ku et al., “Generative speech foundation model pretraining for high-quality speech extraction and restoration,” arXiv:2409.16117.",
    "山中 涼雅 ほか, “Zero-shot音声変換を用いた咽喉マイク音声の気導音復元に関する初期検討,” 日本音響学会 2026年秋季, 1-Q-42.",
    "和田 航次郎 ほか, “咽喉マイク自由発話音声における感情認識の検討,” 日本音響学会 2026年秋季, 1-Q-56.",
    "Sato et al., “Generic speech enhancement with self-supervised representation space loss,” arXiv:2507.07631, 2025.",
    "J. Hauret et al., “Vibravox: A dataset of French speech captured with body-conduction audio sensors,” Speech Communication, 2025.",
  ], { fontSize: 14 }), { x: M, y: 1.6, w: W, h: 5.0, paraSpaceAfter: 10 });
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-10.pptx" }).then(f => console.log("wrote", f));
