// ゼミ進捗報告（2026-10-01、5分）→ progress_slides_2026-10.pptx
// academic-pptx スキル（~/.claude/skills/academic-pptx）の規則で作成:
//   白背景・1書体・3色まで・見出しは結論の1文・1枚1図・結論スライドで終える・カード/色帯/暗背景なし
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
  s.addNotes("【台本】（約10秒）\n進捗を報告します。今日は、予備実験の結果と、関連研究、そこから見えた課題、これからの方針を話します。");
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
    x: M, y: 1.6, w: 7.4, h: 4.6, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 60,
    chartColors: [C.accent, C.alert], catAxisOrientation: "maxMin", catAxisLabelPos: "low",
    catAxisLabelFontFace: FONT, catAxisLabelFontSize: 16, catAxisLabelColor: C.body,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    valAxisMinVal: -25, valAxisMaxVal: 55,
    showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: '+0"%";-0"%";;', dataLabelColor: "FFFFFF", dataLabelFontSize: 15, dataLabelFontBold: true,
    showLegend: false,
  });
  T(s, "誤りの増減（元の音声強調と比べて）", { x: M, y: 6.25, w: 7.4, h: 0.35, fontSize: 14, color: C.muted, align: "center" });
  T(s, bullets([
    ["Whisperと", { b: "同じ入力" }, "の認識器では誤りが減る"],
    ["入力の作りが違う認識器では", { b: "誤りが増える", c: C.alert }],
    ["目標：", { b: "どの音声認識でも誤りが減る前処理" }, "（クラウドなど作り直せない認識器にも使える）"],
  ], { fontSize: 20, color: C.body }), { x: M + 7.9, y: 1.8, w: W - 7.9, h: 4.4, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "卒論の方法：Whisper-small が正しく書き起こせるように音声強調（TAPS SE-Conformer）を追加学習。比べる相手は追加学習前の同じモデル");
  s.addNotes("【台本】（約50秒）\n喉マイクは騒音に強い一方、こもった音になるので、音声認識の誤りが多くなります。そこで、認識の前に音を直す「音声強調」を研究しています。\n\n卒論では、Whisperが正しく書き起こせるように音声強調を学習しました。Whisperや、Whisperと同じ入力を使う認識器では誤りが15%ほど減りましたが、入力の作りが違う認識器では、逆に26〜46%増えました。\n\nそこで目標を、どの音声認識でも誤りが減る前処理にしました。これができれば、クラウドのように作り直せない音声認識にもそのまま使えます。\n\n【補足】数値は CE-SE（λ=10）の TAPS SE-Conformer 比（test、句読点除去CER）。MMS はほぼ±0。");
}

// 3. 結果（生成モデル）
{
  const s = newSlide();
  title(s, "生成モデルの調整で、9つの音声認識すべての誤りが減った");
  const rows = [["Zipformer", 54.2, 45.1], ["MMS-1B", 35.0, 29.7], ["Whisper-base", 27.3, 21.8], ["Whisper-small", 23.0, 18.0], ["XLS-R", 22.2, 16.9],
                ["Whisper-medium", 19.6, 15.9], ["Whisper-large-v3-turbo", 17.0, 13.7], ["Whisper（喉マイクで再学習）", 14.6, 11.0], ["Qwen3-ASR", 13.6, 10.4]];
  const lw = 3.0, cx = M + lw + 0.15, cw = 4.4, sc = cw / 60, y0 = 1.95, rh = 0.47;
  [0, 20, 40, 60].forEach(v => {
    s.addShape(pptx.ShapeType.line, { x: cx + v * sc, y: y0 - 0.1, w: 0, h: rows.length * rh + 0.05, line: { color: C.grid, width: 1 } });
    T(s, `${v}%`, { x: cx + v * sc - 0.35, y: y0 + rows.length * rh + 0.02, w: 0.7, h: 0.3, fontSize: 14, color: C.muted, align: "center" });
  });
  rows.forEach(([n, a, b], i) => {
    const y = y0 + i * rh;
    T(s, n, { x: M, y, w: lw, h: 0.3, fontSize: 15, align: "right", valign: "middle" });
    s.addShape(pptx.ShapeType.line, { x: cx + b * sc, y: y + 0.15, w: (a - b) * sc, h: 0, line: { color: C.rule, width: 2.5 } });
    s.addShape(pptx.ShapeType.ellipse, { x: cx + a * sc - 0.09, y: y + 0.06, w: 0.18, h: 0.18, fill: { color: "A6A6A6" }, line: { color: "A6A6A6" } });
    s.addShape(pptx.ShapeType.ellipse, { x: cx + b * sc - 0.1, y: y + 0.05, w: 0.2, h: 0.2, fill: { color: C.primary }, line: { color: C.primary } });
  });
  T(s, [{ text: "●", options: { color: "A6A6A6" } }, { text: " 元の音声強調　", options: {} }, { text: "●", options: { color: C.primary } }, { text: " 生成モデル　（文字の誤り率）", options: {} }],
    { x: cx - 0.3, y: 1.5, w: 5.5, h: 0.35, fontSize: 14, color: C.muted });
  s.addShape(pptx.ShapeType.roundRect, { x: cx + 2.2, y: y0 + 6 * rh + 0.05, w: 2.2, h: 0.55, fill: { color: C.highlight }, line: { color: "E6C800", width: 1 }, rectRadius: 0.06 });
  T(s, "全9つで 15〜24% 減", { x: cx + 2.2, y: y0 + 6 * rh + 0.05, w: 2.2, h: 0.55, fontSize: 16, bold: true, color: "7A5200", align: "center", valign: "middle" });
  const rx = M + 7.95;
  T(s, bullets([
    ["大量の音声で事前学習した「音声を作るAI」を、喉マイクと普通のマイクの", { b: "同時録音で調整" }],
    [{ b: "音声認識は学習に使っていない" }],
    ["テスト話者", { b: "10人全員" }, "で改善"],
  ], { fontSize: 20 }), { x: rx, y: 1.7, w: W - 7.95, h: 4.6, paraSpaceAfter: 18, lineSpacingMultiple: 1.2 });
  cite(s, "テストデータで1回生成した分のみ（開発データ・再生成での確認は実行中）。Zipformer は空白込み。生成モデル: NVIDIA NeMo flow matching（Ku et al., arXiv:2409.16117）");
  s.addNotes("【台本】（約60秒）\n今回一番よかったのが、大量の音声で事前に学習された「音声を作るAI」、生成モデルを、喉マイクと普通のマイクの同時録音で調整したものです。音声認識は学習に一切使っていません。\n\n（図を指して）これは9つの音声認識の文字の誤り率で、灰色が元の音声強調、紺色が生成モデルです。すべての認識器で誤りが15〜24%減り、テストの10人全員で改善しました。喉マイクで学習し直したWhisperでも、14.6%から11.0%に下がっています。\n\nほかに、複数の音声認識に合わせて学習する方法も試しましたが、学習に使っていない認識器ではやはり悪化しました。ただし、この結果はテストデータで1回生成した分だけで、再現性は確認中です。\n\n【想定質問】\nQ. 作り話（ありもしない内容）をしていないか？\nA. Whisper-smallで確認した範囲では、崩れた発話は0、出力の長さも変わらず、置き換え・挿入の誤りはむしろ減っている。\nQ. Qwen3＋生成モデル（10.4%）は、学習し直したWhisper（SEなし13.8%）より良いのでは？\nA. Qwen3は元の音声強調でも13.6%で、差の大半は認識器の強さ。公平に言えるのは「学習し直したWhisper自身も13.8→11.0%に下がる」こと。\nQ. 元の音声強調とは？\nA. TAPSデータセットの論文で公開されている SE-Conformer（1200万パラメータ）。\nQ. ほかの方法は？\nA. 複数の認識器の損失で学習 → 学習外の Zipformer で悪化。Sato ら型（WavLM 表現の損失）→ 開発データの4認識器で −7〜−13%（同じ条件での比較はこれから）。");
}

// 4. 関連研究
{
  const s = newSlide();
  title(s, "聞こえを良くする研究はあるが、どの認識器でも効くかは未確認");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 16, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }] } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 17, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
  s.addTable([
    [hdr("研究"), hdr("対象"), hdr("目的"), hdr("音声認識の誤り")],
    [cell("山中ら（東大, 2026）"), cell("喉マイク（TAPS・日本語）"), cell("普通のマイクらしい音へ"), cell("どの方法でも増加", { color: C.alert, bold: true })],
    [cell("Sato ら（2025）"), cell("普通のマイクの雑音"), cell("認識器に頼らない音声強調"), cell("複数の認識器で減少")],
    [cell("本研究", { bold: true, color: C.primary }), cell("喉マイク（TAPS）"), cell("どの音声認識でも誤りを減らす"), cell("入力の違う9つすべてで減少", { bold: true, color: C.primary })],
  ], { x: M, y: 1.75, w: W, colW: [2.6, 3.4, 3.6, W - 9.6], fontFace: FONT, rowH: 0.75, valign: "middle", margin: [4, 8, 4, 8] });
  T(s, bullets([
    ["山中らの方が進んでいる点：", { b: "別の喉マイク・日本語でも検証" }, "、同時録音のデータが不要"],
  ], { fontSize: 20 }), { x: M, y: 5.2, w: W, h: 0.9 });
  cite(s, "山中ら, 日本音響学会 2026年秋 1-Q-42（誤り率は気導音の書き起こしを基準とし、本研究とは測り方が異なる）／Sato et al., arXiv:2507.07631／他: 和田ら 1-Q-56");
  s.addNotes("【台本】（約60秒）\n関連研究です。一番近いのは、今回の音響学会で東大の山中さんたちが発表した研究で、同じTAPSのデータを使っています。声質変換などで喉マイクの音を普通のマイクの音に近づけていて、聞こえは良くなりますが、音声認識の誤りはどの方法でも増えていました。聞こえを良くしても認識は良くならない点は、私の卒論の結果と同じです。\n\n一方、向こうは別の喉マイクや日本語でも確かめていて、同時録音のデータも要りません。ここは私に足りない点です。\n\nSatoらは普通のマイクの雑音除去で、特定の認識器に頼らない方法を提案していて、今回比較対象として再現しました。\n\n喉マイクで、入力の違う複数の音声認識で誤りが減るかを確かめた例は、調べた範囲では見当たりません。\n\n【補足】山中ら：無処理0.233、Sidon 0.308、Kanade(ft) 0.550。和田ら（神戸大・三菱電機）：喉マイクの感情認識でも汎用の雑音除去で悪化。");
}

// 5. 課題
{
  const s = newSlide();
  title(s, "効果は大きいが、理由・確かさ・別のマイクでの効果・新しさが課題");
  const items = [["① なぜ効いたか", "大量の音声での事前学習のおかげか、モデルが大きい（元の約35倍）だけか"],
                 ["② 結果は確かか", "テストデータで1回生成しただけ。先行研究の方法との同条件比較もまだ"],
                 ["③ 別のマイクで効くか", "データセット1つ・喉マイク1種類でしか確かめていない"],
                 ["④ 自分の工夫はどこか", "公開モデルを調整しただけで、手法としての新しさがない"]];
  items.forEach(([h, b], i) => {
    const y = 1.75 + i * 1.2;
    T(s, h, { x: M, y, w: 3.6, h: 0.9, fontSize: 22, bold: true, color: C.primary, valign: "middle" });
    T(s, b, { x: M + 3.8, y, w: W - 3.8, h: 0.9, fontSize: 20, valign: "middle" });
    if (i < items.length - 1) s.addShape(pptx.ShapeType.line, { x: M, y: y + 1.05, w: W, h: 0, line: { color: C.grid, width: 1 } });
  });
  s.addNotes("【台本】（約50秒）\n課題は4つです。\n\n①なぜ効いたのかが、まだ分かっていません。大量の音声での事前学習が効いたのか、単にモデルが大きいからなのかを切り分ける必要があります。\n\n②結果が確かかどうか。今の結果はテストデータで1回生成しただけなので、開発データや生成し直した場合でも同じになるか、先行研究の方法と同じ条件で比べる必要があります。\n\n③別のマイクで効くか。今は1つのデータセット、1種類の喉マイクでしか確かめていません。\n\n④一番大きいのがこれで、今は公開されているモデルを調整しただけなので、手法としての新しさがありません。計算が重いことも課題です。\n\n【補足】生成モデル：4.3億パラメータ、非商用ライセンス（CC-BY-NC-SA）。元の音声強調は1200万パラメータ。1発話の変換に20段階の計算。事前学習なしの対照実験は研究室PCの停止で中断中。");
}

// 6. 方針
{
  const s = newSlide();
  title(s, "まず結果を確かめ、そのうえで手法の工夫に進む");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.primary, fontSize: 18, border: [{ type: "none" }, { type: "none" }, { pt: 1.5, color: C.primary }, { type: "none" }] } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 18, color: C.body, border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: C.rule }, { type: "none" }], ...o } });
  s.addTable([
    [hdr("時期"), hdr("課題"), hdr("やること")],
    [cell("〜1週間", { bold: true }), cell("①②"), cell("事前学習なしで同じモデルを学習して比較／開発データ・生成し直しで再確認")],
    [cell("〜2週間", { bold: true }), cell("③"), cell("別の喉マイク（仏語の公開データ VibraVox）で検証")],
    [cell("その先", { bold: true }), cell("④"), cell("マイクの情報を手がかりに少ないデータで別のマイクへ対応／生成を複数回行って平均し安定化")],
  ], { x: M, y: 1.8, w: W, colW: [1.9, 1.3, W - 3.2], fontFace: FONT, rowH: 0.95, valign: "middle", margin: [4, 8, 4, 8] });
  s.addNotes("【台本】（約45秒）\nこれからの方針です。まず1週間ほどで、課題①②の確認をします。事前学習なしで同じモデルを学習して比べることと、開発データや生成し直しでの再確認です。\n\n次の1週間で、③別のマイクで効くかを、フランス語の公開データで確かめます。\n\nその先で、④手法としての工夫を入れたいと考えています。候補は、マイクの情報を手がかりとして与えて少ないデータで別のマイクに対応する方法と、生成を複数回行って平均し、誤りを安定して減らす方法です。\n\n【補足】比較手法として汎用の音声復元ツール（Sidon）と Sato ら型も同条件で評価し、処理時間も測る。VibraVox はマイクと言語が同時に変わるため、切り分けには日本語の自前収録が有効（相談1の理由）。");
}

// 7. まとめ（質疑中に表示）
{
  const s = newSlide();
  title(s, "まとめと相談したいこと");
  T(s, [
    { text: "1. ", options: { bold: true, color: C.primary } }, { text: "Whisperに合わせた音声強調は、入力の違う認識器で逆効果", options: { breakLine: true } },
    { text: "2. ", options: { bold: true, color: C.primary } }, { text: "生成モデルの調整で、9つの音声認識すべての誤りが15〜24%減った", options: { bold: true, breakLine: true } },
    { text: "3. ", options: { bold: true, color: C.primary } }, { text: "理由・確かさ・別のマイクでの効果・自分の工夫を、これから確かめる" },
  ], { x: M, y: 1.7, w: W, h: 2.4, fontSize: 22, paraSpaceAfter: 14, lineSpacingMultiple: 1.2 });
  s.addShape(pptx.ShapeType.line, { x: M, y: 4.35, w: W, h: 0, line: { color: C.rule, width: 1 } });
  T(s, "相談したいこと", { x: M, y: 4.55, w: W, h: 0.45, fontSize: 20, bold: true, color: C.primary });
  T(s, bullets([
    "別のマイクでの検証のため、日本語で自分でも録音すべきか",
    "手法の新しさをどこに置くのがよいか",
    "非商用ライセンスの公開モデルを土台にしてよいか",
  ], { fontSize: 20 }), { x: M, y: 5.05, w: W, h: 1.7, paraSpaceAfter: 6 });
  s.addNotes("【台本】（約25秒）\nまとめると、Whisperに合わせた音声強調は他の認識器で逆効果でしたが、生成モデルを喉マイクに合わせると、9つの音声認識すべてで誤りが減りました。ただ、その理由や確かさはこれから確かめます。\n\n最後に、ここに挙げた3つについてご意見をいただけると助かります。以上です。");
}

// 8. 参考文献（付録）
{
  const s = newSlide();
  title(s, "参考文献");
  T(s, bullets([
    "Y. Kim et al., “Throat and acoustic paired speech dataset for deep learning-based speech enhancement,” Scientific Data, 2026.（TAPS）",
    "Ku et al., “Generative speech foundation model pretraining for high-quality speech extraction and restoration,” arXiv:2409.16117.",
    "山中 涼雅 ほか, “Zero-shot音声変換を用いた咽喉マイク音声の気導音復元に関する初期検討,” 日本音響学会 2026年秋季, 1-Q-42.",
    "和田 航次郎 ほか, “咽喉マイク自由発話音声における感情認識の検討,” 日本音響学会 2026年秋季, 1-Q-56.",
    "Sato et al., “Generic speech enhancement with self-supervised representation space loss,” arXiv:2507.07631, 2025.",
    "J. Hauret et al., “Vibravox: A dataset of French speech captured with body-conduction audio sensors,” Speech Communication, 2025.",
  ], { fontSize: 14 }), { x: M, y: 1.6, w: W, h: 5.0, paraSpaceAfter: 10 });
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-10.pptx" }).then(f => console.log("wrote", f));
