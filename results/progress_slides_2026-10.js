// ゼミ進捗報告（2026-10-01、5分）→ progress_slides_2026-10.pptx
// 数値の出典: results/GENERATIVE_RESULTS_2026-09-30.md
// 前回の progress_slides_2026-09.pptx（手動編集済み）とは別ファイル
const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in

const F = "Yu Gothic";
const C = { dark: "1F2A37", light: "F7F6F2", light2: "EEF1F4", card: "FBFAF7", sub: "4A5568",
            orange: "B45F1E", orangeL: "E8A365", orangeB: "D9772B", blue: "2F6FB0", green: "2E7D4F",
            gray: "9AA3AE", line: "D8D3C8", cardDark: "2A3747", textDark: "DCE2E9" };
const M = 0.75;
const W = 13.333 - 2 * M;

const T = (s, t, o) => s.addText(t, { fontFace: F, color: C.dark, valign: "top", margin: 0, ...o });
const title = (s, t, color = C.dark) => T(s, t, { x: M, y: 0.5, w: W, h: 0.95, fontSize: 26, bold: true, color, valign: "middle" });
const rich = (parts, base = {}) => parts.map(p => typeof p === "string" ? { text: p, options: base } : { text: p.b, options: { ...base, bold: true, ...(p.c ? { color: p.c } : {}) } });
const card = (s, x, y, w, h, fill = C.card, bar) => {
  s.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: fill }, line: { color: fill } });
  if (bar) s.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.06, fill: { color: bar }, line: { color: bar } });
};
const foot = (s, t) => T(s, t, { x: M, y: 6.85, w: W, h: 0.35, fontSize: 11, color: C.sub });

// 1. 表紙
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.17, h: 7.5, fill: { color: C.orangeB }, line: { color: C.orangeB } });
  T(s, "ゼミ 進捗報告", { x: M + 0.2, y: 2.1, w: W, h: 0.4, fontSize: 16, bold: true, color: C.orange, charSpacing: 2 });
  T(s, "喉マイクの音声を\n「どの音声認識でも読める音」に直す", { x: M + 0.2, y: 2.65, w: W, h: 1.9, fontSize: 40, bold: true, lineSpacingMultiple: 1.1 });
  T(s, "生成モデルの予備実験・課題・関連研究・これからの方針", { x: M + 0.2, y: 4.6, w: W, h: 0.4, fontSize: 18, color: C.sub });
  T(s, "春日 裕次　／　2026年10月1日", { x: M + 0.2, y: 5.3, w: W, h: 0.4, fontSize: 15, color: C.sub });
  s.addNotes("【台本】（約10秒）\n進捗を報告します。今日は、予備実験の結果と、そこから見えた課題、関連研究、これからの方針を話します。");
}

// 2. 前提と結論
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "Whisperに合わせた音声強調は、作りの違う音声認識で逆効果だった");
  // 流れ図
  const boxes = [["喉マイクの音", "騒音に強いが、こもった音"], ["音声強調（前処理）", "認識の前に音を直す"], ["音声認識", "どれにでもつなげられる"]];
  const bw = 3.3, gap = 0.55; let x = M;
  boxes.forEach(([a, b], i) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y: 1.7, w: bw, h: 1.0, fill: { color: i === 1 ? "FBF1E7" : C.card }, line: { color: i === 1 ? C.orangeB : C.line, width: i === 1 ? 2 : 1 }, rectRadius: 0.08 });
    T(s, [{ text: a, options: { bold: true, breakLine: true, fontSize: 17 } }, { text: b, options: { fontSize: 13, color: C.sub } }],
      { x, y: 1.7, w: bw, h: 1.0, align: "center", valign: "middle" });
    if (i < 2) s.addShape(pptx.ShapeType.rightArrow, { x: x + bw + 0.12, y: 2.05, w: 0.32, h: 0.3, fill: { color: C.gray }, line: { color: C.gray } });
    x += bw + gap;
  });
  // 卒論の結果（棒）
  T(s, "卒論の方法：Whisperが正しく書き起こせるように音声強調を学習 → 誤りの増減（元の音声強調と比べて）",
    { x: M, y: 3.0, w: W, h: 0.35, fontSize: 14, bold: true, color: C.sub });
  const bars = [["Whisper-small", -15, "Whisperと同じ入力"], ["Qwen3-ASR", -16, "Whisperと同じ入力"], ["XLS-R", 26, "入力が違う"], ["Zipformer", 46, "入力が違う"]];
  const x0 = M + 5.2, scale = 0.075; let y = 3.5;
  s.addShape(pptx.ShapeType.line, { x: x0, y: 3.4, w: 0, h: 2.05, line: { color: C.gray, width: 1 } });
  bars.forEach(([n, v, g]) => {
    const col = v < 0 ? C.green : C.orangeB;
    T(s, [{ text: n, options: { bold: true, breakLine: true } }, { text: g, options: { fontSize: 11, color: C.sub } }], { x: M, y: y - 0.05, w: 2.6, h: 0.5, fontSize: 14 });
    const w = Math.abs(v) * scale;
    s.addShape(pptx.ShapeType.rect, { x: v < 0 ? x0 - w : x0, y: y + 0.05, w, h: 0.32, fill: { color: col }, line: { color: col } });
    T(s, `${v > 0 ? "+" : "−"}${Math.abs(v)}%`, { x: v < 0 ? x0 - w - 1.0 : x0 + w + 0.1, y: y + 0.05, w: 0.9, h: 0.32, fontSize: 15, bold: true, color: col, align: v < 0 ? "right" : "left", valign: "middle" });
    y += 0.5;
  });
  T(s, "← 誤りが減る　　誤りが増える →", { x: x0 - 2.0, y: 5.5, w: 4.0, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
  // 目標
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: 5.85, w: W, h: 1.0, fill: { color: C.dark }, line: { color: C.dark }, rectRadius: 0.08 });
  T(s, [{ text: "目標：", options: { bold: true, color: C.orangeL } }, { text: "どの音声認識でも誤りが減る前処理（クラウドなど作り直せない認識器にもそのまま使える）", options: { color: C.light, breakLine: true } },
         { text: "→ 今回、生成モデルで9つすべての誤りが減った", options: { bold: true, color: C.orangeL } }],
    { x: M + 0.3, y: 5.85, w: W - 0.6, h: 1.0, fontSize: 16, valign: "middle", lineSpacingMultiple: 1.3 });
  s.addNotes("【台本】（約50秒）\n喉マイクは騒音に強い一方、こもった音になるので、音声認識の誤りが多くなります。そこで、認識の前に音を直す「音声強調」を研究しています。\n\n卒論では、Whisperが正しく書き起こせるように音声強調を学習しました。Whisperや、Whisperと同じ入力を使う認識器では誤りが15%ほど減りましたが、入力の作りが違う認識器では、逆に26〜46%増えました。\n\nそこで目標を、どの音声認識でも誤りが減る前処理にしました。これができれば、クラウドのように作り直せない音声認識にもそのまま使えます。\n\n結論を先に言うと、今回、生成モデルを使って9つすべての認識器で誤りが減りました。\n\n【補足】数値はWhisper-smallの学習で得た音声強調（CE-SE、λ=10）の、TAPSのSE-Conformer比。MMSはほぼ±0。");
}

// 3. 結果
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "生成モデルで、9つの音声認識すべての誤りが減った");
  // [名前, 元の音声強調 CER%, 生成モデル CER%, TAPS比（%、丸め前の値から計算）]
  const rows = [["Zipformer", 54.2, 45.1, 16.9], ["MMS-1B", 35.0, 29.7, 15.0], ["Whisper-base", 27.3, 21.8, 20.1], ["Whisper-small", 23.0, 18.0, 21.8], ["XLS-R", 22.2, 16.9, 23.8],
                ["Whisper-medium", 19.6, 15.9, 19.2], ["Whisper-large-v3-turbo", 17.0, 13.7, 19.3], ["Whisper（喉マイクで学習し直し）", 14.6, 11.0, 24.4], ["Qwen3-ASR", 13.6, 10.4, 23.8]];
  const lx = M, lw = 2.95, cx = M + 3.05, cw = 4.3, sc = cw / 60;
  T(s, "文字の誤り率（%）", { x: cx, y: 1.55, w: cw, h: 0.3, fontSize: 12, color: C.sub });
  [0, 20, 40, 60].forEach(v => {
    s.addShape(pptx.ShapeType.line, { x: cx + v * sc, y: 1.9, w: 0, h: 4.55, line: { color: "E2DED5", width: 1, dashType: "dash" } });
    T(s, `${v}`, { x: cx + v * sc - 0.3, y: 6.47, w: 0.6, h: 0.25, fontSize: 11, color: C.sub, align: "center" });
  });
  let y = 2.0; const rh = 0.5;
  rows.forEach(([n, a, b, r]) => {
    T(s, n, { x: lx, y, w: lw, h: 0.3, fontSize: 12.5, align: "right", valign: "middle" });
    s.addShape(pptx.ShapeType.line, { x: cx + b * sc, y: y + 0.15, w: (a - b) * sc, h: 0, line: { color: C.gray, width: 2 } });
    s.addShape(pptx.ShapeType.ellipse, { x: cx + a * sc - 0.09, y: y + 0.06, w: 0.18, h: 0.18, fill: { color: C.gray }, line: { color: C.gray } });
    s.addShape(pptx.ShapeType.ellipse, { x: cx + b * sc - 0.1, y: y + 0.05, w: 0.2, h: 0.2, fill: { color: C.orangeB }, line: { color: C.orangeB } });
    T(s, `−${Math.round(r)}%`, { x: cx + cw + 0.1, y, w: 0.75, h: 0.3, fontSize: 12.5, bold: true, color: C.green, valign: "middle" });
    y += rh;
  });
  // 凡例
  s.addShape(pptx.ShapeType.ellipse, { x: cx + 1.6, y: 1.62, w: 0.16, h: 0.16, fill: { color: C.gray }, line: { color: C.gray } });
  T(s, "元の音声強調", { x: cx + 1.8, y: 1.55, w: 1.2, h: 0.3, fontSize: 11, color: C.sub });
  s.addShape(pptx.ShapeType.ellipse, { x: cx + 3.0, y: 1.62, w: 0.16, h: 0.16, fill: { color: C.orangeB }, line: { color: C.orangeB } });
  T(s, "生成モデル", { x: cx + 3.2, y: 1.55, w: 1.1, h: 0.3, fontSize: 11, color: C.sub });
  // 右側
  const rx = M + 8.35, rw = W - 8.35;
  card(s, rx, 1.6, rw, 2.6, C.card, C.orangeB);
  T(s, "何をしたか", { x: rx + 0.25, y: 1.8, w: rw - 0.5, h: 0.35, fontSize: 15, bold: true, color: C.orange });
  T(s, rich(["大量の音声で事前に学習された", { b: "「音声を作るAI」（生成モデル）" }, "を、喉マイクと普通のマイクの同時録音で調整。", { b: "音声認識は学習に使っていない" }]),
    { x: rx + 0.25, y: 2.25, w: rw - 0.5, h: 1.9, fontSize: 14, lineSpacingMultiple: 1.35 });
  card(s, rx, 4.4, rw, 2.05, C.card, C.gray);
  T(s, "ほかに試した方法", { x: rx + 0.25, y: 4.6, w: rw - 0.5, h: 0.35, fontSize: 15, bold: true, color: C.sub });
  T(s, [{ text: "複数の音声認識に合わせて学習", options: { bold: true, breakLine: true } },
        { text: "→ 学習に使っていない認識器で悪化", options: { breakLine: true } },
        { text: "音声の特徴を普通のマイクに近づける（Sato ら型）", options: { bold: true, breakLine: true } },
        { text: "→ 4つの認識器で改善（同じ条件での比較はこれから）" }],
    { x: rx + 0.25, y: 5.0, w: rw - 0.5, h: 1.4, fontSize: 12.5, lineSpacingMultiple: 1.3 });
  foot(s, "テスト話者10人全員で改善（全認識器）。ただしテストデータで1回生成した分のみ。開発データ・再生成での確認は実行中。Zipformer は空白込みの誤り率");
  s.addNotes("【台本】（約65秒）\n今回一番よかったのが、大量の音声で事前に学習された「音声を作るAI」、生成モデルを、喉マイクと普通のマイクの同時録音で調整したものです。音声認識は学習に一切使っていません。\n\n（図を指して）これは9つの音声認識の文字の誤り率で、灰色が元の音声強調、オレンジが生成モデルです。すべての認識器で誤りが15〜24%減り、テストの10人全員で改善しました。喉マイクで学習し直したWhisperでも、14.6%から11.0%に下がっています。\n\n（右下を指して）ほかに、複数の音声認識に合わせて学習する方法も試しましたが、学習に使っていない認識器ではやはり悪化しました。\n\nただし、この結果はテストデータで1回生成した分だけで、再現性は確認中です。\n\n【想定質問】\nQ. 作り話（ありもしない内容）をしていないか？\nA. Whisper-smallで確認した範囲では、崩れた発話は0、出力の長さも変わらず、置き換え・挿入の誤りはむしろ減っている。\nQ. Qwen3＋生成モデル（10.4%）は、学習し直したWhisper（SEなし13.8%）より良いのでは？\nA. Qwen3は元の音声強調でも13.6%で、差の大半は認識器の強さ。公平に言えるのは「学習し直したWhisper自身も13.8→11.0%に下がる」こと。\nQ. 元の音声強調とは？\nA. TAPSデータセットの論文で公開されている SE-Conformer（1200万パラメータ）。");
}

// 4. 関連研究
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "聞こえを良くする研究はあるが、どの音声認識でも効くかは未確認");
  const head = ["研究", "対象", "目的・方法", "結果"];
  const cw = [2.6, 2.4, 3.3, W - 8.3];
  const rows = [
    [rich([{ b: "山中ら" }, "\n東大・2026秋 音響学会"]), "喉マイク\n（TAPS＋日本語）", "普通のマイクらしい音へ\n（声質変換・帯域拡張）", rich(["聞こえは改善、", { b: "認識の誤りはどの方法でも増加", c: C.orange }, "\n※誤りの測り方は本研究と異なる"])],
    [rich([{ b: "Sato ら" }, "\n2025"]), "普通のマイク\n（雑音除去）", "特定の認識器に頼らない\n音声強調", "複数の認識器で改善\n→ 本研究で比較対象として再現"],
    [rich([{ b: "本研究", c: C.orange }]), "喉マイク\n（TAPS）", "どの音声認識でも\n誤りを減らす", rich([{ b: "入力の違う9つの認識器すべてで改善", c: C.green }])],
  ];
  let x = M;
  head.forEach((h, i) => { T(s, h, { x: x + 0.15, y: 1.65, w: cw[i] - 0.3, h: 0.35, fontSize: 13, bold: true, color: C.sub }); x += cw[i]; });
  let y = 2.05;
  rows.forEach((r, ri) => {
    s.addShape(pptx.ShapeType.rect, { x: M, y, w: W, h: 1.05, fill: { color: ri === 2 ? "FBF1E7" : C.card }, line: { color: C.line, width: 0.75 } });
    x = M;
    r.forEach((c, i) => { T(s, c, { x: x + 0.15, y, w: cw[i] - 0.3, h: 1.05, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.2 }); x += cw[i]; });
    y += 1.15;
  });
  card(s, M, 5.6, W, 1.05, C.light2, C.blue);
  T(s, rich([{ b: "山中らが上の点：", c: C.blue }, "別の喉マイク・日本語でも検証／同時録音のデータが要らない　→　本研究の課題（次ページ）"]),
    { x: M + 0.3, y: 5.72, w: W - 0.6, h: 0.85, fontSize: 15, valign: "middle" });
  foot(s, "その他：和田ら（神戸大・三菱電機、2026秋 音響学会）喉マイクの感情認識でも、汎用の雑音除去を通すと認識が悪化");
  s.addNotes("【台本】（約65秒）\n関連研究です。一番近いのは、今回の音響学会で東大の山中さんたちが発表した研究で、同じTAPSのデータを使っています。声質変換などで喉マイクの音を普通のマイクの音に近づけていて、聞こえは良くなりますが、音声認識の誤りはどの方法でも増えていました。誤りの測り方が違うので直接は比べられませんが、聞こえを良くしても認識は良くならない点は、私の卒論の結果と同じです。\n\n一方、向こうは別の喉マイクや日本語でも確かめていて、同時録音のデータも要りません。ここは私に足りない点です。\n\nSatoらは普通のマイクの雑音除去で、特定の認識器に頼らない方法を提案していて、今回比較対象として再現しました。\n\n喉マイクで、入力の違う複数の音声認識で誤りが減るかを確かめた例は、調べた範囲では見当たりません。\n\n【補足】山中ら：1-Q-42。CERは気導音をWhisperで書き起こした結果を基準にしている（正解テキストではない）。無処理0.233、Sidon 0.308、Kanade(ft) 0.550。\nSatoら：Sato, Ochiai, Delcroix ら, Generic Speech Enhancement with Self-Supervised Representation Space Loss（2025）。本研究の再現はWavLM-Large全層のMSE。");
}

// 5. 課題
{
  const s = pptx.addSlide(); s.background = { color: C.dark };
  T(s, "現状の課題", { x: M, y: 0.45, w: W, h: 0.35, fontSize: 15, bold: true, color: C.orangeL, charSpacing: 2 });
  T(s, "効果は大きいが、理由・確かさ・広がり・自分の工夫がまだ", { x: M, y: 0.85, w: W, h: 0.8, fontSize: 28, bold: true, color: C.light });
  const items = [["①", "なぜ効いたか", "「大量の音声での事前学習」のおかげか、単にモデルが大きい（元の約35倍）からか"],
                 ["②", "結果は確かか", "テストデータで1回生成しただけ。開発データ・再生成での確認、先行研究の方法との同じ条件での比較がまだ"],
                 ["③", "別のマイクで効くか", "データセット1つ・喉マイク1種類でしか確かめていない"],
                 ["④", "自分の工夫はどこか", "公開モデルを調整しただけで、手法としての新しさがない（計算が重いことも課題）"]];
  const cg = 0.25, cw = (W - cg) / 2, ch = 2.15;
  items.forEach(([n, h, b], i) => {
    const x = M + (i % 2) * (cw + cg), y = 1.95 + Math.floor(i / 2) * (ch + cg);
    s.addShape(pptx.ShapeType.roundRect, { x, y, w: cw, h: ch, fill: { color: C.cardDark }, line: { color: C.cardDark }, rectRadius: 0.08 });
    T(s, `${n} ${h}`, { x: x + 0.3, y: y + 0.25, w: cw - 0.6, h: 0.45, fontSize: 22, bold: true, color: C.orangeL });
    T(s, b, { x: x + 0.3, y: y + 0.85, w: cw - 0.6, h: 1.2, fontSize: 17, color: C.textDark, lineSpacingMultiple: 1.35 });
  });
  s.addNotes("【台本】（約50秒）\n課題は4つです。\n\n①なぜ効いたのかが、まだ分かっていません。大量の音声での事前学習が効いたのか、単にモデルが大きいからなのかを切り分ける必要があります。\n\n②結果が確かかどうか。今の結果はテストデータで1回生成しただけなので、開発データや生成し直した場合でも同じになるか、先行研究の方法と同じ条件で比べる必要があります。\n\n③別のマイクで効くか。今は1つのデータセット、1種類の喉マイクでしか確かめていません。\n\n④一番大きいのがこれで、今は公開されているモデルを調整しただけなので、手法としての新しさがありません。\n\n【補足】生成モデル：NVIDIA NeMo の flow matching（4.3億パラメータ、非商用ライセンス）。元の音声強調は1200万パラメータ。1発話の変換に20段階の計算。\n事前学習なしの対照実験は約7000/20000ステップで研究室PCが停止し中断。");
}

// 6. 方針
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "まず結果を確かめ、そのうえで手法の工夫に進む");
  const cols = [["〜1週間", "①②", C.blue, ["事前学習なしで同じモデルを学習して比較", "開発データ・生成し直しで再確認", "先行研究の方法と同じ条件で比較"]],
                ["〜2週間", "③", C.orangeB, ["別の喉マイク：仏語の公開データ（VibraVox）", "汎用の音声復元ツール（Sidon）とも比較", "処理時間を測る"]],
                ["その先", "④", C.green, ["マイクの情報を手がかりに、少ないデータで別のマイクへ対応", "生成を複数回行って平均し、誤りを安定して減らす"]]];
  const cg = 0.25, cw = (W - 2 * cg) / 3; let x = M;
  cols.forEach(([h, tag, col, items], i) => {
    card(s, x, 1.65, cw, 3.2, C.card, col);
    T(s, rich([{ b: h }, `　課題${tag}`], { color: col }), { x: x + 0.25, y: 1.85, w: cw - 0.5, h: 0.4, fontSize: 17, bold: true });
    T(s, items.map((t, j) => ({ text: t, options: { bullet: { indent: 15 }, breakLine: j < items.length - 1 } })),
      { x: x + 0.25, y: 2.4, w: cw - 0.5, h: 2.35, fontSize: 14, lineSpacingMultiple: 1.35, paraSpaceAfter: 6 });
    if (i < 2) s.addShape(pptx.ShapeType.rightArrow, { x: x + cw + 0.02, y: 3.1, w: 0.2, h: 0.3, fill: { color: C.gray }, line: { color: C.gray } });
    x += cw + cg;
  });
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: 5.1, w: W, h: 1.55, fill: { color: C.dark }, line: { color: C.dark }, rectRadius: 0.08 });
  T(s, "相談したいこと", { x: M + 0.3, y: 5.22, w: W - 0.6, h: 0.35, fontSize: 15, bold: true, color: C.orangeL });
  T(s, [{ text: "1. 別のマイクでの検証のために、日本語で自分でも録音すべきか", options: { breakLine: true } },
        { text: "2. 手法の新しさをどこに置くのがよいか", options: { breakLine: true } },
        { text: "3. 非商用ライセンスの公開モデルを土台にしてよいか" }],
    { x: M + 0.3, y: 5.6, w: W - 0.6, h: 1.0, fontSize: 14.5, color: C.light, lineSpacingMultiple: 1.25 });
  s.addNotes("【台本】（約60秒）\nこれからの方針です。課題の番号に対応させています。\n\nまず1週間ほどで、①②の確認をします。事前学習なしで同じモデルを学習して比べることと、開発データや生成し直しでの再確認、先行研究の方法との同じ条件での比較です。\n\n次の1週間で、③別のマイクで効くかを、フランス語の公開データで確かめます。汎用の音声復元ツールとの比較と、処理時間の計測もします。\n\nその先で、④手法としての工夫を入れたいと考えています。候補は、マイクの情報を手がかりとして与えて少ないデータで別のマイクに対応する方法と、生成を複数回行って平均し、誤りを安定して減らす方法です。\n\n最後に相談したいことが3つあります。別のマイクでの検証のために日本語で自分でも録音すべきか、手法の新しさをどこに置くのがよいか、そして非商用ライセンスの公開モデルを土台にしてよいか。ご意見をいただけると助かります。以上です。\n\n【補足】VibraVoxはマイクと言語が同時に変わるため、効果の切り分けには日本語の自前収録が有効（相談1の理由）。複数回の平均は、以前4つの音声強調の出力を平均すると全認識器で改善した知見から。");
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-10.pptx" }).then(f => console.log("wrote", f));
