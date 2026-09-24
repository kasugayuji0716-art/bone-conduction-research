// ゼミ進捗報告（2026-09）スライド → progress_slides_2026-09.pptx
// Artifact版 https://claude.ai/artifact/UWmWDDKcFj3q7hhBnQuDSd と同内容
const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in

const F = "Yu Gothic";
const C = { dark: "1F2A37", light: "F7F6F2", light2: "EEF1F4", card: "FBFAF7", sub: "4A5568",
            orange: "B45F1E", orangeL: "E8A365", orangeB: "D9772B", blue: "2F6FB0", cardDark: "2A3747",
            line: "D8D3C8", textDark: "DCE2E9" };
const M = 0.89; // 128px 相当の余白
const W = 13.333 - 2 * M;

const T = (s, t, o) => s.addText(t, { fontFace: F, color: C.dark, valign: "top", margin: 0, ...o });
const eyebrow = (s, t, color, y = 0.7) => T(s, t, { x: M, y, w: W, h: 0.35, fontSize: 15, bold: true, color, charSpacing: 2 });
const title = (s, t, y = 0.7, color = C.dark, size = 30) => T(s, t, { x: M, y, w: W, h: 0.8, fontSize: size, bold: true, color });
const rich = (parts, base = {}) => parts.map(p => typeof p === "string" ? { text: p, options: base } : { text: p.b, options: { ...base, bold: true } });

// 1. 表紙
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.17, h: 7.5, fill: { color: C.orangeB }, line: { color: C.orangeB } });
  T(s, "卒論 進捗報告", { x: M, y: 2.2, w: W, h: 0.4, fontSize: 15, bold: true, color: C.orange, charSpacing: 2 });
  T(s, "ASR損失で学習した\n喉マイク音声強調の効果と汎化範囲", { x: M, y: 2.75, w: W, h: 1.9, fontSize: 40, bold: true, lineSpacingMultiple: 1.1 });
  T(s, "春日 裕次　／　2026年9月 ゼミ", { x: M, y: 4.9, w: W, h: 0.4, fontSize: 15, color: C.sub });
  s.addNotes("今日は卒論の進捗を5分ほどで報告します。結論から言うと、実験はほぼ揃っていて、今は評価の数値を正しく計算し直している段階です。");
}

// 2. 問いと手法
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "問いと手法");
  const lx = M, lw = 6.3;
  T(s, "背景", { x: lx, y: 1.75, w: lw, h: 0.35, fontSize: 15, bold: true, color: C.orange });
  T(s, rich(["喉マイクは騒音に強いが、4 kHz以上がなくASR精度が低い。SEは", { b: "どのASRにも前処理としてつなげる" }, "のが利点。"]),
    { x: lx, y: 2.15, w: lw, h: 1.3, fontSize: 16, lineSpacingMultiple: 1.4 });
  T(s, "問い", { x: lx, y: 3.65, w: lw, h: 0.35, fontSize: 15, bold: true, color: C.orange });
  T(s, "ASR損失で学習したSEの改善は、\n別のASRにも通用するか？", { x: lx, y: 4.05, w: lw, h: 1.3, fontSize: 20, bold: true, lineSpacingMultiple: 1.3 });
  const rx = M + lw + 0.35, rw = W - lw - 0.35;
  s.addShape(pptx.ShapeType.roundRect, { x: rx, y: 1.75, w: rw, h: 4.6, fill: { color: C.card }, line: { color: "E2DED5", width: 1 }, rectRadius: 0.1 });
  T(s, "手法", { x: rx + 0.33, y: 2.05, w: rw - 0.66, h: 0.35, fontSize: 15, bold: true, color: C.orange });
  T(s, rich(["TAPSのSE-Conformerを、", { b: "凍結したWhisper-small" }, "の交差エントロピー損失で追加学習"]),
    { x: rx + 0.33, y: 2.5, w: rw - 0.66, h: 1.1, fontSize: 16, lineSpacingMultiple: 1.4 });
  s.addShape(pptx.ShapeType.roundRect, { x: rx + 0.33, y: 3.75, w: rw - 0.66, h: 1.1, fill: { color: C.light2 }, line: { color: C.light2 }, rectRadius: 0.08 });
  T(s, "L = 再構成損失 + λ × CE(Whisper(SE出力), 正解)", { x: rx + 0.5, y: 3.85, w: rw - 1, h: 0.9, fontSize: 15, valign: "middle" });
  T(s, "データ：韓国語TAPS（喉マイク／気導マイクの同時収録）", { x: rx + 0.33, y: 5.1, w: rw - 0.66, h: 0.6, fontSize: 13, color: C.sub });
  s.addNotes("予備実験でWhisperのFTが強いことはわかっていますが、SEはどのASRにもつなげられるのが利点なので、SE側を改善する方針です。今回はSEをWhisperの損失で直接学習させました。問題は、その改善が学習に使っていないASRにも通用するかどうかです。");
}

// 3. ASR損失のしくみ
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "ASR損失による学習のしくみ");
  const boxes = [["喉マイク音声", "", C.card, C.line], ["SE-Conformer", "学習する", "FBF1E7", C.orangeB],
                 ["強調音声 ŝ", "→ log-mel", C.card, C.line], ["Whisper-small", "凍結", C.light2, "8A94A3"], ["各トークンの", "予測確率", C.card, C.line]];
  const bw = 1.95, gap = 0.44; let x = M;
  boxes.forEach(([a, b, fill, ln], i) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y: 1.75, w: bw, h: 1.0, fill: { color: fill }, line: { color: ln, width: i === 1 ? 2 : 1 }, rectRadius: 0.08 });
    T(s, [{ text: a, options: { bold: true, breakLine: !!b } }, ...(b ? [{ text: b, options: { bold: true, color: i === 1 ? C.orange : C.sub } }] : [])],
      { x, y: 1.75, w: bw, h: 1.0, fontSize: 13, align: "center", valign: "middle" });
    if (i < boxes.length - 1) s.addShape(pptx.ShapeType.rightArrow, { x: x + bw + 0.08, y: 2.12, w: 0.28, h: 0.26, fill: { color: "8A94A3" }, line: { color: "8A94A3" } });
    x += bw + gap;
  });
  const cards = [
    ["再構成損失", C.dark, C.sub, "ŝ と同時収録の気導音声の差（L1波形＋多解像度STFT）。音として崩れすぎないための歯止め。TAPS SEもこれだけで学習。", 1],
    ["CE損失（ASR損失）", C.orange, C.orangeB, rich(["正解テキストをWhisperのトークン列にし、1つ前までの", { b: "正解" }, "を与えて次のトークンを予測させる（teacher forcing）。正解トークンの確率の −log を全トークンで平均。", { b: "Whisperが正解を書き起こしやすい音" }, "ほど小さい。"]), 1.3],
    ["更新されるのはSEだけ", C.blue, C.blue, "勾配はWhisper → log-mel変換 → SEへ逆伝播。Whisperの重みは変えない。TAPS SEから初期化し追加学習。", 1],
  ];
  const tot = cards.reduce((a, c) => a + c[4], 0), cg = 0.2, avail = W - cg * 2; x = M;
  cards.forEach(([h, hc, bc, body, fl]) => {
    const w = avail * fl / tot;
    s.addShape(pptx.ShapeType.rect, { x, y: 3.1, w, h: 2.55, fill: { color: C.card }, line: { color: C.card } });
    s.addShape(pptx.ShapeType.rect, { x, y: 3.1, w, h: 0.06, fill: { color: bc }, line: { color: bc } });
    T(s, h, { x: x + 0.22, y: 3.3, w: w - 0.44, h: 0.4, fontSize: 16, bold: true, color: hc });
    T(s, body, { x: x + 0.22, y: 3.8, w: w - 0.44, h: 1.8, fontSize: 13, lineSpacingMultiple: 1.35 });
    x += w + cg;
  });
  s.addShape(pptx.ShapeType.roundRect, { x: M, y: 5.95, w: W, h: 0.6, fill: { color: C.light2 }, line: { color: C.light2 }, rectRadius: 0.08 });
  T(s, "L = 再構成損失 ＋ λ × CE損失　（λ：損失の重み、devで選択）", { x: M + 0.2, y: 5.95, w: W - 0.4, h: 0.6, fontSize: 15, bold: true, valign: "middle" });
  s.addNotes("ASR損失の中身です。SEの出力を凍結したWhisper-smallに入れ、正解テキストを1トークンずつ当てさせます。Whisperが正しく書き起こしやすい音ほど損失が小さくなります。Whisperの重みは固定で、勾配はWhisperを通り抜けてSEだけを更新します。音が崩れすぎないよう、気導音声との差を測る再構成損失も足しています。この仕組みだと、SEはWhisper専用に最適化されます。");
}

// 4. わかったこと
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  title(s, "わかったこと");
  const rows = [
    ["学習に使ったWhisper", C.blue, rich(["CERは", { b: "改善する" }, "。Whisper-mediumにも持ち越されるが、baseでははっきりしない"])],
    ["Whisper以外のASR", C.orange, rich(["wav2vec2系では", { b: "改善しない" }, "。XLS-Rでははっきり悪化、MMSではほぼ変わらない"])],
    ["改善の中身", C.dark, rich(["一部は、Whisperが", { b: "句読点を出さなくなった" }, "だけ（正解テキストに句読点がないため）。評価をそろえると改善は小さくなるが、残る"])],
    ["知覚品質", C.dark, rich(["STOI・PESQは下がる。気導音声らしい音ではなく、", { b: "Whisperが読みやすい音" }, "を作っている"])],
  ];
  let y = 1.75; const rh = 1.12;
  rows.forEach(([k, kc, v], i) => {
    s.addShape(pptx.ShapeType.line, { x: M, y, w: W, h: 0, line: { color: C.line, width: 1 } });
    T(s, k, { x: M, y: y + 0.12, w: 3.0, h: rh - 0.24, fontSize: 17, bold: true, color: kc, valign: "middle" });
    T(s, v, { x: M + 3.2, y: y + 0.12, w: W - 3.2, h: rh - 0.24, fontSize: 17, lineSpacingMultiple: 1.3, valign: "middle" });
    y += rh;
  });
  s.addShape(pptx.ShapeType.line, { x: M, y, w: W, h: 0, line: { color: C.line, width: 1 } });
  s.addNotes("細かい数字は省いて傾向だけ話します。学習に使ったWhisperでは改善し、10人の話者全員で改善しました。Whisper-mediumにも効きますが、baseでは差がはっきりしません。ところがwav2vec2系のASRでは改善しません。XLS-Rでははっきり悪化し、MMSではほぼ変わりませんでした。改善の一部は、正解テキストに句読点がないせいでWhisperが句読点を出さなくなっただけで、評価の句読点をそろえると改善は小さくなりますが、残ります。音としての品質指標は下がっていて、気導音声に近づけるのではなく、Whisperが読みやすい音を作っていることがわかりました。");
}

// 5. 現状の評価
{
  const s = pptx.addSlide(); s.background = { color: C.dark };
  eyebrow(s, "現状の評価", C.orangeL);
  title(s, "提案手法として押す方向は、行き詰まっている", 1.15, C.light, 32);
  const cards = [["新しさが小さい", "凍結したWhisperの損失で前処理を学習する手法は、先行研究（Dissen 2024）がある"],
                 ["精度でFTに届かない", "Whisperに特化するなら、Whisper自体を適応させる（FT・Adapter）ほうが明らかに上"],
                 ["SEの長所を失う", "SEの売りは「どのASRにも使える」こと。ASR損失で学習すると、それが消える"]];
  const cg = 0.22, cw = (W - cg * 2) / 3; let x = M;
  cards.forEach(([h, b]) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y: 2.35, w: cw, h: 2.6, fill: { color: C.cardDark }, line: { color: C.cardDark }, rectRadius: 0.1 });
    T(s, h, { x: x + 0.28, y: 2.6, w: cw - 0.56, h: 0.45, fontSize: 18, bold: true, color: C.orangeL });
    T(s, b, { x: x + 0.28, y: 3.2, w: cw - 0.56, h: 1.6, fontSize: 14, color: C.textDark, lineSpacingMultiple: 1.4 });
    x += cw + cg;
  });
  T(s, "「Whisper用ならFT、汎用ならベースラインのSE」で済んでしまい、この手法の居場所がない。", { x: M, y: 5.35, w: W, h: 0.8, fontSize: 16, color: C.textDark, lineSpacingMultiple: 1.4 });
  s.addNotes("正直な評価として、この手法を提案手法として押す方向は行き詰まっていると考えています。理由は3つです。まず、凍結したWhisperの損失で前処理を学習する手法には先行研究があります。次に、Whisperに特化するなら、Whisper自体をファインチューニングするほうが精度が上です。そして、SEの唯一の長所である「どのASRにも使える」ことを、ASR損失で学習すると失ってしまいます。λや損失を工夫しても、この構造は変わらないと思います。");
}

// 6. 行き詰まっていない部分
{
  const s = pptx.addSlide(); s.background = { color: C.light2 };
  eyebrow(s, "行き詰まっていない部分", C.blue);
  title(s, "わかったことには価値がある", 1.15);
  const cards = [["分析の成果", C.dark, C.blue, "「ASR損失で学習すると汎用性を失う」「評価にはテキスト正規化と別系列のASRが必要」", "→ 卒論・学会発表には十分"],
                 ["次の目標がはっきりした", C.orange, C.orangeB, rich(["SEの長所（汎用性）を", { b: "保ったまま" }, "、ASRに効くSEをどう作るか"]), "→ これが一番大きな収穫"]];
  const cg = 0.22, cw = (W - cg) / 2; let x = M;
  cards.forEach(([h, hc, bc, b, f]) => {
    s.addShape(pptx.ShapeType.rect, { x, y: 2.3, w: cw, h: 3.2, fill: { color: C.card }, line: { color: C.card } });
    s.addShape(pptx.ShapeType.rect, { x, y: 2.3, w: cw, h: 0.06, fill: { color: bc }, line: { color: bc } });
    T(s, h, { x: x + 0.3, y: 2.6, w: cw - 0.6, h: 0.5, fontSize: 18, bold: true, color: hc });
    T(s, b, { x: x + 0.3, y: 3.25, w: cw - 0.6, h: 1.4, fontSize: 15, lineSpacingMultiple: 1.4 });
    T(s, f, { x: x + 0.3, y: 4.75, w: cw - 0.6, h: 0.5, fontSize: 14, color: C.sub });
    x += cw + cg;
  });
  s.addNotes("一方で、研究そのものが行き詰まったわけではありません。ASR損失で学習すると汎用性を失うこと、評価にはテキストの正規化と別系列のASRが必要なことは、分析の成果として卒論や学会発表には十分だと考えています。そして一番大きいのは、次に何を目指すべきかがはっきりしたことです。汎用性を保ったまま、ASRに効くSEをどう作るかです。");
}

// 7. 次の方向：本命
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  eyebrow(s, "次の方向の候補：本命", C.orange);
  title(s, "特定のASRに依存しない、喉マイク用のSE", 1.15);
  const rows = [
    ["動機", rich(["現場には", { b: "重みを変えられないASR" }, "が多い（クラウドAPI、共用のASR）。FTもASR損失の学習もできないので、", { b: "ASRを選ばないSEしか選択肢がない" }, "。ここがSEの本当の存在意義"])],
    ["手がかり", rich(["気導音声の表現に近づける損失（Enc L1）は、Whisper以外でも悪化しなかった。汎用性を保てるのは「特定のASRの出力」ではなく", { b: "「音声の表現」に近づける" }, "方向（自己教師あり表現を損失に使う研究もある）"])],
    ["評価の枠組み", rich(["学習に使っていない系列のASRで評価する、句読点をそろえる。", { b: "卒論で作ったものがそのまま使える" }])],
  ];
  let y = 2.2;
  rows.forEach(([k, v]) => {
    const h = 1.28;
    s.addShape(pptx.ShapeType.rect, { x: M, y, w: W, h, fill: { color: C.card }, line: { color: C.card } });
    s.addShape(pptx.ShapeType.rect, { x: M, y, w: 0.06, h, fill: { color: C.orangeB }, line: { color: C.orangeB } });
    T(s, k, { x: M + 0.3, y, w: 2.2, h, fontSize: 16, bold: true, valign: "middle" });
    T(s, v, { x: M + 2.6, y: y + 0.1, w: W - 2.85, h: h - 0.2, fontSize: 14, lineSpacingMultiple: 1.35, valign: "middle" });
    y += h + 0.14;
  });
  s.addNotes("次の方向の本命は、特定のASRに依存しない喉マイク用のSEです。動機として、実際の現場ではクラウドのAPIや共用のASRのように、重みを変えられないASRが多いです。その場合はFTもASR損失での学習もできないので、ASRを選ばないSEしか選択肢がありません。ここがFTに対するSEの本当の存在意義だと考えています。手がかりもあって、気導音声の表現に近づける損失で学習したSEは、Whisper以外のASRでも悪化しませんでした。評価の枠組みは卒論で作ったものがそのまま使えます。");
}

// 8. 次の方向：その他
{
  const s = pptx.addSlide(); s.background = { color: C.light };
  eyebrow(s, "次の方向の候補：その他", C.orange);
  title(s, "ほかに考えられる選択肢", 1.15);
  const hdr = ["方向", "見込み", "懸念"].map(t => ({ text: t, options: { bold: true, color: C.light, fill: { color: C.dark } } }));
  const data = [["複数のASRの損失を混ぜて学習", "自然な発展", "平均的になるだけかも。学習に使っていない系列には効かない可能性"],
                ["実際の雑音環境で評価", "喉マイク本来の強み（騒音に強い）を活かせる", "データの収録が必要。競合研究がある"],
                ["入力側の適応とFTの比較（なぜ効くか）", "分析として面白い", "手法の提案にはならない"]]
    .map((r, i) => r.map(t => ({ text: t, options: { fill: { color: i % 2 ? C.light2 : C.card } } })));
  s.addTable([hdr, ...data], { x: M, y: 2.2, w: W, colW: [W * 0.34, W * 0.33, W * 0.33], fontFace: F, fontSize: 15, color: C.dark,
    border: { type: "solid", color: C.line, pt: 0.75 }, margin: 0.12, valign: "middle", rowH: [0.55, 0.95, 0.95, 0.95] });
  s.addNotes("ほかの選択肢も3つ考えました。複数のASRの損失を混ぜる方法は自然な発展ですが、平均的になるだけで、学習に使っていない系列には効かない可能性があります。実際の雑音環境での評価は喉マイクの本来の強みを活かせますが、データの収録が必要で、競合研究もあります。入力側の適応とFTの比較は分析としては面白いものの、手法の提案にはなりません。これらについてもご意見をいただきたいです。");
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-09.pptx" }).then(f => console.log("Created:", f));
