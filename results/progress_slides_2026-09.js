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
  s.addNotes("【台本】\nよろしくお願いします。卒論の進捗を報告します。\nタイトルは「ASR損失で学習した喉マイク音声強調の効果と汎化範囲」です。\n\n今日の話は3つです。1つ目は、今やっている実験で何がわかったか。2つ目は、それを踏まえて、今のやり方を提案手法として押すのは正直行き詰まっている、という現状の評価。3つ目は、その先の方向について相談したいことです。\n\n先に結論を言うと、卒論に必要な実験はほぼ揃っていて、論文も4ページの形になっています。ただ、途中で評価の方法に問題を見つけたので、数字を全部計算し直しました。その話も正直にします。\n\n（目安：30秒）");
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
  s.addNotes("【台本】\nまず問いと手法です。\n\n（左側を指して）喉マイクは、喉の振動を直接拾うので騒音に強いのが利点です。ただ、今使っているTAPSという韓国語のデータセットでは、喉マイクの音は4kHz以上がもともと入っていません。そのため、普通の音声認識にかけると精度がかなり落ちます。\n\nこれを改善する方法は大きく2つあります。音声認識モデルそのものを喉マイクに合わせて学習し直すファインチューニングと、音声認識の前に音をきれいにする音声強調、SEです。精度だけならファインチューニングの方が強いことは予備実験でわかっています。それでもSEに注目しているのは、SEは前処理なので、どの音声認識にもつなげられるという利点があるからです。\n\n（右側を指して）そこで今回は、TAPSの論文で公開されているSE-Conformerというモデルを、凍結したWhisper-smallの損失で追加学習しました。つまり「Whisperが正しく書き起こせる音」を出すようにSEを学習させています。\n\n（問いを指して）問いは、こうやってASRの損失で学習したSEの改善が、学習に使っていない別のASRにも通用するのか、です。SEの利点が「どのASRにもつなげられること」なので、ここが本質的な問いだと考えています。\n\n（目安：1分15秒）");
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
  s.addNotes("【台本】\nASR損失の仕組みをもう少し詳しく説明します。\n\n（上の図を左から指して）喉マイクの音をSEに入れて、出てきた強調音声をメルスペクトログラムに変換し、Whisper-smallに入れます。Whisperは、各トークンの予測確率を出します。\n\n（真ん中のカードを指して）ここでCE損失、交差エントロピー損失を計算します。正解テキストをWhisperのトークン列にして、1つ前までの正解を与えたうえで、次のトークンを当てさせます。正解のトークンに高い確率を出せるほど損失は小さくなります。つまり、Whisperが正しく書き起こしやすい音ほど損失が小さい、という損失です。\n\n（右のカードを指して）大事なのは、Whisperの重みは固定していて、更新するのはSEだけという点です。勾配はWhisperを通り抜けてSEまで戻ります。\n\n（左のカードを指して）ASRの損失だけだと音が崩れてしまうので、同時に録った気導マイクの音との差、再構成損失も足しています。元のTAPSのSEはこの再構成損失だけで学習されています。\n\n（下の式を指して）この2つを、重みλで足し合わせたものが全体の損失です。λは開発用データで選びました。\n\nこの仕組みだと、SEはWhisperという特定のモデルに合わせて最適化されます。これが次のスライドの結果につながります。\n\n（目安：1分15秒）");
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
  s.addNotes("【台本】\nわかったことです。細かい数字は省いて、傾向だけお話しします。\n\n（1行目）まず、学習に使ったWhisper-smallでは、CERは確かに改善しました。テスト用の10話者全員で改善しています。同じWhisper系列のmediumにも改善は持ち越されましたが、小さいbaseでははっきりした差が出ませんでした。\n\n（2行目）一方で、Whisperとは作りが違うwav2vec2系の音声認識では改善しませんでした。XLS-Rでははっきり悪化し、MMSではほぼ変わりませんでした。\n\n（3行目）それから、評価の途中で問題を見つけました。TAPSの正解テキストには句読点がないのですが、Whisperは普通、文末に句点を付けます。ASR損失で学習すると、SEはWhisperが句読点を出さなくなるような音を作るようになっていました。評価で句読点をそろえていなかったので、その分が改善として数えられていました。句読点をそろえて計算し直すと改善は小さくなりますが、それでも残ります。\n\n（4行目）最後に、STOIやPESQといった音の品質の指標は、むしろ下がっていました。つまり、気導マイクらしい自然な音に近づけているのではなく、Whisperが読みやすい別の音を作っている、ということです。\n\n（補足：聞かれたら）句読点をそろえたCERで、Whisper-smallはTAPSのSEの0.230から0.196、15%の改善です。改善幅のうち約3分の1が句読点によるものでした。XLS-Rは約26%悪化しています。\n\n（目安：1分30秒）");
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
  s.addNotes("【台本】\nここからが、正直な現状の評価です。\n\nこの手法を「提案手法」として押す方向は、行き詰まっていると考えています。理由は3つあります。\n\n（左のカード）1つ目は、新しさが小さいことです。凍結したWhisperの損失で前処理を学習する手法は、2024年のInterspeechで、Dissenらがすでに発表しています。\n\n（真ん中のカード）2つ目は、精度でファインチューニングに届かないことです。Whisperに特化するのであれば、Whisper自体を喉マイクに合わせてファインチューニングする方が、明らかに精度が高いです。\n\n（右のカード）3つ目が一番大きくて、SEの長所を失うことです。SEの売りは、どのASRにもつなげられることでした。ところがASR損失で学習すると、さっきの結果のとおり、他の系列のASRでは効かなくなります。\n\n（下の文）まとめると、「Whisper用ならファインチューニング、汎用ならもともとのTAPSのSE」で済んでしまって、この手法の居場所がありません。損失の重みを変えたり損失を工夫したりしても、この構造自体は変わらないと思っています。\n\n（目安：1分）\n\n【想定質問】\nQ. 学習をやり直せば改善するのでは？\nA. λを大きくするほど、句読点の抑制や、出力に現れる人工的な成分が強くなりました。ASR損失の比重を上げるほどWhisperへの合わせ込みが進む、という構造的な問題だと考えています。");
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
  s.addNotes("【台本】\nただ、研究そのものが行き詰まったわけではないと考えています。\n\n（左のカード）1つ目に、分析の成果としての価値があります。ASR損失で学習するとSEの汎用性が失われること、そしてこういうSEを評価するには、テキストの正規化と、学習に使っていない系列のASRでの評価が必要だということです。なお、ASR損失で学習したSEが別のASRで悪化するという傾向は、NTTのSatoらの2025年の論文や、D4AMという2023年の研究でも報告されていて、今回は喉マイクという帯域が欠けた条件でも同じことが起きることを、句読点の影響も含めて確かめた、という位置づけです。卒論や学会発表としては、この分析でまとめられると考えています。\n\n（右のカード）2つ目、こちらの方が大きいのですが、次に何を目指すべきかがはっきりしました。SEの長所である汎用性を保ったまま、ASRに効くSEをどう作るか、です。\n\n（目安：1分）\n\n【想定質問】\nQ. 先行研究と同じ結論なら、卒論の新規性は？\nA. 喉マイクへの適用と、句読点による見かけの改善の定量化、それに出力に現れる250Hz間隔の人工的な成分の分析が新しい部分です。");
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
  s.addNotes("【台本】\n次の方向の本命として考えているのは、特定のASRに依存しない、喉マイク用のSEです。\n\n（動機）実際の現場では、クラウドのAPIや、他のサービスと共用しているASRのように、重みを変えられないASRが多いです。その場合は、ファインチューニングも、今回のようなASR損失での学習もできません。つまり、ASRを選ばないSEしか選択肢がない。ここが、ファインチューニングに対するSEの本当の存在意義だと考えています。\n\n（手がかり）手がかりもあります。今回、比較手法として、気導マイクの音の表現に近づける損失でも学習しました。こちらは改善は小さかったのですが、Whisper以外のASRでも悪化しませんでした。汎用性を保てるのは、特定のASRの出力に合わせる方向ではなく、音声の表現そのものに近づける方向だと考えています。実際、WavLMのような自己教師あり学習モデルの表現を損失に使うと、学習に使っていないASRでも改善したという報告があります。\n\n（評価の枠組み）評価については、学習に使っていない系列のASRで評価する、句読点をそろえる、といった枠組みを卒論で作ったので、それがそのまま使えます。\n\n（目安：1分15秒）");
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
  s.addNotes("【台本】\n本命以外にも、考えられる選択肢を3つ挙げます。\n\n（1行目）1つ目は、複数のASRの損失を混ぜて学習する方法です。自然な発展ですが、平均的な結果になるだけで、学習に使っていない系列には効かない可能性があります。\n\n（2行目）2つ目は、実際の騒音環境で評価することです。喉マイクの本来の強みは騒音に強いことなので、それを活かせます。ただ、データを自分で収録する必要がありますし、競合する研究もあります。\n\n（3行目）3つ目は、入力側での適応とファインチューニングを比べて、なぜ効くのかを分析する方向です。分析としては面白いのですが、手法の提案にはなりません。\n\n以上です。卒論は分析としてまとめ、この先は本命の方向に切り替えたいと考えています。この方向でよいか、また他に考えるべき選択肢があるか、ご意見をいただけるとありがたいです。ありがとうございました。\n\n（目安：1分）");
}

pptx.writeFile({ fileName: __dirname + "/progress_slides_2026-09.pptx" }).then(f => console.log("Created:", f));
