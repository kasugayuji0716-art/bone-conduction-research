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
  s.addNotes("【台本】（約50秒）\n喉マイクは首の振動を拾うので騒音に強いのですが、TAPS では 8 kHz で収録されていて、4 kHz 以上の成分が最初からありません。そのまま Whisper に入れると誤り率は 0.45 です。\nそこで、NVIDIA が公開している事前学習済みの生成モデルを、喉マイクから気導マイクへの対で追加学習し、欠けた成分を補います。さらに、Whisper の損失で学習した音声強調の出力と、振幅を半分ずつ混ぜます。これを融合Aと呼んでいます。混ぜるだけなので新しい学習はいりません。\n評価は9種の認識器で行い、損失と設定の選択には Whisper-small だけを使っています。\n\n【質問されたら：データ】\n・TAPS（Kim et al., Scientific Data 2026）：韓国語の読み上げ（ニュース文・日常文）。喉マイク（加速度センサ、8 kHz 収録→16 kHz 配布）と気導マイクの同時収録。train 40話者・4,000発話（約10.2時間）、dev 10話者・1,000発話、test 10話者・1,000発話（約2.6時間）。各話者100発話、平均約9秒。話者は分割間で重ならない（test は声の高さで見て低い声5人・高い声5人）\n・test の文で train と完全に一致するのは5種類だけ\n\n【質問されたら：生成モデル】\n・NVIDIA NeMo の sr_ssl_flowmatching_16k_430m（Ku et al., arXiv:2409.16117）。複素 STFT 上の flow matching、4.3億パラメータ。大規模音声で、マスクした区間を復元するタスクで事前学習済み\n・追加学習：入力（条件）＝喉マイク、目標＝同じ発話の気導マイク。TAPS train 4,000対、20k step、batch 8、6.14秒のランダム切り出し、学習率 1e-4、EMA、32ビット浮動小数点。ASR の損失は使っていない\n・重みは学習終了時（最終ステップ）のものを使用。途中の検証は dev の100発話で PESQ を見ていただけで、重みの選択には使っていない\n・推論：標準正規分布の雑音から出発し、喉マイクを条件に ODE を Euler 法20ステップで解く。乱数を変えて4回生成し、STFT 振幅を平均（位相は1つ目）。4回平均で 0.180→0.169（Whisper-small）\n・学習時間：RTX PRO 6000 で 20k step 約3.5時間\n・同じ構造を乱数初期値から同じ量だけ学習すると 0.551 と大きく悪化 → 効いているのは事前学習（ただし学習量不足の可能性もある）\n\n【質問されたら：Whisper 用の強調（CE-SE）】\n・TAPS 公開の SE-Conformer（Kim & Seo, Interspeech 2021。1210万パラメータ）を初期値に追加学習\n・損失＝L1 波形損失＋多重解像度 STFT 損失（気導音に対して）＋ λ × Whisper-small の交差エントロピー（正解テキスト、教師強制、言語は韓国語）。Whisper は凍結し、強調器だけを更新。対数メルは Whisper と同じ計算を微分できる形で実装\n・λ は dev の CER で {0, 0.1, 0.5, 1, 2, 5, 10} から選んで 10。Adam、学習率 3e-4、batch 4、最大50エポック、dev の損失で early stopping。15秒を超える発話は学習から除外\n\n【質問されたら：融合】\n・STFT（512点、シフト128点）の振幅を (1−w)|G| + w|C| で平均し、位相は生成モデル。w は dev で Whisper-small だけを見て {0.25, 0.5, 0.75} から選び 0.5（CER 0.202 / 0.196 / 0.198）\n・なぜ振幅だけ：位相の違う2つの波形をそのまま足すと打ち消し合う");
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
  s.addNotes("【台本】（約60秒）\n主な結果です。左から、処理なし、TAPS が公開している従来の音声強調、生成モデル、融合Aです。\n生成モデルは、系統の違う認識器も含めた9種すべてで、従来の強調より誤りを 19〜32% 減らしました。\n特に見てほしいのは太字の行で、喉マイクの音で追加学習した Whisper です。以前の実験では、この Whisper に音声強調を足すと逆に悪くなっていましたが、生成モデルなら 0.138 から 0.092 まで下がりました。認識器を学習し直した後でも、前処理で上乗せできます。\n融合A は Whisper 系にさらに 2〜8% 効きますが、作りの違う Zipformer では悪化します。融合の上乗せは、学習を3回やり直しても再現しました。\n\n【質問されたら：認識器】\n・Whisper（small / base / medium / large-v3-turbo）：faster-whisper（CTranslate2）、言語を韓国語に指定\n・FT Whisper：Whisper-small を TAPS の喉マイク train で追加学習（学習率 1e-5、batch 16、dev で early stopping）。気導マイクは使っていない\n・Qwen3-ASR-1.7B：エンコーダは Whisper と別だが、入力は同じ形式の対数メル（128次元）\n・系統が違う認識器：XLSR（kresnik/wav2vec2-large-xlsr-korean。XLSR-53 を Zeroth Korean で追加学習した波形入力の CTC）、MMS-1B（韓国語アダプタ）、Zipformer（sherpa-onnx、KsponSpeech のみで学習、Kaldi 型 fbank 入力）\n・損失と設定の選択に使ったのは Whisper-small だけ。他の8種は評価専用\n\n【質問されたら：評価】\n・CER は正解と認識結果の両方から句読点を除き、発話ごとに 1.0 で打ち切って平均（上限付き発話平均 CER）。総編集距離÷総文字数の通常の CER でも主な傾向は同じ\n・検定：同じ話者の発話は独立でないので、話者ごとの平均 CER（10人）で両側 Wilcoxon。条件の対ごとに認識器をまたいで Holm 補正。10人全員が同じ向きのとき未補正 p≈0.002、補正後 p≈0.018\n・再現性：CE-SE と生成モデルを2回ずつ学習し直し、生成×4 はほぼ同じ（多くの認識器で差 0.001 程度）、融合の上乗せは small・medium・turbo・FT で3回とも有意\n・注意：融合を採用したことや内挿への変更は test の他の認識器も見て決めたので、TAPS 上の分析は探索的と論文に明記している\n\n【質問されたら：処理なしの FT Whisper が TAPS SE より良いのはなぜ】\n・FT Whisper は喉マイクの生音声で学習しているので、生音声が得意（0.138）。従来の強調を通すと学習時と違う音になり悪化（0.146）。生成モデルの出力では、学習時と違う音なのに良くなった（0.100）");
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
  s.addNotes("【台本】（約45秒）\n評価が TAPS の中で閉じているので、別のデータセットでも確かめました。フランス語の VibraVox です。\nTAPS で作った部品をそのまま使うと、どの処理も処理なしより悪くなりました。音量や帯域をそろえても変わらなかったので、機器の違いが効いていると考えています。\n一方、VibraVox の対データで同じ手順の適応をやり直すと、生成モデルで誤りが 70〜82% 減り、融合A でさらに減りました。手法の作り方はデータセットを越えて成り立ちますが、部品は機器ごとに適応が必要です。ただ、1,000 対、1.3 時間ほどのデータで、全データの改善量の 86% が得られています。\n\n【質問されたら：VibraVox のデータ】\n・VibraVox（Hauret et al., Speech Communication 2025）の speech_clean：仏語の読み上げ、体内伝導センサー5種＋ヘッドセット（気導）の同時収録。今回は喉マイクとヘッドセットの対を使用\n・train 20,981対（約26.3時間、149話者）、dev（validation）2,523発話、test 3,064発話・21話者。話者は分割間で重ならない。test の文で train と完全に一致するのは1文だけ\n・CER は句読点除去に加えて小文字化\n\n【質問されたら：適応のやり方】\n・生成モデル：同じ NVIDIA の事前学習済みモデルから、VibraVox train で 20k step 追加学習（TAPS と同じ設定）\n・強調：TAPS SE を VibraVox で再構成損失だけで追加学習（VibraVox SE）→ それを初期値に、仏語の Whisper-small の CE（λ=10）で学習\n・λ=10、w=0.5、N=4 は TAPS で決めた値に固定。内挿の α だけ VibraVox の dev で選び直した（TAPS と同じ 0.7）\n・系統が違う認識器：XLSR 仏語版（jonatasgrosman/wav2vec2-large-xlsr-53-french、言語モデルなしの貪欲復号）、MMS-1B（仏語アダプタ）\n\n【質問されたら：そのまま使うと悪化した理由】\n・音量を TAPS にそろえ、4 kHz で帯域制限した入力でも結果はほぼ同じ → 単純な音量・帯域の違いではない\n・言語の違いか機器の違いかは、まだ切り分けていない。同じ VibraVox の別センサー（耳栓型・額）に喉マイク用の部品を使っても悪化したので、言語が同じでも機器が変われば失敗しうる\n\n【質問されたら：少ない対データ】\n・train から無作為に選んだ一部の対で、生成モデルだけを 10k step 適応（全データは 20k step なので学習量も違う）。乱数は1通りで各1試行\n・Whisper-small：200対（約15分）0.272、1,000対（約1.3時間）0.148、5,000対（約6.3時間）0.131、全データ 0.102（処理なし 0.422）\n・今後：乱数を変えた繰り返し、少ない話者で集めた場合、学習量の調整");
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
