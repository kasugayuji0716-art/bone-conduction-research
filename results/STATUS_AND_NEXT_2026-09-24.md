# 研究の現状・課題・次にやること（2026-09-24 時点）

関連: 確定数値 → `results/RESULTS_SUMMARY.md` 末尾「確定値」節 ／ 文献 → `results/literature_survey_2026-09-24.md`

---

## 1. 現状

### 1.1 研究の位置づけ
- 手法: TAPS pretrained SE-Conformer（12.1M）を、L = L1+MR-STFT（対気導） + λ·CE(凍結Whisper-small(ŝ), 正解テキスト) で追加学習（CE-SE）。λ=10（dev選択）
- 評価（2026-09-24 以降の原則）: 正解・認識結果の両方から句読点を除去したCER（cap 1.0）、test 1000発話、話者単位Wilcoxon（n=10）、hyp を必ず保存（scripts 62/64 形式）

### 1.2 確定した結果（句読点除去後CER、比較相手は TAPS SE）
| 項目 | 結果 |
|---|---|
| Whisper-small（学習に使用） | 0.230 → 0.196（−15.0%、10/10話者、p=0.002） |
| λ=0（追加学習のみ） | 0.235（改善なし）→ 改善は CE 損失由来 |
| Whisper-medium | −14.2%（10/10） |
| Whisper-base | −1.8%（6/10、n.s.） |
| XLS-R（CTC） | +25.8%（0/10）、λ=0比でも +20.9% |
| MMS-1B（CTC） | +1.0%（n.s.） |
| Enc L1（Whisper encoder特徴を気導に近づける） | 全ASRで TAPS SE とほぼ同じ（改善も悪化もしない。TAPS SEからほぼ変化していない） |
| FT Whisper（dl-box5で再学習） | SEなし 0.138 が最良。+CE-SE は平均 0.121 だが 5/10 話者で n.s.。CE-SE 単体は FT に及ばない |
| 句読点 | CE-SE は Whisper の句読点出力を抑制（文末句点率 94%→1%）。未正規化CERの改善幅の34%はこの効果 |
| 250 Hz 櫛状成分 | CE-SE 出力の 3–7.75 kHz に 250 Hz 間隔のピーク（λに単調増加、λ=0/TAPS/気導には無い）。ノッチ除去で +4.3%（改善の約1/4を担う）。250 Hz = 最深層の時間解像度（原因は仮説） |
| 帯域 | 改善の約83%は 4 kHz 以下の変化。TAPS SE の 4 kHz 以上を除去すると Whisper CER が 0.230→0.221 に改善（合成高域は Whisper に不利） |
| STOI/PESQ・encoder距離 | CE-SE で低下（STOI 0.892→0.792、PESQ 1.975→1.215）、encoder L1 は増加（0.200→0.326） |

### 1.3 評価（結論）
- **CE-SE を「提案手法」として押す方向は行き詰まり**
  1. 新規性が小さい: 凍結Whisper損失で前処理を学習する手法は Dissen et al., Interspeech 2024 が既出。ASR損失で学習したSEが学習外ASRで悪化することも Sato et al. 2025、D4AM（ICLR 2023）が報告済み（※後者2本は原典の該当箇所を未確認、エージェント報告ベース）
  2. 精度で FT / Adapter に届かない
  3. SE の唯一の利点（ASR非依存）を失う
- 卒論で新しいと言える点: 帯域が欠けた喉マイクでの確認、250 Hz 間隔の人工成分の分析（句読点による見かけの改善の定量化は評価上の注意点であり、新規性としては弱い）

### 1.4 成果物の状態
| 成果物 | 状態 |
|---|---|
| 卒論 `results/thesis_v3.js` → docx/pdf | 4ページ、確定値で全表・本文を記入済み。**ただし 2章以下の課題A（下記）を未反映** |
| ゼミ発表 `results/progress_slides_2026-09.pptx` | 7枚、台本入りノート。**PowerPointで手動編集済み → 生成スクリプトを再実行すると上書きされる**。句読点の話は画面から削除（ノートの想定質問に残置） |
| Artifact版スライド | 古い版（台本・先行研究の反映なし）。使わない |
| `CLAUDE.md` / `RESULTS_SUMMARY.md` | 確定値に更新済み |
| 実験スクリプト | 51（CE学習）・53（λ探索）・61（櫛周期性）・62（非Whisper ASR・帯域/ノッチ）・63（句読点正規化再採点）・64（全条件の再推論、hyp保存） |

### 1.5 計算環境
- 新GPU PC（RTX PRO 6000 96GB、共用WSL Ubuntu-24.04、`ssh labgpu` / `labgpu-ts`）へ移行中。旧 DL-Box5（RTX 4500 24GB）は以後使わない
- 未完了: DL-Box5 から checkpoints / data の移行、`whisper_ft_ct2` の変換（checkpoints 移行後）
- 96GB になったため、複数ASR・大型SSLを同時に載せる学習が現実的になった

---

## 2. 課題

### A. 卒論（thesis_v3）で直すべき点（優先度高）
1. **Sato 2025 と D4AM（ICLR 2023）の原典を確認**し、序論「十分に検証されていない」と考察を修正・引用追加。主張を「先行研究と同じ傾向を喉マイクで確認し、帯域・人工成分の観点で分析した」に弱める
2. 考察の「Enc L1 は認識器を選ばない代わりに改善も小さい」→「Enc L1 は TAPS SE からほぼ変化せず、改善も悪化もしなかった」
3. 学習ラベルに `<|ko|><|transcribe|>` が無い、log-mel のパディングが faster-whisper と異なる（結果を水増しする方向ではない）→ thesis_v3 に「前処理を一致させた」旨の記述が無いことは確認済み。限界として触れるかは任意
4. 限界の明記: 学習1回（seed分散未評価）、λ が探索範囲の端（10）、韓国語のみ、雑音なし音声のみ
5. 参考文献に追加候補: Sato 2025、D4AM、Ravenscroft 2024、Suzuki 2019（喉マイクKD）

### B. 研究上の未解決点
- 250 Hz 成分の原因（転置畳み込み仮説）は未検証
- FT + CE-SE の平均改善（0.138→0.121）が一部話者に偏る理由は未分析
- CE-SE がなぜ XLS-R で大きく悪化し MMS では変わらないのか（同じCTC系で差がある）

### C. 発表・資料
- スライド3（ASR損失のしくみ）: 「ŝ」を「SEの出力」に言い換え、CE損失カードを簡略化する案あり（手動編集版を直接書き換える必要があるため未実施）

---

## 3. 次にやること（修士に向けた研究）

目標: **重みや勾配を使えない ASR にも効く、ASR非依存の喉マイクSE**。

### Step 0（今すぐ・学習不要、新GPU PC）
- checkpoints / data を新GPU PC へ移行し、script 64 の一部条件で旧結果が再現するか確認
- **残差スイープ**: x = TAPS + α(CE − TAPS)（α∈[0, 1.5]）を 6 ASR で評価。Whisper で改善・XLS-R で悪化が単調なら「Whisper専用成分」を直接示せる
- **カットオフ × 認識器の行列**: TAPS / CE 出力に 3–8 kHz のローパス、櫛ノッチ版も全ASRで評価（現状 Whisper-small のみ）

### Step 1（本命）: SSL表現損失による喉マイクSE
- 損失: L = L1 + MR-STFT + λ · mean_{l>N/2} ‖φ_l(ŝ) − φ_l(s_air)‖²（Sato 2025 方式）
- φ: WavLM-Large（英語）と多言語SSL（mHuBERT-147 / w2v-BERT 2.0 等）を比較
- 実装: script 51 の CE 項を差し替え
- 評価: Whisper-base/small/medium、MMS、XLS-R。φ と同系列のASRは評価から除外（系列単位の leave-one-out）
- リスク: Enc L1 と同じく「ほぼ変化なし」に終わる可能性、英語SSLと韓国語の不一致、位置埋め込みの近道（対策 Soft-DTW、Meghanani & Hain 2026）

### Step 2: 系列の異なる複数損失 + leave-one-family-out
- 損失: recon + λ_W·CE(Whisper) + λ_C·CTC(MMS or XLS-R)、各項を勾配ノルムで正規化、ASR勾配と recon 勾配の衝突成分を射影（D4AM）
- 早期停止は学習に使っていない認識器の dev CER で判定、dropout 有効化（Olivier 2023）

### Step 3: ブラックボックスASR向け（先行研究の空白）
- (a) 複数SE出力（TAPS / CE / Enc L1 等）の発話単位選択を ASR 信頼度で決める（最安）
- (b) 複数ASRのCERを回帰する代理判別器を通してSEを学習（Sawata 2022 / MetricGAN+ 型）
- (c) 低次元の後処理パラメータ（帯域ゲイン・OA係数など）を CMA-ES でクラウドAPIのCERに直接最適化
- 喉マイクSE × 商用ASR API の研究は調査範囲で見当たらない

### その他の候補
- 喉マイクへのテスト時適応（SUTA、Whisper向けEM）: 先行例なし、ただし ASR の重みが必要
- 雑音環境での評価（喉マイク本来の利点）: データ収録が必要
- SSL トークン推定フロントエンド（Ashihara ICASSP 2026）、Mimi FT（Hauret 2025、HF公開モデルあり）

### 実験共通の原則
- 句読点除去CER（raw も併記）、学習に使っていない系列のASRで評価、話者単位検定、hyp 保存、可能なら seed 複数
