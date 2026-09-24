# 骨伝導マイク音声認識研究 — プロジェクト引き継ぎ

## 研究概要

> ⚠️ 2026-09-24: フェーズ1〜4および旧CE実験のCERは句読点未正規化（raw）。Pretrained Whisperは句読点を出力しFT済みWhisperは出力しないため、FT比較にも句読点差が混入している。正規化後の確定値（test 1000発話）: Pretrained Whisper-small 0.446 → FT 0.138（−69%）。最新の確定値は末尾「CE損失 ASR-aware SE：確定値」節

**統一RQ**: 「喉マイク音声のASR改善には、音声強調（SE→ASR）とASRモデル適応（FT/PEFT）のどちらが有効か」
→ **結論: ASRモデル適応が圧倒的に有効（CER 0.54→0.15、72.4%改善）。ドメイン外SEは全条件で逆効果。ドメイン適合SEは未適応ASRを改善するが（0.47→0.25）、FT済みASRには逆効果かつFT単体（0.14）に及ばず**

**フェーズ1**: SE→ASRパイプライン評価（気導マイク向けSE） → 全34条件でCER悪化。三重パラドックス確認
**フェーズ2**: ASRモデル適応の評価 → Full FT: CER 72.4%改善、Adapter: 0.5%パラメータで同等精度
**フェーズ3**: 言語間転移・PEFT比較 → 韓国語FTが仏語も改善（0.47→0.39）。KD v1・v2失敗→打ち切り
**フェーズ4**: TAPSベースラインSE（喉マイク向け）×ASRモデル3種の公平比較完了 → 設計原則導出

**主要発見（フェーズ1）**:
- DSP・GTCRNともに全ノイズ条件でCERが悪化する
- GTCRNはSTOI↑・PESQ↑でありながらCER↑という三重パラドックスが観測（全ノイズ条件で一貫）
- Wilcoxon検定: 30検定中26件でp<0.05（4件はn.s.、-5dBの天井効果による）

**スペクトル分析による機構解明（フェーズ1補足）**:
- GTCRN適用後の帯域別エネルギー変化（話者p00・50発話）
  - 0–50 Hz: **+15.04 dB**（極低域にアーティファクト追加。ASR音韻識別には無関係な帯域）
  - 100 Hz以上: すべて削除方向
  - 1–4 kHz: −2.23 dB（フォルマント主要帯域を抑制）
  - 4–8 kHz: **−12.73 dB**（高域を壊滅的に削除）
- 喉マイクは4 kHz以上が構造的に無音（後述）→ GTCRNがさらに除去する逆方向の動作

---

## データセット

### TAPSデータセット詳細

- **正式名称**: Throat and Acoustic Pairing Speech Dataset
- **言語**: 韓国語（ニュース文・日常文）
- **HuggingFace**: `yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset`
- **収録マイク**: 喉マイク + 気導マイクの**ペア同時収録**
- **話者独立設計**: train / dev / test で話者IDが完全に重複しない

| Split | 話者数 | 発話数 | 総時間 | 話者ID例 |
|-------|--------|--------|--------|----------|
| train | 40名 | 4,000発話 | 10.2時間 | p01, p03, p06… |
| dev   | 10名 | 1,000発話 | — | — |
| test  | 10名 | 1,000発話 | 2.6時間 | p00, p02, p04, p21, p24, p26, p32, p38, p41, p56 |

各話者ちょうど100発話。平均発話長 約9秒。

**重要な注意：サンプルレートについて**
- 配布ファイルは 16 kHz で保存されているが、**実質 8 kHz 相当**
- スペクトログラムで 4 kHz に直線的なカットオフが現れる → 元の収録が 8 kHz（ナイキスト = 4 kHz）で行われ、16 kHz にアップサンプリングされた痕跡
- 3–4 kHz vs 4–5 kHz のエネルギー比が 43〜273倍（自然な物理ロールオフではなくフィルタによる急峻なカットオフ）
- 喉マイクの骨伝導特性による自然な高域減衰（2〜3 kHz以降）と、アップサンプリングによる 4 kHz 以上の完全無音が重なっている
- **重要な用語訂正**: 「4kHz以上が欠落」は不正確。正確には「8kHz収録（ナイキスト=4kHz）により、4kHz以上は収録時点から不在」。欠落（存在したものが失われた）ではなく不在（最初から存在しない）が正確
- TAPS論文（arXiv:2502.11478、p.3）に明記: "The accelerometer was configured with 8 kHz sampling rate"、Post-processingでFourier-based resamplingを確認済み

**TAPSデータセット引用状況（2026年5月時点）**
- 引用数: 5件（主にPOSTECHグループ）
  - BAF-Net（Kim & Chung, Interspeech 2025, arXiv:2508.17336）: **デュアルマイクSEフレームワーク**（後述）
  - LAU-Net（Song et al., 2025）: 喉マイク強調ネットワーク
  - その他3件: サーベイ・データセット論文
- 外部研究者によるASR応用はほぼ未着手のフロンティア領域

**気導マイクデータの使用状況**
- Whisper FT・CER評価：**未使用**（喉マイクのみ）
- STOI・PESQ計算：**参照信号として使用**（同発話の気導マイク版を参照）
- ベースラインCER（0.131）：**使用**（気導マイクで認識した場合の参照値）

---

## 実験条件（フェーズ1: 34条件）

| カテゴリ | 条件 |
|---|---|
| ベースライン | baseline_acoustic, baseline_throat |
| SE × クリーン | dsp_only/clean, gtcrn/clean |
| ノイズのみ（SE未適用）| no_se / {white,pink} / snr_{-5,+0,+5,+10,+20}dB |
| DSPのみ × ノイズ | dsp_only / {white,pink} / snr_{-5,+0,+5,+10,+20}dB |
| GTCRN × ノイズ | gtcrn / {white,pink} / snr_{-5,+0,+5,+10,+20}dB |

- **ノイズ**: 白色・ピンクノイズを後付けで混合
- **ASR**: faster-whisper small、言語=ko
- **指標**: CER・STOI・PESQ

---

## SEモデル

### DSP-only（scripts/03_apply_se.py）
- ハイパスフィルタ（Butterworth 6次、300Hz）
- プリエンファシス（0.97）
- RMS正規化

### GTCRN
- リポジトリ: `gtcrn/`（github.com/Xiaobin-Rong/gtcrn）、gitサブモジュール登録済み
- 重み: `gtcrn/checkpoints/model_trained_on_dns3.tar`
- 48.2Kパラメータの超軽量ニューラルSE
- DNS3（気導マイク・英語）で学習 → 喉マイクはドメイン外（これがパラドックスの原因）

---

## 完了済み作業

| スクリプト | 内容 | 出力 |
|---|---|---|
| 01_download_taps.py | TAPSデータ取得（test 50件のみ） | data/raw/taps/ |
| 01b_download_taps_full.py | TAPS全split取得（train/dev/test 計6,000件） | data/raw/taps/{throat,acoustic}/{train,dev,test}/ |
| 02_add_noise.py | ノイズ付加 | data/processed/noisy/ |
| 03_apply_se.py | SE適用 | data/processed/se/ |
| 05b_evaluate_gtcrn_noisy.py | 全34条件CER計測 | results/summary.csv |
| 06_visualize.py | CER可視化 | results/figures/ |
| 07_statistical_test.py | Wilcoxon符号順位検定 | results/stat_test.csv, per_sample_cer.csv |
| 08_stoi.py | STOI計測 | results/stoi_results.csv |
| 08b_pesq.py | PESQ計測（WBモード） | results/pesq_results.csv |
| 09_visualize_stoi_cer.py | STOI vs CER図 | results/figures/ |
| 14_visualize_pesq_stoi_cer.py | PESQ/STOI/CER統合可視化 | results/figures/ |
| 15_paper_asj_final.py | 日本音響学会フォーマット論文docx | results/paper_asj_final.docx |
| 16_slides.py | 研究紹介スライド（29枚） | results/slides.pptx |
| 20_finetune_gtcrn.py | GTCRNファインチューニング（失敗） | checkpoints/gtcrn_taps_finetuned.tar |
| 21_evaluate_finetuned.py | ファインチューニング後CER評価 | results/phase2_cer.csv, phase2_comparison.csv |
| 22_finetune_whisper.py | Whisper smallファインチューニング（epochs=20, batch=16, lr=1e-5） | checkpoints/whisper_throat_finetuned/ |
| 23_evaluate_whisper_ft.py | ファインチューニング済みWhisper評価（testセット話者p00 100件） | results/phase2_whisper_cer.csv, phase2_whisper_summary.csv |
| 25_overview_slides.py | 非専門家向け概要スライド（7枚） | results/overview_slides.pptx |
| 26_evaluate_multispeaker.py | 全10話者・4条件評価（DNN PCで実行） | results/multispeaker_*.csv |
| 27_spectrum_analysis.py | スペクトル分析・アーティファクト可視化 | results/figures/spectrum_analysis.png |
| 28_nir_evaluation.py | NIR（Noise Immunity Retention）計測 | results/nir_results.csv |
| 29_finetune_whisper_adapter.py | Encoder-only Adapter学習（r=64） | checkpoints/whisper_encoder_adapter/ |
| 30_finetune_whisper_lora.py | Encoder+Decoder LoRA学習（r=16） | checkpoints/whisper_encoder_decoder_lora/ |
| 31_download_vibravox.py | VibraVox（仏語）ダウンロード・前処理 | data/raw/vibravox/ |
| 32_evaluate_crosslingual.py | Cross-lingual評価（韓国語+仏語、5モデル対応） | results/crosslingual_*.csv |
| 33_postprocess_cap_cer.py | CER cap@1.0後処理・サマリー再計算 | results/crosslingual_summary.csv（更新） |
| 34_finetune_whisper_kd.py | KD Adapter v1学習（epoch6で停止、Best Dev CER=0.1448） | checkpoints/whisper_kd_adapter/ |
| 35_finetune_whisper_acoustic.py | 気導マイクWhisper FT（KD v2のTeacher用） | checkpoints/whisper_acoustic_finetuned/ |
| 36_finetune_whisper_kd_v2.py | KD Adapter v2学習（β=2.0、acoustic teacher、Best Dev CER=0.1407） | checkpoints/whisper_kd_adapter_v2/ |
| research_progress_slides.js | 研究進捗スライド（12枚、pptxgenjs） | results/research_progress.pptx |
| 37_evaluate_taps_se_baselines.py | TAPSベースラインSE（Demucs/SE-Conformer/TSTNN）×ASRモデル3種の公平比較（DNN PCで実行） | results/taps_se_comparison.csv |

### 成果物（文書）

| ファイル | 内容 |
|---|---|
| 2027年度_研究計画書_春日裕次_第6稿.docx | 大学院研究計画書（最新版・第6稿、矛盾修正・設計原則明示済み） |
| 2027年度_研究計画書.docx | 大学院研究計画書（旧版） |
| results/research_progress.pptx | 仮説検証フロー整理スライド（12枚、スピーカーノート付き） |
| results/slides.pptx | 研究紹介スライド（29枚） |
| results/overview_slides.pptx | 非専門家向け概要スライド（7枚） |
| results/figures/research_overview.png | SE vs ASR適応の2アプローチ比較図 |

---

## 主要結果

### フェーズ1（summary.csv より）

```
condition                   CER
baseline_acoustic           0.131  ← 気導マイク参照
baseline_throat             0.269  ← 喉マイク生音声
dsp_only/clean              0.416  ← DSP逆効果（クリーンでも）
gtcrn/clean                 0.296  ← GTCRNはクリーンで微改善

no_se/white/snr_+20dB       0.330
gtcrn/white/snr_+20dB       0.378  ← ノイズ下ではGTCRNも逆効果
gtcrn/white/snr_+0dB        1.214  ← SNR 0dBで最大悪化
```

**三重パラドックス（GTCRNノイズ条件）**
- STOI: no_se(0.58) → gtcrn(0.72) ↑改善
- PESQ: 全条件でGTCRN > no_se ↑改善
- CER:  no_se(0.88) → gtcrn(1.21) ↑悪化
→ 知覚品質指標と実用ASR性能が全ノイズ条件で乖離

### フェーズ2（2×2比較・全10話者・1,000発話）

| 条件 | CER | 備考 |
|------|-----|------|
| A: 未学習Whisper + No SE | 0.5436 | baseline |
| B: 未学習Whisper + GTCRN | 0.6347 | +0.091（SE逆効果） |
| C: FT済みWhisper + No SE | 0.1498 | **72.4%改善** |
| D: FT済みWhisper + GTCRN | 0.1860 | +0.036（SE悪化縮小） |

- FT: 教師あり学習（喉マイク音声 + 韓国語テキストのペア）
- 評価データは学習に未使用（話者独立・発話独立）
- 気導マイクデータはFT学習に使用していない

**NIR（Noise Immunity Retention）= 1.81**
- Pretrained Whisperのノイズ感度: 0.274
- FT済みWhisperのノイズ感度: 0.151
- NIR > 1.0 → クリーン音声のみでFTしてもノイズ耐性が向上（逆説的）

### フェーズ3（Cross-lingual評価・cap@1.0適用後）

**データ**: 韓国語TAPS test 1,000件 + 仏語VibraVox test 3,064件

| モデル | Korean CER | Korean std | French CER | French std |
|--------|-----------|-----------|-----------|-----------|
| A: Pretrained Whisper | 0.4896 | 0.190 | 0.4713 | 0.300 |
| B: Full FT | 0.1498 | 0.091 | **0.3883** | 0.285 |
| C: Encoder-only Adapter | 0.1469 | 0.082 | 0.4480 | 0.277 |
| D: Enc+Dec LoRA | 0.2159 | 0.120 | 0.4220 | 0.291 |
| **E: KD Adapter** | **0.1455** | **0.079** | 0.4524 | 0.284 |

- CER cap@1.0適用済み（French: A=5.0%、B=1.2%、C/D/E=1.9%がハルシネーション除外）
- 仮説「Encoder-only Adapterが仏語転移に有利」は棄却 → Full FTが仏語でも最良
- Korean最良: KD Adapter（0.1455）、stdも最小（0.079）→ 最も安定
- KD Adapterの仏語: Adapterよりわずかに悪化（0.4480→0.4524）→ KDが仏語転移を改善しない
- 発見: 韓国語FTが仏語ゼロショットを改善（0.471→0.388）→ 音響適応は言語非依存の示唆

### フェーズ4（TAPSベースラインSE × ASRモデル公平比較）

**データ**: TAPS test 1,000発話（全10話者）

| ASRモデル | SE | CER |
|---|---|---|
| Pretrained Whisper | No SE | 0.471 |
| Pretrained Whisper | SE:Demucs | 0.278 |
| Pretrained Whisper | **SE:SE-Conformer** | **0.253** |
| Pretrained Whisper | SE:TSTNN | 0.473 |
| **FT済みWhisper** | **No SE** | **0.136** |
| FT済みWhisper | SE:SE-Conformer | 0.143 |
| FT済みWhisper | SE:TSTNN | 0.137 |
| Encoder Adapter | No SE | 0.149 |
| Encoder Adapter | SE:SE-Conformer | 0.187 |

**主要な発見**:
- ドメイン適合SE（SE-Conformer）は未適応Whisperを大幅改善（0.47→0.25）するが、FT単体（0.136）に及ばない
- FT済みWhisperにSEを適用すると逆効果（0.136→0.143）：FTが学習した喉マイク信号特性をSEが変換するため
- TSTNN（マスキング型）はほぼ効果なし（0.471→0.473、誤差範囲）
- Demucs（マッピング型）は中程度の改善（0.471→0.278）

**OOD度による統一的解釈**:
- SEがASRを改善するかはASRモデルへの入力のOOD度に依存する
- 未適応Whisper（気導学習）に喉マイク（OOD）→ ドメイン適合SEがOODを緩和 → 改善
- FT済みWhisper（喉マイク適応済み）→ 喉マイクがin-domain → SEが信号を変えると悪化
- 先行研究（Ochiai/Mawalim）が気導マイクでSE逆効果を確認したのも同原理（気導Whisper×気導音声=in-domain）

---

## 設計原則（実験から導出）

**原則1（適用条件）**: ASR未適応モデルに対してはドメイン適合SEが一定の改善をもたらすが、ASR適応済みモデルに対してはSEが逆効果となる。

**原則2（最適化目標）**: 知覚品質指標（STOI・PESQ）の改善はCERの改善と必ずしも一致しない。ASR精度改善が目的なら、ASR損失を直接最適化するモデル適応が有効。

**原則3（組み合わせ）**: SEとASRモデル適応は相補的でなく競合的。両者を組み合わせるとモデル適応単体より精度が低下する（FT単体0.136 < FT+SE-Conformer 0.143）。

**原則4（軽量化）**: 全パラメータFTと同等の精度が0.5%パラメータ更新（Encoder-only Adapter）で達成できる。

---

## フェーズ2: Whisperファインチューニングの詳細

### 学習設定
- **入力**: 喉マイク音声のみ（`throat/train/`、4,000件、10.2時間）
- **教師ラベル**: 韓国語テキスト（metadata_train.csv の text列）
- **検証**: dev split でエポックごとにCER計算
- **モデル**: `openai/whisper-small`（HuggingFaceから取得）
- **前処理**: WhisperProcessor で log-mel spectrogram（80次元）に変換、言語=korean
- **epochs**: 最大20（early stopping patience=3、実際はepoch3で停止）
- **batch_size**: 16、lr=1e-5、warmup_steps=500、fp16=True
- **気導マイクデータ**: 学習に未使用

### GTCRNファインチューニング（失敗・参考）
- 試み: 「喉マイク入力 → 気導マイク出力」で再学習
- 失敗原因: GTCRNは「削る」モデル。存在しない高周波を「生成」するのは逆方向の操作
- 損失が収束せず（HybridLoss 97〜99）、CER≈1.0
- ログ: `checkpoints/training_log.csv`（Mac上で確認可能）

### Whisper FTのエポック停止について
- early stopping（patience=3）でepoch 3に停止したのは**過学習ではなく早期収束**と考えられる
- 根拠: pretrained Whisperは韓国語ASRを既に学習済み → 喉マイクドメイン適応は少ないepochで完了する
- FT学習ログ（epoch別CER）はDNN PC側に存在: `~/kasuga/bone_conduction_research/checkpoints/whisper_throat_finetuned/trainer_state.json`

---

## 研究の新規性・限界・今後の展望

### 新規性の正直な評価（2026年5月時点）

**各発見の新規性レベル**:

| 主張 | 新規性 | 正直な評価 |
|---|---|---|
| 三重パラドックス（喉マイク版） | **低** | 気導マイクで既知（Ochiai TASLP 2024、Mawalim Interspeech 2024）。喉マイクでの再確認に過ぎない |
| スペクトル分析による機構解明 | **中** | 4–8kHz −12.73dB削除の実測は喉マイク固有。ただし「SEがASRに有害な周波数を削る」という概念はOPD枠組みで既出 |
| FTでCER 72.4%改善 | **低** | Whisper FTは広く知られた手法。喉マイクでの実証は新しいが手法的新規性なし |
| NIR=1.81 | **低〜中** | FT後のベースラインCER差から来る数学的帰結の可能性。本質的にノイズ耐性が向上したかは要検証 |
| Cross-lingual transfer | **中** | 韓国語FTが仏語を改善（0.471→0.388）は興味深いが、音響適応の言語非依存性自体は他ドメインで報告あり |
| KD Adapter | **失敗** | v1・v2ともにKD損失が効かず、実質Adapterのみ。手法提案として成立しない |

**喉マイク × 電話音声（狭帯域ASR）との関係**:
- 電話音声（8kHz収録、4kHz以上不在）と信号レベルで類似
- ただし電話音声研究では「SEとASRの関係」はほぼ分析されていない
- 電話音声研究は「大量データFTで解決」が主流、SE+ASRの乖離は研究の隙間

**喉マイクの物理的特性**:
- 骨伝導は物理的ローパスフィルタ（3〜4kHz以上で大幅減衰）
- これはハードウェア限界であり、どの喉マイクでも共通
- TAPS 8kHz設定はこの物理限界に合わせた合理的選択

**検討・棄却したアプローチ**:
- Conv1D前アダプタ（mel入力レベル周波数アダプタ）: 検討→棄却
  - 理由: 高域binは既に≈0、ゲインを掛けても0×gain≈0。Full FTがConv1D重みで既に学習済み。160パラメータの線形変換は入力正規化と等価で新規性なし

**懸念（査読者・教授からの指摘リスク）**:
- 全フェーズを通じて「既存手法を新ドメインで試しただけ」という批判に弱い
- KDが2回失敗 → 手法提案としての柱がない
- BAF-Netとは評価条件が異なりすぎて直接比較不能
- 三重パラドックスは気導マイクで既知現象の追認
- **SE vs FT比較の公平性問題**: SEは音声品質改善が目的でASR最適化ではない。FTはASR損失を直接最適化。「FTが勝って当然」と言われるリスク
- **72.4%改善の解釈**: Pretrained WhisperのCER 0.54自体がOODで異常に高い。in-domain FTで下がるのは当然の帰結
- **「言語非依存音響適応」の主張の飛躍**: Full FTはDecoderも変更しており、Encoder-only Adapterの仏語CERは0.448でFull FTの0.388より大幅劣後。Encoder側の音響適応だけでは言語間転移は不十分であり、Decoder変更の貢献を分離できていない
- **日本語検証の具体性不足**: 収録計画（マイク・話者数・発話数・環境・テキスト）が未定

### 今後の方針（2027年度研究計画書 第6稿に準拠）
- **KDアプローチは打ち切り**（v1・v2で2回失敗）
- **TAPSベースラインSE比較実験は完了**（フェーズ4、results/taps_se_comparison.csv）
- **修士論文の方向性**: SE vs ASRモデル適応の体系的比較・効果解析
  - 比較実験は学部での予備実験として完了済み
  - 修士課程では効果解析・設計原則の体系化・査読論文発表が主目標
  - 「比較実験は済んでいるが、効果解析レポートの論文化は未完了」という防衛ロジックで計画書成立
- **研究計画書防衛の核心**: テーマ「比較・効果解析」のうち「比較」は予備実験として完了、「効果解析」（なぜ効果差が生じるかの体系的説明・査読論文化）は未完了 → 修士課程の目標として成立

### 喉マイク→気導マイク変換の関連研究（2026年5月調査）
- TAPS論文自身がベースライン変換実験を含む: Demucs(mapping) > TSTNN(masking) for CER
- arXiv:2508.02974 (Hauret et al., 2025): VibraVoxでMimi（Neural Audio Codec）をFTしリアルタイム変換
- 本研究との関係: 先行研究はSE側アプローチ、本研究はASR側適応。同一データでの直接比較が未踏

---

## フェーズ3: KD Adapter（実装・実行完了）

### 設計思想

**問題**: BAF-Net（Interspeech 2025）は推論時に喉マイク+気導マイクの両方が必要 → 喉マイクを使う動機（高ノイズ環境）と矛盾
**提案**: 学習時のみTAPSペアデータを使い、推論時は喉マイク単体で動作するKD Adapter

### アーキテクチャ・学習設定

```
学習時:
  喉マイク → Student (Whisper + Encoder Adapter) → Encoder出力S
  気導マイク → Teacher (Pretrained Whisper, 凍結) → Encoder出力T
  Loss = 1.0×CE損失(デコーダ出力 vs テキスト) + 0.5×KD損失(cosine: S vs T)

推論時:
  喉マイク単体 → Student → 書き起こし（気導マイク不要）
```

- Teacher: Pretrained Whisper-small（凍結）
- Student: Whisper-small + Encoder Adapter（Adapterのみ学習、全体の0.5%）
- r=64, α=1.0, β=0.5, kd_loss=cosine, lr=1e-3
- epoch6でearly stopping、Best Dev CER=0.1448
- 出力: `checkpoints/whisper_kd_adapter/`

### 学習ログ要約

| Epoch | CE損失 | KD損失 | Dev CER |
|---|---|---|---|
| 1 | 1.011 | 0.009 | 0.1756 |
| 2 | 0.420 | 0.003 | 0.1579 |
| **3** | **0.288** | **0.002** | **0.1448** ← best |
| 4 | 0.205 | 0.002 | 0.1504 |
| 5 | 0.146 | 0.002 | 0.1471 |
| 6 | 0.106 | 0.002 | 0.1455 → early stop |

KD損失はCE損失の約1/50と極めて小さく、実質的にAdapterのみの学習に近い状態だった。

### KD Adapter v2（改良版・scripts 35+36）

**v1の問題点と改善策**:
- Teacher（Pretrained Whisper）がTAPSドメインを知らない → 気導マイクFT済みWhisperをTeacherに変更
- β=0.5では弱い → β=2.0に拡大

**v2 学習ログ**:

| Epoch | CE損失 | KD損失 | KD/CE比 | Dev CER |
|---|---|---|---|---|
| 1 | 1.006 | 0.0030 | 0.003 | 0.1754 |
| 2 | 0.419 | 0.0009 | 0.002 | 0.1591 |
| 3 | 0.287 | 0.0008 | 0.003 | 0.1500 |
| **4** | **0.204** | **0.0007** | **0.004** | **0.1407** ← best |
| 5 | 0.148 | 0.0007 | 0.005 | 0.1434 |
| 6 | 0.109 | 0.0007 | 0.006 | 0.1491 |
| 7 | 0.080 | 0.0006 | 0.008 | 0.1527 → early stop |

**v2の結論: KDは依然として効いていない**
- KD/CE比: 0.3〜0.8%（v1の2%と同水準、β=2.0に上げても改善せず）
- Dev CER 0.1407 vs v1の0.1448 → 微改善だがKDの貢献かは不明
- Teacher変更・β拡大の両方を試しても本質的にKD損失の絶対値が小さすぎる
- **KDアプローチは2回試行して2回とも失敗。この方向での追加投資は打ち切り**

### BAF-Netとの差別化

| | BAF-Net | KD Adapter（提案） |
|---|---|---|
| 学習時 | ペアデータ使用 | ペアデータ使用 |
| **推論時** | **喉+気導の両方必要** | **喉マイク単体のみ** |
| ASRモデル | Whisper-large-v3-turbo | Whisper-small |
| 評価環境 | 合成ノイズ下のみ | クリーン+ノイズ(NIR) |

### BAF-Net 実態メモ（arXiv:2508.17336 精読済み）
- 正式名: Body-Acoustic Fusion Network
- BMS(喉)→SE-conformer（高域復元）+ AMS(気導)→DCCRN（ノイズ除去）→ FC-Netで動的融合
- 実際のCER: 22.2%（SNR -20dB）〜 16.7%（SNR +15dB）
- ASRモデル: Whisper-large-v3-turbo + Korean Zeroth FTを固定使用
- 評価は合成ノイズ下のみ（クリーン評価なし）
- 「CER 84.4%→24.4%」という数字は誤記。実際はAMS単体86.1%→BMS SE-conformer 24.4%の対比

---

## Git / GitHub

- GitHubリポジトリ: `https://github.com/kasugayuji0716-art/bone-conduction-research`
- `gtcrn/` はサブモジュール → クローン時は `git clone --recurse-submodules <URL>`
- `.gitignore` で除外済み: `data/raw/throat・acoustic/`・`data/processed/`・`venv/`
- GPU PC（RTX PRO 6000、WSL Ubuntu-24.04、`~/kasuga/`）にクローン・環境構築済み（詳細は「環境メモ」）

---

## 参照論文

1. **TASLP 2024**: Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?" (arXiv:2404.14860)
   - Artifact errorがSE逆効果の主因と特定（OPD枠組み）
2. **Interspeech 2024**: Mawalim, Okada, Unoki (JAIST), "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?" (DOI: 10.21437/Interspeech.2024-129)
   - STOI改善・ASR悪化の乖離を実証
3. **TAPS論文**: Kim et al., Scientific Data (Nature), 2026 (arXiv:2502.11478)
   - TAPSデータセットの詳細・収録条件
   - ベースラインSEモデル（Demucs / SE-conformer / TSTNN）の評価含む
   - 公開コード: github.com/yskim3271/taps-baselines
4. **BAF-Net**: Kim & Chung, Interspeech 2025, arXiv:2508.17336
   - Body-Acoustic Fusion Network: 喉マイク+気導マイクのデュアルマイクSE
   - **推論時も気導マイクが必須**（我々の単一マイク設定とは問題設定が異なる）
   - Whisper-large-v3-turboを固定ASRとして使用。合成ノイズ下のみ評価
   - CER: 22.2%（SNR -20dB）〜 16.7%（SNR +15dB）
5. **"When De-noising Hurts"** (arXiv:2512.17562) — 参照のみ、主軸には据えない（未査読）
6. **VibraVox**: Hauret et al., Interspeech 2024 (doi:10.21437/Interspeech.2024-1797)
   - フランス語体内収録音声188話者・45時間、CC-BY-4.0
7. **Adapter原論文**: Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," ICML 2019
8. **LoRA原論文**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022
9. **リアルタイム喉マイクSE**: Hauret et al., arXiv:2508.02974, 2025
   - VibraVoxでMimi（Neural Audio Codec）をFT、リアルタイム喉→気導変換

---

## 環境メモ

- **macOS（開発・執筆用）**: Python 3.13
- **GPU PC（2026-09-24〜、現行）**: NVIDIA RTX PRO 6000 Blackwell Max-Q（96GB、sm_120）、研究室の**共用Windows PC**上のWSL2
  - 自分専用ディストリ **Ubuntu-24.04**（Python 3.12）。既定ディストリ（Ubuntu-22.04）とWindows側の設定は他の人のものなので触らない
  - リポジトリ: `~/kasuga/bone-conduction-research`、venvは同ディレクトリの `venv/`
  - 仮想ディスクはFドライブに配置（Cドライブは空きが少ない）。データはWSL内（`~/`）に置き、`/mnt/c` 以下には置かない
  - Macからの接続: `ssh labgpu`（研究室Wi-Fi内）/ `ssh labgpu-ts`（Tailscale経由、学外から）。設定はMacの `~/.ssh/config`。鍵認証のみ
  - WSLはウィンドウを全部閉じると止まる → Ubuntu-24.04のウィンドウを開いたままにする。止まったらリモートデスクトップで `wsl -d Ubuntu-24.04`
  - sudoはパスワード入力が必要（Claudeからは実行できない）
  - 長時間の処理は必ずtmuxの中で実行する
- **旧DNN PC**: dl-box3（TITAN RTX × 2）、DL-Box5（RTX 4500 Ada、WSL）。DL-Box5のcheckpoints・TAPS公式重み・results・logsは2026-09-24に新GPU PCへ転送（VibraVoxとdata/processedはDL-Box5になく、必要なら再生成）。以後は使わない
- `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`（**Blackwellはcu128以上が必須**、cu121では動かない）
- `pip install faster-whisper jiwer soundfile librosa scipy pystoi einops pesq transformers accelerate peft "datasets<4" python-pptx python-docx`
  - librosaはdatasetsで音声をデコードするときに必要
- **faster-whisper（CTranslate2）がGPUで `libcublas.so.12 is not found` になる場合**: pipで入るcuBLAS/cuDNNを見つけられていない。venvの `bin/activate` の末尾で `LD_LIBRARY_PATH` に `venv/lib/python3.12/site-packages/nvidia/{cublas,cudnn}/lib` を追加する（GPU PCでは設定済み。venvを作り直したら再設定）
- CT2モデルの実体は `~/models/` に置き、スクリプトが参照する `/tmp/whisper-small-ct2`・`/tmp/whisper_small_ct2`・`/tmp/whisper_ft_ct2` へシンボリックリンクを貼る（`/tmp` はWSL再起動で消えるため、GPU PCでは `~/.bashrc` から `~/link_models.sh` を呼んで毎回貼り直している）
- 移行確認（2026-09-24、GPU PC）: whisper-small × test × No SE の句読点除去後CER = 0.4467（確定値0.446と一致）
- datasets は `<4.0`（3.x系）を使用 — 4.x系はtorchcodecが必要で動作しない
- GTCRNはPyTorch `return_complex=True` API（旧APIは廃止済み）
- pesqはPython3.13でコンパイルに `sudo xcodebuild -license accept` が必要（macOS）

---

## 修士課程の研究方向性（2026年7月確定）

**最終目標：喉マイク向け音声強調（SE）の精度向上によるASR性能改善**

### 研究の目標設定（重要）

**「気導マイク相当の音声を復元すること」ではなく「ASR性能（CER）が最も高くなる音声処理を設計すること」が目標。**

- 従来SE：STOI・PESQ・SI-SDRを最適化（気導音声との波形一致）
- 本研究：CERを最小化する処理を設計（ASR最適化SE）
- 根拠：三重パラドックス（STOI↑PESQ↑なのにCER↑）= 「気導相当に近い ≠ ASRに良い」

### 研究の軸
落合ら（TASLP 2024）の問題設定を喉マイクドメインへ拡張 + ASR-aware学習を提案

### 実験の進捗（2026年8月31日）

| スクリプト | 内容 | 結果 |
|---|---|---|
| 40 | OA後処理検証 | 喉マイクでは逆効果（ω↑→CER↓、逆方向） |
| 41 | ASR-aware SE再学習 | SI-SDR + λ×L1 encoder距離で学習 |
| 42 | CER比較評価 | ASR-aware(0.239) > SI-SDR only(0.283) |
| 43 | Encoder距離 vs CER相関分析 | **r=0.737（発話）、r=0.960（条件別）** |
| 44 | 全条件公平比較 | 同一ASRで5条件比較 |
| 45 | TAPS公式実装との比較 | forward不一致を発見（max diff=0.94） |
| 46 | TAPS公式SE CER測定 | **TAPS公式SE: CER=0.252** |

### 核心的発見

**① Encoder距離はCERの予測指標（r=0.737）**
- 従来指標（STOI・PESQ）→ CERと相関しない（三重パラドックス）
- Whisper encoder距離 → CERと強い相関
- 先行研究: Perceive & Predict（ICASSP 2023）が類似概念だがASR性能は未評価

**② ASR-aware学習でTAPS公式SEを上回る**

| 条件 | CER | 備考 |
|---|---|---|
| No SE | 0.472 | ベースライン |
| TAPS SE-Conformer（公式実装） | 0.252 | 従来手法 |
| **ASR-aware（L1, λ=1.0）** | **0.239** | **本研究提案（5.2%改善）** |
| FT Whisper（参考） | 0.136 | ASRモデル適応の上限 |

**③ TAPS公式実装との不一致問題**
- チェックポイントのハイパーパラメータはTAPS公式コードのデフォルトと異なる
- hidden=64, conformer_dim=512, ffn_dim=64, depth=4, conv_kernel=15（実際）
- hidden=32, conformer_dim=256, ffn_dim=256, depth=2, conv_kernel=31（公式デフォルト）
- 我々のforward計算はTAPS公式と異なる（max diff=0.94）
- SI-SDR再学習・ASR-awareは我々のforwardで学習→評価しており一貫性あり

### SEに焦点を置く理由
- SEはモデル非依存でどのASRにも使える
- FTはモデルごとに再学習が必要
- → SE改善は汎用的な価値がある

### CE損失 ASR-aware SE：確定値（2026-09-24、scripts 51/53/62/63/64）

> ⚠️ **評価方法の変更**: TAPSの正解テキストは句読点なし。旧CERは句読点未正規化で、CE-SEがWhisperの句読点出力を抑制する分（文末句点率 TAPS 94% → CE 1%）が改善に混入していた（raw改善の34%）。**以下はすべて句読点除去後のCER（nopunct, cap 1.0）、test 1000発話、話者単位Wilcoxon（n=10）**。v1（λ=2.0最良, 0.229）や raw の 0.252→0.200（−20.7%）は旧値。詳細: results/RESULTS_SUMMARY.md「確定値」節

**λ探索（Whisper-small、devで選択）**

| 条件 | dev | test | test(raw) | 句点率 |
|---|---|---|---|---|
| TAPS pretrained | 0.265 | 0.230 | 0.252 | 94% |
| λ=0（再構成のみ） | 0.265 | 0.235 | 0.257 | 95% |
| CE λ=1 | 0.244 | 0.208 | 0.228 | 90% |
| CE λ=2 | 0.231 | 0.200 | 0.216 | 67% |
| CE λ=5 | 0.232 | 0.199 | 0.203 | 0% |
| **CE λ=10（選択）** | **0.227** | **0.196** | 0.200 | 1% |
| CE only λ=10 | 0.238 | 0.203 | 0.205 | 0% |

- CE λ=10: TAPS比 **−15.0%**、10/10話者、p(spk)=0.002。λ=2〜10はdevでほぼ横ばい
- λ=0は改善なし（+2.2%）→ 改善はCE損失由来。CE onlyより recon併用が良い

**認識器ごとのCER（汎化）**

| SE | W-base | W-small* | W-medium | FT Whisper | MMS-1B | XLS-R |
|---|---|---|---|---|---|---|
| No SE | 0.655 | 0.446 | 0.337 | 0.138 | 0.497 | 0.584 |
| TAPS pretrained | 0.273 | 0.230 | 0.197 | 0.146 | 0.350 | 0.222 |
| Enc L1 (λ=5) | 0.277 | 0.230 | 0.197 | — | 0.353 | 0.220 |
| λ=0 | — | 0.235 | — | — | 0.358 | 0.231 |
| **CE λ=10** | 0.268 | **0.196** | **0.169** | 0.121 | 0.354 | **0.279** |

- W-medium −14.2%（10/10）、**W-base −1.8%（6/10, n.s.）**
- **XLS-R +25.8%（0/10）**、λ=0比でも+20.9% → CE損失がWhisper以外を悪化させる。MMS +1.0%（n.s.）
- Enc L1は全認識器でTAPSとほぼ同等（改善も悪化もしない）
- FT Whisper（dl-box5で再学習、dev最良epoch 2）: SEなし0.138が最良。+CE-SEは平均0.121だが5/10話者でn.s.。CE-SE単体はFTに及ばない

**帯域・櫛の分析（Whisper-small）**
- CE出力に250 Hz格子の櫛状ピーク（3–7.75 kHz、λに単調増加、script 61）。250 Hz = SE-Conformer最深層の時間解像度
- 櫛ノッチ: 0.196→0.204（+4.3%, 0/10）→ **櫛は改善の約1/4を担う**
- CE低域＋TAPS高域: 0.201（改善の約83%を保持）→ 改善の大部分は4 kHz以下
- TAPSの4 kHz以上を除去: 0.230→0.221（10/10改善）→ TAPSの高域補間はWhisperに不利

**知覚品質・encoder距離（v2）**: STOI TAPS 0.892 → CE 0.792、PESQ 1.975 → 1.215、Whisper-small encoder L1（対気導）0.200 → 0.326。CEは気導音声に近づかずWhisperが読みやすい別の信号を作る

**現状の評価（2026-09-24）**
- CE-SEを「提案手法」として押すのは行き詰まり: 凍結Whisper損失で前処理を学習する先行研究あり（Dissen et al., Interspeech 2024）、精度でFTに劣る、SEの長所（ASR非依存）を失う
- 卒論は分析としてまとめる（results/thesis_v3.*、案A「ASR損失で学習した喉マイク音声強調の効果と汎化範囲」、4ページ）
- 修士の本命: 重みを変えられないASR（クラウドAPI・共用ASR）向けの、ASRに依存しない喉マイクSE。SSL表現系の損失（Sato 2025等）が手がかり。Enc L1はTAPSからほぼ変化せず手がかりとしては弱い
- 評価の原則: テキスト正規化（句読点除去）必須、学習に使っていない系列のASRで評価、話者単位の検定、hypを必ず保存（scripts 62/64形式）

→ 詳細: results/RESULTS_SUMMARY.md, memory/thesis-robustness-status.md
