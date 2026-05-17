# 骨伝導マイク音声認識研究 — プロジェクト引き継ぎ

## 研究概要

**テーマ**: 喉マイク（骨伝導マイク）音声に対するSpeech Enhancement（SE）がWhisperのCERに与える影響の定量的評価、および喉マイク特化モデルの開発

**フェーズ1 RQ**: 「どの条件でSEが喉マイク音声のASR（Whisper）性能を悪化させるか」
**フェーズ2 RQ**: 「Whisper smallを喉マイク音声でファインチューニングするとCERは改善するか」→ **Yes（10話者・1,000発話でCER 0.5436 → 0.1498、72.4%改善）**
**フェーズ3 RQ**: 「韓国語TAPS喉マイクで学習したモデルはフランス語VibraVoxに転移するか、またKDはAdapter単体より改善するか」→ **転移確認済み。KD実験はDNN PC実行待ち**

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
| 34_finetune_whisper_kd.py | **KD Adapter学習（DNN PC実行待ち）** | checkpoints/whisper_kd_adapter/ |

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

| モデル | Korean CER | French CER | 備考 |
|--------|-----------|-----------|------|
| A: Pretrained Whisper | 0.490 | 0.471 | baseline |
| B: Full FT | 0.150 | **0.388** | French最良 |
| C: Encoder-only Adapter | **0.147** | 0.448 | Korean最良・パラメータ効率最高 |
| D: Enc+Dec LoRA | 0.216 | 0.422 | |
| E: KD Adapter | — | — | **DNN PC実行待ち** |

- CER > 1.0 のハルシネーション除外（French: A=5.0%、B=1.2%、C/D=1.9%）
- 仮説「Encoder-only Adapterが仏語転移に有利」は棄却 → Full FTが仏語でも最良
- 発見: 韓国語FTが仏語ゼロショットを改善（0.471→0.388）→ 音響適応は言語非依存の示唆

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

### 新規性の正直な評価

**強み（本物の発見）**:
1. **三重パラドックス + 機構解明**（フェーズ1）: STOI↑PESQ↑CER↑が喉マイクドメインで一貫。4–8 kHz での −12.73 dB 削除というメカニズムを実測で特定。気導マイク向け先行研究（Ochiai/Mawalim）には存在しない発見
2. **NIR=1.81**（フェーズ2）: クリーン音声のみのFTがノイズ耐性まで向上させるという逆説的結果
3. **KD Adapter**（フェーズ3、実行待ち）: 学習時のみペアデータを使い推論時は喉マイク単体。BAF-Net（推論時もデュアルマイク必須）との明確な差別化

**懸念（査読者からの指摘リスク）**:
- フェーズ1+2は「既存手法を新ドメインで試しただけ」という批判に弱い
- KDが明確な改善を示せなければPhase3の貢献が消える
- BAF-Netとは評価条件が異なりすぎて直接比較不能

### 今後の方針
- **最優先**: KD Adapter（script 34）をDNN PCで実行し結果確認
- KDがAdapterを上回れば → 「推論時単体動作＋ペアデータ活用KD」という新規手法として論文化
- KDが改善しなければ → フェーズ1の三重パラドックス論文（分析論文として）に絞る
- **ターゲット会議**: ICASSP 2027（締め切り2026年9月頃）

---

## フェーズ3: KD Adapter（実装済み・DNN PC実行待ち）

### 設計思想

**問題**: BAF-Net（Interspeech 2025）は推論時に喉マイク+気導マイクの両方が必要 → 喉マイクを使う動機（高ノイズ環境）と矛盾
**提案**: 学習時のみTAPSペアデータを使い、推論時は喉マイク単体で動作するKD Adapter

### アーキテクチャ

```
学習時:
  喉マイク → Student (Whisper + Encoder Adapter) → Encoder出力S
  気導マイク → Teacher (Pretrained Whisper, 凍結) → Encoder出力T
  Loss = α×CE損失(デコーダ出力 vs テキスト) + β×KD損失(cosine/MSE: S vs T)

推論時:
  喉マイク単体 → Student → 書き起こし（気導マイク不要）
```

- Teacher: Pretrained Whisper-small（凍結）
- Student: Whisper-small + Encoder Adapter（Adapterのみ学習、全体の約0.3%）
- デフォルト: r=64, α=1.0, β=0.5, kd_loss=cosine
- 出力: `checkpoints/whisper_kd_adapter/`

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
- DNN PC（dl-box3, ~/kasuga/）にクローン・環境構築済み

---

## 参照論文

1. **TASLP 2024**: Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?" (arXiv:2404.14860)
   - Artifact errorがSE逆効果の主因と特定（OPD枠組み）
2. **Interspeech 2024**: Mawalim, Okada, Unoki (JAIST), "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?" (DOI: 10.21437/Interspeech.2024-129)
   - STOI改善・ASR悪化の乖離を実証
3. **TAPS論文**: Kim et al., arXiv:2502.11478, 2025
   - TAPSデータセットの詳細・収録条件
4. **BAF-Net**: Kim & Chung, Interspeech 2025, arXiv:2508.17336
   - Body-Acoustic Fusion Network: 喉マイク+気導マイクのデュアルマイクSE
   - **推論時も気導マイクが必須**（我々の単一マイク設定とは問題設定が異なる）
   - Whisper-large-v3-turboを固定ASRとして使用。合成ノイズ下のみ評価
   - CER: 22.2%（SNR -20dB）〜 16.7%（SNR +15dB）
5. **"When De-noising Hurts"** (arXiv:2512.17562) — 参照のみ、主軸には据えない（未査読）

---

## 環境メモ

- **macOS（開発・執筆用）**: Python 3.13
- **DNN PC (dl-box3, Linux)**: TITAN RTX × 2（各24GB VRAM）、CUDA 13.1
- `pip install faster-whisper jiwer soundfile scipy pystoi einops pesq python-pptx python-docx`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`（DNN PC）
- datasets は `<4.0`（3.x系）を使用 — 4.x系はtorchcodecが必要で動作しない
- GTCRNはPyTorch `return_complex=True` API（旧APIは廃止済み）
- pesqはPython3.13でコンパイルに `sudo xcodebuild -license accept` が必要（macOS）
