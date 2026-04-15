# 骨伝導マイク音声認識研究 — プロジェクト引き継ぎ

## 研究概要

**テーマ**: 喉マイク（骨伝導マイク）音声に対するSpeech Enhancement（SE）がWhisperのCERに与える影響の定量的評価、および喉マイク特化モデルの開発

**フェーズ1 RQ**: 「どの条件でSEが喉マイク音声のASR（Whisper）性能を悪化させるか」
**フェーズ2 RQ**: 「Whisper smallを喉マイク音声でファインチューニングするとCERは改善するか」→ **Yes（10話者・1,000発話でCER 0.5436 → 0.1498、72.4%改善）**

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

---

## 研究の新規性・限界・今後の展望

### 新規性
1. **喉マイク特有ドメインでのSE逆効果の系統的実証**
   - Ochiai・Mawalimらの先行研究は気導マイク対象。高周波が構造的に欠落したドメインでの検証は本研究が初
2. **三重パラドックスの定量化**
   - STOI↑PESQ↑CER↑が全ノイズ条件で一貫することを30条件・統計検定で示した
3. **スペクトル分析によるメカニズムの特定**
   - 4–8 kHz での −12.73 dB 削除という具体的な原因を実測で特定

### 限界
1. 使用したSEモデル（GTCRN）はドメイン外学習済みモデルのみ。喉マイク向け再設計SEとの比較なし
2. 評価データはTAPS（韓国語）のみ。他言語・他データセットへの汎化性は未検証
3. Whisper FTはクリーン音声のみ。ノイズ下での頑健性評価が不足
4. ペアデータ（気導マイク）を学習に活用していない
5. 音素レベルの誤認識分析なし

### 今後の展望
1. ドメイン適合型SE（Kim et al. 2025 BAF-Net）との組み合わせ
2. Whisper large-v3 でのFT検証
3. ノイズ下でのFT評価（MUSAN・DEMANDなど実録音ノイズ）
4. Knowledge Distillation（気導マイクモデル → 喉マイクモデル）
5. 音素誤認識分析による学習戦略の改善

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
4. **BAF-Net**: Kim & Chung, Interspeech 2025
   - TAPSでドメイン適合型SE-conformer → CER 84.4%→24.4%
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
