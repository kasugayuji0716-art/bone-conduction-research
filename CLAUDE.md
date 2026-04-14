# 骨伝導マイク音声認識研究 — プロジェクト引き継ぎ

## 研究概要

**テーマ**: 喉マイク（骨伝導マイク）音声に対するSpeech Enhancement（SE）がWhisperのCERに与える影響の定量的評価、および喉マイク特化モデルの開発

**フェーズ1 RQ**: 「どの条件でSEが喉マイク音声のASR（Whisper）性能を悪化させるか」
**フェーズ2 RQ（修正後）**: 「Whisper smallを喉マイク音声でファインチューニングするとCERは改善するか」→ **Yes（CER 0.269 → 0.095、64.6%改善）**

**主要発見（フェーズ1）**:
- DSP・GTCRNともに全ノイズ条件でCERが悪化する
- GTCRNはSTOI↑・PESQ↑でありながらCER↑という三重パラドックスが観測（全ノイズ条件で一貫）
- Wilcoxon検定: 30検定中26件でp<0.05（4件はn.s.、-5dBの天井効果による）

---

## データセット

- **TAPS Dataset**: Korean paired throat mic + acoustic mic
  - HuggingFace: `yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset`
  - **全体規模**: 60話者・6,000発話・15.3時間（train:4,000 / dev:1,000 / test:1,000）
  - 注意: HuggingFace上のsplit名は `validation` ではなく `dev`
  - フェーズ1で使用: testセット50サンプル（話者p00のみ）
  - `data/raw/taps/throat/{train,dev,test}/` — 喉マイク音声（DNN PCにダウンロード済み）
  - `data/raw/taps/acoustic/{train,dev,test}/` — 気導マイク音声（DNN PCにダウンロード済み）
  - `data/raw/taps/metadata_{train,dev,test,all}.csv` — split別メタデータ
  - 注意: 喉マイク・気導マイクともに16kHzで保存（データセット説明の「8kHz」は誤り、実測で確認済み）

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
| 16_slides.py | 研究紹介スライド（25枚） | results/slides.pptx |
| 20_finetune_gtcrn.py | GTCRNファインチューニング（失敗） | checkpoints/gtcrn_taps_finetuned.tar |
| 21_evaluate_finetuned.py | ファインチューニング後CER評価 | results/phase2_cer.csv, phase2_comparison.csv |
| 22_finetune_whisper.py | Whisper smallファインチューニング（epochs=20, batch=16, lr=1e-5） | checkpoints/whisper_throat_finetuned/ |
| 23_evaluate_whisper_ft.py | ファインチューニング済みWhisper評価（testセット話者p00 100件） | results/phase2_whisper_cer.csv, phase2_whisper_summary.csv |

---

## 主要結果（summary.csv より）

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

---

## フェーズ2: GTCRNファインチューニング（失敗・方針転換）

### 試みた内容
- GTCRNを「喉マイク入力 → 気導マイク出力」で15エポック学習
- train 4,000件 / dev 1,000件 / batch_size=4 / lr=1e-4

### 失敗の原因
1. **タスクのミスマッチ**: GTCRNは「ノイズを除去（スペクトルを削る）」モデル。喉マイクに存在しない高周波を生成する（逆方向の操作）には不適
2. **損失が収束しない**: HybridLoss が97〜99で推移（正常時は1〜5程度）。喉・気導マイクのスペクトル差が大きすぎてmag MSEが巨大
3. **出力が壊れた音声に**: 事前学習重みから引き離されたまま停止 → CER≈1.0（完全に聞き取れない音声）

### 結果
```
condition              CER
baseline_throat        0.269
gtcrn/clean (original) 0.296
gtcrn/clean (ft)       1.051  ← 大幅悪化
```

### 方針転換: Whisperのファインチューニング（成功）
「音声を修正する」ではなく「ASRモデルを喉マイクに慣れさせる」アプローチへ変更

- **入力**: 喉マイク音声（throat/train, 4,000件）
- **教師ラベル**: 韓国語テキスト（metadata_train.csv）
- **モデル**: Whisper small（openai/whisper-small）
- **学習設定**: epochs=20, batch_size=16, lr=1e-5, warmup_steps=500, fp16=True
- **早期停止**: epoch5で停止（patience=3）、best checkpoint = epoch3

### フェーズ2 最終結果（testセット 話者p00 100件）

```
condition                          CER
baseline_throat (Phase1, 未学習)  0.2690
whisper_small_finetuned (Phase2)   0.0952  ← 64.6%改善
```

→ フェーズ2 RQ: **Whisper FTによりCERはbaseline_throatの約1/3に改善（0.269 → 0.095）**

---

## Git / GitHub

- GitHubリポジトリ: `https://github.com/kasugayuji0716-art/bone-conduction-research`
- `gtcrn/` はサブモジュール → クローン時は `git clone --recurse-submodules <URL>`
- `.gitignore` で除外済み: `data/raw/throat・acoustic/`・`data/processed/`・`venv/`
- DNN PC（dl-box3, ~/kasuga/）にクローン・環境構築済み

---

## 参照論文

1. **TASLP 2024**: Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?" (arXiv:2404.14860)
   - Artifact errorがSE逆効果の主因と特定
2. **Interspeech 2024**: Mawalim, Okada, Unoki (JAIST), "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?" (DOI: 10.21437/Interspeech.2024-129)
   - STOI改善・ASR悪化の乖離を実証
3. **"When De-noising Hurts"** (arXiv:2512.17562) — 参照のみ、主軸には据えない（未査読）

---

## 環境メモ

- **macOS（開発・執筆用）**: Python 3.13
- **DNN PC (dl-box3, Linux)**: TITAN RTX × 2（各24GB VRAM）、CUDA 13.1
- `pip install faster-whisper jiwer soundfile scipy pystoi einops pesq python-pptx python-docx`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`（DNN PC）
- datasets は `<4.0`（3.x系）を使用 — 4.x系はtorchcodecが必要で動作しない
- GTCRNはPyTorch `return_complex=True` API（旧APIは廃止済み）
- pesqはPython3.13でコンパイルに `sudo xcodebuild -license accept` が必要（macOS）
