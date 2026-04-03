# 骨伝導マイク音声認識研究 — プロジェクト引き継ぎ

## 研究概要

**テーマ**: 喉マイク（骨伝導マイク）音声に対するSpeech Enhancement（SE）がWhisperのCERに与える影響の定量的評価、および喉マイク特化SEモデルの開発

**フェーズ1 RQ**: 「どの条件でSEが喉マイク音声のASR（Whisper）性能を悪化させるか」
**フェーズ2 RQ**: 「TAPSペアデータで特化ファインチューニングしたSEはCERを改善できるか」

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
  - `data/raw/taps/throat/{train,dev,test}/` — 喉マイク音声（ダウンロード済み・3.3GB）
  - `data/raw/taps/acoustic/{train,dev,test}/` — 気導マイク音声（ダウンロード済み）
  - `data/raw/taps/metadata_{train,dev,test}.csv` — split別メタデータ
  - `data/raw/taps/metadata_all.csv` — 全split統合メタデータ
  - 注意: 喉マイク・気導マイクともに16kHzで保存されている（データセット説明の「8kHz」は誤り、実測で確認済み）

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

## 次のステップ（フェーズ2）

**メイン: 喉マイク特化SEモデルのファインチューニング**

- TAPSの train split（40話者・4,000発話）を使用、dev split（1,000件）で学習中評価
- GTCRNを「喉マイク入力 → 気導マイク出力」でファインチューニング
- DNN用PCで学習（Chrome Remote Desktop経由）
- GitHubでコードを共有
- 評価: ファインチューニング後モデルのCER・STOI・PESQをフェーズ1と比較

**サブ候補（優先度低）**
- Whisper large-v3 での再評価（ASRモデル依存性の検証）
- DSPパラメータ感度分析（HPFカットオフ 100/200/300/500Hz）

---

## Git / GitHub

- ローカルgit初期化済み（main ブランチ）
- `gtcrn/` はサブモジュール（`git submodule update --init` で取得）
- `.gitignore` で除外済み: `data/raw/throat・acoustic/`・`data/processed/`・`venv/`
- GitHubリポジトリ: **未作成（ユーザーが手動作成してpush予定）**
  - 作成後: `git remote add origin <URL> && git push -u origin main`
  - DNN PC側: `git clone --recurse-submodules <URL>`

---

## 参照論文

1. **TASLP 2024**: Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?" (arXiv:2404.14860)
   - Artifact errorがSE逆効果の主因と特定
2. **Interspeech 2024**: Mawalim, Okada, Unoki (JAIST), "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?" (DOI: 10.21437/Interspeech.2024-129)
   - STOI改善・ASR悪化の乖離を実証
3. **"When De-noising Hurts"** (arXiv:2512.17562) — 参照のみ、主軸には据えない（未査読）

---

## 環境メモ

- Python 3.13（macOS / DNN PC両対応）
- `pip install faster-whisper jiwer soundfile scipy pystoi einops pesq python-pptx python-docx`
- datasets は `<4.0`（3.x系）を使用すること — 4.x系はtorchcodecが必要でWSL2環境で動作しない
- GTCRNはPyTorch `return_complex=True` API（旧APIは廃止済み）
- pesqはPython3.13でコンパイルに `sudo xcodebuild -license accept` が必要（macOS）
- DNN PC側はCUDA対応PyTorchを別途インストール
