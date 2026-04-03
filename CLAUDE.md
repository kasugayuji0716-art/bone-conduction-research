# 骨伝導マイク音声認識研究 — プロジェクト引き継ぎ

## 研究概要

**テーマ**: 喉マイク（骨伝導マイク）音声に対するSpeech Enhancement（SE）がWhisperのCERに与える影響の定量的評価

**リサーチクエスチョン**: 「どの条件でSEが喉マイク音声のASR（Whisper）性能を悪化させるか」

**主要発見**: DSP・GTCRNともに全ノイズ条件でCERが悪化する。GTCRNはSTOIを改善しながらCERを悪化させるパラドックスが観測された。

---

## データセット

- **TAPS Dataset**: Korean paired throat mic + acoustic mic
  - HuggingFace: `yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset`
  - 使用: テストセット50サンプル（話者p00）
  - `data/raw/taps/throat/` — 喉マイク音声
  - `data/raw/taps/acoustic/` — 気導マイク音声（参照ベースライン）
  - `data/raw/taps/metadata.csv` — 正解テキスト（韓国語）

---

## 実験条件（34条件）

| カテゴリ | 条件 |
|---|---|
| ベースライン | baseline_acoustic, baseline_throat |
| SE × クリーン | dsp_only/clean, gtcrn/clean |
| ノイズのみ（SE未適用）| no_se / {white,pink} / snr_{-5,+0,+5,+10,+20}dB |
| DSPのみ × ノイズ | dsp_only / {white,pink} / snr_{-5,+0,+5,+10,+20}dB |
| GTCRN × ノイズ | gtcrn / {white,pink} / snr_{-5,+0,+5,+10,+20}dB |

- **ノイズ**: 白色・ピンクノイズを後付けで混合
- **ASR**: faster-whisper small、言語=ko
- **指標**: CER（文字誤り率）、STOI

---

## SEモデル

### DSP-only（scripts/03_apply_se.py）
- ハイパスフィルタ（Butterworth 6次、300Hz）
- プリエンファシス（0.97）
- RMS正規化

### GTCRN
- リポジトリ: `gtcrn/`（github.com/Xiaobin-Rong/gtcrn）
- 重み: `gtcrn/checkpoints/model_trained_on_dns3.tar`
- 48.2Kパラメータの超軽量ニューラルSE
- DNS3（気導マイク・英語）で学習 → 喉マイクはドメイン外

---

## 完了済み実験

| スクリプト | 内容 | 出力 |
|---|---|---|
| 01_download_taps.py | TAPSデータ取得 | data/raw/taps/ |
| 02_add_noise.py | ノイズ付加 | data/processed/noisy/ |
| 03_apply_se.py | SE適用 | data/processed/se/ |
| 05b_evaluate_gtcrn_noisy.py | 全34条件CER計測 | results/summary.csv |
| 06_visualize.py | CER可視化 | results/figures/4枚 |
| 08_stoi.py | STOI計測 | results/stoi_results.csv |
| 09_visualize_stoi_cer.py | STOI vs CER図 | results/figures/3枚 |

---

## 主要結果（summary.csv より）

```
condition                   CER
baseline_acoustic           0.131  ← 気導マイク参照
baseline_throat             0.269  ← 喉マイク生音声
dsp_only/clean              0.416  ← DSP逆効果（クリーンでも）
gtcrn/clean                 0.296  ← GTCRNはクリーンで改善

no_se/white/snr_+20dB       0.330
gtcrn/white/snr_+20dB       0.378  ← ノイズ下ではGTCRNも逆効果
gtcrn/white/snr_+0dB        1.214  ← SNR 0dBで最大悪化
```

**STOI vs CER パラドックス（GTCRNノイズ条件）**
- STOI: no_se(0.58) → gtcrn(0.72) ↑改善
- CER:  no_se(0.88) → gtcrn(1.21) ↑悪化
→ 知覚品質改善・ASR性能悪化の乖離が全ノイズ条件で一貫して観測

---

## 進行中

- `07_statistical_test.py` — nohupで実行中（完了時: results/per_sample_cer.csv, stat_test.csv）
  - 完了確認: `tail /tmp/stat_test_progress.log`

---

## 次のステップ（フェーズ2）

1. **統計検定結果確認**（完了待ち）
2. **Whisper large-v3 再評価** — ASRモデル依存性の検証
3. **PESQ計測** — `sudo xcodebuild -license` 後に `pip install pesq` → `08b_pesq.py`
4. **DSPパラメータ感度分析** — HPFカットオフ 100/200/300/500Hz
5. **DSP+GTCRN直列パイプライン**

---

## 参照論文

1. **TASLP 2024**: Ochiai et al., "Rethinking Processing Distortions: How Do They Affect the Downstream ASR Performance?" (arXiv:2404.14860)
   - Artifact errorがSE逆効果の主因と特定
2. **Interspeech 2024**: Mawalim, Okada, Unoki (JAIST), "Are Recent Deep Learning-Based Speech Enhancement Methods Ready to Confront Real-World Noisy Environments?" (DOI: 10.21437/Interspeech.2024-129)
   - STOI改善・ASR悪化の乖離を実証
3. **"When De-noising Hurts"** (arXiv:2512.17562) — 踏み台として参照、主軸には据えない（未査読）

---

## 環境メモ

- Python 3.13（macOS）
- `pip3 install faster-whisper jiwer soundfile scipy pystoi einops`
- GTCRNはPyTorch `return_complex=True` API（旧APIは廃止）
- pesqはPython3.13でコンパイルエラー → `sudo xcodebuild -license` が必要
