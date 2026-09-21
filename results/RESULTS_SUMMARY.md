# 実験結果サマリー（2026年9月20日時点）

## 研究概要

**目標**: 喉マイク音声のASR精度（CER）を改善する音声強調（SE）の設計
**提案手法**: Whisperの交差エントロピー（CE）損失をSE学習に直接組み込むASR-aware SE
**ベースライン**: TAPS公式SE-Conformerの事前学習済みモデル（追加学習なし）
**提案**: ベースラインの重みから初期化し、再構成損失+CE損失で追加学習

### 損失関数

```
再構成損失（TAPS論文と同一、追加学習時の正則化として使用）:
  L_recon = L1(SE出力, 気導音声) + MultiResolutionSTFT(SE出力, 気導音声)

Encoder距離損失（比較手法）:
  L_enc = L_recon + λ × L1(Whisper_encoder(SE出力), Whisper_encoder(気導音声))

CE損失（提案手法）:
  L_CE = L_recon + λ × CrossEntropy(Whisper(SE出力), 正解テキスト)
```

- Whisperは全パラメータ凍結、SEモデルのみ学習
- 勾配はCE損失 → decoder → encoder → mel → SEモデルへ逆伝播

### 手法の関係

```
TAPS pretrained（ベースライン）= SE-Conformerを再構成損失で200epoch学習（TAPS論文の成果物、追加学習なし）
    ↓ この重みから初期化して追加学習
Enc L1 = L_recon + λ×encoder距離で追加学習（比較手法）
CE     = L_recon + λ×CE損失で追加学習（提案手法）
```

- アーキテクチャは全て同一（SE-Conformer 12.1M）
- 違いは追加学習の損失関数のみ

---

## データセット: TAPS

- 正式名称: Throat and Acoustic Pairing Speech Dataset
- 韓国語喉マイク＋気導マイクペア同時収録
- Train: 40話者・4,000発話、Dev: 10話者・1,000発話、Test: 10話者・1,000発話
- 喉マイクは8kHz収録→16kHzアップサンプリング（4kHz以上は不在）
- 話者独立分割（train/dev/testで話者重複なし）

---

## 1. Encoder距離損失 λ探索（script 49、DNN PC dl-box3）

ASR: Whisper-small、評価: test 1,000発話

| 条件 | CER | vs TAPS pretrained |
|---|---|---|
| No SE | 0.471 | — |
| TAPS pretrained | 0.2519 | — |
| λ=0.0（再構成損失のみで追加学習） | 0.2549 | +0.003（悪化） |
| λ=0.1 | 0.2516 | -0.000 |
| λ=0.5 | 0.2490 | -0.003 |
| λ=1.0 | 0.2485 | -0.003 |
| λ=2.0 | 0.2490 | -0.003 |
| **λ=5.0** | **0.2362** | **-0.016（-6.2%）** |
| λ=10.0 | 0.2592 | +0.007（悪化） |
| λ=20.0 | 0.2381 | -0.014（-5.5%） |

- λ=5.0が最良
- λ=0.0（ASR損失なし）はTAPS pretrainedより微悪化 → 再構成損失だけの追加学習は効果なし
- 注: この結果はdl-box3で実行。dl-box5で再学習したEnc L1の値は異なる（後述）

---

## 2. CE損失 λ探索（script 53、DNN PC dl-box5）

ASR: Whisper-small、評価: test 1,000発話

| 条件 | CER | vs TAPS pretrained |
|---|---|---|
| No SE | 0.4692 | — |
| TAPS pretrained | 0.2518 | — |
| CE λ=0.1 | 0.2433 | -3.4% |
| CE λ=0.5 | 0.2360 | -6.3% |
| CE λ=1.0 | 0.2294 | -8.9% |
| **CE λ=2.0** | **0.2285** | **-9.3%** |
| CE λ=5.0 | 0.2404 | -4.5% |
| CE λ=10.0 | 0.2329 | -7.5% |

- CE λ=2.0が最良（CER 0.2285）
- Encoder距離（λ=5.0、CER 0.2362）より3.3%改善

---

## 3. 手法比較まとめ

| 手法 | 最良λ | CER | vs TAPS |
|---|---|---|---|
| TAPS pretrained（ベースライン） | — | 0.2518 | — |
| Encoder L1距離（比較手法） | 5.0 | 0.2362 | -6.2% |
| **CE損失（提案手法）** | **2.0** | **0.2285** | **-9.3%** |
| FT Whisper（参考上限） | — | 0.136 | — |

---

## 4. Ablation: 再構成損失の必要性（script 57、dl-box5）

CE λ=2.0で再構成損失（L1+STFT）の有無を比較

| 条件 | 損失関数 | CER |
|---|---|---|
| TAPS pretrained | L1+STFT（追加学習なし） | 0.2518 |
| CE only（再構成損失なし） | CE | 0.2374 |
| **CE+recon** | **L1+STFT+CE** | **0.2285** |

- CE損失のみでもTAPSを上回る（-5.7%）
- 再構成損失を加えるとさらに改善（-9.3%）
- 両損失は相補的に機能

---

## 5. 別ASRモデルでの評価（script 54、dl-box5）

CE-aware SE（Whisper-smallで学習）を3種のASRモデルで評価（汎用性検証）

| SE条件 | Whisper-base | Whisper-small | Whisper-medium |
|---|---|---|---|
| No SE | 0.676 | 0.471 | 0.360 |
| TAPS pretrained | 0.294 | 0.252 | 0.219 |
| Enc L1（λ=5.0） | 0.308 | 0.257 | 0.225 |
| **CE（λ=2.0）** | **0.284** | **0.229** | **0.206** |

- **CE損失は全3モデルでTAPS pretrainedを上回る** → Whisper-smallへの過学習なし
- **Enc L1はWhisper-base/smallでTAPSより悪化** → encoder表現への過学習
- 注: Enc L1のCER（0.257）はdl-box5で再学習したモデル。dl-box3のλ探索（0.2362）とは異なる

---

## 6. STOI・PESQ評価（script 55、dl-box5）

気導マイク音声を参照信号として計測

| 条件 | STOI | PESQ | CER (small) |
|---|---|---|---|
| No SE | 0.697 | 1.224 | 0.471 |
| TAPS pretrained | 0.892 | 1.975 | 0.252 |
| Enc L1（λ=5.0） | 0.871 | 1.750 | 0.257 |
| CE（λ=2.0） | 0.825 | 1.256 | 0.229 |

- CE-awareはSTOI/PESQがTAPSより低下するが、CERは改善
- 従来の三重パラドックス（STOI↑PESQ↑なのにCER↑）の逆パターン
- 知覚品質とASR精度は根本的に異なる最適化目標であることの証拠

---

## 7. 統計的検定（script 56、dl-box5）

### Wilcoxon符号順位検定

TAPS pretrained vs CE（λ=2.0）の1,000発話ペアCER比較

- **W = 119478.0、p = 2.75×10⁻²⁵（p < 0.001）**
- 統計的に極めて有意な差

### 話者別分析

全10話者でCE（λ=2.0）< TAPS pretrained：

| 話者 | TAPS CER | CE CER | 改善 |
|---|---|---|---|
| p00 | 0.2083 | 0.1972 | YES |
| p02 | 0.2377 | 0.2084 | YES |
| p04 | 0.2257 | 0.2080 | YES |
| p21 | 0.2291 | 0.2026 | YES |
| p24 | 0.2384 | 0.2057 | YES |
| p26 | 0.3065 | 0.2717 | YES |
| p32 | 0.2441 | 0.2215 | YES |
| p38 | 0.2738 | 0.2520 | YES |
| p41 | 0.2851 | 0.2641 | YES |
| p56 | 0.2697 | 0.2536 | YES |

**全10話者で一貫して改善（10/10）**

---

## 8. Encoder距離分析（script 58、dl-box5）★核心的発見

各SE条件について、3つのASRモデル（Whisper-base/small/medium）のencoder表現間L1距離を計測。
SE出力と気導音声を各encoderに入力し、出力ベクトルのL1距離を計算。

### Encoder L1距離

| SE条件 | base L1 | small L1 | medium L1 |
|---|---|---|---|
| No SE | 0.367 | 0.386 | 0.408 |
| TAPS pretrained | 0.157 | 0.201 | 0.226 |
| Enc L1（λ=5.0） | 0.163 | 0.199 | 0.230 |
| CE（λ=2.0） | 0.215 | 0.290 | 0.343 |

### Encoder距離 vs CER 相関係数

| ASRモデル | r（相関係数） | n |
|---|---|---|
| whisper-base | 0.9519 | 4 |
| whisper-small | 0.8200 | 4 |
| whisper-medium | 0.7258 | 4 |

### 核心的発見

**1. CE損失はencoder距離を増加させているのにCERは改善**

| SE条件 | small encoder L1 | small CER | 解釈 |
|---|---|---|---|
| TAPS pretrained | 0.201 | 0.252 | ベースライン |
| Enc L1 | 0.199（↓縮小） | 0.257（↑悪化） | 距離を縮めてもCER悪化 |
| CE | 0.290（↑増加） | 0.229（↓改善） | 距離が増えてもCER改善 |

→ 「気導音声にencoder空間で近づける = ASR改善」は**誤り**

**2. Enc L1は学習したWhisper-smallでもほとんど距離を縮めていない**
- base: 0.157 → 0.163（+0.006、増加）
- small: 0.201 → 0.199（-0.002、ほぼ変化なし）
- medium: 0.226 → 0.230（+0.004、増加）

**3. CE損失は「気導音声に似た音声」ではなく「ASRが正しく読める別の表現」を生成**
- CE損失のSE出力はencoder空間で気導音声からむしろ離れている
- それにも関わらずCERは全ASRモデルで最良
- 気導音声の復元を目標とする従来SEの前提が、ASR応用には最適でない

**4. encoder距離とCERの相関はモデルが大きいほど弱い**
- base: r=0.95 → medium: r=0.73
- 大きいモデルほどencoder距離がCERの予測指標として不十分

---

## 論文の構成（thesis.docx）

| セクション | 内容 |
|---|---|
| 1. はじめに | 背景・課題・提案の概要 |
| 2.1 ベースライン | TAPS SE-Conformer（追加学習なし） |
| 2.2 提案手法 | (a) Encoder距離損失 (b) CE損失 |
| 3. 実験条件 | TAPS・SE-Conformer・評価設定 |
| 4.1 λ探索+ablation | 表1 |
| 4.2 別ASRモデル | 表2（CER） |
| 4.3 Encoder距離分析 | 表3（距離）★核心 |
| 4.4 知覚品質評価 | 表4（STOI/PESQ/CER） |
| 5. 考察とまとめ | 結論 |
| 参考文献 | 4件 |

---

## 学習設定

| 項目 | Enc L1（script 47） | CE損失（script 51） |
|---|---|---|
| SEモデル | SE-Conformer（12.1M） | SE-Conformer（12.1M） |
| 初期化 | TAPS pretrained | TAPS pretrained |
| ASRモデル（凍結） | Whisper-small encoder | Whisper-small（full） |
| Optimizer | Adam | Adam |
| lr | 3×10⁻⁴ | 3×10⁻⁴ |
| betas | (0.9, 0.99) | (0.9, 0.99) |
| batch_size | 4 | 4 |
| epochs | 50（early stopping） | 50（early stopping） |
| patience | 5 | 5 |
| AMP | なし | Whisper forward部分のみ |

### SE-Conformer設定（TAPS公式チェックポイント準拠）

```
hidden=64, conformer_dim=512, conformer_ffn_dim=64,
conformer_depth=4, depthwise_conv_kernel_size=15
```

### 評価設定

- ASR: faster-whisper（CTranslate2形式）
- 言語: 韓国語（ko）
- beam_size: 5
- CER: jiwer.cer、cap@1.0

---

## 実行環境

| 実験 | マシン | GPU |
|---|---|---|
| Enc L1 λ探索 | dl-box3 | TITAN RTX ×2（各24GB） |
| CE λ探索・cross-ASR・STOI/PESQ・統計検定・ablation・encoder距離分析 | dl-box5 | RTX 4500 Ada（24GB） |

### 注意: マシン間のCER差異
- Enc L1 λ=5.0: dl-box3で0.2362、dl-box5で0.257（再学習で収束が異なる）
- 論文のcross-ASR・STOI/PESQ・encoder距離は全てdl-box5のモデルで統一
- dl-box3のλ探索結果（0.2362）は論文の表2-4には使用しない

---

## スクリプト一覧

| スクリプト | 内容 |
|---|---|
| 47_train_se_official.py | Encoder距離損失でのASR-aware SE学習 |
| 48_eval_official.py | 公式forward評価 |
| 49_lambda_search.py | Encoder距離のλ探索（学習+評価） |
| 51_train_se_ce_loss.py | CE損失でのASR-aware SE学習（AMP対応） |
| 52_eval_ce_loss.py | CE損失モデルCER評価 |
| 53_ce_lambda_search.py | CE損失のλ探索（学習+評価） |
| 54_eval_cross_asr.py | 別ASRモデル（base/small/medium）でのCER評価 |
| 55_eval_stoi_pesq.py | STOI・PESQ評価 |
| 56_statistical_analysis.py | Wilcoxon検定・話者別分析 |
| 57_eval_ablation.py | CE only vs CE+recon ablation評価 |
| 58_encoder_distance_analysis.py | Encoder距離分析（3モデル×4条件） |

---

## 参考文献

1. T. Ochiai et al., "Rethinking Processing Distortions: Disentangling the Impact of SE Errors on ASR," IEEE/ACM Trans. ASLP, vol. 32, 2024
2. C.O. Mawalim, S. Okada, and M. Unoki, Proc. Interspeech, DOI: 10.21437/Interspeech.2024-129, 2024
3. G. Close, W. Ravenscroft, T. Hain, and S. Goetze, "Perceive and Predict," Proc. ICASSP, 2023
4. Y. Kim et al., "TAPS Dataset," Scientific Data (Nature), DOI: 10.1038/s41597-026-07268-2, 2026

---

# ★ v2 修正版結果（2026年9月22日）

## v1→v2の修正内容

| # | 修正 | 影響 |
|---|---|---|
| 1 | WhisperLogMel: Slaney mel scale/norm + 波形パディング | 前処理がWhisper標準に一致 |
| 2 | CE教師ラベル: EOS保持、BOS除去 | 正しい教師信号 |
| 3 | 15秒超の発話を除外（切り出し→除外） | 音声-テキスト不一致解消 |
| 4 | STFT損失パラメータをTAPS公式に合わせる | 公平な比較 |
| 5 | λ選択をdevで行い、testは最終評価のみ | 正しい実験プロトコル |

## v2 CE損失λ探索（devで選択→testで評価）

| 条件 | dev CER | test CER | vs TAPS test |
|---|---|---|---|
| No SE | 0.431 | 0.471 | — |
| TAPS pretrained | 0.285 | 0.252 | — |
| CE λ=0.1 | 0.278 | 0.241 | -4.3% |
| CE λ=0.5 | 0.266 | 0.232 | -7.8% |
| CE λ=1.0 | 0.262 | 0.228 | -9.4% |
| CE λ=2.0 | 0.246 | 0.216 | -14.3% |
| CE λ=5.0 | 0.235 | 0.202 | -19.7% |
| **CE λ=10.0** | **0.232** | **0.200** | **-20.7%** |

best on dev: λ=10.0

## v2 Cross-ASR（script 54）

| SE条件 | W-base | W-small | W-medium |
|---|---|---|---|
| No SE | 0.671 | 0.470 | 0.360 |
| TAPS pretrained | 0.294 | 0.252 | 0.219 |
| Enc L1 (λ=5.0) | 0.298 | 0.251 | 0.218 |
| **CE (λ=10.0)** | **0.287** | **0.200** | **0.189** |

## v2 STOI/PESQ（script 55）

| 条件 | STOI | PESQ | CER(small) |
|---|---|---|---|
| No SE | 0.697 | 1.224 | 0.471 |
| TAPS pretrained | 0.892 | 1.975 | 0.252 |
| Enc L1 (λ=5.0) | 0.878 | 1.775 | 0.251 |
| CE (λ=10.0) | 0.792 | 1.215 | 0.200 |

## v2 Encoder距離分析（script 58）

| SE条件 | base L1 | small L1 | medium L1 |
|---|---|---|---|
| No SE | 0.367 | 0.386 | 0.408 |
| TAPS pretrained | 0.157 | 0.200 | 0.226 |
| Enc L1 (λ=5.0) | 0.157 | 0.192 | 0.224 |
| CE (λ=10.0) | 0.236 | 0.326 | 0.346 |

核心的発見が再確認: CE損失はencoder距離を大幅に増加させているのにCER最良

## v2 統計検定（script 56）
- Wilcoxon: W=43350.5, p=3.38×10⁻⁹³

## v2 Ablation（script 57）
- CE+recon (λ=10.0): CER 0.200
- CE only (λ=10.0): CER 0.205

## v1 → v2 比較

| 指標 | v1（バグあり） | v2（修正後） |
|---|---|---|
| best λ | 2.0（test選択） | **10.0（dev選択）** |
| test CER | 0.229 | **0.200** |
| vs TAPS | -9.3% | **-20.7%** |
| Wilcoxon p | 2.75×10⁻²⁵ | **3.38×10⁻⁹³** |

v2チェックポイント名: `ce_v2_lambda_*`, `enc_v2_lambda_5.0`, `ce_v2_only_lambda_10.0`
