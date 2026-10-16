# レビュー用の案内（2026-10-17）

外部の査読・批評のための入口です。

## 論文（正本）
- `results/paper_asj/main.tex`：本文（LaTeX、日本音響学会講演論文の形式、XeLaTeX）
- `results/paper_asj/refs.bib`：参考文献
- `results/paper_asj/main.pdf`：ビルド済み PDF（5ページ）
- 図: `results/figures/method_overview.png`（scripts/87）、`results/figures/fusion_relative.png`（scripts/83）
- `results/paper_asj_2026-10.docx` は古い版（使わない）

## 結果のまとめ（数値の出典）
- `results/GENERATIVE_RESULTS_2026-09-30.md`：全実験の結果と読み取り（§1〜§7.11）。論文の数値はすべてここから
  - §7.2 融合、§7.4 比較手法、§7.5 分解、§7.7 内挿・ケプストラム、§7.8 失敗した案、§7.9 内挿の詰め、§7.10 VibraVox（そのまま）、§7.11 VibraVox（適応）
- `results/CROSS_DOMAIN_IDEAS_2026-10-08.md`：他分野からの手法案と順位

## 認識結果の集計（CSV）
- `results/lambda_asr/summary.csv`、`pairs.csv`：TAPS test（`summary_dev.csv` は dev）。列 asr / cond / n / cer_nopunct。pairs は TAPS SE 比と話者単位の検定
- `results/lambda_asr_vibravox/`：VibraVox test（喉マイク、仏語）
- `results/lambda_asr_vibravox_lvl/`：VibraVox を TAPS の音量・帯域にそろえた入力
- `results/lambda_asr_vibravox_soft_in_ear/`：VibraVox の別センサー（柔らかい耳栓型マイク）。額センサーは評価中

条件名の対応（cond）
| cond | 論文での名前 |
|---|---|
| taps | TAPS SE |
| ce10.0 | CE-SE（λ=10） |
| ssl_wavlm_0.3 | SSL-MSE |
| fm_taps / fm_taps_avg4 | 生成×1 / 生成×4 |
| fm_avg4_mavg_ce10 | 融合A |
| wise_a0.7 | 内挿（α=0.7） |
| fm_avg4_lmavg_wise | 融合B |
| fm_fuse_lo / fm_fuse_hi / fm_fuse_notch | 4 kHz 未満だけ / 以上だけ / 櫛除去して融合 |
| fm_ceps_env / fm_ceps_fine | 包絡だけ / 細部だけ融合 |
| vbx_se0 / vbx_ce10 / vbx_wise / fm_vbx_avg4 / fm_vbx_avg4_mavg_ce10 / fm_vbx_avg4_lmavg_wise | VibraVox で適応した各条件 |

## 主なスクリプト
- `scripts/75_flowmatching_se.py`：生成モデルの追加学習・推論（NeMo）
- `scripts/51_train_se_ce_loss.py`：Whisper の CE 損失での SE 学習
- `scripts/78_whisper_select.py`：振幅融合（`fuse`、`--log` で対数振幅）
- `scripts/84_cheap_fusion_analysis.py`：重みの内挿・ケプストラム融合
- `scripts/66_lambda_cross_asr.py`：認識と集計（条件の一覧は SETS）
- `scripts/dataset_cfg.py`：データセットの切り替え（`BCR_DATASET`）
