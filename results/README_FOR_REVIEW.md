# レビュー用の案内（2026-10-20 更新）

外部の査読・批評のための入口です。

## 論文（正本）
- `results/paper_asj/main.tex`：本文（LaTeX、日本音響学会講演論文の形式、XeLaTeX）
- `results/paper_asj/refs.bib`：参考文献
- `results/paper_asj/main.pdf`：ビルド済み PDF（6ページ）
- 図: `results/figures/method_overview.png`（scripts/87）、`results/figures/fusion_relative.png`（scripts/83）
- `results/paper_asj_2026-10.docx` は古い版（使わない）

## 結果のまとめ（数値の出典）
- `results/GENERATIVE_RESULTS_2026-09-30.md`：全実験の結果と読み取り（§1〜§7.16）。論文の数値はすべてここから
  - §7.2 融合、§7.4 比較手法、§7.5 分解、§7.7 内挿・ケプストラム、§7.8 失敗した案、§7.9 内挿の詰め、§7.10 VibraVox（そのまま）、§7.11 VibraVox（適応）、§7.12 融合の対照・別センサー・少量適応、§7.13 仏語 CTC、§7.14 学習し直し、§7.15 融合の蒸留、§7.16 学習し直した 2 回での融合B
- `results/CROSS_DOMAIN_IDEAS_2026-10-08.md`：他分野からの手法案と順位

## 認識結果の集計（CSV）
- `results/lambda_asr/summary.csv`、`pairs.csv`：TAPS test（`summary_dev.csv` は dev）。列 asr / cond / n / cer_nopunct。pairs は TAPS SE 比と話者単位の検定
- `results/eq_control/`：TAPS test の処理なし（`no_se`、論文 表1「なし」）と、ASR 損失なしの SE 同士の融合の対照（script 68）
- `results/lambda_asr_vibravox/`：VibraVox test（喉マイク、仏語）
- `results/lambda_asr_vibravox_lvl/`：VibraVox を TAPS の音量・帯域にそろえた入力
- `results/lambda_asr_vibravox_soft_in_ear/`、`results/lambda_asr_vibravox_forehead/`：VibraVox の別センサー（柔らかい耳栓型マイク、額の加速度センサー）
- 発話ごとの認識結果は各ディレクトリの `hyp_<認識器>.csv`（列 utt / spk / cond / hyp。認識器はファイル名で、dev はファイル名末尾の `_dev`）。正解は TAPS `data/raw/taps/metadata_*.csv`、VibraVox `data/raw/vibravox/metadata_*.csv`

## 査読への対応
- `results/reviews/`：査読（10-17）と再査読（10-19）
- `results/review_response/response_2026-10-17.md`、`response_2026-10-19.md`：対応表
- `results/review_response/stats_{taps,vibravox}.csv`：主な比較の効果量・改善した話者数・話者単位 Wilcoxon（未補正 p と、条件対ごとに認識器をまたいだ Holm 補正後 p）。`scripts/88_review_stats.py`
- `results/review_response/cer_variants_{taps,vibravox}.csv`：上限付き発話平均 CER / 通常の CER / 空白除去 CER
- `results/review_response/provenance.md`：重みのパス・更新時刻・SHA-256、生成モデルの最終 ckpt、音声長、評価件数。`scripts/90_provenance.py`

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
| fm_ceps_geo / fm_ceps_env / fm_ceps_fine | 対数振幅の融合 / 包絡だけ / 細部だけ融合 |
| fm_avg4_mavg_taps / fm_avg4_mavg_ce0 / fm_avg4_mavg_ssl | 生成×4 と TAPS SE / 再構成のみ SE（λ=0）/ SSL-MSE の融合（対照） |
| ce_s2, ce_s3 / fm_s2_avg4, fm_s3_avg4 / fm_s{2,3}_avg4_mavg_ce_s{2,3} | 学習し直した CE-SE / 生成×4 / 融合A（同じ回どうし） |
| wise_s2_a0.7, wise_s3_a0.7 / fm_s{2,3}_avg4_lmavg_wise_s{2,3} | 学習し直した CE-SE の内挿（α=0.7 固定）/ 融合B（同じ回どうし） |
| no_se | 処理なし |
| dfuse_l0 / dfuse_l1 | 融合A を SE-Conformer に蒸留（λ=0 / 1、script 89） |
| vbx_se0 / vbx_ce10 / vbx_wise / fm_vbx_avg4 / fm_vbx_avg4_mavg_ce10 / fm_vbx_avg4_lmavg_wise | VibraVox で適応した各条件 |

## 主なスクリプト
- `scripts/75_flowmatching_se.py`：生成モデルの追加学習・推論（NeMo）
- `scripts/51_train_se_ce_loss.py`：Whisper の CE 損失での SE 学習
- `scripts/78_whisper_select.py`：振幅融合（`fuse`、`--log` で対数振幅）
- `scripts/84_cheap_fusion_analysis.py`：重みの内挿・ケプストラム融合
- `scripts/66_lambda_cross_asr.py`：認識と集計（条件の一覧は SETS）
- `scripts/dataset_cfg.py`：データセットの切り替え（`BCR_DATASET`）
