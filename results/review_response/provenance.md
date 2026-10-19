# 確認記録（scripts/90_provenance.py が生成）

生成日時: 2026-10-20 00:10

## SE-Conformer の重み

| 名前 | パス | 大きさ | 更新時刻 | SHA-256（先頭16桁） |
|---|---|---|---|---|
| TAPS SE | `taps-baselines/pretrained/seconformer.th` | 48.4 MB | 2025-02-14 15:37 | 77a72482ea0f9941 |
| CE-SE（元） | `checkpoints/ce_v2_lambda_10.0/best.th` | 48.4 MB | 2026-09-21 01:28 | f2e3e66f85fbb296 |
| λ=0 SE | `checkpoints/ce_v2_lambda_0.0/best.th` | 48.4 MB | 2026-09-22 03:27 | 979d5e19bd2508ef |
| SSL-MSE | `checkpoints/ssl_wavlm_0.3/best.th` | 48.4 MB | 2026-09-30 01:21 | 371e3c24dfc94462 |
| 内挿 α=0.7 | `checkpoints/wise_a0.7/best.th` | 48.4 MB | 2026-10-08 09:44 | 33516ebda5b78d73 |
| CE-SE（再学習 s2） | `checkpoints/ce_s2/best.th` | 48.4 MB | 2026-10-18 05:43 | 73c7b1f0cb271f77 |
| CE-SE（再学習 s3） | `checkpoints/ce_s3/best.th` | 48.4 MB | 2026-10-18 06:57 | 12885637e23168c6 |
| VibraVox SE（再構成のみ） | `checkpoints/vbx_se0/best.th` | 48.4 MB | 2026-10-15 09:40 | 3ca55774ebc250e0 |
| VibraVox CE-SE | `checkpoints/vbx_ce10/best.th` | 48.4 MB | 2026-10-15 14:48 | 0a3f73280d915420 |
| VibraVox 内挿 | `checkpoints/vbx_wise/best.th` | 48.4 MB | 2026-10-15 16:49 | e826989c7b296622 |

## 生成モデル（NeMo）の重み

評価には各 tag の下で最も新しい .nemo（script 75 `find_nemo`）を用いる。TRAIN_DONE は学習コマンドが正常終了した印。

| 名前 | .nemo | 大きさ | 更新時刻 | SHA-256 | TRAIN_DONE | 最終 ckpt |
|---|---|---|---|---|---|---|
| 生成（元、TAPS） | `checkpoints/fm_taps/fm_taps/2026-09-29_22-22-12/checkpoints/fm_taps.nemo` | 1723.4 MB | 2026-09-30 06:13 | e1f9bbd67c867d8c | なし | fm_taps--val_pesq=2.0960-epoch=39-last.ckpt |
| 生成（再学習 s2） | `checkpoints/fm_s2/fm_s2/2026-10-18_07-17-19/checkpoints/fm_s2.nemo` | 1723.4 MB | 2026-10-18 10:42 | 253e7d00195c08e2 | あり | fm_s2--val_pesq=2.1073-epoch=39-last.ckpt |
| 生成（再学習 s3） | `checkpoints/fm_s3/fm_s3/2026-10-18_11-44-15/checkpoints/fm_s3.nemo` | 1723.4 MB | 2026-10-18 15:08 | 882296c7cefc3507 | あり | fm_s3--val_pesq=2.1203-epoch=39-last.ckpt |
| 生成（VibraVox） | `checkpoints/fm_vbx/fm_vbx/2026-10-15_16-49-58/checkpoints/fm_vbx.nemo` | 1723.4 MB | 2026-10-15 17:29 | 7eaaf820258e8396 | あり | fm_vbx--val_pesq=2.4478-epoch=7-last.ckpt |

## 音声の件数と長さ（喉マイク）

| データ | split | 発話数 | 最長（秒） | 15秒超 |
|---|---|---|---|---|
| TAPS | dev | 1000 | 17.9 | 9 |
| TAPS | test | 1000 | 16.6 | 9 |
| VibraVox | dev | 2523 | 10.6 | 0 |
| VibraVox | test | 3064 | 10.3 | 0 |

## 主な条件の評価件数（summary.csv の n）

### TAPS（`results/lambda_asr/summary.csv`）

| cond | mms-1b-all | qwen3-asr-1.7b | whisper-base | whisper-ft | whisper-large-v3-turbo | whisper-medium | whisper-small | xlsr-korean | zipformer-ko |
|---|---|---|---|---|---|---|---|---|---|
| taps | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| ce10.0 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| ssl_wavlm_0.3 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_taps | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_taps_avg4 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_avg4_mavg_ce10 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| wise_a0.7 | — | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_avg4_lmavg_wise | — | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_avg4_mavg_taps | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_avg4_mavg_ce0 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_avg4_mavg_ssl | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| ce_s2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| ce_s3 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_s2_avg4 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_s3_avg4 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_s2_avg4_mavg_ce_s2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| fm_s3_avg4_mavg_ce_s3 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |

### VibraVox（`results/lambda_asr_vibravox/summary.csv`）

| cond | mms-1b-all | qwen3-asr-1.7b | whisper-base | whisper-large-v3-turbo | whisper-medium | whisper-small | xlsr-french |
|---|---|---|---|---|---|---|---|
| no_se | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| taps | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| fm_taps_avg4 | — | 3064 | 3064 | 3064 | 3064 | 3064 | — |
| fm_avg4_lmavg_wise | — | 3064 | 3064 | 3064 | 3064 | 3064 | — |
| vbx_se0 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| vbx_ce10 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| vbx_wise | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| fm_vbx_avg4 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| fm_vbx_avg4_mavg_ce10 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |
| fm_vbx_avg4_lmavg_wise | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 | 3064 |

