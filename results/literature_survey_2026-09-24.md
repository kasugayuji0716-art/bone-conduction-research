# 文献調査：ASR非依存の喉マイクSEに使えるアプローチ（2026-09-24）

6観点の並列調査（各エージェントが原典を開いて確認。未読は「未確認」）の統合メモ。
※ 本メモの記述はエージェントの読解に基づく。論文に引用する前に原典で再確認すること。

## 本研究の結果と直接関係する先行研究（要引用・要確認）
| 文献 | 要点 | 本研究との関係 |
|---|---|---|
| Sato et al., Frontiers Signal Proc. 2025 (arXiv 2507.07631) | WavLM等の後半層表現のMSE（SSL-MSE）でSE学習。学習外Whisper-mediumでWER 12.2→9.6%。**ASR損失で学習したSEは学習外Whisperを悪化させた** | 本研究の「ASR損失SEは他ASRで悪化」と同じ現象を既に報告 → 新規性の主張を弱める必要 |
| Lee, Tsao et al., ICLR 2023 D4AM (arXiv 2311.16595) | 代理ConformerのASR損失＋回帰損失、勾配衝突を射影。**学習外のTransformer・RNN・wav2vec2-CTC・Google API・DNN-HMMで評価**。ASR損失のみ（CLSO）は汎化しない | 「別系列ASRで評価した先行研究はない」は誤り。held-out評価の模範 |
| Ravenscroft, Close et al., Interspeech 2024 (arXiv 2406.08914) | wav2vec2-CTCエンコーダ出力MSE損失 → 学習外Whisper large-v2でcpWER −8.5% | 表現損失の転移例 |
| Plantinga et al., arXiv 2112.06068 | mimic loss拡張、別途学習のKaldiで評価 | held-out評価 |
| Dissen et al., Interspeech 2024 | 凍結Whisper前段。base→large-v2へ転移（同系列内のみ） | 既引用 |
| Suzuki et al., Interspeech 2019 | 喉マイク特徴→気導BN特徴への蒸留、CER 14.3→12.9% | 喉マイクKDの先行研究（KD Adapter章で引用） |
| Huo et al., arXiv 2607.11157（未査読） | wav2vec2は強い補正、Whisperは弱い補正を好む | 認識器ごとの最適な処理強度 |
| de Oliveira et al., arXiv 2605.12107（未査読） | テキスト正規化で順位16〜18%入替 | 既引用 |
| AP-BWE, Lu et al., arXiv 2401.06387 | 8→16 kHz BWEはWhisper WERを改善せず、生成型は悪化 | 「TAPSの合成高域はWhisperに不利」と整合 |
| Olivier et al., ICLR 2023 (arXiv 2209.13523); Liao et al., arXiv 2606.05678 | 波形摂動はWhisper↔SSL-CTC間で転移しにくい、SSL特徴空間の摂動は転移する | CE-SEがXLS-Rに転移しない理由の説明 |

訂正: arXiv 2503.19591 は「Whisper→CTC転移」ではなく、代理DeepSpeech2→商用API/Whisper-large-v3（WavLM低層損失で転移向上）。

## 転移性の原理（④のまとめ）
- P1 最終出力（トークン）の損失は代理モデル固有の近道を学ぶ（句読点抑制・250 Hz櫛）。低〜中間層の表現損失は共有されやすい
- P2 複数モデル＋学習外モデルで早期停止
- P3 dropout有効・SAMなど平坦性
- P4 転移範囲は事前学習パラダイムとデータの共有で決まる（Whisper系 vs SSL-CTC系）
- P5 再構成損失は正則化。ASR勾配と衝突する成分を射影（D4AM）

## 候補アプローチ（優先順）
1. **学習不要の診断**（すぐできる）
   - 残差スイープ: x = TAPS + α(CE − TAPS), α∈[0,1.5] を6 ASRで評価（Huo流）
   - カットオフ×認識器行列: TAPS/CE出力に3–8 kHz LPF、全ASRで評価。櫛ノッチ版をXLS-Rでも
2. **SSL表現損失SE**（修士の本命候補）: L = L1+MRSTFT + λ·mean_{l>N/2}‖φ_l(ŝ)−φ_l(s_air)‖²、φ=WavLM-Large（または多言語SSL）。script 51 のCE項の差し替えで実装可。損失に使った系列は評価から除外
3. **異系列の複数損失＋leave-one-family-out**: CE(Whisper)+CTC(MMS or XLS-R)＋勾配ノルム正規化＋D4AM勾配射影、学習外ASRのdevで早期停止
4. **ブラックボックスASR向け**: (a) 複数SE出力の発話単位選択をASR信頼度で決める（安価）(b) 複数ASRのCERを模倣する代理判別器（Sawata/MetricGAN+型）(c) 低次元後処理パラメータのCMA-ES。喉マイクSE×商用APIの研究は見つからず＝空白
5. その他: 喉マイクへのテスト時適応（SUTA/Whisper EM、要重み）、SSLトークン推定フロントエンド（Ashihara ICASSP 2026）、Mimi FT（Hauret 2508.02974、HF公開あり）

## 主なURL
- https://arxiv.org/html/2507.07631 ・ https://arxiv.org/html/2311.16595 ・ https://arxiv.org/html/2406.08914
- https://arxiv.org/pdf/2112.06068 ・ https://arxiv.org/html/2607.11157 ・ https://arxiv.org/html/2401.06387
- https://arxiv.org/abs/2209.13523 ・ https://arxiv.org/html/2606.05678 ・ https://arxiv.org/html/2503.19591
- https://arxiv.org/abs/2110.05968 ・ https://arxiv.org/abs/2104.03538 ・ https://arxiv.org/html/2602.20967
- https://arxiv.org/abs/2203.14222 ・ https://arxiv.org/abs/2605.08186 ・ https://arxiv.org/abs/2602.04217
- https://www.isca-archive.org/interspeech_2019/suzuki19_interspeech.html ・ https://arxiv.org/html/2508.02974v1
