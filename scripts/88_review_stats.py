"""
スクリプト88: 査読（results/reviews/asj_manuscript_review_2026-10-17.md）への対応のための集計

  指摘5・6: 主な比較の効果量（発話平均 CER の相対変化）と、話者単位の Wilcoxon 符号順位検定（両側、丸めない p 値、
           比較ごとに認識器をまたいだ Holm 補正）。改善した話者数も出す
  指摘12 : 論文の CER（発話ごとに 1.0 で打ち切った平均）に加えて、通常の CER（総編集距離 / 総文字数）と、
           空白を除いた通常の CER を出す。どちらも句読点除去（VibraVox は小文字化も）後

使い方（GPU 不要）
    BCR_DATASET=taps     python scripts/88_review_stats.py
    BCR_DATASET=vibravox python scripts/88_review_stats.py
出力: results/review_response/stats_<dataset>.csv, cer_variants_<dataset>.csv
"""

import csv
import os
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import jiwer
import numpy as np
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
import dataset_cfg as DS
s64 = import_module('64_retranscribe_all')
s66 = import_module('66_lambda_cross_asr')
OUT = BASE_DIR / 'results' / 'review_response'

if DS.DATASET == 'taps':
    ASRS = ['whisper-small', 'whisper-base', 'whisper-medium', 'whisper-large-v3-turbo', 'whisper-ft', 'qwen3-asr-1.7b',
            'xlsr-korean', 'mms-1b-all', 'zipformer-ko']
    COMPARE = [('taps', 'fm_taps'), ('fm_taps', 'fm_taps_avg4'), ('fm_taps_avg4', 'fm_avg4_mavg_ce10'),
               ('ce10.0', 'fm_avg4_mavg_ce10'), ('ssl_wavlm_0.3', 'fm_avg4_mavg_ce10'), ('taps', 'ce10.0'),
               ('ce10.0', 'wise_a0.7'), ('fm_avg4_mavg_ce10', 'fm_avg4_lmavg_wise'), ('fm_taps_avg4', 'fm_avg4_lmavg_wise'),
               ('fm_taps_avg4', 'fm_fuse_lo'), ('fm_taps_avg4', 'fm_fuse_hi'), ('fm_taps_avg4', 'fm_fuse_notch'),
               ('fm_taps_avg4', 'fm_ceps_geo'), ('fm_taps_avg4', 'fm_ceps_env'), ('fm_taps_avg4', 'fm_ceps_fine'),
               ('fm_taps_avg4', 'fm_avg4_mavg_taps'), ('fm_taps_avg4', 'fm_avg4_mavg_ce0'), ('fm_taps_avg4', 'fm_avg4_mavg_ssl'),
               ('fm_avg4_mavg_taps', 'fm_avg4_mavg_ce10'), ('fm_avg4_mavg_ce0', 'fm_avg4_mavg_ce10'),
               # 学習し直した 2 回分（s2, s3）
               ('fm_s2_avg4', 'fm_s2_avg4_mavg_ce_s2'), ('fm_s3_avg4', 'fm_s3_avg4_mavg_ce_s3'),
               ('taps', 'ce_s2'), ('taps', 'ce_s3'), ('taps', 'fm_s2_avg4'), ('taps', 'fm_s3_avg4')]
    CER_CONDS = ['taps', 'ce10.0', 'ssl_wavlm_0.3', 'fm_taps', 'fm_taps_avg4', 'fm_avg4_mavg_ce10', 'wise_a0.7', 'fm_avg4_lmavg_wise']
else:
    ASRS = ['whisper-small', 'whisper-base', 'whisper-medium', 'whisper-large-v3-turbo', 'qwen3-asr-1.7b', 'xlsr-french', 'mms-1b-all']
    COMPARE = [('no_se', 'taps'), ('no_se', 'fm_taps_avg4'), ('no_se', 'fm_avg4_lmavg_wise'),
               ('no_se', 'vbx_se0'), ('vbx_se0', 'vbx_ce10'), ('vbx_ce10', 'vbx_wise'), ('no_se', 'fm_vbx_avg4'),
               ('vbx_se0', 'fm_vbx_avg4'), ('fm_vbx_avg4', 'fm_vbx_avg4_mavg_ce10'),
               ('fm_vbx_avg4_mavg_ce10', 'fm_vbx_avg4_lmavg_wise'), ('no_se', 'fm_vbx_avg4_mavg_ce10')]
    CER_CONDS = ['no_se', 'taps', 'fm_taps_avg4', 'fm_avg4_lmavg_wise', 'vbx_se0', 'vbx_ce10', 'vbx_wise', 'fm_vbx_avg4',
                 'fm_vbx_avg4_mavg_ce10', 'fm_vbx_avg4_lmavg_wise']


def holm(ps):
    """Holm 補正（nan は除いて補正し、元の位置に戻す）"""
    idx = [i for i, p in enumerate(ps) if not np.isnan(p)]
    order = sorted(idx, key=lambda i: ps[i])
    adj, run = [np.nan] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = s64.load_samples('test')
    refs = {s['utt']: s['text'] for s in samples}
    hyp = {}   # (asr, cond) -> {utt: (spk, hyp)}
    for a in ASRS:
        p = s66.hyp_path(a, 'test')
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['utt'] in refs:
                hyp.setdefault((a, r['cond']), {})[r['utt']] = (r['spk'], r['hyp'])
    full = {k: v for k, v in hyp.items() if len(v) == len(refs)}

    def capped(a, c):
        return {u: s64.capped(s64.norm(refs[u]), s64.norm(h)) for u, (_, h) in full[(a, c)].items()}

    rows = []
    for A, B in COMPARE:
        block = []
        for a in ASRS:
            if (a, A) not in full or (a, B) not in full:
                continue
            ca, cb = capped(a, A), capped(a, B)
            spk = defaultdict(lambda: [[], []])
            for u in ca:
                s = full[(a, A)][u][0]
                spk[s][0].append(ca[u]); spk[s][1].append(cb[u])
            ks = sorted(spk)
            ma = np.array([np.mean(spk[k][0]) for k in ks]); mb = np.array([np.mean(spk[k][1]) for k in ks])
            p = wilcoxon(ma, mb).pvalue if np.any(ma != mb) else 1.0
            ua, ub = np.mean(list(ca.values())), np.mean(list(cb.values()))
            block.append(dict(dataset=DS.DATASET, A=A, B=B, asr=a, cer_A=round(ua, 4), cer_B=round(ub, 4),
                              rel_pct=round(100 * (ub / ua - 1), 2), spk_B_better=int((mb < ma).sum()), n_spk=len(ks),
                              p_wilcoxon=float(f'{p:.3g}')))
        for r, adj in zip(block, holm([r['p_wilcoxon'] for r in block])):
            r['p_holm'] = float(f'{adj:.3g}')
        rows += block
    with open(OUT / f'stats_{DS.DATASET}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    vrows = []
    for c in CER_CONDS:
        for a in ASRS:
            if (a, c) not in full:
                continue
            us = sorted(full[(a, c)])
            R = [s64.norm(refs[u]) for u in us]; H = [s64.norm(full[(a, c)][u][1]) for u in us]
            keep = [i for i, r in enumerate(R) if r]
            R = [R[i] for i in keep]; H = [H[i] for i in keep]
            vrows.append(dict(dataset=DS.DATASET, cond=c, asr=a,
                              cer_capped_mean=round(np.mean([s64.capped(r, h) for r, h in zip(R, H)]), 4),
                              cer_corpus=round(jiwer.cer(R, H), 4),
                              cer_corpus_nospace=round(jiwer.cer([r.replace(' ', '') for r in R],
                                                                 [h.replace(' ', '') if h.replace(' ', '') else ' ' for h in H]), 4)))
    with open(OUT / f'cer_variants_{DS.DATASET}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(vrows[0])); w.writeheader(); w.writerows(vrows)
    print(f'saved → {OUT}/stats_{DS.DATASET}.csv, cer_variants_{DS.DATASET}.csv ({len(rows)} / {len(vrows)} rows)')


if __name__ == '__main__':
    main()
