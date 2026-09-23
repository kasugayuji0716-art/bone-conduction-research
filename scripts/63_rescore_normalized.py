"""
スクリプト63: 句読点・空白を正規化したCERで再採点
背景: TAPS正解テキストは句読点なし。従来のCERは句読点未正規化のため、
      Whisperの句読点出力の有無がCERに混入していた疑い（コード監査で発見）。

入力: results/robustness_per_utt.csv（script 62、hyp列を含む）
      書き込み中でも読める（読み取り専用、不完全な最終行は無視）
出力: results/rescore_normalized.csv

CER 3種:
  raw      : 従来通り（jiwer.cer, cap 1.0）
  nopunct  : 両側から句読点（Unicode P*）を除去、空白を1つに
  nospace  : さらに空白も除去（分かち書きの差も無視）

使い方（Mac / DNN PC どちらでも）:
    python scripts/63_rescore_normalized.py
    python scripts/63_rescore_normalized.py --csv results/robustness_per_utt_trial.csv
"""

import argparse, csv, re, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np
from jiwer import cer
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
META = BASE_DIR / 'data' / 'raw' / 'taps' / 'metadata_test.csv'


def norm(s, drop_space=False):
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.replace(' ', '') if drop_space else s


def capped(ref, hyp):
    return min(cer(ref, hyp), 1.0) if ref else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(BASE_DIR / 'results' / 'robustness_per_utt.csv'))
    args = ap.parse_args()

    refs = {f"{r['speaker_id']}_{r['sentence_id']}": r['text']
            for r in csv.DictReader(open(META, encoding='utf-8'))}

    scores = defaultdict(dict)   # (asr, cond) -> utt -> (raw, nopunct, nospace)
    spk_of = {}
    punct_end = defaultdict(list)
    bad = 0
    with open(args.csv, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('hyp') is None or r['utt'] not in refs or r.get('cond') is None:
                bad += 1
                continue
            ref, hyp = refs[r['utt']], r['hyp']
            scores[(r['asr'], r['cond'])][r['utt']] = (
                capped(ref, hyp), capped(norm(ref), norm(hyp)), capped(norm(ref, 1), norm(hyp, 1)))
            spk_of[r['utt']] = r['spk']
            punct_end[(r['asr'], r['cond'])].append(hyp.rstrip()[-1:] in '.?!。' if hyp.strip() else False)
    if bad:
        print(f'(skipped {bad} incomplete rows)')

    out = []
    print(f'\n{"ASR":<14} {"condition":<22} {"n":>5} {"raw":>7} {"nopunct":>8} {"nospace":>8} {"文末句点":>7}')
    for k in sorted(scores):
        v = np.array(list(scores[k].values()))
        m = v.mean(axis=0)
        pe = np.mean(punct_end[k])
        out.append(dict(asr=k[0], cond=k[1], n=len(v), cer_raw=round(m[0], 4),
                        cer_nopunct=round(m[1], 4), cer_nospace=round(m[2], 4),
                        final_punct_rate=round(pe, 3)))
        print(f'{k[0]:<14} {k[1]:<22} {len(v):>5} {m[0]:>7.4f} {m[1]:>8.4f} {m[2]:>8.4f} {pe:>7.0%}')

    # 主要比較: taps vs ce10（各ASR）、ce0 vs ce10
    print(f'\n{"ASR":<14} {"A→B":<16} {"metric":<8} {"A":>7} {"B":>7} {"B-A":>8} '
          f'{"p(utt)":>9} {"p(spk)":>8} {"spk改善":>6}')
    for asr in sorted({a for a, _ in scores}):
        for a, b in [('taps', 'ce10'), ('ce0', 'ce10')]:
            da, db = scores.get((asr, a)), scores.get((asr, b))
            if not da or not db:
                continue
            keys = sorted(set(da) & set(db))
            for mi, mname in enumerate(['raw', 'nopunct', 'nospace']):
                xa = np.array([da[u][mi] for u in keys]); xb = np.array([db[u][mi] for u in keys])
                p_u = wilcoxon(xa, xb).pvalue if np.any(xa != xb) else 1.0
                spks = sorted({spk_of[u] for u in keys})
                sa = np.array([np.mean([da[u][mi] for u in keys if spk_of[u] == s]) for s in spks])
                sb = np.array([np.mean([db[u][mi] for u in keys if spk_of[u] == s]) for s in spks])
                p_s = wilcoxon(sa, sb).pvalue if len(spks) >= 5 and np.any(sa != sb) else float('nan')
                print(f'{asr:<14} {a + "→" + b:<16} {mname:<8} {xa.mean():>7.4f} {xb.mean():>7.4f} '
                      f'{xb.mean() - xa.mean():>+8.4f} {p_u:>9.1e} {p_s:>8.3f} '
                      f'{int((sb < sa).sum()):>3}/{len(spks)}')

    dst = BASE_DIR / 'results' / 'rescore_normalized.csv'
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(f'\nSaved: {dst}')


if __name__ == '__main__':
    main()
