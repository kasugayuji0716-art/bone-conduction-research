"""
スクリプト66: CE λ 違いの SE を 6 認識器で評価（学習不要の分析）

目的
  script 65 の残差スイープで「CE が TAPS に加えた変化は、少しだけ加えると非Whisper系にも効き、
  加えすぎると Whisper 専用になる」傾向が出た。出力を混ぜる操作を使わずにこれを確かめるため、
  すでに学習済みの CE v2（λ = 0, 0.1, 0.5, 1, 2, 5, 10）を認識器ごとに評価する。
  λ が小さいほど非Whisper系で良いなら、「ASR損失での追加学習は効く方向に進むが、進みすぎる」
  という解釈を学習そのものの設定で裏づけられる。
  これまで λ 違いは whisper-small（学習に使った認識器）でしか評価していなかった。

条件: taps, ce0.0 … ce10.0（script 64 の SE_CKPTS と同じチェックポイント）
  --set avg: 2つのSE出力の平均（対照実験）。script 65 の「α=0.5 で非Whisper系も改善」が
  CE の変化の方向によるものか、単に異なる2モデルの出力を平均した効果（アンサンブル）かを切り分ける。
  avg_taps_ce0.0 は CE 損失なし（λ=0）で追加学習したモデルとの平均。これでも改善するなら平均の効果
認識器: script 65 と同じ 6 つ（whisper-base/small/medium/ft, mms-1b-all, xlsr-korean）
CER は句読点除去後（script 63 の norm/capped）。検定は話者単位（n=10）。

出力
  results/lambda_asr/hyp_<asr>.csv（dev は hyp_<asr>_dev.csv）  utt, spk, cond, hyp
  results/lambda_asr/summary[_dev].csv、pairs[_dev].csv

使い方（GPU PC）
    python scripts/66_lambda_cross_asr.py asr --asr xlsr-korean [--split dev] [--limit 3]
    python scripts/66_lambda_cross_asr.py summary [--split dev]
"""

import argparse, csv, sys, time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import spearmanr, wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s64 = import_module('64_retranscribe_all')          # load_se, load_samples, SE_CKPTS, norm, capped
s65 = import_module('65_step0_residual_cutoff')     # load_asr, ASRS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = BASE_DIR / 'results' / 'lambda_asr'
LAMBDAS = ['0.0', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']
CONDS = ['taps'] + [f'ce{l}' for l in LAMBDAS]
AVGS = {'avg_taps_ce0.0': ('taps', 'ce0.0'), 'avg_taps_ce10.0': ('taps', 'ce10.0'),
        'avg_ce0.0_ce10.0': ('ce0.0', 'ce10.0')}
ALL_CONDS = CONDS + list(AVGS)
ASRS = s65.ASRS


def hyp_path(name, split, limit=0):
    tag = '' if split == 'test' else f'_{split}'
    return OUT / f'hyp_{name}{tag}{"_trial" if limit else ""}.csv'


def asr(name, split, limit, conds_all=CONDS):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = hyp_path(name, split, limit)
    done = set()
    if out_path.exists() and not limit:
        done = {(r['utt'], r['cond']) for r in csv.DictReader(open(out_path, encoding='utf-8'))}
    samples = s64.load_samples(split)[:limit or None]
    need = sorted({m for c in conds_all for m in AVGS.get(c, (c,))})
    se = {c: s64.load_se(s64.SE_CKPTS[c]) for c in need}
    run = s65.load_asr(name)
    new = not out_path.exists() or limit
    f = open(out_path, 'w' if limit else 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['utt', 'spk', 'cond', 'hyp'])
    if new:
        w.writeheader()
    todo = sum(1 for s in samples for c in conds_all if (s['utt'], c) not in done)
    print(f'{name} / {split}: {todo} jobs', flush=True)
    t0, n = time.time(), 0
    for i, s in enumerate(samples):
        conds = [c for c in conds_all if (s['utt'], c) not in done]
        if not conds:
            continue
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outs = {m: se[m](x).squeeze().cpu().numpy().astype(np.float32)
                    for m in need if any(m in AVGS.get(c, (c,)) for c in conds)}
        for c in conds:
            if c in AVGS:
                a, b = (outs[m] for m in AVGS[c])
                n_ = min(len(a), len(b))
                audio = (0.5 * (a[:n_] + b[:n_])).astype(np.float32)
            else:
                audio = outs[c]
            hyp = run(audio)
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
            n += 1
            if limit and i < 1:
                cer = s64.capped(s64.norm(s['text']), s64.norm(hyp))
                print(f'  [{s["utt"]}] {c:<8} CER={cer:.3f} | {hyp[:40]}')
        f.flush()
        if (i + 1) % 100 == 0:
            rate = n / (time.time() - t0)
            print(f'  {name} {i + 1}/{len(samples)}  残り約{(todo - n) / rate / 60:.0f}分', flush=True)
    f.close()
    print(f'{name} done → {out_path}')


def summary(split):
    tag = '' if split == 'test' else f'_{split}'
    refs = {s['utt']: s['text'] for s in s64.load_samples(split)}
    sc = {}
    for name in ASRS:
        p = hyp_path(name, split)
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding='utf-8')):
            sc.setdefault((name, r['cond']), {})[(r['utt'], r['spk'])] = s64.capped(
                s64.norm(refs[r['utt']]), s64.norm(r['hyp']))

    def spk_means(d):
        by = defaultdict(list)
        for (_, spk), v in d.items():
            by[spk].append(v)
        return np.array([np.mean(by[k]) for k in sorted(by)])

    rows, pairs = [], []
    for name in ASRS:
        for c in ALL_CONDS:
            d = sc.get((name, c))
            if d:
                rows.append(dict(asr=name, cond=c, n=len(d), cer_nopunct=round(np.mean(list(d.values())), 4)))
        base = sc.get((name, 'taps'))
        for c in ALL_CONDS[1:]:
            d = sc.get((name, c))
            if not base or not d or len(d) != len(base):
                continue
            ma, mb = spk_means(base), spk_means(d)
            p = wilcoxon(ma, mb).pvalue if np.any(ma != mb) else 1.0
            pairs.append(dict(asr=name, A='taps', B=c, cer_A=round(ma.mean(), 4), cer_B=round(mb.mean(), 4),
                              rel=round(100 * (mb.mean() / ma.mean() - 1), 1),
                              spk_B_better=int((mb < ma).sum()), n_spk=len(ma), p_spk=round(p, 4)))
    if not rows:
        print('no rows yet'); return
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, data in ((f'summary{tag}.csv', rows), (f'pairs{tag}.csv', pairs)):
        if data:
            with open(OUT / fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    cer = {(r['asr'], r['cond']): r['cer_nopunct'] for r in rows if r['n'] == len(refs)}
    print(f'\nCE λ 違い × 認識器  CER（句読点除去、{split}）')
    print(f'{"ASR":<15}{"TAPS":>8}' + ''.join(f'{"λ=" + l:>8}' for l in LAMBDAS) + f'{"ρ(λ,CER)":>10}')
    for name in ASRS:
        ys = [cer.get((name, c)) for c in CONDS]
        if all(y is None for y in ys):
            continue
        ce_ys = ys[1:]
        rho = spearmanr([float(l) for l in LAMBDAS], ce_ys).statistic if None not in ce_ys else float('nan')
        print(f'{name:<15}' + ''.join(f'{y:>8.4f}' if y is not None else f'{"-":>8}' for y in ys) + f'{rho:>10.2f}')
    print(f'\n2つのSE出力の平均（対照）  CER  （括弧内は TAPS 比）')
    for name in ASRS:
        t = cer.get((name, 'taps'))
        items = [(c, cer.get((name, c))) for c in AVGS]
        if t is None or all(v is None for _, v in items):
            continue
        print(f'{name:<15}' + ''.join(f'  {c}={v:.4f}({100 * (v / t - 1):+.1f}%)' for c, v in items if v is not None))
    print(f'\nSaved: {OUT}/summary{tag}.csv, pairs{tag}.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['asr', 'summary'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--split', choices=['test', 'dev'], default='test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--set', choices=['lambda', 'avg'], default='lambda')
    a = ap.parse_args()
    if a.stage == 'asr':
        if not a.asr:
            ap.error('--asr が必要')
        asr(a.asr, a.split, a.limit, CONDS if a.set == 'lambda' else ['taps'] + list(AVGS))
    else:
        summary(a.split)


if __name__ == '__main__':
    main()
