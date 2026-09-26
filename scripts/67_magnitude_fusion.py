"""
スクリプト67: ASR損失を使わない複数SEの振幅領域での統合（学習不要の分析）

背景（results/STEP0_RESULTS_2026-09-26.md §2b）
  ASR損失を使わない TAPS SE-Conformer と TAPS Demucs の出力を、振幅スペクトルだけ平均し位相は TAPS を
  使って合成すると、6認識器すべてで TAPS より改善した（波形の平均は位相差で打ち消し合い悪化）。
  ここでは統合のしかたを広げ、どの設定が認識器に依らず効くかを調べる。統合の元はすべて ASR 損失なし。

条件（STFT 512/128, Hann。位相は断りがなければ TAPS）
  taps                      基準
  p2_w25 / p2_w50 / p2_w75  TAPS と Demucs の振幅の重み付き平均（数字は Demucs の重み %）
  p2_w50_phD                p2_w50 で位相を Demucs にしたもの（位相の影響）
  p2_log                    TAPS と Demucs の対数振幅の平均（幾何平均）
  m3_amp / m3_pow / m3_log / m3_med
                            TAPS・Demucs・TSTNN の振幅平均 / パワー平均 / 対数平均 / 中央値
  m4_amp / m4_med           m3 に CE0（λ=0、ASR損失なしで追加学習した SE-Conformer）を加えた4つ
認識器: script 65 と同じ6つ。CER は句読点除去後、検定は話者単位（n=10）

出力
  results/fusion/hyp_<asr>[_dev].csv、summary[_dev].csv、pairs[_dev].csv

使い方（GPU PC）
    python scripts/67_magnitude_fusion.py asr --asr xlsr-korean [--split dev] [--limit 1]
    python scripts/67_magnitude_fusion.py summary [--split dev]
"""

import argparse, csv, sys, time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s64 = import_module('64_retranscribe_all')          # load_samples, norm, capped
s65 = import_module('65_step0_residual_cutoff')     # load_asr, ASRS
s66 = import_module('66_lambda_cross_asr')          # load_any_se

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = BASE_DIR / 'results' / 'fusion'
ASRS = s65.ASRS
SOURCES = ['taps', 'demucs', 'tstnn', 'ce0.0']
CONDS = ['taps', 'p2_w25', 'p2_w50', 'p2_w75', 'p2_w50_phD', 'p2_log',
         'm3_amp', 'm3_pow', 'm3_log', 'm3_med', 'm4_amp', 'm4_med']
N_FFT, HOP = 512, 128
_WIN = torch.hann_window(N_FFT).to(DEVICE)
EPS = 1e-8


def stft(x):
    return torch.stft(x, N_FFT, HOP, window=_WIN, return_complex=True)


def istft(X, n):
    return torch.istft(X, N_FFT, HOP, window=_WIN, length=n).cpu().numpy().astype(np.float32)


def fuse_all(outs):
    """outs: {name: (T,) tensor}（長さ揃え済み）→ {cond: np.float32 波形}"""
    n = outs['taps'].shape[-1]
    S = {k: stft(v) for k, v in outs.items()}
    A = {k: v.abs() for k, v in S.items()}
    ph_t, ph_d = torch.exp(1j * S['taps'].angle()), torch.exp(1j * S['demucs'].angle())
    m3 = torch.stack([A['taps'], A['demucs'], A['tstnn']])
    m4 = torch.stack([A['taps'], A['demucs'], A['tstnn'], A['ce0.0']])
    mags = {
        'p2_w25': 0.75 * A['taps'] + 0.25 * A['demucs'],
        'p2_w50': 0.5 * (A['taps'] + A['demucs']),
        'p2_w75': 0.25 * A['taps'] + 0.75 * A['demucs'],
        'p2_log': torch.exp(0.5 * (torch.log(A['taps'] + EPS) + torch.log(A['demucs'] + EPS))),
        'm3_amp': m3.mean(0),
        'm3_pow': (m3 ** 2).mean(0).sqrt(),
        'm3_log': torch.exp(torch.log(m3 + EPS).mean(0)),
        'm3_med': m3.median(0).values,
        'm4_amp': m4.mean(0),
        'm4_med': m4.median(0).values,     # 偶数個のときは torch.median は下側の値
    }
    out = {'taps': outs['taps'].cpu().numpy().astype(np.float32)}
    for c, M in mags.items():
        out[c] = istft(M * ph_t, n)
    out['p2_w50_phD'] = istft(mags['p2_w50'] * ph_d, n)
    return out


def hyp_path(name, split, limit=0):
    tag = '' if split == 'test' else f'_{split}'
    return OUT / f'hyp_{name}{tag}{"_trial" if limit else ""}.csv'


def asr(name, split, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = hyp_path(name, split, limit)
    done = set()
    if out_path.exists() and not limit:
        done = {(r['utt'], r['cond']) for r in csv.DictReader(open(out_path, encoding='utf-8'))}
    samples = s64.load_samples(split)[:limit or None]
    se = {c: s66.load_any_se(c) for c in SOURCES}
    run = s65.load_asr(name)
    new = not out_path.exists() or limit
    f = open(out_path, 'w' if limit else 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['utt', 'spk', 'cond', 'hyp'])
    if new:
        w.writeheader()
    todo = sum(1 for s in samples for c in CONDS if (s['utt'], c) not in done)
    print(f'{name} / {split}: {todo} jobs', flush=True)
    t0, n_done = time.time(), 0
    for i, s in enumerate(samples):
        conds = [c for c in CONDS if (s['utt'], c) not in done]
        if not conds:
            continue
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outs = {k: m(x).reshape(-1) for k, m in se.items()}   # SE ごとに出力の次元が違うので1次元に揃える
            n = min(v.shape[-1] for v in outs.values())
            audio = fuse_all({k: v[:n] for k, v in outs.items()})
        for c in conds:
            hyp = run(audio[c])
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
            n_done += 1
            if limit and i < 1:
                cer = s64.capped(s64.norm(s['text']), s64.norm(hyp))
                print(f'  [{s["utt"]}] {c:<11} CER={cer:.3f} | {hyp[:40]}')
        f.flush()
        if (i + 1) % 100 == 0:
            rate = n_done / (time.time() - t0)
            print(f'  {name} {i + 1}/{len(samples)}  残り約{(todo - n_done) / rate / 60:.0f}分', flush=True)
    f.close()
    print(f'{name} done → {out_path}')


def summary(split):
    tag = '' if split == 'test' else f'_{split}'
    refs = {s['utt']: s['text'] for s in s64.load_samples(split)}
    sc = {}
    for name in ASRS:
        p = hyp_path(name, split)
        if p.exists():
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
        for c in CONDS:
            d = sc.get((name, c))
            if d:
                rows.append(dict(asr=name, cond=c, n=len(d), cer_nopunct=round(np.mean(list(d.values())), 4)))
        for base in ('taps', 'p2_w50'):
            db = sc.get((name, base))
            for c in CONDS:
                d = sc.get((name, c))
                if c == base or not db or not d or len(d) != len(db):
                    continue
                ma, mb = spk_means(db), spk_means(d)
                p = wilcoxon(ma, mb).pvalue if np.any(ma != mb) else 1.0
                pairs.append(dict(asr=name, A=base, B=c, cer_A=round(ma.mean(), 4), cer_B=round(mb.mean(), 4),
                                  rel=round(100 * (mb.mean() / ma.mean() - 1), 1),
                                  spk_B_better=int((mb < ma).sum()), n_spk=len(ma), p_spk=round(p, 4)))
    if not rows:
        print('no rows yet'); return
    for fname, data in ((f'summary{tag}.csv', rows), (f'pairs{tag}.csv', pairs)):
        if data:
            with open(OUT / fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    cer = {(r['asr'], r['cond']): r['cer_nopunct'] for r in rows if r['n'] == len(refs)}
    names = [a for a in ASRS if (a, 'taps') in cer]
    print(f'\n振幅領域の統合  TAPS 比 [%]（{split}、句読点除去CER。括弧なし = 全{len(refs)}発話そろった認識器のみ）')
    print(f'{"cond":<12}' + ''.join(f'{a.replace("whisper-", "W-")[:10]:>11}' for a in names) + f'{"最悪":>8}')
    for c in CONDS[1:]:
        vals = [100 * (cer[(a, c)] / cer[(a, 'taps')] - 1) if (a, c) in cer else None for a in names]
        worst = max(v for v in vals if v is not None) if any(v is not None for v in vals) else float('nan')
        print(f'{c:<12}' + ''.join(f'{v:>+11.1f}' if v is not None else f'{"-":>11}' for v in vals) + f'{worst:>+8.1f}')
    print(f'\nSaved: {OUT}/summary{tag}.csv, pairs{tag}.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['asr', 'summary'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--split', choices=['test', 'dev'], default='test')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'asr':
        if not a.asr:
            ap.error('--asr が必要')
        asr(a.asr, a.split, a.limit)
    else:
        summary(a.split)


if __name__ == '__main__':
    main()
