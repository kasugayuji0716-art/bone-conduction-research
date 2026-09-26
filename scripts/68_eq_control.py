"""
スクリプト68: 振幅統合の効果はイコライザ（長時間スペクトルの形）や平滑化で説明できるか（対照実験）

背景
  script 67 で、ASR損失を使わない複数SEの振幅平均（m4_amp: TAPS/Demucs/TSTNN/CE0、p2_w50: TAPS/Demucs）が
  6認識器で TAPS より改善した。しかし TAPS の合成高域は Whisper 系に有害で（script 65 のカットオフ行列）、
  録音経路やスペクトル傾斜だけで ASR 精度が大きく変わる報告もある（Huang et al. ASRU 2025、Boril & Hansen 2009）。
  振幅平均が「単に TAPS の長時間スペクトルを整えているだけ」なら、アンサンブルとしての解釈は成り立たない。

条件（STFT 512/128, Hann。位相はすべて TAPS）
  taps, m4_amp, p2_w50        基準（script 67 と同じ定義）
  eq_m4_dev / eq_p2_dev       dev 全体で求めた固定ゲイン G(f) = mean|統合| / mean|TAPS| を TAPS に掛ける
  eq_m4_utt / eq_p2_utt       同じ比を発話ごとに求めて掛ける（その発話の統合出力を使うオラクル。時間変化だけを除いた対照）
  smooth_t3 / smooth_f5       TAPS の振幅を時間方向3フレーム / 周波数方向5ビンで移動平均（歪みの平滑化の対照）
  oa_w20 / oa_w40             観測（喉マイク生音声）を波形で足す: (1-w)·TAPS + w·観測（Iwamoto et al. の observation adding）
  moa_w25                     観測との振幅平均（重み 0.25、位相は TAPS）

ステージ
  ltas     dev で TAPS と統合出力の平均振幅スペクトルを求め results/eq_control/ltas_dev.npz に保存（eq_*_dev に必要）
  asr      書き起こし（results/eq_control/hyp_<asr>.csv）
  summary  TAPS 比と、統合出力（m4_amp / p2_w50）との比較

使い方（GPU PC）
    python scripts/68_eq_control.py ltas
    python scripts/68_eq_control.py asr --asr xlsr-korean [--limit 1]
    python scripts/68_eq_control.py summary
"""

import argparse, csv, sys, time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s64 = import_module('64_retranscribe_all')
s65 = import_module('65_step0_residual_cutoff')
s66 = import_module('66_lambda_cross_asr')
s67 = import_module('67_magnitude_fusion')          # stft/istft と SOURCES

DEVICE = s67.DEVICE
OUT = BASE_DIR / 'results' / 'eq_control'
LTAS = OUT / 'ltas_dev.npz'
ASRS = s65.ASRS
CONDS = ['taps', 'm4_amp', 'p2_w50', 'eq_m4_dev', 'eq_m4_utt', 'eq_p2_dev', 'eq_p2_utt',
         'smooth_t3', 'smooth_f5', 'oa_w20', 'oa_w40', 'moa_w25']


def se_outputs(se, wav):
    x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outs = {k: m(x).reshape(-1) for k, m in se.items()}
    n = min(min(v.shape[-1] for v in outs.values()), len(wav))
    return {k: v[:n] for k, v in outs.items()}, torch.from_numpy(wav[:n]).to(DEVICE), n


def magnitudes(outs):
    S = {k: s67.stft(v) for k, v in outs.items()}
    A = {k: v.abs() for k, v in S.items()}
    m4 = torch.stack([A['taps'], A['demucs'], A['tstnn'], A['ce0.0']]).mean(0)
    p2 = 0.5 * (A['taps'] + A['demucs'])
    return S, A, m4, p2


def ltas():
    se = {c: s66.load_any_se(c) for c in s67.SOURCES}
    acc = {k: torch.zeros(s67.N_FFT // 2 + 1, device=DEVICE, dtype=torch.float64) for k in ('taps', 'm4', 'p2')}
    frames = 0
    for s in s64.load_samples('dev'):
        wav, _ = sf.read(s['path'], dtype='float32')
        outs, _, _ = se_outputs(se, wav)
        _, A, m4, p2 = magnitudes(outs)
        acc['taps'] += A['taps'].sum(1); acc['m4'] += m4.sum(1); acc['p2'] += p2.sum(1)
        frames += A['taps'].shape[1]
    mean = {k: (v / frames).cpu().numpy() for k, v in acc.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(LTAS, **mean)
    g4, g2 = mean['m4'] / mean['taps'], mean['p2'] / mean['taps']
    freqs = np.arange(len(g4)) * 16000 / s67.N_FFT
    print('dev 固定ゲイン G(f) [dB]（統合 / TAPS）')
    for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8001)):
        b = (freqs >= lo) & (freqs < hi)
        print(f'  {lo:>4}-{hi:<4} Hz  m4 {20 * np.log10(g4[b]).mean():+6.2f}  p2 {20 * np.log10(g2[b]).mean():+6.2f}')
    print(f'Saved: {LTAS}')


def make_conds(outs, obs, n, gains):
    S, A, m4, p2 = magnitudes(outs)
    ph = torch.exp(1j * S['taps'].angle())
    At = A['taps']
    ist = lambda M: s67.istft(M * ph, n)
    out = {'taps': outs['taps'].cpu().numpy().astype(np.float32), 'm4_amp': ist(m4), 'p2_w50': ist(p2)}
    out['eq_m4_dev'] = ist(At * gains['m4'][:, None])
    out['eq_p2_dev'] = ist(At * gains['p2'][:, None])
    eps = 1e-10
    out['eq_m4_utt'] = ist(At * (m4.mean(1) / (At.mean(1) + eps))[:, None])
    out['eq_p2_utt'] = ist(At * (p2.mean(1) / (At.mean(1) + eps))[:, None])
    out['smooth_t3'] = ist(F.avg_pool1d(At.unsqueeze(0), 3, 1, 1, count_include_pad=False).squeeze(0))
    out['smooth_f5'] = ist(F.avg_pool1d(At.T.unsqueeze(0), 5, 1, 2, count_include_pad=False).squeeze(0).T)
    taps_w = outs['taps']
    for w in (0.2, 0.4):
        out[f'oa_w{int(w * 100)}'] = ((1 - w) * taps_w + w * obs).cpu().numpy().astype(np.float32)
    Ao = s67.stft(obs).abs()
    out['moa_w25'] = ist(0.75 * At + 0.25 * Ao)
    return out


def hyp_path(name, limit=0):
    return OUT / f'hyp_{name}{"_trial" if limit else ""}.csv'


def asr(name, limit):
    if not LTAS.exists():
        raise SystemExit(f'{LTAS} がない。先に ltas ステージを実行する')
    z = np.load(LTAS)
    gains = {k: torch.from_numpy(z[k] / z['taps']).float().to(DEVICE) for k in ('m4', 'p2')}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = hyp_path(name, limit)
    done = set()
    if out_path.exists() and not limit:
        done = {(r['utt'], r['cond']) for r in csv.DictReader(open(out_path, encoding='utf-8'))}
    samples = s64.load_samples('test')[:limit or None]
    se = {c: s66.load_any_se(c) for c in s67.SOURCES}
    run = s65.load_asr(name)
    new = not out_path.exists() or limit
    f = open(out_path, 'w' if limit else 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['utt', 'spk', 'cond', 'hyp'])
    if new:
        w.writeheader()
    todo = sum(1 for s in samples for c in CONDS if (s['utt'], c) not in done)
    print(f'{name}: {todo} jobs', flush=True)
    t0, n_done = time.time(), 0
    for i, s in enumerate(samples):
        conds = [c for c in CONDS if (s['utt'], c) not in done]
        if not conds:
            continue
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        outs, obs, n = se_outputs(se, wav)
        with torch.no_grad():
            audio = make_conds(outs, obs, n, gains)
        for c in conds:
            hyp = run(audio[c])
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
            n_done += 1
            if limit and i < 1:
                cer = s64.capped(s64.norm(s['text']), s64.norm(hyp))
                print(f'  [{s["utt"]}] {c:<10} CER={cer:.3f} | {hyp[:40]}')
        f.flush()
        if (i + 1) % 100 == 0:
            rate = n_done / (time.time() - t0)
            print(f'  {name} {i + 1}/{len(samples)}  残り約{(todo - n_done) / rate / 60:.0f}分', flush=True)
    f.close()
    print(f'{name} done → {out_path}')


def summary():
    refs = {s['utt']: s['text'] for s in s64.load_samples('test')}
    sc = {}
    for name in ASRS:
        p = hyp_path(name)
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
        for base in ('taps', 'm4_amp', 'p2_w50'):
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
    for fname, data in (('summary.csv', rows), ('pairs.csv', pairs)):
        if data:
            with open(OUT / fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    cer = {(r['asr'], r['cond']): r['cer_nopunct'] for r in rows if r['n'] == len(refs)}
    names = [a for a in ASRS if (a, 'taps') in cer]
    print('\nイコライザ・平滑化・観測加算の対照  TAPS 比 [%]（test、句読点除去CER）')
    print(f'{"cond":<11}' + ''.join(f'{a.replace("whisper-", "W-")[:10]:>11}' for a in names))
    for c in CONDS[1:]:
        print(f'{c:<11}' + ''.join(f'{100 * (cer[(a, c)] / cer[(a, "taps")] - 1):>+11.1f}' if (a, c) in cer
                                   else f'{"-":>11}' for a in names))
    print(f'\nSaved: {OUT}/summary.csv, pairs.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['ltas', 'asr', 'summary'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'ltas':
        ltas()
    elif a.stage == 'asr':
        if not a.asr:
            ap.error('--asr が必要')
        asr(a.asr, a.limit)
    else:
        summary()


if __name__ == '__main__':
    main()
