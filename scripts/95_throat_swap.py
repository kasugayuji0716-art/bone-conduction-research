"""
スクリプト95: 喉マイク音声の何が認識を難しくするのか（入れ替え実験、計画: results/PLAN_THROAT_DIFFICULTY_2026-11-05.md）

実験 0（前提の確認）と、実験 1〜3 の音声作り・集計を実装している。

  align     喉マイクと気導マイクの時間のずれを発話ごとに求める（TAPS test）→ results/throat_swap/align.csv
            - wave: 100〜1000 Hz に帯域制限した波形の相互相関（±50 ms）。両方に声のエネルギーがある帯域
            - env : 同じ帯域の包絡（整流して 30 Hz で平滑化）の相互相関（±50 ms）。位相の違いに強い
            正の lag は、気導音が喉マイクより遅れていることを表す
  make0     実験 0 の音声を、script 92 と同じく BCR_DATASET 用のデータセットとして作る（data/raw/<name>/throat/test）
            - taps_airlp : 気導音の 4 kHz 以上を 0 にしたもの（以下、気導LP）
            - taps_threq : 喉マイク音声に 1/H(f) をかけて平均特性を気導にそろえ、4 kHz 以上を 0 にしたもの（補正喉マイク）
            H(f) は script 92 の channels.npz（TAPS train、喉/気導の長時間平均パワー比）。音量はどちらも
            train の喉マイクの RMS の中央値にそろえる（script 92 の疑似喉マイクと同じ）
  summary0  実験 0 の CER を集計する → results/throat_swap/exp0.md
            条件: 気導（taps_air）、気導LP、疑似喉マイク（taps_simtaps）、補正喉マイク、本物の喉マイク（taps の no_se）

実験 1〜3: 気導LP（A）と補正喉マイク（T）の STFT を部分的に入れ替える。d = 壊す向き（A の一部を T に）、
r = 救出の向き（T の一部を A に）。各条件は data/raw/taps_sw_<条件> に作る
  stats     train 500 対で、区間（無音・有声・無声）ごとの R(f) = log|T| − log|A| の平均を求める → results/throat_swap/class_R.npz
  make      test の全条件（31）の音声を作る。区間の割合とずらしたマスクの重なりを results/throat_swap/masks.csv に書く
            実験 1（時間の細かさ）: R を時間方向に幅 τ でなめらかにした R_τ を使う
              d_t_<τ>: |A|·exp(R_τ)·e^{j∠A}、r_t_<τ>: |T|·exp(−R_τ)·e^{j∠T}
              τ = utt（発話全体の平均）、cls（train の区間ごとの平均）、1000・300・100・30 ms、0（d のみ）
              d_phase: |A|·e^{j∠T}（位相だけ入れ替え。r_t_0 と同じ音声）
            実験 2（区間）: マスク m（3 フレームの移動平均でなめらかにする）で複素係数を混ぜる
              d_seg_<区間>: (1−m)A + mT、r_seg_<区間>: (1−m)T + mA。区間 = unv（無声）、voi（有声）、sil（無音）、
              unvsh・voish（同じマスクを発話ごとの乱数で巡回シフトしたもの）
            実験 3（帯域）: 振幅の 1 帯域だけを入れ替える（境目は 1 ビン幅でなめらかにする）
              d_b<k>: |A| の帯域 k を |T| に、位相 ∠A。r_b<k>: |T| の帯域 k を |A| に、位相 ∠T
              B1 0〜0.5、B2 0.5〜1、B3 1〜2、B4 2〜4 kHz
  summary   全条件の CER、g（壊す向き）・h（救出の向き）と、事前に決めた比較の検定 → results/throat_swap/summary.md

認識は script 66 で行う（GPU PC）:
    BCR_DATASET=taps_airlp python scripts/66_lambda_cross_asr.py asr --asr whisper-small --conds no_se

使い方
    python scripts/95_throat_swap.py align
    python scripts/95_throat_swap.py make0
    python scripts/95_throat_swap.py summary0
    python scripts/95_throat_swap.py stats
    python scripts/95_throat_swap.py make [--limit 3]
    python scripts/95_throat_swap.py summary
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, sosfiltfilt
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
RAW = BASE_DIR / 'data' / 'raw'
OUT = BASE_DIR / 'results' / 'throat_swap'
N_FFT, HOP, SR = 512, 128, 16000
WIN = torch.hann_window(N_FFT)
HZ = np.fft.rfftfreq(N_FFT, 1 / SR)
MAX_LAG = int(0.05 * SR)
ASRS = ['whisper-small', 'whisper-large-v3-turbo', 'qwen3-asr-1.7b', 'mms-1b-all']

_BP = butter(4, [100, 1000], btype='band', fs=SR, output='sos')
_LP30 = butter(4, 30, btype='low', fs=SR, output='sos')


def test_pairs(split='test'):
    rows = list(csv.DictReader(open(RAW / 'taps' / f'metadata_{split}.csv', encoding='utf-8')))
    out = []
    for r in rows:
        fn = f"{r['speaker_id']}_{r['sentence_id']}.wav"
        t, a = RAW / 'taps' / 'throat' / split / fn, RAW / 'taps' / 'acoustic' / split / fn
        if t.exists() and a.exists():
            out.append((r, t, a))
    return out


def xcorr_lag(x, y):
    """y が x よりどれだけ遅れているか（サンプル）と、その正規化相関"""
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    c = fftconvolve(y, x[::-1], mode='full') / len(x)
    mid = len(x) - 1
    seg = c[mid - MAX_LAG: mid + MAX_LAG + 1]
    k = int(np.argmax(np.abs(seg)))
    return k - MAX_LAG, float(seg[k])


def align(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, t, a in test_pairs():
        xt, _ = sf.read(t, dtype='float32')
        xa, _ = sf.read(a, dtype='float32')
        m = min(len(xt), len(xa))
        bt, ba = sosfiltfilt(_BP, xt[:m]), sosfiltfilt(_BP, xa[:m])
        lw, rw = xcorr_lag(bt, ba)
        le, re_ = xcorr_lag(sosfiltfilt(_LP30, np.abs(bt)), sosfiltfilt(_LP30, np.abs(ba)))
        rows.append(dict(utt=f"{r['speaker_id']}_{r['sentence_id']}", spk=r['speaker_id'],
                         len_throat=len(xt), len_air=len(xa), lag_wave=lw, r_wave=rw, lag_env=le, r_env=re_))
    with open(OUT / 'align.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} 発話 → {OUT / "align.csv"}')
    print(f'長さが違う発話: {sum(r["len_throat"] != r["len_air"] for r in rows)}')
    for k in ('wave', 'env'):
        lag = np.array([r[f'lag_{k}'] for r in rows])
        rr = np.array([r[f'r_{k}'] for r in rows])
        q = np.percentile(lag, [5, 25, 50, 75, 95])
        print(f'{k}: lag [サンプル] 5/25/50/75/95% = {q.astype(int).tolist()}  '
              f'|lag|>128（8 ms）: {np.mean(np.abs(lag) > HOP) * 100:.1f}%  '
              f'|lag|>16（1 ms）: {np.mean(np.abs(lag) > 16) * 100:.1f}%  相関の中央値 {np.median(rr):.3f}')
    by = defaultdict(list)
    for r in rows:
        by[r['spk']].append(r['lag_wave'])
    print('話者ごとの lag_wave の中央値: ' + ', '.join(f'{s} {int(np.median(v))}' for s, v in sorted(by.items())))


def stft(x):
    return torch.stft(torch.from_numpy(x), N_FFT, HOP, window=WIN, return_complex=True)


def shape(x, gain, rms):
    """STFT の各周波数に gain（振幅倍率、4 kHz 以上は 0 にしておく）をかけて戻し、RMS をそろえる"""
    y = torch.istft(stft(x) * torch.from_numpy(gain).float()[:, None], N_FFT, HOP, window=WIN, length=len(x)).numpy()
    return (y * rms / (np.sqrt(np.mean(y ** 2)) + 1e-9)).astype(np.float32)


LO = HZ < 4000


def exp0_gains():
    """気導LP・補正喉マイクの周波数ごとの倍率と、そろえる RMS"""
    ch = np.load(BASE_DIR / 'results' / 'lang_vs_device' / 'channels.npz')
    h = 10 ** (ch['taps_gain_db'] / 20)          # 喉 / 気導 の振幅比
    return {'taps_airlp': np.where(LO, 1.0, 0.0), 'taps_threq': np.where(LO, 1 / h, 0.0)}, float(ch['taps_rms']), h


def make0(args):
    gains, rms, h = exp0_gains()
    lo = LO
    print(f'H(f)（4 kHz 未満）: {20 * np.log10(h[lo]).min():+.1f}〜{20 * np.log10(h[lo]).max():+.1f} dB → 1/H の上限は設けない')
    srcs = {'taps_airlp': 'acoustic', 'taps_threq': 'throat'}
    pairs = test_pairs()
    for name, g in gains.items():
        dst = RAW / name / 'throat' / 'test'
        dst.mkdir(parents=True, exist_ok=True)
        for r, t, a in pairs:
            fn = t.name
            if (dst / fn).exists():
                continue
            x, _ = sf.read(t if srcs[name] == 'throat' else a, dtype='float32')
            sf.write(dst / fn, shape(x, g, rms), SR)
        with open(RAW / name / 'metadata_test.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0][0]))
            w.writeheader(); w.writerows(r for r, _, _ in pairs)
        print(f'{name}: {len(pairs)} 発話 → {dst}')


EXP0 = [('taps_air', '気導'), ('taps_airlp', '気導LP'), ('taps_simtaps', '疑似喉マイク'),
        ('taps_threq', '補正喉マイク'), ('taps', '本物の喉マイク')]


def load_cer(ds, asr):
    os.environ['BCR_DATASET'] = ds          # 63 の norm は仏語のときだけ小文字化する。ここは韓国語のみ
    r63 = import_module('63_rescore_normalized')
    sfx = '' if ds == 'taps' else f'_{ds}'
    p = BASE_DIR / 'results' / f'lambda_asr{sfx}' / f'hyp_{asr}.csv'
    if not p.exists():
        return None
    refs = {f"{r['speaker_id']}_{r['sentence_id']}": r['text']
            for r in csv.DictReader(open(RAW / 'taps' / 'metadata_test.csv', encoding='utf-8'))}
    out = {}
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if r['cond'] == 'no_se' and r['utt'] in refs:
            out[r['utt']] = (r['spk'], r63.capped(r63.norm(refs[r['utt']]), r63.norm(r['hyp'])))
    return out


def spk_mean(d):
    by = defaultdict(list)
    for spk, c in d.values():
        by[spk].append(c)
    return {s: float(np.mean(v)) for s, v in by.items()}


def summary0(args):
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ['# 実験 0: 基準の CER（TAPS test、句読点除去・発話ごとに 1.0 で打ち切り）', '',
             '| 認識器 | ' + ' | '.join(lab for _, lab in EXP0) + ' | g(疑似) | 補正 対 本物 p | 気導LP 対 気導 p |',
             '|---|' + '---|' * (len(EXP0) + 3)]
    for asr in ASRS:
        cer = {ds: load_cer(ds, asr) for ds, _ in EXP0}
        cells, mean = [], {}
        for ds, _ in EXP0:
            d = cer[ds]
            if not d:
                cells.append('--'); continue
            mean[ds] = float(np.mean([c for _, c in d.values()]))
            cells.append(f'{mean[ds]:.3f}' + ('' if len(d) == 1000 else f' (n={len(d)})'))

        def p(a, b):
            if not (cer[a] and cer[b]):
                return '--'
            sa, sb = spk_mean(cer[a]), spk_mean(cer[b])
            ks = sorted(set(sa) & set(sb))
            return f'{wilcoxon([sa[k] for k in ks], [sb[k] for k in ks]).pvalue:.3f}'

        g = '--'
        if all(k in mean for k in ('taps_airlp', 'taps_simtaps', 'taps_threq')):
            g = f"{(mean['taps_simtaps'] - mean['taps_airlp']) / (mean['taps_threq'] - mean['taps_airlp']):.2f}"
        lines.append(f'| {asr} | ' + ' | '.join(cells) + f' | {g} | {p("taps_threq", "taps")} | {p("taps_airlp", "taps_air")} |')
    lines += ['', 'g(疑似) = (疑似喉マイク − 気導LP) / (補正喉マイク − 気導LP)。p は話者ごとの平均 CER の両側 Wilcoxon（n = 10、Holm 補正前）']
    (OUT / 'exp0.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


# ── 実験 1〜3 ──────────────────────────────────────────────

TAUS = {'1000': 125, '300': 38, '100': 13, '30': 4}      # なめらかにする幅（フレーム数、1 フレーム = 8 ms）
BANDS = {'b1': (0, 500), 'b2': (500, 1000), 'b3': (1000, 2000), 'b4': (2000, 4000)}
SEGS = ['unv', 'voi', 'sil', 'unvsh', 'voish']
SWAP = ([f'd_t_{t}' for t in ('utt', 'cls', '1000', '300', '100', '30', '0')] + ['d_phase']
        + [f'r_t_{t}' for t in ('utt', 'cls', '1000', '100', '30')]
        + [f'{d}_seg_{s}' for d in 'dr' for s in SEGS]
        + [f'{d}_{b}' for d in 'dr' for b in BANDS])
CLASS_R = OUT / 'class_R.npz'


def frame_classes(airlp, A):
    """フレームごとの区間: 0 無音、1 有声、2 無声。無音は気導LP のエネルギーが最大から 40 dB 未満、有声は pYIN"""
    import librosa
    _, vflag, _ = librosa.pyin(airlp, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=HOP, center=True)
    n = A.shape[1]
    v = np.zeros(n, bool)
    v[:min(n, len(vflag))] = vflag[:n]
    e = 10 * np.log10((np.abs(A) ** 2).sum(0) + 1e-12)
    return np.where(e < e.max() - 40, 0, np.where(v, 1, 2))


def log_ratio(A, T):
    """R = log|T| − log|A|（自然対数）。両方に同じ下限（大きい方の最大から −80 dB）を設け、4 kHz 以上は 0"""
    fl = 1e-4 * max(np.abs(A).max(), np.abs(T).max())
    R = np.log(np.maximum(np.abs(T), fl)) - np.log(np.maximum(np.abs(A), fl))
    R[~LO] = 0
    return R


def _stats_one(pair):
    torch.set_num_threads(1)
    _, t, a = pair
    gains, rms, _ = exp0_gains()
    airlp = shape(sf.read(a, dtype='float32')[0], gains['taps_airlp'], rms)
    threq = shape(sf.read(t, dtype='float32')[0], gains['taps_threq'], rms)
    A, T = stft(airlp).numpy(), stft(threq).numpy()
    R, c = log_ratio(A, T), frame_classes(airlp, A)
    return np.stack([R[:, c == k].sum(1) for k in range(3)]), np.array([(c == k).sum() for k in range(3)])


def stats(args):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = test_pairs('train')
    np.random.default_rng(0).shuffle(pairs)
    pairs = pairs[:500]
    with Pool(args.workers) as p:
        res = p.map(_stats_one, pairs)
    sums = sum(r[0] for r in res)
    cnt = sum(r[1] for r in res)
    R_cls = sums / np.maximum(cnt, 1)[:, None]
    np.savez(CLASS_R, R_cls=R_cls, count=cnt, n_utt=len(pairs), hz=HZ)
    print(f'train {len(pairs)} 発話。フレームの割合 無音/有声/無声 = ' + ' / '.join(f'{x:.1%}' for x in cnt / cnt.sum()))
    db = 20 / np.log(10)
    for k, lab in enumerate(['無音', '有声', '無声']):
        print(f'  {lab} の R [dB]: ' + ', '.join(f'{lo / 1000:g}-{hi / 1000:g}k {R_cls[k][(HZ >= lo) & (HZ < hi)].mean() * db:+.1f}'
                                            for lo, hi in BANDS.values()))
    print('saved →', CLASS_R)


def _band_weight(lo, hi):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(((HZ >= lo) & (HZ < hi)).astype(np.float32), 3, mode='nearest')[:, None]


def _make_one(job):
    from scipy.ndimage import uniform_filter1d
    torch.set_num_threads(1)
    i, fn, rms = job
    airlp = sf.read(RAW / 'taps_airlp' / 'throat' / 'test' / fn, dtype='float32')[0]
    threq = sf.read(RAW / 'taps_threq' / 'throat' / 'test' / fn, dtype='float32')[0]
    A, T = stft(airlp).numpy(), stft(threq).numpy()
    mA, mT = np.abs(A), np.abs(T)
    pA, pT = np.exp(1j * np.angle(A)), np.exp(1j * np.angle(T))
    R, c = log_ratio(A, T), frame_classes(airlp, A)
    n = A.shape[1]
    Rt = {'utt': np.repeat(R.mean(1, keepdims=True), n, 1), 'cls': np.load(CLASS_R)['R_cls'][c].T}
    Rt.update({t: uniform_filter1d(R, k, axis=1, mode='nearest') for t, k in TAUS.items()})
    out = {}
    for t, Rx in Rt.items():
        out[f'd_t_{t}'] = mA * np.exp(Rx) * pA
        if t != '300':
            out[f'r_t_{t}'] = mT * np.exp(-Rx) * pT
    out['d_t_0'], out['d_phase'] = mT * pA, mA * pT
    rng = np.random.default_rng(i)
    base = {'unv': c == 2, 'voi': c == 1, 'sil': c == 0}
    base['unvsh'] = np.roll(base['unv'], rng.integers(1, n))
    base['voish'] = np.roll(base['voi'], rng.integers(1, n))
    for s, m in base.items():
        m = uniform_filter1d(m.astype(np.float32), 3, mode='nearest')[None, :]
        out[f'd_seg_{s}'] = (1 - m) * A + m * T
        out[f'r_seg_{s}'] = (1 - m) * T + m * A
    for b, (lo, hi) in BANDS.items():
        w = _band_weight(lo, hi)
        out[f'd_{b}'] = ((1 - w) * mA + w * mT) * pA
        out[f'r_{b}'] = ((1 - w) * mT + w * mA) * pT
    for cond, Y in out.items():
        Y = torch.from_numpy((Y * LO[:, None]).astype(np.complex64))
        y = torch.istft(Y, N_FFT, HOP, window=WIN, length=len(airlp)).numpy()
        y = (y * rms / (np.sqrt(np.mean(y ** 2)) + 1e-9)).astype(np.float32)
        sf.write(RAW / f'taps_sw_{cond}' / 'throat' / 'test' / fn, y, SR)
    ov = lambda a, b: float((base[a] & base[b]).sum() / max(base[a].sum(), 1))
    return dict(utt=fn[:-4], n_frames=n, sil=float(base['sil'].mean()), voi=float(base['voi'].mean()),
                unv=float(base['unv'].mean()), unvsh_overlap=ov('unv', 'unvsh'), voish_overlap=ov('voi', 'voish'))


def make(args):
    from multiprocessing import Pool
    _, rms, _ = exp0_gains()
    pairs = test_pairs()[:args.limit or None]
    for cond in SWAP:
        (RAW / f'taps_sw_{cond}' / 'throat' / 'test').mkdir(parents=True, exist_ok=True)
    jobs = [(i, t.name, rms) for i, (_, t, _) in enumerate(pairs)]
    with Pool(args.workers) as p:
        rows = p.map(_make_one, jobs, chunksize=4)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / ('masks_trial.csv' if args.limit else 'masks.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    if not args.limit:
        for cond in SWAP:
            with open(RAW / f'taps_sw_{cond}' / 'metadata_test.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(pairs[0][0]))
                w.writeheader(); w.writerows(r for r, _, _ in pairs)
    m = {k: np.mean([r[k] for r in rows]) for k in ('sil', 'voi', 'unv', 'unvsh_overlap', 'voish_overlap')}
    print(f'{len(rows)} 発話 × {len(SWAP)} 条件。区間の割合 無音 {m["sil"]:.1%}・有声 {m["voi"]:.1%}・無声 {m["unv"]:.1%}。'
          f'ずらしたマスクが元と重なる割合 無声 {m["unvsh_overlap"]:.1%}・有声 {m["voish_overlap"]:.1%}')


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (len(ps) - r)))
        out[i] = run
    return out


def summary(args):
    names = ['taps_airlp', 'taps_threq'] + [f'taps_sw_{c}' for c in SWAP]
    cer = {a: {n: load_cer(n, a) for n in names} for a in ASRS}
    asrs = [a for a in ASRS if cer[a]['taps_airlp'] and cer[a]['taps_threq']]
    mean = {a: {n: float(np.mean([c for _, c in d.values()])) for n, d in cer[a].items() if d and len(d) == 1000} for a in asrs}
    lines = ['# 実験 1〜3: 入れ替え実験の CER（TAPS test 1,000 発話）', '',
             'g = (条件 − 気導LP) / (補正喉 − 気導LP)（d: 壊す向き）。h = (補正喉 − 条件) / (補正喉 − 気導LP)（r: 救出の向き）。'
             '1,000 発話がそろっていない条件は -- にする', '',
             '| 条件 | ' + ' | '.join(asrs) + ' |', '|---|' + '---|' * len(asrs)]
    for n in names:
        cells = []
        for a in asrs:
            if n not in mean[a]:
                cells.append('--'); continue
            lo, hi, x = mean[a]['taps_airlp'], mean[a]['taps_threq'], mean[a][n]
            c = n.replace('taps_sw_', '')
            ratio = ((x - lo) / (hi - lo) if c.startswith('d_') else (hi - x) / (hi - lo) if c.startswith('r_') else None)
            cells.append(f'{x:.3f}' + (f' ({"g" if c.startswith("d_") else "h"}={ratio:.2f})' if ratio is not None else ''))
        lines.append(f'| {n.replace("taps_sw_", "")} | ' + ' | '.join(cells) + ' |')
    comps = ([(f'taps_sw_{d}_seg_{s}', f'taps_sw_{d}_seg_{s}sh') for d in 'dr' for s in ('unv', 'voi')]
             + [(f'taps_sw_{c}', 'taps_airlp' if c.startswith('d_') else 'taps_threq') for c in SWAP])
    lines += ['', '## 検定（話者ごとの平均 CER、両側 Wilcoxon、n = 10、比較ごとに認識器の間で Holm 補正）', '',
              '最初の 4 行が事前に決めた主な比較（ずらしたマスクとの直接比較）。残りは各条件と出発点の音声との比較', '',
              '| 比較 | ' + ' | '.join(asrs) + ' |', '|---|' + '---|' * len(asrs)]
    for x, y in comps:
        ps, diffs, have = [], [], []
        for a in asrs:
            if x in mean[a] and y in mean[a]:
                sx, sy = spk_mean(cer[a][x]), spk_mean(cer[a][y])
                ks = sorted(set(sx) & set(sy))
                ps.append(wilcoxon([sx[k] for k in ks], [sy[k] for k in ks]).pvalue)
                diffs.append(mean[a][x] - mean[a][y]); have.append(a)
        adj = dict(zip(have, zip(diffs, holm(ps)))) if ps else {}
        cells = [f'{adj[a][0]:+.3f} (p={adj[a][1]:.3f})' if a in adj else '--' for a in asrs]
        lines.append(f'| {x.replace("taps_sw_", "")} − {y.replace("taps_sw_", "")} | ' + ' | '.join(cells) + ' |')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['align', 'make0', 'summary0', 'stats', 'make', 'summary'])
    ap.add_argument('--limit', type=int, default=0, help='make: 先頭の発話だけ作る（確認用。メタデータは書かない）')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    {'align': align, 'make0': make0, 'summary0': summary0,
     'stats': stats, 'make': make, 'summary': summary}[args.stage](args)


if __name__ == '__main__':
    main()
