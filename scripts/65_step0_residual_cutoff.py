"""
スクリプト65: Step 0 — 残差スイープ × カットオフ行列 × 6認識器（学習不要の分析）

目的（results/STATUS_AND_NEXT_2026-09-24.md の Step 0）
  CE-SE（λ=10）が TAPS SE に加えた変化のうち、どれが「Whisper専用」でどれが認識器に依らないかを切り分ける。

  1. 残差スイープ: x = TAPS + α(CE − TAPS)、α ∈ {0, .25, .5, .75, 1, 1.25, 1.5}
       α=0 は TAPS、α=1 は CE。Whisper で単調に改善し XLS-R で単調に悪化すれば、
       CE が加えた変化そのものが Whisper 専用であることを直接示せる。
  2. カットオフ行列: TAPS / CE に 3–7 kHz のローパス（8 kHz = 無処理）
  3. 櫛ノッチ（script 62 と同じ 250 Hz 格子）と 4 kHz 境界の低域/高域入れ替え
  script 62 では 2・3 を whisper-small でしか評価していなかった。

認識器: whisper-base / small / medium（faster-whisper, beam 5）、FT Whisper、
        MMS-1B（target_lang=kor）、XLS-R Korean（CTC, greedy）
CER は句読点除去後（script 63 の norm/capped）を主、raw も保存。検定は話者単位（n=10）。

ステージ（すべて再開可能）
  gen      : test 1000発話の SE 出力と全条件の波形を data/processed/step0/<utt>.npz に保存
  asr      : --asr で指定した認識器で全条件を書き起こし、results/step0/hyp_<asr>.csv に追記
             （認識器ごとに別プロセスで並列に回せる）
  summary  : results/step0/summary.csv（条件別CER）と pairs.csv（話者単位の比較）

使い方（GPU PC）
    python scripts/65_step0_residual_cutoff.py gen
    python scripts/65_step0_residual_cutoff.py asr --asr whisper-small [--limit 5]
    python scripts/65_step0_residual_cutoff.py summary
"""

import argparse, csv, sys, time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr, wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s62 = import_module('62_robustness_eval')     # comb_notch, split4k, load_ctc, ctc_transcribe
s64 = import_module('64_retranscribe_all')    # load_se, load_samples, asr_source, norm, capped

SR = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE = BASE_DIR / 'data' / 'processed' / 'step0'
OUT = BASE_DIR / 'results' / 'step0'

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
CUTOFFS = [3000, 4000, 5000, 6000, 7000]
MIX = [f'mix{a:.2f}' for a in ALPHAS]                       # mix0.00 = TAPS, mix1.00 = CE
LPF = [f'{src}_lpf{fc // 1000}k' for src in ('taps', 'ce') for fc in CUTOFFS]
OTHER = ['taps_notch', 'ce_notch', 'hyb_tapsLow_ceHigh', 'hyb_ceLow_tapsHigh']
CONDS = MIX + LPF + OTHER

WHISPER = ['whisper-base', 'whisper-small', 'whisper-medium', 'whisper-ft']
ASRS = WHISPER + list(s62.CTC_MODELS)

_SOS = {fc: butter(8, fc, btype='low', fs=SR, output='sos') for fc in CUTOFFS}


def lowpass(x, fc):
    return sosfiltfilt(_SOS[fc], x).astype(np.float32)


def make_conds(taps, ce):
    n = min(len(taps), len(ce))
    taps, ce = taps[:n].astype(np.float32), ce[:n].astype(np.float32)
    out = {f'mix{a:.2f}': (taps + a * (ce - taps)).astype(np.float32) for a in ALPHAS}
    for fc in CUTOFFS:
        out[f'taps_lpf{fc // 1000}k'] = lowpass(taps, fc)
        out[f'ce_lpf{fc // 1000}k'] = lowpass(ce, fc)
    tl, th = s62.split4k(taps)
    cl, ch = s62.split4k(ce)
    out.update(taps_notch=s62.comb_notch(taps), ce_notch=s62.comb_notch(ce),
               hyb_tapsLow_ceHigh=tl + ch, hyb_ceLow_tapsHigh=cl + th)
    return out


# ---------------- gen ----------------
def gen(limit):
    CACHE.mkdir(parents=True, exist_ok=True)
    samples = s64.load_samples('test')[:limit or None]
    se_taps = s64.load_se(s64.SE_CKPTS['taps'])
    se_ce = s64.load_se(s64.SE_CKPTS['ce10.0'])
    t0 = time.time()
    for i, s in enumerate(samples):
        path = CACHE / f"{s['utt']}.npz"
        if path.exists():
            continue
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            taps = se_taps(x).squeeze().cpu().numpy()
            ce = se_ce(x).squeeze().cpu().numpy()
        np.savez(path, **make_conds(taps, ce))
        if (i + 1) % 100 == 0:
            print(f'  gen {i + 1}/{len(samples)}  {(time.time() - t0) / 60:.1f}min', flush=True)
    print(f'gen done: {len(samples)} utts → {CACHE}')


# ---------------- asr ----------------
def load_asr(name):
    if name in WHISPER:
        from faster_whisper import WhisperModel
        src = s64.asr_source(name)
        if src is None:
            raise SystemExit(f'{name}: モデルが見つからない')
        model = WhisperModel(src, device=DEVICE, compute_type='float16')

        def run(wav):
            segs, _ = model.transcribe(wav, language='ko', beam_size=5)
            return ''.join(g.text for g in segs).strip()
        return run
    proc, model = s62.load_ctc(name, s62.CTC_MODELS[name])
    return lambda wav: s62.ctc_transcribe(proc, model, wav)


def asr(name, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / (f'hyp_{name}.csv' if not limit else f'hyp_{name}_trial.csv')
    done = set()
    if out_path.exists() and not limit:
        done = {(r['utt'], r['cond']) for r in csv.DictReader(open(out_path, encoding='utf-8'))}
    samples = s64.load_samples('test')[:limit or None]
    run = load_asr(name)
    new = not out_path.exists() or limit
    f = open(out_path, 'w' if limit else 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['utt', 'spk', 'cond', 'hyp'])
    if new:
        w.writeheader()
    t0, n = time.time(), 0
    todo = sum(1 for s in samples for c in CONDS if (s['utt'], c) not in done)
    print(f'{name}: {todo} jobs', flush=True)
    for i, s in enumerate(samples):
        conds = [c for c in CONDS if (s['utt'], c) not in done]
        if not conds:
            continue
        z = np.load(CACHE / f"{s['utt']}.npz")
        for c in conds:
            hyp = run(z[c])
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
            n += 1
            if limit and i < 2:
                cer = s64.capped(s64.norm(s['text']), s64.norm(hyp))
                print(f'  [{s["utt"]}] {c:<20} CER={cer:.3f} | {hyp[:40]}')
        f.flush()
        if (i + 1) % 50 == 0:
            rate = n / (time.time() - t0)
            print(f'  {name} {i + 1}/{len(samples)}  残り約{(todo - n) / rate / 60:.0f}分', flush=True)
    f.close()
    print(f'{name} done → {out_path}')


# ---------------- summary ----------------
def spk_means(d):
    by = defaultdict(list)
    for (utt, spk), v in d.items():
        by[spk].append(v)
    spk = sorted(by)
    return spk, np.array([np.mean(by[k]) for k in spk])


def summary():
    refs = {s['utt']: s['text'] for s in s64.load_samples('test')}
    sc = {}   # (asr, cond) -> {(utt, spk): (nopunct, raw)}
    for name in ASRS:
        p = OUT / f'hyp_{name}.csv'
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding='utf-8')):
            ref, h = refs[r['utt']], r['hyp']
            sc.setdefault((name, r['cond']), {})[(r['utt'], r['spk'])] = (
                s64.capped(s64.norm(ref), s64.norm(h)), s64.capped(ref, h))

    rows = []
    for (name, cond), d in sorted(sc.items(), key=lambda kv: (ASRS.index(kv[0][0]), CONDS.index(kv[0][1]))):
        v = np.array(list(d.values()))
        rows.append(dict(asr=name, cond=cond, n=len(d), cer_nopunct=round(v[:, 0].mean(), 4),
                         cer_raw=round(v[:, 1].mean(), 4)))
    with open(OUT / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def cmp(name, a, b):
        da, db = sc.get((name, a)), sc.get((name, b))
        if not da or not db or len(da) != len(db):
            return None
        _, ma = spk_means({k: v[0] for k, v in da.items()})
        _, mb = spk_means({k: v[0] for k, v in db.items()})
        p = wilcoxon(ma, mb).pvalue if np.any(ma != mb) else 1.0
        return dict(asr=name, A=a, B=b, cer_A=round(ma.mean(), 4), cer_B=round(mb.mean(), 4),
                    rel=round(100 * (mb.mean() / ma.mean() - 1), 1),
                    spk_B_better=int((mb < ma).sum()), n_spk=len(ma), p_spk=round(p, 4))

    pairs = []
    for name in ASRS:
        pairs += [cmp(name, 'mix0.00', c) for c in MIX[1:]]                       # α vs TAPS
        pairs += [cmp(name, 'mix0.00', f'taps_lpf{fc // 1000}k') for fc in CUTOFFS]  # LPF の効果（TAPS）
        pairs += [cmp(name, 'mix1.00', f'ce_lpf{fc // 1000}k') for fc in CUTOFFS]    # LPF の効果（CE）
        pairs += [cmp(name, f'taps_lpf{fc // 1000}k', f'ce_lpf{fc // 1000}k') for fc in CUTOFFS]
        pairs += [cmp(name, 'mix0.00', 'taps_notch'), cmp(name, 'mix1.00', 'ce_notch'),
                  cmp(name, 'mix0.00', 'hyb_tapsLow_ceHigh'), cmp(name, 'mix0.00', 'hyb_ceLow_tapsHigh')]
    pairs = [p for p in pairs if p]
    with open(OUT / 'pairs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)

    # 表示: 残差スイープ（相対変化 vs TAPS）とカットオフ行列
    cer = {(r['asr'], r['cond']): r['cer_nopunct'] for r in rows}
    names = [a for a in ASRS if (a, 'mix0.00') in cer]
    print('\n残差スイープ  CER（句読点除去）  α=0: TAPS, α=1: CE λ=10')
    print(f'{"ASR":<15}' + ''.join(f'{a:>8.2f}' for a in ALPHAS) + f'{"ρ(α,CER)":>10}')
    for a in names:
        ys = [cer.get((a, c)) for c in MIX]
        rho = spearmanr(ALPHAS, ys).statistic if None not in ys else float('nan')
        print(f'{a:<15}' + ''.join(f'{y:>8.4f}' if y is not None else f'{"-":>8}' for y in ys) + f'{rho:>10.2f}')
    print('\nカットオフ行列  CER（句読点除去）  8k = 無処理')
    for src, full in (('taps', 'mix0.00'), ('ce', 'mix1.00')):
        print(f'[{src}] {"ASR":<15}' + ''.join(f'{fc // 1000}k'.rjust(8) for fc in CUTOFFS) + '8k'.rjust(8)
              + 'notch'.rjust(8))
        for a in names:
            ys = [cer.get((a, f'{src}_lpf{fc // 1000}k')) for fc in CUTOFFS] + [cer.get((a, full)),
                                                                                 cer.get((a, f'{src}_notch'))]
            print(f'       {a:<15}' + ''.join(f'{y:>8.4f}' if y is not None else f'{"-":>8}' for y in ys))
    print(f'\nSaved: {OUT / "summary.csv"}, {OUT / "pairs.csv"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gen', 'asr', 'summary'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'gen':
        gen(a.limit)
    elif a.stage == 'asr':
        if not a.asr:
            ap.error('--asr が必要')
        asr(a.asr, a.limit)
    else:
        summary()


if __name__ == '__main__':
    main()
