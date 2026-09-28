"""
スクリプト72: CE-SE の効果が「Whisper の特徴量を使う認識器」にだけ転移する仕組みの検証（学習不要）

背景
  CE（Whisper-small の CE 損失で学習した SE、λ=10）は、Whisper の特徴量（WhisperFeatureExtractor: n_fft 400,
  hop 160, log10 mel, 最大値 −8 で切り捨て）を使う認識器（Whisper 各サイズ、Qwen3-ASR）では改善し、
  それ以外（XLS-R: 波形入力、Zipformer: Kaldi fbank）では悪化した。何が転移を決めるのかを2つの仮説で調べる。

仮説1: フレーム格子への依存
  CE は Whisper の 160 サンプル（10 ms）刻みのフレーム格子に合わせた加工をしている。
  → 出力を k サンプル遅らせる（先頭に無音を足す）。k=80（半フレーム）で効果が消え、k=160（1フレーム）で
    戻るなら格子依存。TAPS にも同じずらしを掛けて、ずらし自体の影響を差し引く。
仮説2: Whisper が見ない部分（切り捨て領域）の改変
  Whisper は log10 mel の最大値 −8（約 80 dB 下）より小さい値を切り捨てる。CE はこの見えない部分を
  自由に変えており、それが他の認識器を悪化させている。
  → STFT（n_fft 400, hop 160, Hann。Whisper と同じ格子）の各ビンを、CE の出力について
    「Whisper に見える（log10 パワー ≥ 発話の最大 log10 mel − 8）」「見えない」に分け、
      ce_visible   : 見える部分は CE、見えない部分は TAPS の複素 STFT
      ce_invisible : 見える部分は TAPS、見えない部分は CE
    を作る。ce_visible が Whisper 系で改善を保ち、XLS-R / Zipformer で悪化しなければ仮説2を支持
    （しかも CE を認識器に依らず安全にする後処理になる）
  さらに CE − TAPS の差のエネルギーのうち、見えない部分に入る割合を発話ごとに記録する

条件: taps, ce, ce_s40, ce_s80, ce_s160, taps_s80, taps_s160, ce_visible, ce_invisible
認識器: whisper-small（CE の学習に使用）, qwen3-asr-1.7b（Whisper 特徴量・別モデル）, xlsr-korean, zipformer-ko
出力: results/mechanism/hyp_<asr>.csv, invisible_energy.csv, summary.csv, pairs.csv

使い方（GPU PC）
    python scripts/72_mechanism.py asr --asr whisper-small [--limit 2]
    venv-qwen/bin/python scripts/72_mechanism.py asr --asr qwen3-asr-1.7b
    python scripts/72_mechanism.py summary
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
s64 = import_module('64_retranscribe_all')
s65 = import_module('65_step0_residual_cutoff')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = BASE_DIR / 'results' / 'mechanism'
ASRS = ['whisper-small', 'qwen3-asr-1.7b', 'xlsr-korean', 'zipformer-ko']
SHIFTS = [40, 80, 160]
CONDS = (['taps', 'ce'] + [f'ce_s{k}' for k in SHIFTS] + ['taps_s80', 'taps_s160']
         + ['ce_visible', 'ce_invisible'])
N_FFT, HOP = 400, 160
WIN = torch.hann_window(N_FFT).to(DEVICE)
MEL = None


def whisper_mel_filters():
    global MEL
    if MEL is None:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')   # 80 mel（学習に使った Whisper-small と同じ）
        MEL = torch.tensor(np.array(fe.mel_filters), dtype=torch.float32, device=DEVICE)   # (201, 80)
    return MEL


def visibility_mask(y):
    """Whisper-small の特徴量で『切り捨てられない』STFT ビンのマスク（1=見える）"""
    S = torch.stft(y, N_FFT, HOP, window=WIN, return_complex=True)
    P = S.abs() ** 2                                          # (201, T)
    logmel = torch.log10((P.T @ whisper_mel_filters()).clamp(min=1e-10))   # (T, 80)
    floor = logmel.max() - 8.0
    return (torch.log10(P.clamp(min=1e-10)) >= floor).float(), S


def make_conds(taps, ce):
    n = min(len(taps), len(ce))
    t = torch.from_numpy(taps[:n]).to(DEVICE)
    c = torch.from_numpy(ce[:n]).to(DEVICE)
    out = {'taps': taps[:n], 'ce': ce[:n]}
    for k in SHIFTS:
        out[f'ce_s{k}'] = np.concatenate([np.zeros(k, np.float32), ce[:n]])
    for k in (80, 160):
        out[f'taps_s{k}'] = np.concatenate([np.zeros(k, np.float32), taps[:n]])
    vis, Sc = visibility_mask(c)
    St = torch.stft(t, N_FFT, HOP, window=WIN, return_complex=True)
    ist = lambda X: torch.istft(X, N_FFT, HOP, window=WIN, length=n).cpu().numpy().astype(np.float32)
    out['ce_visible'] = ist(Sc * vis + St * (1 - vis))
    out['ce_invisible'] = ist(St * vis + Sc * (1 - vis))
    d = (Sc - St).abs() ** 2
    stats = dict(frac_bins_invisible=round(1 - vis.mean().item(), 4),
                 frac_diff_energy_invisible=round(((d * (1 - vis)).sum() / d.sum().clamp(min=1e-12)).item(), 4))
    return out, stats


def hyp_path(name, limit=0):
    return OUT / f'hyp_{name}{"_trial" if limit else ""}.csv'


def asr(name, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = hyp_path(name, limit)
    done = set()
    if out_path.exists() and not limit:
        done = {(r['utt'], r['cond']) for r in csv.DictReader(open(out_path, encoding='utf-8'))}
    samples = s64.load_samples('test')[:limit or None]
    se = {k: s64.load_se(s64.SE_CKPTS[c]) for k, c in (('taps', 'taps'), ('ce', 'ce10.0'))}
    run = s65.load_asr(name)
    new = not out_path.exists() or limit
    f = open(out_path, 'w' if limit else 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['utt', 'spk', 'cond', 'hyp'])
    if new:
        w.writeheader()
    stat_path = OUT / 'invisible_energy.csv'
    need_stats = not stat_path.exists() and not limit      # 見えない部分の統計は最初に回した認識器で1回だけ記録
    stat_rows = []
    todo = sum(1 for s in samples for c in CONDS if (s['utt'], c) not in done)
    print(f'{name}: {todo} jobs', flush=True)
    t0, n_done = time.time(), 0
    for i, s in enumerate(samples):
        conds = [c for c in CONDS if (s['utt'], c) not in done]
        if not conds and not need_stats:
            continue
        wav, _ = sf.read(s['path'], dtype='float32')
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            taps = se['taps'](x).reshape(-1).cpu().numpy().astype(np.float32)
            ce = se['ce'](x).reshape(-1).cpu().numpy().astype(np.float32)
            audio, st = make_conds(taps, ce)
        stat_rows.append(dict(utt=s['utt'], spk=s['spk'], **st))
        for c in conds:
            hyp = run(audio[c])
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
            n_done += 1
            if limit and i < 1:
                cer = s64.capped(s64.norm(s['text']), s64.norm(hyp))
                print(f'  [{s["utt"]}] {c:<13} CER={cer:.3f} | {hyp[:40]}')
        f.flush()
        if (i + 1) % 100 == 0 and n_done:
            rate = n_done / (time.time() - t0)
            print(f'  {name} {i + 1}/{len(samples)}  残り約{(todo - n_done) / rate / 60:.0f}分', flush=True)
    f.close()
    if limit:
        print('  invisible stats:', stat_rows[:2])
    elif need_stats and stat_rows:
        with open(stat_path, 'w', newline='') as g:
            wr = csv.DictWriter(g, fieldnames=list(stat_rows[0])); wr.writeheader(); wr.writerows(stat_rows)
    print(f'{name} done → {out_path}')


def summary():
    refs = {s['utt']: s['text'] for s in s64.load_samples('test')}
    rows, pairs = [], []
    for name in ASRS:
        p = hyp_path(name)
        if not p.exists():
            continue
        ns = name == 'zipformer-ko'
        sc = defaultdict(dict)
        for r in csv.DictReader(open(p, encoding='utf-8')):
            sc[r['cond']][(r['utt'], r['spk'])] = s64.capped(s64.norm(refs[r['utt']], ns), s64.norm(r['hyp'], ns))

        def spk(d):
            by = defaultdict(list)
            for (_, sp), v in d.items():
                by[sp].append(v)
            return np.array([np.mean(by[k]) for k in sorted(by)])
        for c in CONDS:
            if c in sc:
                rows.append(dict(asr=name, cond=c, n=len(sc[c]), cer=round(np.mean(list(sc[c].values())), 4)))
        for a, b in [('taps', 'ce'), ('taps', 'ce_visible'), ('taps', 'ce_invisible'), ('ce', 'ce_visible'),
                     ('taps_s80', 'ce_s80'), ('taps_s160', 'ce_s160'), ('ce', 'ce_s40'), ('ce', 'ce_s80'),
                     ('ce', 'ce_s160'), ('taps', 'taps_s80')]:
            if a in sc and b in sc and len(sc[a]) == len(sc[b]):
                ma, mb = spk(sc[a]), spk(sc[b])
                pv = wilcoxon(ma, mb).pvalue if np.any(ma != mb) else 1.0
                pairs.append(dict(asr=name, A=a, B=b, cer_A=round(ma.mean(), 4), cer_B=round(mb.mean(), 4),
                                  rel=round(100 * (mb.mean() / ma.mean() - 1), 1), spk_B_better=int((mb < ma).sum()),
                                  p_spk=round(pv, 4)))
    OUT.mkdir(parents=True, exist_ok=True)
    for fn, data in (('summary.csv', rows), ('pairs.csv', pairs)):
        if data:
            with open(OUT / fn, 'w', newline='', encoding='utf-8') as g:
                wr = csv.DictWriter(g, fieldnames=list(data[0])); wr.writeheader(); wr.writerows(data)
    cer = {(r['asr'], r['cond']): r['cer'] for r in rows if r['n'] == len(refs)}
    names = [a for a in ASRS if (a, 'taps') in cer]
    print('CER（句読点除去、Zipformer は空白も除去）  括弧は TAPS 比')
    print(f'{"cond":<14}' + ''.join(f'{a[:14]:>22}' for a in names))
    for c in CONDS:
        print(f'{c:<14}' + ''.join(
            f'{cer[(a, c)]:>12.4f} ({100 * (cer[(a, c)] / cer[(a, "taps")] - 1):+5.1f}%)' if (a, c) in cer else f'{"-":>22}'
            for a in names))
    sp = OUT / 'invisible_energy.csv'
    if sp.exists():
        st = list(csv.DictReader(open(sp)))
        print(f"\nWhisper に見えないビンの割合 {np.mean([float(r['frac_bins_invisible']) for r in st]):.3f}、"
              f"CE−TAPS の差のエネルギーのうち見えない部分 {np.mean([float(r['frac_diff_energy_invisible']) for r in st]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['asr', 'summary'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    asr(a.asr, a.limit) if a.stage == 'asr' else summary()


if __name__ == '__main__':
    main()
