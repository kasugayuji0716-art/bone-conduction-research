"""
スクリプト73: 雑音下での評価 — 気導マイク＋ASR と 喉マイク＋SE＋ASR のどちらが強いか

目的
  喉マイクを使う理由は、周囲がうるさくても声を拾えること。雑音が大きくなったとき、喉マイク＋SE が
  気導マイク（同時収録の TAPS acoustic）を上回るのはどの SNR からか、それが SE の種類でどう変わるかを調べる。

雑音
  DEMAND（Thiemann et al. 2013、CC-BY-4.0、16 kHz 版の ch01）。17 環境のうち test には 8 環境を使う:
  PCAFETER, STRAFFIC, TMETRO, TBUS, PSTATION, DKITCHEN, OMEETING, NRIVER（残り 9 環境は dev 用に取っておく）
  発話ごとに、環境と開始位置を乱数で選ぶ（発話 ID から決まるシード、再現可能）。SNR は発話全体のパワーで定義

条件（test 1000 発話）
  気導マイク（SE なし）: air_clean, air_snr{20,10,5,0,-5}
  喉マイク＋SE: {taps, ce, m4amp} × 喉マイク側 SNR {clean, 20, 10, 5}
    - 喉マイクは周囲の音を強く減衰して拾う。漏れの大きさは実測していないので、喉マイク側の SNR を直接振る
      （例: 気導 0 dB のとき喉 20 dB なら、漏れが気導より 20 dB 小さい状況に相当）。論文では感度分析として扱う
    - 喉マイクの雑音は、喉マイク音声と同じく 16k → 8k → 16k を通して 4 kHz 以上をなくしてから足す（TAPS の喉マイクは
      8 kHz 収録で 4 kHz 以上は記録されないため。全帯域の雑音を足すと、実際には入らない帯域に雑音が入り不当に不利になる。
      2026-09-29 の試行で気付き修正）
    - ce: CE λ=10（Whisper-small の CE 損失で学習）、m4amp: 4つの SE（TAPS/Demucs/TSTNN/CE0）の振幅平均（位相 TAPS）
認識器: whisper-small, whisper-large-v3-turbo, qwen3-asr-1.7b, xlsr-korean, zipformer-ko
CER: 句読点除去後（Zipformer は空白も除去）。検定は話者単位

ステージ
  prep    DEMAND の test/dev 環境をダウンロードし ch01 を ~/data_demand/<ENV>.wav に保存
  gen     全条件の波形を data/processed/noise73/<utt>.npz に保存（SE を1回だけ動かす。再開可能）
  asr     書き起こし（results/noise/hyp_<asr>.csv、再開可能）
  summary results/noise/summary.csv と表示

使い方（GPU PC）
    python scripts/73_noise_eval.py prep
    python scripts/73_noise_eval.py gen [--limit 3]
    python scripts/73_noise_eval.py asr --asr whisper-small [--limit 1]
    venv-qwen/bin/python scripts/73_noise_eval.py asr --asr qwen3-asr-1.7b
    python scripts/73_noise_eval.py summary
"""

import argparse, csv, io, sys, time, urllib.request, zipfile, zlib
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s64 = import_module('64_retranscribe_all')
s65 = import_module('65_step0_residual_cutoff')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEMAND_DIR = Path.home() / 'data_demand'
CACHE = BASE_DIR / 'data' / 'processed' / 'noise73'
OUT = BASE_DIR / 'results' / 'noise'
TEST_ENVS = ['PCAFETER', 'STRAFFIC', 'TMETRO', 'TBUS', 'PSTATION', 'DKITCHEN', 'OMEETING', 'NRIVER']
DEV_ENVS = ['PRESTO', 'DWASHING', 'OOFFICE', 'OHALLWAY', 'DLIVING', 'NPARK', 'NFIELD', 'SPSQUARE', 'TCAR']
AIR_SNRS = [20, 10, 5, 0, -5]
THROAT_SNRS = [20, 10, 5]
SES = ['taps', 'ce', 'm4amp']
CONDS = (['air_clean'] + [f'air_snr{s}' for s in AIR_SNRS]
         + [f'throat_{se}_{lv}' for se in SES for lv in ['clean'] + [f'snr{s}' for s in THROAT_SNRS]])
ASRS = ['whisper-small', 'whisper-large-v3-turbo', 'qwen3-asr-1.7b', 'xlsr-korean', 'zipformer-ko']
ZENODO = 'https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1'


def prep():
    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    for env in TEST_ENVS + DEV_ENVS:
        out = DEMAND_DIR / f'{env}.wav'
        if out.exists():
            continue
        for attempt in range(3):                 # Zenodo は途中で切れることがある
            try:
                data = urllib.request.urlopen(ZENODO.format(env=env), timeout=600).read()
                break
            except Exception as e:
                print(f'  {env}: retry ({e.__class__.__name__})', flush=True)
        else:
            print(f'  {env}: skipped'); continue
        z = zipfile.ZipFile(io.BytesIO(data))
        name = next(n for n in z.namelist() if n.endswith('ch01.wav'))
        x, sr = sf.read(io.BytesIO(z.read(name)), dtype='float32')
        assert sr == 16000, sr
        sf.write(out, x, 16000)
        print(f'{env}: {len(x) / sr / 60:.1f} min', flush=True)
    print('prep done')


_NOISE = {}


def noise_for(utt, n, envs):
    """発話 ID から決まるシードで環境と開始位置を選び、長さ n の雑音を返す"""
    rng = np.random.default_rng(zlib.crc32(utt.encode()))
    env = envs[rng.integers(len(envs))]
    if env not in _NOISE:
        _NOISE[env] = sf.read(DEMAND_DIR / f'{env}.wav', dtype='float32')[0]
    z = _NOISE[env]
    start = rng.integers(0, len(z) - n) if len(z) > n else 0
    seg = z[start:start + n] if len(z) > n else np.resize(z, n)
    return seg.astype(np.float32), env


def add_noise(clean, noise, snr_db):
    ps = np.mean(clean.astype(np.float64) ** 2)
    pn = np.mean(noise.astype(np.float64) ** 2) + 1e-12
    return (clean + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def gen(limit):
    s66 = import_module('66_lambda_cross_asr')
    s67 = import_module('67_magnitude_fusion')
    CACHE.mkdir(parents=True, exist_ok=True)
    se = {c: s66.load_any_se(c) for c in s67.SOURCES}
    ce = s66.load_any_se('ce10.0')
    meta_rows = []
    t0 = time.time()
    samples = s64.load_samples('test')[:limit or None]
    for i, s in enumerate(samples):
        path = CACHE / f"{s['utt']}.npz"
        throat, _ = sf.read(s['path'], dtype='float32')
        air, _ = sf.read(str(s['path']).replace('/throat/', '/acoustic/'), dtype='float32')
        n = min(len(throat), len(air))
        throat, air = throat[:n], air[:n]
        noise, env = noise_for(s['utt'], n, TEST_ENVS)
        meta_rows.append(dict(utt=s['utt'], env=env))
        if path.exists():
            continue
        out = {'air_clean': air}
        for snr in AIR_SNRS:
            out[f'air_snr{snr}'] = add_noise(air, noise, snr)
        # 喉マイクには、同じ環境の別区間の雑音（気導とは独立）を使う
        noise_t, _ = noise_for(s['utt'] + '#throat', n, TEST_ENVS)
        noise_t = resample_poly(resample_poly(noise_t, 1, 2), 2, 1)[:n].astype(np.float32)   # 4 kHz 以上をなくす（喉マイクの収録帯域）
        for lv in ['clean'] + [f'snr{v}' for v in THROAT_SNRS]:
            x = throat if lv == 'clean' else add_noise(throat, noise_t, int(lv[3:]))
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outs = {k: m(xt).reshape(-1) for k, m in se.items()}
                m_ = min(v.shape[-1] for v in outs.values())
                S = {k: s67.stft(v[:m_]) for k, v in outs.items()}
                A = torch.stack([S[k].abs() for k in s67.SOURCES]).mean(0)
                out[f'throat_m4amp_{lv}'] = s67.istft(A * torch.exp(1j * S['taps'].angle()), m_)
                out[f'throat_taps_{lv}'] = outs['taps'].cpu().numpy().astype(np.float32)
                out[f'throat_ce_{lv}'] = ce(xt).reshape(-1).cpu().numpy().astype(np.float32)
        np.savez(path, **out)
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f'  gen {i + 1}/{len(samples)}  {(time.time() - t0) / 60:.1f} min', flush=True)
    if not limit:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / 'noise_assignment.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['utt', 'env']); w.writeheader(); w.writerows(meta_rows)
    print(f'gen done: {len(samples)} utts → {CACHE}')


def hyp_path(name, limit=0):
    return OUT / f'hyp_{name}{"_trial" if limit else ""}.csv'


def asr(name, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = hyp_path(name, limit)
    done = set()
    if out_path.exists() and not limit:
        done = {(r['utt'], r['cond']) for r in csv.DictReader(open(out_path, encoding='utf-8'))}
    samples = s64.load_samples('test')[:limit or None]
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
        z = np.load(CACHE / f"{s['utt']}.npz")
        for c in conds:
            hyp = run(z[c])
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
            n_done += 1
            if limit and i < 1:
                ns = name == 'zipformer-ko'
                cer = s64.capped(s64.norm(s['text'], ns), s64.norm(hyp, ns))
                print(f'  [{s["utt"]}] {c:<20} CER={cer:.3f} | {hyp[:36]}')
        f.flush()
        if (i + 1) % 100 == 0 and n_done:
            rate = n_done / (time.time() - t0)
            print(f'  {name} {i + 1}/{len(samples)}  残り約{(todo - n_done) / rate / 60:.0f}分', flush=True)
    f.close()
    print(f'{name} done → {out_path}')


def summary():
    refs = {s['utt']: s['text'] for s in s64.load_samples('test')}
    rows = []
    for name in ASRS:
        p = hyp_path(name)
        if not p.exists():
            continue
        ns = name == 'zipformer-ko'
        sc = defaultdict(dict)
        for r in csv.DictReader(open(p, encoding='utf-8')):
            sc[r['cond']][r['utt']] = s64.capped(s64.norm(refs[r['utt']], ns), s64.norm(r['hyp'], ns))
        for c in CONDS:
            if len(sc.get(c, {})) == len(refs):
                rows.append(dict(asr=name, cond=c, n=len(sc[c]), cer=round(float(np.mean(list(sc[c].values()))), 4)))
    if not rows:
        print('no rows yet'); return
    with open(OUT / 'summary.csv', 'w', newline='', encoding='utf-8') as g:
        wr = csv.DictWriter(g, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    cer = {(r['asr'], r['cond']): r['cer'] for r in rows}
    names = [a for a in ASRS if any(k[0] == a for k in cer)]
    print('CER（句読点除去、Zipformer は空白も除去）')
    print(f'{"cond":<22}' + ''.join(f'{a[:16]:>18}' for a in names))
    for c in CONDS:
        print(f'{c:<22}' + ''.join(f'{cer[(a, c)]:>18.4f}' if (a, c) in cer else f'{"-":>18}' for a in names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prep', 'gen', 'asr', 'summary'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'prep':
        prep()
    elif a.stage == 'gen':
        gen(a.limit)
    elif a.stage == 'asr':
        if not a.asr:
            ap.error('--asr が必要')
        asr(a.asr, a.limit)
    else:
        summary()


if __name__ == '__main__':
    main()
