"""
スクリプト76: 方向A（分析）— 出力が「自然な声（気導マイクの音）」に近いほど、どの認識器でも誤りが減るか

作業仮説: どの認識器にも効く音 = 自然な声に近い音、特定の認識器にだけ効く音 = その癖に合わせた音。
これまでに評価した全条件（TAPS、CE λ違い、Demucs/TSTNN、2出力平均、振幅統合、蒸留、生成モデル…）について、
test 1000発話の出力を作り直し、気導マイクの音との距離を測って、9認識器のCER（句読点除去）との関係を見る。

距離（どれも認識器・学習に使っていない表現で測る）:
  d_hubert : HuBERT-Large（facebook/hubert-large-ll60k、自己教師あり、ASR未学習）の全層について、
             同じ発話の気導音とのフレームごとのコサイン距離の平均（対応あり）
  d_mel    : 対数メルスペクトログラム（80次元）の L1 距離（対応あり、音響的な近さ）
  fad      : HuBERT 中間層（12層目）の発話平均ベクトルの分布と、気導音の分布の Fréchet 距離（対応なし、"自然さ"）

使い方（GPU PC）:
    python scripts/76_naturalness_vs_cer.py feat [--limit N] [--conds ...]
    python scripts/76_naturalness_vs_cer.py analyze
出力: results/naturalness/{metrics_utt.csv, pooled.npz, by_cond.csv, corr.csv}
"""

import argparse
import csv
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s64 = import_module('64_retranscribe_all')
s66 = import_module('66_lambda_cross_asr')
s67 = import_module('67_magnitude_fusion')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = BASE_DIR / 'results' / 'naturalness'
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
SUMMARIES = [BASE_DIR / 'results' / 'lambda_asr' / 'summary.csv', BASE_DIR / 'results' / 'fusion' / 'summary.csv']
FUSION = [c for c in s67.CONDS if c != 'taps']
SSL = 'facebook/hubert-large-ll60k'
MID = 12


def cer_table():
    """{(asr, cond): CER}。test 1000発話そろっているものだけ"""
    t = {}
    for p in SUMMARIES:
        if p.exists():
            for r in csv.DictReader(open(p, encoding='utf-8')):
                if int(r['n']) == 1000:
                    t[(r['asr'], r['cond'])] = float(r['cer_nopunct'])
    return t


def all_conds():
    return sorted({c for (_, c) in cer_table()} | {'no_se'})


class Features:
    def __init__(self):
        from transformers import AutoFeatureExtractor, HubertModel
        fe = AutoFeatureExtractor.from_pretrained(SSL)
        self.norm = getattr(fe, 'do_normalize', False)
        self.m = HubertModel.from_pretrained(SSL).to(DEVICE).eval()
        self.mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=80).to(DEVICE)

    @torch.no_grad()
    def __call__(self, wav):
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(DEVICE)[None]
        if self.norm:
            x = (x - x.mean()) / (x.std() + 1e-7)
        hs = self.m(x, output_hidden_states=True).hidden_states[1:]          # 24 x (1, T, 1024)
        lm = torch.log(self.mel(torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(DEVICE)) + 1e-6)
        return [h[0] for h in hs], lm


def paired(fa, fb):
    ha, la = fa
    hb, lb = fb
    T = min(ha[0].shape[0], hb[0].shape[0])
    d = [1 - torch.nn.functional.cosine_similarity(a[:T], b[:T], dim=-1).mean().item() for a, b in zip(ha, hb)]
    F_ = min(la.shape[-1], lb.shape[-1])
    return float(np.mean(d)), (la[:, :F_] - lb[:, :F_]).abs().mean().item()


def feat(args):
    OUT.mkdir(parents=True, exist_ok=True)
    conds = args.conds or all_conds()
    samples = s64.load_samples('test')[:args.limit or None]
    # 出力の作り方: 保存済み音声 / script 66 の SE と平均 / script 67 の振幅統合 / 処理なし
    wavdir = {c: s66.SE_WAV / c / 'test' for c in conds if (s66.SE_WAV / c / 'test').is_dir()}
    need66 = sorted({m for c in conds if c not in wavdir and c not in FUSION and c != 'no_se' for m in s66.AVGS.get(c, (c,))})
    se = {}
    for m in sorted(set(need66) | (set(s67.SOURCES) if any(c in FUSION for c in conds) else set())):
        try:
            se[m] = s66.load_any_se(m)
        except Exception as e:     # 重みがない条件は飛ばす
            print(f'  skip SE {m}: {e}')
    conds = [c for c in conds if c in wavdir or c == 'no_se'
             or (c in FUSION and all(m in se for m in s67.SOURCES))
             or (c not in FUSION and all(m in se for m in s66.AVGS.get(c, (c,))))]
    print(f'{len(conds)} conditions: {conds}', flush=True)
    fx = Features()
    rows, pooled = [], defaultdict(list)
    for i, s in enumerate(samples):
        wav, _ = sf.read(s['path'], dtype='float32')
        air, _ = sf.read(TAPS_DIR / 'acoustic' / 'test' / Path(s['path']).name, dtype='float32')
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outs = {m: f(x).reshape(-1) for m, f in se.items()}
        audio = {'no_se': wav}
        if any(c in FUSION for c in conds):
            n = min(outs[m].shape[-1] for m in s67.SOURCES)
            audio.update(s67.fuse_all({m: outs[m][:n] for m in s67.SOURCES}))
        for c in conds:
            if c in wavdir:
                audio[c] = sf.read(wavdir[c] / f"{s['utt']}.wav", dtype='float32')[0]
            elif c in s66.AVGS:
                a, b = (outs[m].cpu().numpy() for m in s66.AVGS[c])
                n = min(len(a), len(b))
                audio[c] = s66.mag_avg(a[:n], b[:n]) if c in s66.MAVGS else 0.5 * (a[:n] + b[:n])
            elif c not in audio:
                audio[c] = outs[c].cpu().numpy()
        fa = fx(air)
        pooled['air'].append(fa[0][MID - 1].mean(0).cpu().numpy())
        for c in conds:
            f = fx(audio[c])
            dh, dm = paired(f, fa)
            rows.append(dict(utt=s['utt'], spk=s['spk'], cond=c, d_hubert=round(dh, 5), d_mel=round(dm, 5)))
            pooled[c].append(f[0][MID - 1].mean(0).cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(samples)}', flush=True)
    tag = '_trial' if args.limit else ''
    with open(OUT / f'metrics_utt{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT / f'pooled{tag}.npz', **{k: np.stack(v) for k, v in pooled.items()})
    print(f'saved → {OUT}')


def frechet(a, b):
    from scipy.linalg import sqrtm
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = sqrtm(s1 @ s2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def analyze(args):
    from scipy.stats import spearmanr
    tag = '_trial' if args.limit else ''
    m = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(OUT / f'metrics_utt{tag}.csv', encoding='utf-8')):
        for k in ('d_hubert', 'd_mel'):
            m[r['cond']][k].append(float(r[k]))
    P = np.load(OUT / f'pooled{tag}.npz')
    cer = cer_table()
    asrs = sorted({a for (a, _) in cer})
    by = []
    for c in sorted(m):
        row = dict(cond=c, d_hubert=np.mean(m[c]['d_hubert']), d_mel=np.mean(m[c]['d_mel']), fad=frechet(P[c], P['air']))
        rel = [np.log(cer[(a, c)] / cer[(a, 'taps')]) for a in asrs if (a, c) in cer and (a, 'taps') in cer]
        row['n_asr'] = len(rel)
        row['mean_rel'] = 100 * (np.exp(np.mean(rel)) - 1) if rel else np.nan       # 認識器の幾何平均で見た TAPS 比 [%]
        row['worst_rel'] = 100 * (np.exp(np.max(rel)) - 1) if rel else np.nan       # 一番悪化した認識器の TAPS 比 [%]
        for a in asrs:
            row[a] = cer.get((a, c), np.nan)
        by.append(row)
    keys = ['cond', 'n_asr', 'd_hubert', 'd_mel', 'fad', 'mean_rel', 'worst_rel'] + asrs
    with open(OUT / f'by_cond{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in by])
    corr = []
    for metric in ('d_hubert', 'd_mel', 'fad'):
        for target in ['mean_rel', 'worst_rel'] + asrs:
            pts = [(r[metric], r[target]) for r in by if not np.isnan(r[target]) and (target in asrs or r['n_asr'] == 9)]
            if len(pts) >= 5:
                x, y = zip(*pts)
                rho = spearmanr(x, y)
                corr.append(dict(metric=metric, target=target, n_cond=len(pts), rho=round(rho.statistic, 3), p=round(rho.pvalue, 4)))
    with open(OUT / f'corr{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(corr[0])); w.writeheader(); w.writerows(corr)
    print(f'\n条件ごとの距離と CER（{len(by)} 条件）')
    print(f'{"cond":<26}{"n":>3}{"d_hub":>8}{"d_mel":>8}{"fad":>8}{"平均":>8}{"最悪":>8}')
    for r in sorted(by, key=lambda r: r['d_hubert']):
        print(f'{r["cond"]:<26}{r["n_asr"]:>3}{r["d_hubert"]:>8.4f}{r["d_mel"]:>8.3f}{r["fad"]:>8.1f}'
              f'{r["mean_rel"]:>+8.1f}{r["worst_rel"]:>+8.1f}')
    print('\nSpearman ρ（条件をまたいで: 距離が小さいほど CER が低いなら正）')
    for r in corr:
        print(f'  {r["metric"]:<9}{r["target"]:<26} n={r["n_cond"]:<3} ρ={r["rho"]:+.3f}  p={r["p"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['feat', 'analyze'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--conds', nargs='+')
    args = ap.parse_args()
    {'feat': feat, 'analyze': analyze}[args.stage](args)


if __name__ == '__main__':
    main()
