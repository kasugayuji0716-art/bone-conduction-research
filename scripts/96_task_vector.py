"""
スクリプト96: タスク算術（Ilharco et al., ICLR 2023）で CE 損失の効きを伸び縮み・反転する（案 A・B）

  τ_CE = θ_CE10 − θ_λ0（どちらも TAPS SE から追加学習した SE-Conformer）
  θ(α) = θ_λ0 + α τ_CE
    A: α = 0.5, 1.5, 2（縮める・伸ばす。α=0 は λ=0、α=1 は CE-SE と同じ重み）
    B: α = −0.5, −1（CE の方向を逆に進む）
  注意: λ=0 の最良エポックは 7、λ=10 は 14。τ_CE には「再構成損失で余分に 7 エポック学習した差」も入る

  build : checkpoints/tv_a<α>/best.th を作る（α=0, 1 は元の重みと一致するかを確かめる）
  wav   : dev の SE 出力 se_wav/tv_a<α>/dev と、生成×4 との振幅の融合 se_wav/fm_avg4_mavg_tv_a<α>/dev を書く
          （融合は script 66 の mag_avg と同じ: STFT 512/128 Hann、振幅を半々、位相は生成×4）
  認識  : python scripts/66_lambda_cross_asr.py asr --asr <名前> --split dev --conds <CONDS>
  summary: 認識器ごとの CER と、α=1（SE 単体は CE-SE、融合は今の融合）との比較（話者単位 Wilcoxon、n=10）

dev は α を選ぶための探索。test は見ていない。
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
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
from dataset_cfg import SE_WAV

ALPHAS = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
# 2026-11-19: 重みをそのまま外挿すると BatchNorm の running_var が 0 以下になり出力が NaN になった
# （α=−0.05 と 1.1 で既に1層が負）。そこで τ は学習パラメータだけに足し、BatchNorm の統計は
# α ごとに TAPS train の喉マイク音声 N_CALIB 発話で推定し直す（モデル統合で普通の処置）。名前は tvr_。
# 元の重みの α=0（λ=0）と α=1（CE-SE）は tv_a0 / tv_a1 として参照に残す（推定し直しの影響を見るため）
N_CALIB = 400
SE_CONDS = [f'tvr_a{a:g}' for a in ALPHAS]
FUSED = [f'fm_avg4_mavg_tvr_a{a:g}' for a in ALPHAS]
REFS = ['taps', 'fm_taps_avg4', 'tv_a0', 'tv_a1', 'fm_avg4_mavg_tv_a0', 'fm_avg4_mavg_tv_a1']
CONDS = REFS + SE_CONDS + FUSED
ASR_LIST = ['whisper-small', 'whisper-base', 'whisper-medium', 'whisper-large-v3-turbo', 'whisper-ft',
            'qwen3-asr-1.7b', 'xlsr-korean', 'mms-1b-all', 'zipformer-ko']
CKPT = BASE_DIR / 'checkpoints'


def _state(path):
    st = torch.load(path, map_location='cpu', weights_only=False)
    return st['model'] if isinstance(st, dict) and 'model' in st else st


def _is_bn_stat(k):
    return k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))


def build(args):
    s64 = import_module('64_retranscribe_all')
    from models.seconformer import seconformer
    t0 = _state(CKPT / 'ce_v2_lambda_0.0' / 'best.th')
    t1 = _state(CKPT / 'ce_v2_lambda_10.0' / 'best.th')
    assert t0.keys() == t1.keys()
    rng = np.random.default_rng(0)
    train = s64.load_samples('train')
    calib = [train[i] for i in rng.choice(len(train), N_CALIB, replace=False)]
    xs = []
    for s in calib:
        x, _ = sf.read(s['path'], dtype='float32')
        xs.append(torch.from_numpy(x.mean(axis=1) if x.ndim > 1 else x)[None])
    m = seconformer(**s64.TAPS_SE_CONFIG).to(s64.DEVICE)
    bns = [mod for mod in m.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    print(f'BatchNorm 層 {len(bns)} 個、統計の推定に train {N_CALIB} 発話')
    for a in ALPHAS:
        st = {k: (t0[k] + a * (t1[k] - t0[k])) if (t1[k].is_floating_point() and not _is_bn_stat(k)) else t1[k]
              for k in t1}
        m.load_state_dict(st)
        m.eval()
        for bn in bns:              # BatchNorm だけ学習モードにして統計を累積平均で取り直す（dropout は eval のまま）
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        with torch.no_grad():
            for x in xs:
                m(x.to(s64.DEVICE))
        m.eval()
        st = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        out = CKPT / f'tvr_a{a:g}'
        out.mkdir(parents=True, exist_ok=True)
        torch.save(st, out / 'best.th')
        with torch.no_grad():
            y = m(xs[0].to(s64.DEVICE))
        print(f'α={a:g}: saved {out / "best.th"}  確認用の出力 rms={y.pow(2).mean().sqrt().item():.4g}'
              f' nan={int(torch.isnan(y).sum())}', flush=True)
    keys = [k for k in t1 if t1[k].is_floating_point() and not _is_bn_stat(k)]
    norm = sum(((t1[k] - t0[k]).float() ** 2).sum().item() for k in keys) ** 0.5
    base = sum((t0[k].float() ** 2).sum().item() for k in keys) ** 0.5
    print(f'|τ_CE| = {norm:.3f}, |θ_λ0| = {base:.3f}, 比 {norm / base:.4f}（学習パラメータのみ）')


def wav(args):
    s64 = import_module('64_retranscribe_all')
    s66 = import_module('66_lambda_cross_asr')
    split = args.split
    gen_dir = SE_WAV / 'fm_taps_avg4' / split
    assert gen_dir.is_dir(), gen_dir
    for a in ALPHAS:
        se = s64.load_se(CKPT / f'tvr_a{a:g}' / 'best.th')
        d_se, d_fu = SE_WAV / f'tvr_a{a:g}' / split, SE_WAV / f'fm_avg4_mavg_tvr_a{a:g}' / split
        d_se.mkdir(parents=True, exist_ok=True)
        d_fu.mkdir(parents=True, exist_ok=True)
        n_new = 0
        for s in s64.load_samples(split):
            name = f"{s['utt']}.wav"
            if (d_se / name).exists() and (d_fu / name).exists():
                continue
            x, sr = sf.read(s['path'], dtype='float32')
            if x.ndim > 1:
                x = x.mean(axis=1)
            with torch.no_grad():
                y = se(torch.from_numpy(x)[None].to(s64.DEVICE)).reshape(-1).cpu().numpy().astype(np.float32)
            sf.write(d_se / name, y, sr)
            g = sf.read(gen_dir / name, dtype='float32')[0]
            n = min(len(g), len(y))
            sf.write(d_fu / name, s66.mag_avg(g[:n], y[:n]), sr)
            n_new += 1
        print(f'α={a:g}: {n_new} 発話を書いた → {d_se}, {d_fu}', flush=True)
        del se
        torch.cuda.empty_cache()


def summary(args):
    s64 = import_module('64_retranscribe_all')
    s66 = import_module('66_lambda_cross_asr')
    split = args.split
    refs = {s['utt']: s['text'] for s in s64.load_samples(split)}
    n_all = len(refs)
    rows, lines = [], []
    for asr in ASR_LIST:
        p = s66.hyp_path(asr, split)
        if not p.exists():
            continue
        sc = defaultdict(dict)
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['cond'] in CONDS:
                sc[r['cond']][(r['utt'], r['spk'])] = s64.capped(s64.norm(refs[r['utt']]), s64.norm(r['hyp']))

        def spk(d):
            by = defaultdict(list)
            for (_, k), v in d.items():
                by[k].append(v)
            return np.array([np.mean(by[k]) for k in sorted(by)])

        cer = {c: np.mean(list(d.values())) for c, d in sc.items() if len(d) == n_all}
        for group, ref in ((SE_CONDS + ['tv_a0'], 'tv_a1'), (FUSED + ['fm_avg4_mavg_tv_a0'], 'fm_avg4_mavg_tv_a1')):
            if ref not in cer:
                continue
            mr = spk(sc[ref])
            for c in group:
                if c not in cer:
                    continue
                mc = spk(sc[c])
                p_ = wilcoxon(mr, mc).pvalue if np.any(mr != mc) else 1.0
                rows.append(dict(asr=asr, cond=c, ref=ref, cer=round(cer[c], 4), cer_ref=round(cer[ref], 4),
                                 rel=round(100 * (cer[c] / cer[ref] - 1), 1),
                                 spk_better=int((mc < mr).sum()), n_spk=len(mr), p_spk=round(p_, 4)))
        def fmt(c):
            return f'{cer[c]:.4f}' if c in cer else '   -  '
        lines.append(f'{asr:<24}' + ''.join(f'{fmt(c):>9}' for c in REFS + SE_CONDS + FUSED))
    head = f'{"ASR / " + split:<24}' + ''.join(f'{c.replace("fm_avg4_mavg_", "F:"):>9}' for c in REFS + SE_CONDS + FUSED)
    print(head)
    print('\n'.join(lines))
    out = BASE_DIR / 'results' / 'task_vector'
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / f'pairs_{split}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(out / f'summary_{split}.txt', 'w', encoding='utf-8') as f:
        f.write(head + '\n' + '\n'.join(lines) + '\n')
    print(f'Saved: {out}/pairs_{split}.csv, summary_{split}.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'wav', 'summary', 'conds'])
    ap.add_argument('--split', default='dev')
    args = ap.parse_args()
    if args.stage == 'conds':
        print(' '.join(CONDS))
        return
    {'build': build, 'wav': wav, 'summary': summary}[args.stage](args)


if __name__ == '__main__':
    main()
