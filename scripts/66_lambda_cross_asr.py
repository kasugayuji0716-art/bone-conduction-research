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
  --set div: 多様性の対照。ASR損失を使わない別構造のSE（TAPS 公開の Demucs、TSTNN。公式実装）と TAPS の平均。
  TAPS+CE10 の平均が TAPS+CE0 より良かったのが「CE の変化の中身」によるのか「相手が TAPS から遠い
  （多様性が大きい）」だけなのかを切り分ける
  mavg_*: 振幅スペクトルだけを平均し位相は TAPS のものを使う平均（STFT 512/128, Hann）。Demucs/TSTNN は
  TAPS と位相が大きく異なり（差分SNR < 0 dB）、波形の平均では打ち消し合いが起きるため、位相に依らない平均も併記する
  dist: TAPS と各SE出力の違いの大きさ（log-mel L1、TAPS に対する差分のSNR）を測る
認識器: script 65 と同じ 6 つ（whisper-base/small/medium/ft, mms-1b-all, xlsr-korean）
CER は句読点除去後（script 63 の norm/capped）。検定は話者単位（n=10）。

出力
  results/lambda_asr/hyp_<asr>.csv（dev は hyp_<asr>_dev.csv）  utt, spk, cond, hyp
  results/lambda_asr/summary[_dev].csv、pairs[_dev].csv

使い方（GPU PC）
    python scripts/66_lambda_cross_asr.py asr --asr xlsr-korean [--split dev] [--limit 3]
    python scripts/66_lambda_cross_asr.py summary [--split dev]
    python scripts/66_lambda_cross_asr.py asr --asr whisper-small --set div
    python scripts/66_lambda_cross_asr.py dist
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
OTHER_SE = ['demucs', 'tstnn']
AVGS = {'avg_taps_ce0.0': ('taps', 'ce0.0'), 'avg_taps_ce10.0': ('taps', 'ce10.0'),
        'avg_ce0.0_ce10.0': ('ce0.0', 'ce10.0'),
        'avg_taps_demucs': ('taps', 'demucs'), 'avg_taps_tstnn': ('taps', 'tstnn')}
MAVGS = {f'mavg_taps_{p}': ('taps', p) for p in ['ce0.0', 'ce10.0', 'demucs', 'tstnn']}
AVGS.update(MAVGS)
ALL_CONDS = CONDS + OTHER_SE + list(AVGS)
SETS = {'lambda': CONDS,
        'avg': ['taps', 'avg_taps_ce0.0', 'avg_taps_ce10.0', 'avg_ce0.0_ce10.0'],
        'div': ['taps', 'demucs', 'tstnn', 'avg_taps_demucs', 'avg_taps_tstnn',
                'avg_taps_ce0.0', 'avg_taps_ce10.0'] + list(MAVGS),
        # script 69 で振幅統合（m4_med）を1つの SE-Conformer に蒸留したモデル
        'distill': ['taps', 'ce0.0', 'distill_m4_med_dev'],
        'distill_mag': ['taps', 'distill_m4_med_dev_mag'],
        'distill_vbx': ['taps', 'distill_m4_med_vbx', 'distill_m4_med_vbx_mag'],
        # script 74: SSL-MSE（Sato 2025 型）と複数ASR損失（Whisper CE + XLS-R CTC、単純和 / AND-mask）
        'multiloss': ['taps', 'ce10.0', 'ssl_wavlm_0.3', 'ssl_wavlm_0.03', 'ctc4_xlsr',
                      'ce10_ctc4_sum', 'ce10_ctc4_and'],
        # script 75: 生成モデル（NeMo flow matching）を TAPS で追加学習
        'flow': ['taps', 'fm_taps'],
        # 生成のばらつき（シード）と N 個の振幅平均、事前学習なしの対照
        'flow_seed': ['taps', 'fm_taps', 'fm_taps_s1', 'fm_taps_s2', 'fm_taps_avg4'],
        'flow_var': ['taps', 'fm_taps', 'fm_taps_s1', 'fm_taps_s2', 'fm_taps_avg4', 'fm_scratch'],
        # script 77: Whisper 系に特化（生成モデルの出力に Whisper 用 SE をかける／生成モデルを Whisper の CE で追加学習）
        'whisper_spec': ['taps', 'ce10.0', 'fm_taps', 'fm_taps_avg4', 'fm_taps_avg4_ce100'],
        'whisper_fmce': ['taps', 'fm_taps', 'fmce_0.1', 'fmce_0.01'],
        'whisper_diag': ['taps', 'fm_taps', 'fmce_0.1', 'fmce_0.01', 'fmce_sp0', 'fmce_2p0.1'],
        # script 78: 推論時の工夫（Whisper-small の自信度で選ぶ／生成モデルと Whisper 用 SE の振幅平均）
        'whisper_infer': ['taps', 'ce10.0', 'fm_taps', 'fm_taps_avg4', 'fm_sel_conf', 'fm_avg4_mavg_ce10'],
        # script 79: 実際の生成出力に Whisper の損失（後段モジュール / DRaFT-K）と、その対照（dev で λ=0）
        'whisper_real': ['taps', 'fm_taps', 'fm_taps_avg4', 'fm_post_ce', 'fm_avg4_post_ce', 'fm_draft_k1', 'fm_draft_lam0'],
        # 混ぜる比率 w の選択（dev、生成1回の出力で）
        'mix_dev': ['taps', 'ce10.0', 'fm_taps'] + [f'fm_taps_mixce{w}' for w in ('0.25', '0.5', '0.75')],
        # script 80: 相補学習（融合後の音に CE）と対照（dev で CE-SE を追加学習しただけ）。どちらも推論は生成4平均と半々で融合
        'comp': ['taps', 'fm_taps_avg4', 'fm_avg4_mavg_ce10', 'fm_avg4_comp_ce10', 'fm_avg4_devce10'],
        # 相補学習のやり直し: 学習も test と同じく生成4サンプル平均と混ぜる
        'comp2': ['taps', 'fm_taps_avg4', 'fm_avg4_mavg_ce10', 'fm_avg4_comp2_ce10']}
ALL_CONDS += ['distill_m4_med', 'distill_m4_med_dev', 'distill_m4_med_dev_mag',
              'distill_m4_med_vbx', 'distill_m4_med_vbx_mag',
              'ssl_wavlm_0.3', 'ssl_wavlm_0.03', 'ctc4_xlsr', 'ce10_ctc4_sum', 'ce10_ctc4_and', 'fm_taps',
              'fm_taps_s1', 'fm_taps_s2', 'fm_taps_avg4', 'fm_scratch', 'fm_taps_avg4_ce100', 'fmce_0.1', 'fmce_0.01', 'fmce_sp0', 'fmce_2p0.1', 'fm_sel_conf', 'fm_avg4_mavg_ce10',
              'fm_post_ce', 'fm_avg4_post_ce', 'fm_draft_k1', 'fm_draft_lam0',
              'fm_taps_mixce0.25', 'fm_taps_mixce0.5', 'fm_taps_mixce0.75',
              'fm_avg4_comp_ce10', 'fm_avg4_devce10', 'fm_avg4_comp2_ce10']

_WIN = torch.hann_window(512)


def mag_avg(ref, other):
    """振幅スペクトルを平均し、位相は ref のものを使って波形に戻す"""
    r, o = torch.from_numpy(ref), torch.from_numpy(other)
    R = torch.stft(r, 512, 128, window=_WIN, return_complex=True)
    O = torch.stft(o, 512, 128, window=_WIN, return_complex=True)
    Y = 0.5 * (R.abs() + O.abs()) * torch.exp(1j * R.angle())
    return torch.istft(Y, 512, 128, window=_WIN, length=len(ref)).numpy().astype(np.float32)


SE_WAV = BASE_DIR / 'data' / 'processed' / 'se_wav'


def load_any_se(name):
    """(B, T) → (B, T) の SE。TAPS/CE は script 64、Demucs/TSTNN は TAPS 公式実装（taps-baselines/models）"""
    if name in s64.SE_CKPTS:
        return s64.load_se(s64.SE_CKPTS[name])
    ckpt = BASE_DIR / 'checkpoints' / name / 'best.th'     # script 69 などで学習した SE-Conformer
    if ckpt.exists():
        return s64.load_se(ckpt)
    pre = BASE_DIR / 'taps-baselines' / 'pretrained'
    if name == 'demucs':
        from models.demucs import demucs
        m = demucs(hidden=64, causal=False, stride=2, resample=2)   # script 37d と同じ設定
    elif name == 'tstnn':
        from models.tstnn import tstnn
        m = tstnn()
    else:
        raise ValueError(name)
    m.load_state_dict(torch.load(pre / f'{name}.th', map_location='cpu', weights_only=False)['model'])
    m = m.to(DEVICE).eval()
    return lambda x: m(x.unsqueeze(1)).squeeze(1)       # 公式実装は (B, 1, T) を受け取る
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
    wavdir = {m: SE_WAV / m / split for m in need if (SE_WAV / m / split).is_dir()}   # script 75 など保存済み音声
    need = [m for m in need if m not in wavdir]
    se = {c: load_any_se(c) for c in need}
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
        for m, d in wavdir.items():
            if any(m in AVGS.get(c, (c,)) for c in conds):
                outs[m] = sf.read(d / f"{s['utt']}.wav", dtype='float32')[0]
        for c in conds:
            if c in AVGS:
                a, b = (outs[m] for m in AVGS[c])
                n_ = min(len(a), len(b))
                audio = (mag_avg(a[:n_], b[:n_]) if c in MAVGS
                         else (0.5 * (a[:n_] + b[:n_])).astype(np.float32))
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
    print(f'\n単体SEと2つのSE出力の平均（対照）  CER  （括弧内は TAPS 比）')
    for name in ASRS:
        t = cer.get((name, 'taps'))
        items = [(c, cer.get((name, c))) for c in OTHER_SE + list(AVGS)]
        if t is None or all(v is None for _, v in items):
            continue
        print(f'{name:<15}' + ''.join(f'  {c}={v:.4f}({100 * (v / t - 1):+.1f}%)' for c, v in items if v is not None))
    print(f'\nSaved: {OUT}/summary{tag}.csv, pairs{tag}.csv')


def dist(split, limit):
    """TAPS 出力と各SE出力の違い（多様性）: log-mel L1 と、TAPS に対する差分のSNR [dB]（小さいほど違いが大きい）"""
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=80).to(DEVICE)
    logmel = lambda y: torch.log10(mel(y).clamp(min=1e-10))
    partners = ['ce0.0', 'ce10.0', 'demucs', 'tstnn']
    se = {c: load_any_se(c) for c in ['taps'] + partners}
    rows = []
    for s in s64.load_samples(split)[:limit or None]:
        wav, _ = sf.read(s['path'], dtype='float32')
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            t = se['taps'](x)
            for c in partners:
                y = se[c](x)
                n = min(t.shape[-1], y.shape[-1])
                tt, yy = t[..., :n], y[..., :n]
                snr = 10 * torch.log10((tt ** 2).sum() / ((yy - tt) ** 2).sum().clamp(min=1e-12))
                rows.append(dict(utt=s['utt'], spk=s['spk'], partner=c,
                                 logmel_l1=round((logmel(yy) - logmel(tt)).abs().mean().item(), 4),
                                 snr_db=round(snr.item(), 2)))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = '' if split == 'test' else f'_{split}'
    with open(OUT / f'diversity{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nTAPS との違い（{split}, {len(rows) // len(partners)}発話の平均）')
    for c in partners:
        r = [x for x in rows if x['partner'] == c]
        print(f'  {c:<8} log-mel L1={np.mean([x["logmel_l1"] for x in r]):.4f}  '
              f'差分SNR={np.mean([x["snr_db"] for x in r]):.2f} dB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['asr', 'summary', 'dist'])
    ap.add_argument('--asr', choices=ASRS)
    ap.add_argument('--split', choices=['test', 'dev'], default='test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--set', choices=list(SETS), default='lambda')
    a = ap.parse_args()
    if a.stage == 'asr':
        if not a.asr:
            ap.error('--asr が必要')
        asr(a.asr, a.split, a.limit, SETS[a.set])
    elif a.stage == 'dist':
        dist(a.split, a.limit)
    else:
        summary(a.split)


if __name__ == '__main__':
    main()
