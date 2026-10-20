"""
スクリプト91: 融合A はなぜ Whisper 系に効くのか — 音声処理の観点での分析

融合A（生成×4 と CE-SE の振幅平均、位相は生成）が生成×4 から何を変えたかを、信号として調べる。
CER による分解（帯域・ケプストラム、論文 4.3 節）は「どの成分を混ぜると効くか」を見たもので、
ここでは「混ぜた結果、音声がどう変わったか」と「その変化が誤りの減り方と対応するか」を見る。

  feats   test の各発話で、気導音（参照）・各条件の出力から次を求める → results/acoustic/metrics_utt.csv, bands.npz
          - Whisper が実際に見る対数メル（80 帯域、発話ごとに最大値−8 で下限、(x+4)/4）
          - 区間の種類（気導音から決める）: 無音（最大から 40 dB 以上低い）／有声（pYIN の有声判定）／無声
          - 区間ごと・帯域ごとの平均対数メルと、気導音との L1 距離
          - LPC（8 kHz に下げて 10 次）で有声区間の F1〜F3 と帯域幅。気導音とのずれ
          - 気導音の有声区間の基本周波数の中央値（f0_air、話者の声の高さ。山中ら 2026 は女性話者で劣化が大きいと報告）
          - スペクトルの傾き（有声区間、4 kHz 未満の対数メルの回帰直線の傾き）、無音区間の床、正のスペクトルフラックス（立ち上がり）、
            変調スペクトル（4 kHz 未満の帯域包絡の 4〜16 Hz 成分の割合）
  report  条件ごとの平均（話者単位の Wilcoxon で融合A 対 生成×4）と、発話ごとの特徴の変化と CER の変化の Spearman 相関
          → results/acoustic/report.md
  fig     融合A − 生成×4（と 気導 − 生成×4）の対数メル差を、区間の種類ごとに帯域に沿って描く → results/figures/acoustic_fusion_diff.png
  eq      イコライザの対照: 生成×4 に「融合A との差」をどこまで細かく移せば効果が戻るかを見る（認識は script 66 で）
            fm_avg4_eqA_dev  dev 全体で求めた固定ゲイン（周波数だけの関数）
            fm_avg4_eqA_cls  区間の種類（生成×4 自身から判定、気導音は使わない）ごとの固定ゲイン（dev で推定）
            fm_avg4_eqA_utt  発話ごとの固定ゲイン（その発話の融合A を使うオラクル。時間変化だけを除いた対照）
          時間変化する包絡まで移したものは既存の fm_ceps_env（論文 表3「包絡」）

使い方（GPU PC、feats / eq は CPU で可、SE 出力が保存されていない条件は SE を GPU で動かす）
    python scripts/91_acoustic_analysis.py feats
    python scripts/91_acoustic_analysis.py report
    python scripts/91_acoustic_analysis.py fig      # Mac で（results/acoustic/bands.npz を持ってきて）
    python scripts/91_acoustic_analysis.py eq
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
import torchaudio.transforms as Ta
from scipy.signal import resample_poly
from scipy.stats import spearmanr, wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
from dataset_cfg import RAW_DIR, SE_WAV

OUT = BASE_DIR / 'results' / 'acoustic'
CONDS = ['throat', 'taps', 'ce10.0', 'fm_taps_avg4', 'fm_avg4_mavg_ce10']
NAMES = {'air': '気導', 'throat': '処理なし', 'taps': 'TAPS SE', 'ce10.0': 'CE-SE', 'fm_taps_avg4': '生成×4', 'fm_avg4_mavg_ce10': '融合A'}
CLASSES = ['voiced', 'unvoiced', 'silence']
SR = 16000
MEL = Ta.MelSpectrogram(sample_rate=SR, n_fft=400, win_length=400, hop_length=160, n_mels=80, f_min=0.0, f_max=8000.0,
                        power=2.0, window_fn=torch.hann_window, mel_scale='slaney', norm='slaney')
# slaney メル尺度での各帯域の中心周波数（Hz）
MEL_HZ = np.array(__import__('librosa').mel_frequencies(n_mels=82, fmin=0.0, fmax=8000.0, htk=False)[1:-1])
LO = MEL_HZ < 4000


def whisper_logmel(x):
    """Whisper の特徴量と同じ対数メル（30 秒への詰め物はしない）。(80, T)"""
    m = torch.log10(MEL(torch.from_numpy(x)).clamp(min=1e-10))
    m = torch.maximum(m, m.max() - 8.0)
    return ((m + 4.0) / 4.0).numpy()


def frame_classes(air, T):
    """気導音から各フレーム（10 ms、Whisper の特徴量と同じ中心位置）を 有声 / 無声 / 無音 に分ける。
    無音: 最大から 40 dB 以上低い。有声: pYIN（Mauch & Dixon 2014、librosa）の有声判定。F0 も返す"""
    import librosa
    pad = np.pad(air, (200, 200))
    idx = np.arange(T)[:, None] * 160 + np.arange(400)[None]
    fr = pad[np.minimum(idx, len(pad) - 1)] * np.hanning(400)
    e = 10 * np.log10((fr ** 2).sum(1) + 1e-12)
    f0, vflag, _ = librosa.pyin(air, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=160, center=True)
    v = np.zeros(T, bool)
    v[:min(T, len(vflag))] = vflag[:T]
    f0 = np.pad(np.nan_to_num(f0), (0, max(0, T - len(f0))))[:T]
    cls = np.where(e < e.max() - 40, 2, np.where(v, 0, 1))
    return cls, f0


def lpc_formants(x, voiced_frames, order=10):
    """8 kHz に下げた信号の有声フレーム（10 ms 単位の番号）で F1〜F3 と帯域幅（Hz）。無いところは nan"""
    import librosa
    y = resample_poly(x, 1, 2)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    out = np.full((len(voiced_frames), 6), np.nan)
    win = np.hamming(200)
    for k, t in enumerate(voiced_frames):
        s = t * 80
        fr = y[s:s + 200]
        if len(fr) < 200 or np.abs(fr).max() < 1e-5:
            continue
        try:
            a = librosa.lpc(fr * win, order=order)
        except Exception:   # 数値的に解けないフレームは飛ばす
            continue
        r = np.roots(a)
        r = r[np.imag(r) > 0]
        f = np.angle(r) * 8000 / (2 * np.pi)
        bw = -np.log(np.abs(r)) * 8000 / np.pi
        keep = (f > 90) & (f < 3800) & (bw < 600)
        o = np.argsort(f[keep])
        f, bw = f[keep][o][:3], bw[keep][o][:3]
        out[k, :len(f)] = f
        out[k, 3:3 + len(bw)] = bw
    return out


def mod_ratio(M):
    """4 kHz 未満の帯域ごとのパワー包絡の変調スペクトルで、4〜16 Hz（音節の速さ）の成分が 50 Hz までに占める割合"""
    env = 10 ** (4 * M[LO] - 4)
    P = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1)) ** 2
    fq = np.fft.rfftfreq(env.shape[1], 0.01)
    return P[:, (fq >= 4) & (fq <= 16)].sum() / (P[:, (fq > 0) & (fq <= 50)].sum() + 1e-12)


def load_cond(c, s, se_cache):
    if c == 'throat':
        return sf.read(s['path'], dtype='float32')[0]
    d = SE_WAV / c / 'test'
    if d.is_dir():
        return sf.read(d / f"{s['utt']}.wav", dtype='float32')[0]
    if c not in se_cache:
        se_cache[c] = import_module('66_lambda_cross_asr').load_any_se(c)
    x = torch.from_numpy(sf.read(s['path'], dtype='float32')[0])[None].to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        return se_cache[c](x).reshape(-1).cpu().numpy()


def feats(args):
    s64 = import_module('64_retranscribe_all')
    OUT.mkdir(parents=True, exist_ok=True)
    samples = s64.load_samples('test')[:args.limit or None]
    sums = {c: np.zeros((3, 80)) for c in ['air'] + CONDS}     # 区間の種類ごとの対数メルの和
    l1s = {c: np.zeros((3, 80)) for c in CONDS}                # 気導音との差の絶対値の和
    cnt = np.zeros(3)
    rows, se_cache = [], {}
    for i, s in enumerate(samples):
        air = sf.read(RAW_DIR / 'acoustic' / 'test' / f"{s['utt']}.wav", dtype='float32')[0]
        sig = {c: load_cond(c, s, se_cache) for c in CONDS}
        n = min([len(air)] + [len(v) for v in sig.values()])
        air = air[:n]
        sig = {c: v[:n].astype(np.float32) for c, v in sig.items()}
        Ma = whisper_logmel(air)
        T = Ma.shape[1]
        cls, f0 = frame_classes(air, T)
        vf = np.where(cls == 0)[0]
        Fa = lpc_formants(air, vf)
        for k in range(3):
            cnt[k] += (cls == k).sum()
            sums['air'][k] += Ma[:, cls == k].sum(1)
        for c, x in sig.items():
            M = whisper_logmel(x)[:, :T]
            for k in range(3):
                sums[c][k] += M[:, cls == k].sum(1)
                l1s[c][k] += np.abs(M - Ma)[:, cls == k].sum(1)
            Fc = lpc_formants(x, vf)
            r = dict(utt=s['utt'], spk=s['spk'], cond=c,
                     l1_all=np.abs(M - Ma).mean(),
                     **{f'l1_{n_}': np.abs(M - Ma)[:, cls == k].mean() if (cls == k).any() else np.nan
                        for k, n_ in enumerate(CLASSES)},
                     l1_lo=np.abs(M - Ma)[LO].mean(), l1_hi=np.abs(M - Ma)[~LO].mean())
            for j in range(3):
                d = np.abs(Fc[:, j] - Fa[:, j])
                r[f'dF{j + 1}'] = np.nanmedian(d) if np.isfinite(d).any() else np.nan
                r[f'bw{j + 1}'] = np.nanmedian(Fc[:, 3 + j]) if np.isfinite(Fc[:, 3 + j]).any() else np.nan
            V = M[:, cls == 0][LO]
            r['tilt'] = np.polyfit(MEL_HZ[LO] / 1000, V.mean(1), 1)[0] if V.size else np.nan   # 対数メル単位 / kHz
            r['sil_floor'] = M[:, cls == 2].mean() if (cls == 2).any() else np.nan
            r['flux'] = np.clip(np.diff(M, axis=1), 0, None).sum(0).mean()
            r['mod4_16'] = mod_ratio(M)
            rows.append(r)
        # 気導音そのものの値も（参照）
        rows.append(dict(utt=s['utt'], spk=s['spk'], cond='air', l1_all=0.0,
                         **{f'bw{j + 1}': np.nanmedian(Fa[:, 3 + j]) if np.isfinite(Fa[:, 3 + j]).any() else np.nan for j in range(3)},
                         tilt=np.polyfit(MEL_HZ[LO] / 1000, Ma[:, cls == 0][LO].mean(1), 1)[0] if (cls == 0).any() else np.nan,
                         sil_floor=Ma[:, cls == 2].mean() if (cls == 2).any() else np.nan,
                         flux=np.clip(np.diff(Ma, axis=1), 0, None).sum(0).mean(), mod4_16=mod_ratio(Ma),
                         f0_air=np.median(f0[cls == 0]) if (cls == 0).any() else np.nan))
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(samples)}', flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('utt', 'spk', 'cond'), k))
    with open(OUT / 'metrics_utt.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez(OUT / 'bands.npz', cnt=cnt, mel_hz=MEL_HZ,
             **{f'mean_{c}': sums[c] / cnt[:, None] for c in sums}, **{f'l1_{c}': l1s[c] / cnt[:, None] for c in l1s})
    print('saved →', OUT, f'（フレーム: 有声 {cnt[0]:.0f} / 無声 {cnt[1]:.0f} / 無音 {cnt[2]:.0f}）')


def cer_by_utt(asr):
    s64 = import_module('64_retranscribe_all')
    refs = {s['utt']: s['text'] for s in s64.load_samples('test')}
    out = defaultdict(dict)
    p = BASE_DIR / 'results' / 'lambda_asr' / f'hyp_{asr}.csv'
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if r['utt'] in refs and r['cond'] in ('fm_taps_avg4', 'fm_avg4_mavg_ce10'):
            out[r['cond']][r['utt']] = s64.capped(s64.norm(refs[r['utt']]), s64.norm(r['hyp']))
    return out


def report(args):
    rows = list(csv.DictReader(open(OUT / 'metrics_utt.csv', encoding='utf-8')))
    M = [k for k in rows[0] if k not in ('utt', 'spk', 'cond')]
    by = defaultdict(dict)
    for r in rows:
        by[r['cond']][r['utt']] = (r['spk'], {k: float(r[k]) if r[k] not in ('', 'nan') else np.nan for k in M})
    L = ['# 融合A の音声処理的な分析（scripts/91_acoustic_analysis.py）', '',
         'test 1,000 発話。l1_* は Whisper の対数メル（(x+4)/4 の単位）での気導音との平均絶対差、dF は LPC フォルマントの気導音とのずれの中央値（Hz）、'
         'bw は帯域幅の中央値（Hz）、tilt は有声区間の 4 kHz 未満の傾き（/kHz）、sil_floor は無音区間の平均、flux は正のスペクトルフラックス、mod4_16 は変調スペクトルの 4〜16 Hz の割合', '',
         '## 条件ごとの平均', '', '| 指標 | ' + ' | '.join(NAMES[c] for c in ['air'] + CONDS) + ' | 融合A−生成×4 | 改善方向の話者 | p |',
         '|---|' + '---|' * (len(CONDS) + 4)]
    A, G = 'fm_avg4_mavg_ce10', 'fm_taps_avg4'
    for k in M:
        vals = [np.nanmean([v[1][k] for v in by[c].values()]) if c in by else np.nan for c in ['air'] + CONDS]
        spk = defaultdict(lambda: [[], []])
        for u, (s, a) in by[A].items():
            spk[s][0].append(a[k]); spk[s][1].append(by[G][u][1][k])
        da = np.array([np.nanmean(v[0]) - np.nanmean(v[1]) for v in spk.values()])
        p = wilcoxon(da).pvalue if np.any(da != 0) and np.isfinite(da).all() else np.nan
        L.append(f'| {k} | ' + ' | '.join(f'{v:.4g}' for v in vals) + f' | {vals[-1] - vals[-2]:+.4g} | {(da < 0).sum()}/{len(da)} 減 | {p:.3g} |')
    L += ['', '## 発話ごとの「融合A − 生成×4」の特徴の変化と CER の変化の相関（Spearman、1,000 発話）', '',
          '正の ρ = その特徴が増えた発話ほど CER も増えた（悪化した）', '']
    asrs = ['whisper-small', 'whisper-medium', 'whisper-ft', 'qwen3-asr-1.7b', 'zipformer-ko']
    L += ['| 指標 | ' + ' | '.join(asrs) + ' |', '|---|' + '---|' * len(asrs)]
    cers = {a: cer_by_utt(a) for a in asrs}
    for k in M:
        cells = []
        for a in asrs:
            us = [u for u in by[A] if u in cers[a][A] and u in cers[a][G]]
            dx = np.array([by[A][u][1][k] - by[G][u][1][k] for u in us])
            dy = np.array([cers[a][A][u] - cers[a][G][u] for u in us])
            ok = np.isfinite(dx)
            rho, p = spearmanr(dx[ok], dy[ok]) if ok.sum() > 10 else (np.nan, np.nan)
            cells.append(f'{rho:+.3f}' + ('*' if p < 0.01 else ''))
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    L.append('\n*: p<0.01（発話単位なので話者内の相関は考慮していない。目安として見る）')
    # 話者ごとの声の高さ（気導音の F0 中央値）と、融合A の生成×4 に対する CER の変化
    L += ['', '## 話者の声の高さと融合の効果（10 話者、Spearman）', '', '| 認識器 | ρ（F0 と 融合A−生成×4 の CER 差） | p |', '|---|---|---|']
    f0 = defaultdict(list)
    for u, (s_, a_) in by['air'].items():
        f0[s_].append(a_.get('f0_air', np.nan))
    f0 = {k: np.nanmedian(v) for k, v in f0.items()}
    for a in asrs:
        d = defaultdict(list)
        for u in cers[a][A]:
            if u in cers[a][G] and u in by[A]:
                d[by[A][u][0]].append(cers[a][A][u] - cers[a][G][u])
        ks = sorted(set(d) & set(f0))
        rho, p = spearmanr([f0[k] for k in ks], [np.mean(d[k]) for k in ks])
        L.append(f'| {a} | {rho:+.2f} | {p:.3g} |')
    L.append('話者ごとの F0: ' + ', '.join(f'{k} {v:.0f} Hz' for k, v in sorted(f0.items())))
    (OUT / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


def fig(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']   # 日本語のフォントがある Mac で描く（bands.npz だけあればよい）
    z = np.load(OUT / 'bands.npz')
    hz = z['mel_hz'] / 1000
    fig_, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    lab = {'voiced': '有声', 'unvoiced': '無声', 'silence': '無音'}
    for k, (ax, c) in enumerate(zip(axes, CLASSES)):
        ax.axhline(0, color='0.7', lw=0.8)
        ax.plot(hz, z['mean_air'][k] - z['mean_fm_taps_avg4'][k], color='0.55', ls='--', label='気導 − 生成×4')
        ax.plot(hz, z['mean_fm_avg4_mavg_ce10'][k] - z['mean_fm_taps_avg4'][k], color='C3', label='融合A − 生成×4')
        ax.plot(hz, z['mean_ce10.0'][k] - z['mean_fm_taps_avg4'][k], color='C0', lw=0.8, alpha=0.7, label='CE-SE − 生成×4')
        ax.axvline(4, color='0.8', lw=0.8, ls=':')
        ax.set_title(f'{lab[c]}区間（{z["cnt"][k] / z["cnt"].sum() * 100:.0f}%）', fontsize=10)
        ax.set_xlabel('周波数 [kHz]')
    axes[0].set_ylabel('Whisper 対数メルの差')
    axes[0].legend(fontsize=7, frameon=False)
    fig_.tight_layout()
    p = BASE_DIR / 'results' / 'figures' / 'acoustic_fusion_diff.png'
    fig_.savefig(p, dpi=200)
    print('saved →', p)


def eq(args):
    s64 = import_module('64_retranscribe_all')
    win = torch.hann_window(512)
    stft = lambda x: torch.stft(torch.from_numpy(x), 512, 128, window=win, return_complex=True)

    def gen_classes(G):
        """生成×4 の出力自身から区間の種類を決める（気導音は使わない）。STFT フレーム単位"""
        p = (G.abs() ** 2).sum(0).numpy()
        e = 10 * np.log10(p + 1e-12)
        lo = (G.abs()[:64] ** 2).sum(0).numpy() / (p + 1e-12)     # 2 kHz 未満のエネルギーの割合（有声の目安）
        return np.where(e < e.max() - 40, 2, np.where(lo > 0.5, 0, 1))

    def pair(split, utt):
        g = sf.read(SE_WAV / 'fm_taps_avg4' / split / f'{utt}.wav', dtype='float32')[0]
        a = sf.read(SE_WAV / 'fm_avg4_mavg_ce10' / split / f'{utt}.wav', dtype='float32')[0]
        n = min(len(g), len(a))
        return g[:n], stft(g[:n]), stft(a[:n])

    # dev で、対数振幅の差の平均（全体 / 区間の種類ごと）を求める
    s_all, n_all = torch.zeros(257), 0
    s_cls, n_cls = torch.zeros(3, 257), np.zeros(3)
    for s in s64.load_samples('dev'):
        _, G, A = pair('dev', s['utt'])
        d = torch.log(A.abs() + 1e-7) - torch.log(G.abs() + 1e-7)
        cl = gen_classes(G)
        sp = cl < 2
        s_all += d[:, sp].sum(1); n_all += sp.sum()
        for k in range(3):
            s_cls[k] += d[:, cl == k].sum(1); n_cls[k] += (cl == k).sum()
    g_all = s_all / n_all
    g_cls = s_cls / torch.from_numpy(n_cls)[:, None].float()
    np.savez(OUT / 'eq_gains_dev.npz', g_all=g_all.numpy(), g_cls=g_cls.numpy(), n_cls=n_cls)
    outs = {k: SE_WAV / k / 'test' for k in ('fm_avg4_eqA_dev', 'fm_avg4_eqA_cls', 'fm_avg4_eqA_utt')}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    for s in s64.load_samples('test'):
        g, G, A = pair('test', s['utt'])
        lg = torch.log(G.abs() + 1e-7)
        cl = torch.from_numpy(gen_classes(G))
        d_utt = (torch.log(A.abs() + 1e-7) - lg)[:, cl < 2].mean(1)   # 無音を除いた区間で
        mags = {'fm_avg4_eqA_dev': lg + g_all[:, None],
                'fm_avg4_eqA_cls': lg + g_cls[cl].T,
                'fm_avg4_eqA_utt': lg + d_utt[:, None]}
        for k, L_ in mags.items():
            y = torch.istft(torch.exp(L_) * torch.exp(1j * G.angle()), 512, 128, window=win, length=len(g))
            sf.write(outs[k] / f"{s['utt']}.wav", y.numpy().astype(np.float32), SR)
    print('done →', ', '.join(str(d) for d in outs.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['feats', 'report', 'fig', 'eq'])
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    {'feats': feats, 'report': report, 'fig': fig, 'eq': eq}[args.stage](args)


if __name__ == '__main__':
    main()
