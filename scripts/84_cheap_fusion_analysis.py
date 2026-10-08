"""
スクリプト84: 学習なしで試せる2つの案（results/CROSS_DOMAIN_IDEAS_2026-10-08.md の案4・案5）

  wise : 重み空間での内挿（WiSE-FT, Wortsman et al. CVPR 2022）
         θ(α) = (1−α) θ_TAPS + α θ_CE10 の SE-Conformer を作る → checkpoints/wise_a{α}/best.th
         （出力の平均 avg_taps_ce10.0 / mavg_taps_ce10.0 とは別物。dev で Whisper-small だけを見て α を選ぶ）
  ceps : ケプストラム領域での融合（衛星画像のパンシャープニングの考え方: 大まかな形と細部を別々の出力から取る）
         生成4平均 G と CE-SE C の対数振幅を、ケプストラムの低次（スペクトル包絡）と高次（細かい調波構造）に分け、
         どちらを混ぜると Whisper 系の上乗せが出るかを見る（帯域の分解 §7.5 の、もう一つの軸）
           fm_ceps_geo  : 包絡も細部も半々（対数振幅の平均 = 幾何平均）
           fm_ceps_env  : 包絡だけ半々、細部は G
           fm_ceps_fine : 細部だけ半々、包絡は G
         位相はいつも G。分ける次数 K=30（16 kHz で 1.9 ms、ピッチ周期 2.5 ms 以上より短い）は事前に決めた値

使い方（GPU PC、venv）
    python scripts/84_cheap_fusion_analysis.py wise
    python scripts/84_cheap_fusion_analysis.py ceps --split test
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
SE_WAV = BASE_DIR / 'data' / 'processed' / 'se_wav'
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_FFT, HOP = 512, 128


def _state(path):
    st = torch.load(path, map_location='cpu', weights_only=False)
    return st['model'] if isinstance(st, dict) and 'model' in st else st


def wise(args):
    s64 = import_module('64_retranscribe_all')
    a, b = _state(s64.SE_CKPTS['taps']), _state(s64.SE_CKPTS['ce10.0'])
    assert a.keys() == b.keys()
    for al in ALPHAS:
        st = {k: ((1 - al) * a[k] + al * b[k]) if b[k].is_floating_point() else b[k] for k in b}
        out = BASE_DIR / 'checkpoints' / f'wise_a{al:g}'
        out.mkdir(parents=True, exist_ok=True)
        torch.save(st, out / 'best.th')
        print('saved', out / 'best.th')


def split_ceps(logmag, K):
    """(F, T) の対数振幅 → (包絡, 細部)。ケプストラムの 0..K−1 次（と対称な側）を包絡とする"""
    c = torch.fft.irfft(logmag, n=N_FFT, dim=0)
    lift = torch.zeros(N_FFT, 1)
    lift[:K] = 1
    lift[N_FFT - K + 1:] = 1
    env = torch.fft.rfft(c * lift, dim=0).real
    return env, logmag - env


def ceps(args):
    s64 = import_module('64_retranscribe_all')
    s66 = import_module('66_lambda_cross_asr')
    se = s66.load_any_se(args.se)
    win = torch.hann_window(N_FFT)
    outs = {k: SE_WAV / k / args.split for k in ('fm_ceps_geo', 'fm_ceps_env', 'fm_ceps_fine')}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    for smp in s64.load_samples(args.split):
        name = f"{smp['utt']}.wav"
        if all((d / name).exists() for d in outs.values()):
            continue
        gen, sr = sf.read(SE_WAV / args.src / args.split / name, dtype='float32')
        x, _ = sf.read(smp['path'], dtype='float32')
        with torch.no_grad():
            c = se(torch.from_numpy(x)[None].cuda()).reshape(-1).cpu().numpy()
        n = min(len(gen), len(c))
        G = torch.stft(torch.from_numpy(gen[:n]), N_FFT, HOP, window=win, return_complex=True)
        C = torch.stft(torch.from_numpy(c[:n]), N_FFT, HOP, window=win, return_complex=True)
        lg, lc = torch.log(G.abs() + 1e-7), torch.log(C.abs() + 1e-7)
        eg, fg = split_ceps(lg, args.K)
        ec, fc = split_ceps(lc, args.K)
        mixes = {'fm_ceps_geo': 0.5 * (lg + lc),
                 'fm_ceps_env': 0.5 * (eg + ec) + fg,
                 'fm_ceps_fine': eg + 0.5 * (fg + fc)}
        for k, L in mixes.items():
            y = torch.istft(torch.exp(L) * torch.exp(1j * G.angle()), N_FFT, HOP, window=win, length=n)
            sf.write(outs[k] / name, y.numpy().astype(np.float32), sr)
    print('done →', ', '.join(str(d) for d in outs.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['wise', 'ceps'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--src', default='fm_taps_avg4', help='ceps: 生成モデル側の出力')
    ap.add_argument('--se', default='ce10.0', help='ceps: Whisper 用 SE')
    ap.add_argument('--K', type=int, default=30, help='ceps: 包絡とみなすケプストラムの次数')
    args = ap.parse_args()
    {'wise': wise, 'ceps': ceps}[args.stage](args)


if __name__ == '__main__':
    main()
