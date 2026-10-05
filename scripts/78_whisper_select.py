"""
スクリプト78: Whisper 系に特化する推論時の工夫（学習なし）

  select : 生成モデルの複数サンプル（fm_taps / fm_taps_s1 / fm_taps_s2）から、発話ごとに
           Whisper-small が最も自信をもって書き起こせたもの（平均対数確率が最大）を選ぶ。正解テキストは使わない
           → fm_sel_conf
  fuse   : 生成モデル（4サンプル平均）と Whisper 用 SE（CE λ=10、喉マイク入力）の出力を振幅で平均（位相は生成モデル）
           → fm_avg4_mavg_ce10

選択・統合に使うのは Whisper-small だけ。効果は他の Whisper 系（base/medium/turbo/FT・Qwen3）で確かめる。
使い方（GPU PC）: python scripts/78_whisper_select.py select|fuse [--split test]
"""

import argparse
import csv
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
SAMPLES = ['fm_taps', 'fm_taps_s1', 'fm_taps_s2']


def select(args):
    from faster_whisper import WhisperModel
    s64 = import_module('64_retranscribe_all')
    asr = WhisperModel('/tmp/whisper-small-ct2', device='cuda', compute_type='float16')
    out = SE_WAV / 'fm_sel_conf' / args.split
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for i, s in enumerate(s64.load_samples(args.split)):
        name = f"{s['utt']}.wav"
        if (out / name).exists():
            continue
        best, best_lp = None, -1e9
        for c in SAMPLES:
            wav, sr = sf.read(SE_WAV / c / args.split / name, dtype='float32')
            segs, _ = asr.transcribe(wav, language='ko', beam_size=1)
            segs = list(segs)
            n = sum(len(g.tokens) for g in segs)
            lp = sum(g.avg_logprob * len(g.tokens) for g in segs) / max(n, 1)   # 発話全体のトークン平均対数確率
            if lp > best_lp:
                best, best_lp, best_wav = c, lp, wav
        sf.write(out / name, best_wav, sr)
        log.append(dict(utt=s['utt'], chosen=best, logprob=round(best_lp, 4)))
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}', flush=True)
    if log:
        with open(BASE_DIR / 'results' / 'lambda_asr' / f'select_conf_{args.split}.csv', 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(log[0])); w.writerows(log)
    print(f'done → {out}')


def fuse(args):
    """生成モデルの出力と CE-SE の出力の振幅を (1−w):w で混ぜる（位相は生成モデル）。w=0.5 が fm_avg4_mavg_ce10"""
    s64 = import_module('64_retranscribe_all')
    s66 = import_module('66_lambda_cross_asr')
    ce = s66.load_any_se(args.se)
    name_out = args.out or ('fm_avg4_mavg_ce10' if (args.src, args.w) == ('fm_taps_avg4', 0.5) else f'{args.src}_mixce{args.w:g}')
    out = SE_WAV / name_out / args.split
    out.mkdir(parents=True, exist_ok=True)
    win = torch.hann_window(512)
    for s in s64.load_samples(args.split):
        name = f"{s['utt']}.wav"
        if (out / name).exists():
            continue
        gen, sr = sf.read(SE_WAV / args.src / args.split / name, dtype='float32')
        x, _ = sf.read(s['path'], dtype='float32')
        with torch.no_grad():
            c = ce(torch.from_numpy(x)[None].cuda()).reshape(-1).cpu().numpy()
        n = min(len(gen), len(c))
        G = torch.stft(torch.from_numpy(gen[:n]), 512, 128, window=win, return_complex=True)
        C = torch.stft(torch.from_numpy(c[:n]), 512, 128, window=win, return_complex=True)
        Y = ((1 - args.w) * G.abs() + args.w * C.abs()) * torch.exp(1j * G.angle())
        sf.write(out / name, torch.istft(Y, 512, 128, window=win, length=n).numpy().astype(np.float32), sr)
    print(f'done → {out}')


def variants(args):
    """融合の分解: (1) CE-SE の櫛状成分を除いてから融合 (2) 4 kHz 未満だけ混ぜる (3) 4 kHz 以上だけ混ぜる"""
    s66 = import_module('66_lambda_cross_asr')
    s62 = import_module('62_robustness_eval')
    s64 = import_module('64_retranscribe_all')
    ce = s66.load_any_se(args.se)
    win = torch.hann_window(512)
    freqs = torch.fft.rfftfreq(512, 1 / 16000)
    lo = (freqs < 4000).float()[:, None]
    outs = {k: SE_WAV / k / args.split for k in ('fm_fuse_notch', 'fm_fuse_lo', 'fm_fuse_hi')}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    for smp in s64.load_samples(args.split):
        name = f"{smp['utt']}.wav"
        if all((d / name).exists() for d in outs.values()):
            continue
        gen, sr = sf.read(SE_WAV / args.src / args.split / name, dtype='float32')
        x, _ = sf.read(smp['path'], dtype='float32')
        with torch.no_grad():
            c = ce(torch.from_numpy(x)[None].cuda()).reshape(-1).cpu().numpy()
        n = min(len(gen), len(c)); gen, c = gen[:n], c[:n]
        G = torch.stft(torch.from_numpy(gen), 512, 128, window=win, return_complex=True)
        mix = lambda C, wmap: torch.istft(((1 - wmap) * G.abs() + wmap * C.abs()) * torch.exp(1j * G.angle()),
                                          512, 128, window=win, length=n).numpy().astype(np.float32)
        C = torch.stft(torch.from_numpy(c), 512, 128, window=win, return_complex=True)
        Cn = torch.stft(torch.from_numpy(s62.comb_notch(c)), 512, 128, window=win, return_complex=True)
        sf.write(outs['fm_fuse_notch'] / name, mix(Cn, args.w * torch.ones_like(lo)), sr)
        sf.write(outs['fm_fuse_lo'] / name, mix(C, args.w * lo), sr)
        sf.write(outs['fm_fuse_hi'] / name, mix(C, args.w * (1 - lo)), sr)
    print('done →', ', '.join(str(d) for d in outs.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['select', 'fuse', 'variants'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--src', default='fm_taps_avg4', help='fuse: 生成モデル側の出力')
    ap.add_argument('--w', type=float, default=0.5, help='fuse: CE-SE 側の重み')
    ap.add_argument('--out', default='')
    ap.add_argument('--se', default='ce10.0', help='fuse: Whisper 用 SE（script 66 の load_any_se の名前）')
    args = ap.parse_args()
    {'select': select, 'fuse': fuse, 'variants': variants}[args.stage](args)


if __name__ == '__main__':
    main()
