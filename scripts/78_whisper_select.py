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
    s64 = import_module('64_retranscribe_all')
    s66 = import_module('66_lambda_cross_asr')
    ce = s66.load_any_se('ce10.0')
    out = SE_WAV / 'fm_avg4_mavg_ce10' / args.split
    out.mkdir(parents=True, exist_ok=True)
    for s in s64.load_samples(args.split):
        name = f"{s['utt']}.wav"
        if (out / name).exists():
            continue
        gen, sr = sf.read(SE_WAV / 'fm_taps_avg4' / args.split / name, dtype='float32')
        x, _ = sf.read(s['path'], dtype='float32')
        with torch.no_grad():
            c = ce(torch.from_numpy(x)[None].cuda()).reshape(-1).cpu().numpy()
        n = min(len(gen), len(c))
        sf.write(out / name, s66.mag_avg(gen[:n], c[:n]), sr)     # 位相は1つ目（生成モデル）
    print(f'done → {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['select', 'fuse'])
    ap.add_argument('--split', default='test')
    args = ap.parse_args()
    {'select': select, 'fuse': fuse}[args.stage](args)


if __name__ == '__main__':
    main()
