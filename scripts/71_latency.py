"""
スクリプト71: 推論時間の測定 — SE を4つ動かして振幅統合するコストは、ASR と比べてどの程度か

背景
  振幅統合（results/FUSION_RESULTS_2026-09-28.md）は SE を4つ動かす。1つの SE への蒸留（script 69）は
  認識器に依らない効果を保てなかったため、統合をそのまま使う場合の実際のコストを測る。
  SE は 1000万パラメータ前後、ASR は数億〜20億パラメータなので、統合のコストは ASR に比べて小さい可能性がある。

測定（GPU、fp32 の SE / 各 ASR は評価と同じ設定）
  - SE 単体（TAPS / Demucs / TSTNN / CE0）の1発話あたりの時間
  - 統合の追加処理（STFT → 中央値 → iSTFT）
  - 各 ASR の1発話あたりの時間（入力は TAPS の出力）
  いずれも test の先頭 --n 発話、最初の --warmup 発話は捨てる。torch.cuda.synchronize で計測
  RTF = 処理時間 / 音声長。GPU を他の処理と共有しているときは値が大きく出るので、空いているときに測る

使い方（GPU PC）
    python scripts/71_latency.py --asrs whisper-small whisper-medium whisper-large-v3-turbo xlsr-korean mms-1b-all zipformer-ko
    venv-qwen/bin/python scripts/71_latency.py --asrs qwen3-asr-1.7b
"""

import argparse, csv, sys, time
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
s64 = import_module('64_retranscribe_all')
s65 = import_module('65_step0_residual_cutoff')
s66 = import_module('66_lambda_cross_asr')
s67 = import_module('67_magnitude_fusion')

OUT = BASE_DIR / 'results' / 'latency'


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    y = fn()
    torch.cuda.synchronize()
    return y, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asrs', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=5)
    a = ap.parse_args()
    samples = s64.load_samples('test')[:a.n]
    se = {c: s66.load_any_se(c) for c in s67.SOURCES}
    rec = {k: [] for k in list(s67.SOURCES) + ['fusion', 'se_total']}
    dur, taps_out = [], []
    for i, s in enumerate(samples):
        wav, _ = sf.read(s['path'], dtype='float32')
        x = torch.from_numpy(wav).unsqueeze(0).cuda()
        outs, tot = {}, 0.0
        with torch.no_grad():
            for k, m in se.items():
                outs[k], dt = timed(lambda: m(x).reshape(-1))
                tot += dt
                if i >= a.warmup:
                    rec[k].append(dt)
            n = min(v.shape[-1] for v in outs.values())

            def fuse():
                S = {k: s67.stft(v[:n]) for k, v in outs.items()}
                M = torch.stack([S[k].abs() for k in s67.SOURCES]).median(0).values
                return torch.istft(M * torch.exp(1j * S['taps'].angle()), s67.N_FFT, s67.HOP, window=s67._WIN, length=n)
            _, dt = timed(fuse)
        if i >= a.warmup:
            rec['fusion'].append(dt)
            rec['se_total'].append(tot + dt)
            dur.append(len(wav) / 16000)
        taps_out.append(outs['taps'].cpu().numpy().astype(np.float32))
    for name in a.asrs:
        run = s65.load_asr(name)
        ts = []
        for i, y in enumerate(taps_out):
            _, dt = timed(lambda: run(y))
            if i >= a.warmup:
                ts.append(dt)
        rec[name] = ts
        del run
        torch.cuda.empty_cache()
    D = float(np.sum(dur))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [dict(module=k, n=len(v), ms_per_utt=round(1000 * np.mean(v), 1), rtf=round(np.sum(v) / D, 4))
            for k, v in rec.items() if v]
    tag = '_'.join(a.asrs) if len(a.asrs) == 1 else 'main'
    with open(OUT / f'latency_{tag}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'GPU: {torch.cuda.get_device_name(0)}   平均音声長 {np.mean(dur):.1f} s（{len(dur)} 発話）')
    for r in rows:
        print(f"  {r['module']:24s} {r['ms_per_utt']:8.1f} ms/発話   RTF {r['rtf']:.4f}")
    se_ms = 1000 * np.mean(rec['se_total'])
    for name in a.asrs:
        asr_ms = 1000 * np.mean(rec[name])
        taps_ms = 1000 * np.mean(rec['taps'])
        print(f"  {name}: TAPS+ASR {taps_ms + asr_ms:.0f} ms → 4SE統合+ASR {se_ms + asr_ms:.0f} ms（{(se_ms + asr_ms) / (taps_ms + asr_ms):.2f} 倍）")


if __name__ == '__main__':
    main()
