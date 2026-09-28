"""
スクリプト70: VibraVox の喉マイク音声を「ラベルなし」の蒸留用データとして取り出す

目的（script 69）
  統合の元の SE（TAPS SE-Conformer / Demucs / TSTNN / CE0）はすべて TAPS train で学習済みで、train 発話では
  SE 同士の差が小さく、統合の効果を含む教師が作れない。TAPS dev（約2時間）だけでは少ないので、どの SE も
  見ていない別データセットの喉マイク音声で教師を作る。蒸留に書き起こしは不要なので、言語（仏語）は問わない。

処理
  - Cnam-LMSSC/vibravox の speech_clean / train から、話者が偏らないよう --every 個おきに parquet を取得
  - audio.throat_microphone（48 kHz）だけを読み、TAPS の喉マイクと帯域を揃えるため 48k → 8k → 16k に変換
    （TAPS の喉マイクは 8 kHz 収録を 16 kHz に変換したもの。4 kHz 以上は不在）
  - 1〜15 秒の発話だけを data/raw/vibravox_unlab/throat/<speaker>_<sentence>.wav に保存し、parquet は削除
  - 再開可能（取り出し済みの parquet はスキップ）

使い方（GPU PC）
    python scripts/70_vibravox_unlabeled.py --every 3
"""

import argparse, csv, io, os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

BASE_DIR = Path(__file__).parent.parent
OUT = BASE_DIR / 'data' / 'raw' / 'vibravox_unlab'
CACHE = Path.home() / 'data_vibravox'
N_SHARDS = 201


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--min_sec', type=float, default=1.0)
    ap.add_argument('--max_sec', type=float, default=15.0)
    a = ap.parse_args()
    wav_dir = OUT / 'throat'
    wav_dir.mkdir(parents=True, exist_ok=True)
    done_log = OUT / 'shards_done.txt'
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    meta_path = OUT / 'metadata.csv'
    new_meta = not meta_path.exists()
    meta = open(meta_path, 'a', newline='', encoding='utf-8')
    w = csv.writer(meta)
    if new_meta:
        w.writerow(['utt', 'speaker_id', 'sentence_id', 'gender', 'duration', 'shard'])
    total_sec = 0.0
    for k in range(0, N_SHARDS, a.every):
        name = f'speech_clean/train-{k:05d}-of-{N_SHARDS:05d}.parquet'
        if name in done:
            continue
        p = hf_hub_download('Cnam-LMSSC/vibravox', name, repo_type='dataset', local_dir=str(CACHE))
        t = pq.read_table(p, columns=['audio.throat_microphone', 'speaker_id', 'sentence_id', 'gender', 'duration'])
        n_ok = 0
        for r in t.to_pylist():
            if not (a.min_sec <= r['duration'] <= a.max_sec):
                continue
            x, sr = sf.read(io.BytesIO(r['audio.throat_microphone']['bytes']), dtype='float32')
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 48000:
                raise ValueError(f'unexpected sr {sr}')
            y = resample_poly(resample_poly(x, 1, 6), 2, 1).astype(np.float32)   # 48k → 8k → 16k
            utt = f"{r['speaker_id']}_{r['sentence_id']}"
            sf.write(wav_dir / f'{utt}.wav', y, 16000)
            w.writerow([utt, r['speaker_id'], r['sentence_id'], r['gender'], round(r['duration'], 2), k])
            n_ok += 1
            total_sec += r['duration']
        meta.flush()
        os.remove(p)
        with open(done_log, 'a') as f:
            f.write(name + '\n')
        print(f'shard {k}: {n_ok} utts (this run {total_sec / 3600:.2f} h)', flush=True)
    meta.close()
    print('done')


if __name__ == '__main__':
    main()
