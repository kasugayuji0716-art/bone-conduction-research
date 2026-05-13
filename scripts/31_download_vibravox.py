"""
VibraVox データセット取得スクリプト

HuggingFace から Cnam-LMSSC/vibravox の speech_clean を取得し、
throat_microphone チャンネルのみ抽出して保存する。

出力:
  data/raw/vibravox/throat/test/  -- 喉マイク WAV ファイル（16kHz）
  data/raw/vibravox/metadata_test.csv -- 書き起こし CSV

CSV 列:
  sentence_id, speaker_id, text

実行:
  pip install datasets soundfile scipy
  python3 scripts/31_download_vibravox.py
  python3 scripts/31_download_vibravox.py --split test --max_samples 500
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR  = Path(__file__).parent.parent
OUT_DIR   = BASE_DIR / 'data' / 'raw' / 'vibravox'
TARGET_SR = 16000
SRC_SR    = 48000  # VibraVox 収録サンプリングレート


def resample_48k_to_16k(wav: np.ndarray) -> np.ndarray:
    """48kHz → 16kHz（単純デシメーション: 3倍ダウンサンプル）"""
    from scipy.signal import resample_poly
    return resample_poly(wav, TARGET_SR, SRC_SR).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='取得する split')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='取得するサンプル数の上限（None=全件）')
    parser.add_argument('--streaming',   action='store_true',
                        help='ストリーミングモードで取得（ディスク節約）')
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError('datasets が必要です: pip install datasets')

    print(f'VibraVox speech_clean [{args.split}] を取得中...')
    print('（初回はダウンロードに時間がかかります）\n')

    ds = load_dataset(
        'Cnam-LMSSC/vibravox',
        'speech_clean',
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=True,
    )

    out_wav_dir = OUT_DIR / 'throat' / args.split
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    count = 0

    for i, sample in enumerate(ds):
        if args.max_samples is not None and count >= args.max_samples:
            break

        # 書き起こし取得
        text = sample.get('normalized_text', sample.get('text', '')).strip()
        if not text:
            continue

        # 話者 ID・発話 ID
        speaker_id  = sample.get('speaker_id',  f'spk{i:04d}')
        sentence_id = sample.get('sentence_id', f'utt{i:06d}')

        # 音声取得（throat_microphone チャンネル）
        audio_dict = sample.get('audio.throat_microphone',
                     sample.get('throat_microphone', None))
        if audio_dict is None:
            # フラットなキー構造の場合
            audio_keys = [k for k in sample.keys() if 'throat' in k.lower()]
            if not audio_keys:
                continue
            audio_dict = sample[audio_keys[0]]

        wav_array = np.array(audio_dict['array'], dtype=np.float32)
        sr        = audio_dict['sampling_rate']

        # モノラル化
        if wav_array.ndim > 1:
            wav_array = wav_array.mean(axis=1)

        # 16kHz にリサンプル
        if sr != TARGET_SR:
            from scipy.signal import resample_poly
            wav_array = resample_poly(wav_array, TARGET_SR, sr).astype(np.float32)

        # 音量正規化
        peak = np.max(np.abs(wav_array))
        if peak > 0:
            wav_array = wav_array / peak * 0.9

        # 保存
        filename = f'{speaker_id}_{sentence_id}.wav'
        out_path = out_wav_dir / filename
        sf.write(str(out_path), wav_array, TARGET_SR)

        rows.append({
            'sentence_id': sentence_id,
            'speaker_id':  speaker_id,
            'text':        text,
        })

        count += 1
        if count % 100 == 0 or count == 1:
            print(f'  [{count}] {speaker_id} {sentence_id}  "{text[:40]}"')

    # CSV 保存
    out_csv = OUT_DIR / f'metadata_{args.split}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sentence_id', 'speaker_id', 'text'])
        w.writeheader()
        w.writerows(rows)

    print(f'\n完了: {count} サンプル')
    print(f'  WAV : {out_wav_dir}')
    print(f'  CSV : {out_csv}')
    print(f'\n次のステップ:')
    print(f'  python3 scripts/32_evaluate_crosslingual.py')


if __name__ == '__main__':
    main()
