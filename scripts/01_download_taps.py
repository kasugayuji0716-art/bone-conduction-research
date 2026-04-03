"""
TAPS Dataset から test スプリット先頭50件を WAV で保存する
"""
import os
import numpy as np
import soundfile as sf
from datasets import load_dataset

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'taps')
os.makedirs(os.path.join(SAVE_DIR, 'throat'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'acoustic'), exist_ok=True)

N_SAMPLES = 50

print(f"TAPSデータセット test スプリット 先頭{N_SAMPLES}件をダウンロード中...")

ds = load_dataset(
    'yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset',
    split='test',
    streaming=True
)

records = []
for i, sample in enumerate(ds):
    if i >= N_SAMPLES:
        break

    sid    = sample['speaker_id']
    sentid = sample['sentence_id']
    text   = sample['normalized_text']
    dur    = sample['duration']

    # 喉マイク
    throat  = sample['audio.throat_microphone']
    t_arr   = np.array(throat['array'], dtype=np.float32)
    t_sr    = throat['sampling_rate']
    t_path  = os.path.join(SAVE_DIR, 'throat', f'{sid}_{sentid}.wav')
    sf.write(t_path, t_arr, t_sr)

    # 気導マイク
    acoustic = sample['audio.acoustic_microphone']
    a_arr    = np.array(acoustic['array'], dtype=np.float32)
    a_sr     = acoustic['sampling_rate']
    a_path   = os.path.join(SAVE_DIR, 'acoustic', f'{sid}_{sentid}.wav')
    sf.write(a_path, a_arr, a_sr)

    records.append({
        'speaker_id': sid,
        'sentence_id': sentid,
        'text': text,
        'duration': dur,
        'throat_path': t_path,
        'acoustic_path': a_path,
    })

    print(f"  [{i+1:02d}/{N_SAMPLES}] {sid}_{sentid} ({dur:.1f}s) {text[:30]}...")

# メタデータCSVを保存
import csv
meta_path = os.path.join(SAVE_DIR, 'metadata.csv')
with open(meta_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"\n完了: {len(records)}件を保存")
print(f"  喉マイク: {SAVE_DIR}/throat/")
print(f"  気導マイク: {SAVE_DIR}/acoustic/")
print(f"  メタデータ: {meta_path}")
