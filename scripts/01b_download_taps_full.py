"""
TAPS Dataset — train / validation / test 全splitをダウンロード

出力構造:
  data/raw/taps/
    throat/{split}/  {speaker_id}_{sentence_id}.wav
    acoustic/{split}/ {speaker_id}_{sentence_id}.wav
    metadata_{split}.csv
    metadata_all.csv
"""
import os
import csv
import numpy as np
import soundfile as sf
from datasets import load_dataset

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'taps')

SPLITS = {
    'train':      4000,
    'dev':        1000,   # HuggingFace上のsplit名は 'dev'（validationではない）
    'test':       1000,
}

all_records = []

for split, expected in SPLITS.items():
    print(f"\n{'='*50}")
    print(f"Split: {split}  (期待件数: {expected})")
    print('='*50)

    throat_dir  = os.path.join(BASE_DIR, 'throat',  split)
    acoustic_dir = os.path.join(BASE_DIR, 'acoustic', split)
    os.makedirs(throat_dir,   exist_ok=True)
    os.makedirs(acoustic_dir, exist_ok=True)

    ds = load_dataset(
        'yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset',
        split=split,
        streaming=True,
    )

    records = []
    for i, sample in enumerate(ds):
        sid    = sample['speaker_id']
        sentid = sample['sentence_id']
        text   = sample['normalized_text']
        dur    = sample['duration']

        t_path = os.path.join(throat_dir,  f'{sid}_{sentid}.wav')
        a_path = os.path.join(acoustic_dir, f'{sid}_{sentid}.wav')

        # 既存ファイルはスキップ（再開対応）
        if not os.path.exists(t_path):
            throat = sample['audio.throat_microphone']
            sf.write(t_path, np.array(throat['array'], dtype=np.float32), throat['sampling_rate'])

        if not os.path.exists(a_path):
            acoustic = sample['audio.acoustic_microphone']
            sf.write(a_path, np.array(acoustic['array'], dtype=np.float32), acoustic['sampling_rate'])

        records.append({
            'split':       split,
            'speaker_id':  sid,
            'sentence_id': sentid,
            'text':        text,
            'duration':    dur,
            'throat_path': t_path,
            'acoustic_path': a_path,
        })

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1:4d}/{expected}] {sid}_{sentid}  {dur:.1f}s  {text[:40]}")

    # split別CSV
    meta_path = os.path.join(BASE_DIR, f'metadata_{split}.csv')
    with open(meta_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"\n  完了: {len(records)}件  → {meta_path}")
    all_records.extend(records)

# 全split統合CSV
all_meta_path = os.path.join(BASE_DIR, 'metadata_all.csv')
with open(all_meta_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
    writer.writeheader()
    writer.writerows(all_records)

print(f"\n{'='*50}")
print(f"全split完了: 合計 {len(all_records)} 件")
print(f"  統合メタデータ: {all_meta_path}")
print(f"  喉マイク: {BASE_DIR}/throat/{{train,validation,test}}/")
print(f"  気導マイク: {BASE_DIR}/acoustic/{{train,validation,test}}/")
