"""
フェーズ2: ファインチューニング済みWhisperの評価
フェーズ1と同条件（test split 話者p00 50サンプル）でCERを計測して比較

使い方:
  python scripts/23_evaluate_whisper_ft.py

出力:
  results/phase2_whisper_cer.csv     — サンプルごとCER
  results/phase2_whisper_summary.csv — フェーズ1との比較サマリー
"""

import os
import csv
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import cer

BASE_DIR   = Path(__file__).parent.parent
TAPS_DIR   = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT_DIR   = BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned'
RESULT_DIR = BASE_DIR / 'results'
TARGET_SR  = 16000

# フェーズ1の結果（比較用）
PHASE1_BASELINE_THROAT = 0.269


def load_model(ckpt_dir):
    print(f"モデルロード: {ckpt_dir}")
    processor = WhisperProcessor.from_pretrained(str(ckpt_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(ckpt_dir))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return processor, model, device


def transcribe(wav_path, processor, model, device):
    wav, sr = sf.read(wav_path, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        wav = resample_poly(wav, TARGET_SR, sr).astype(np.float32)

    inputs = processor.feature_extractor(
        wav, sampling_rate=TARGET_SR, return_tensors='pt'
    ).input_features.to(device)

    with torch.no_grad():
        pred_ids = model.generate(
            inputs,
            language='korean',
            task='transcribe',
            max_new_tokens=225,
        )

    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]


def main():
    processor, model, device = load_model(CKPT_DIR)

    # テストセット: 話者p00の50サンプル（フェーズ1と同条件）
    meta_path = TAPS_DIR / 'metadata_test.csv'
    samples = []
    with open(meta_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == 'p00':
                throat_path = TAPS_DIR / 'throat' / 'test' / f"p00_{row['sentence_id']}.wav"
                if throat_path.exists():
                    samples.append({'path': str(throat_path), 'text': row['text'],
                                    'sentence_id': row['sentence_id']})

    if not samples:
        # フェーズ1と同じディレクトリ構造（testがflatな場合）
        meta_path = TAPS_DIR / 'metadata.csv'
        with open(meta_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                throat_path = TAPS_DIR / 'throat' / f"p00_{row['sentence_id']}.wav"
                if throat_path.exists():
                    samples.append({'path': str(throat_path), 'text': row['text'],
                                    'sentence_id': row['sentence_id']})

    print(f"評価サンプル数: {len(samples)}")

    rows = []
    for i, s in enumerate(samples):
        hyp = transcribe(s['path'], processor, model, device)
        ref = s['text']
        c   = cer(ref, hyp)
        rows.append({
            'sentence_id': s['sentence_id'],
            'reference': ref,
            'hypothesis': hyp,
            'cer': round(c, 4),
        })
        print(f"[{i+1:2d}/{len(samples)}] {s['sentence_id']}  CER={c:.4f}")

    avg_cer = float(np.mean([r['cer'] for r in rows]))
    print(f"\n平均CER（ファインチューニング後）: {avg_cer:.4f}")
    print(f"平均CER（フェーズ1 未学習）      : {PHASE1_BASELINE_THROAT:.4f}")
    diff = avg_cer - PHASE1_BASELINE_THROAT
    print(f"差分: {diff:+.4f}  ({'改善' if diff < 0 else '悪化'})")

    # サンプルごと結果
    out1 = RESULT_DIR / 'phase2_whisper_cer.csv'
    with open(out1, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence_id', 'reference', 'hypothesis', 'cer'])
        writer.writeheader()
        writer.writerows(rows)

    # サマリー比較
    out2 = RESULT_DIR / 'phase2_whisper_summary.csv'
    with open(out2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'avg_cer', 'vs_baseline'])
        writer.writeheader()
        writer.writerow({'condition': 'whisper_small_pretrained (Phase1)',
                         'avg_cer': PHASE1_BASELINE_THROAT, 'vs_baseline': '—'})
        writer.writerow({'condition': 'whisper_small_finetuned (Phase2)',
                         'avg_cer': round(avg_cer, 4),
                         'vs_baseline': f'{diff:+.4f}'})

    print(f"\n保存: {out1}")
    print(f"保存: {out2}")


if __name__ == '__main__':
    main()
