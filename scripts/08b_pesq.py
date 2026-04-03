"""
PESQ（Perceptual Evaluation of Speech Quality）計測
参照信号: data/raw/taps/throat/（クリーン喉マイク）
出力: results/pesq_results.csv
"""
import os, csv
import numpy as np
import soundfile as sf
from pesq import pesq

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
META_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
CLEAN_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat')
NOISY_BASE = os.path.join(BASE_DIR, 'data', 'processed', 'noisy')
SE_BASE    = os.path.join(BASE_DIR, 'data', 'processed', 'se')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# PESQは8kHz(nb)または16kHz(wb)のみ対応
MODE = 'wb'  # wideband（16kHz）


def compute_pesq(ref_path, deg_path):
    ref, sr_r = sf.read(ref_path, dtype='float32')
    deg, sr_d = sf.read(deg_path, dtype='float32')
    assert sr_r == sr_d == 16000, f"16kHz必須: {sr_r}, {sr_d}"
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    return pesq(sr_r, ref, deg, MODE)


def measure_dir(audio_dir, records):
    scores = []
    for rec in records:
        fname    = f"{rec['speaker_id']}_{rec['sentence_id']}.wav"
        deg_path = os.path.join(audio_dir, fname)
        ref_path = os.path.join(CLEAN_DIR, fname)
        if not os.path.exists(deg_path) or not os.path.exists(ref_path):
            continue
        try:
            s = compute_pesq(ref_path, deg_path)
            scores.append(s)
        except Exception as e:
            print(f'  PESQ error: {fname} — {e}')
    return float(np.mean(scores)) if scores else None


def build_conditions():
    conds = [('throat_clean_ref', CLEAN_DIR)]
    for m in ['dsp_only', 'gtcrn']:
        conds.append((f'{m}/clean', os.path.join(SE_BASE, m, 'clean')))
    for nt in sorted(os.listdir(NOISY_BASE)):
        snr_base = os.path.join(NOISY_BASE, nt)
        if not os.path.isdir(snr_base): continue
        for snr_cond in sorted(os.listdir(snr_base)):
            conds.append((f'no_se/{nt}/{snr_cond}',
                          os.path.join(snr_base, snr_cond)))
    for m in ['dsp_only', 'gtcrn']:
        for nt in sorted(os.listdir(NOISY_BASE)):
            snr_base = os.path.join(NOISY_BASE, nt)
            if not os.path.isdir(snr_base): continue
            for snr_cond in sorted(os.listdir(snr_base)):
                d = os.path.join(SE_BASE, m, nt, snr_cond)
                if os.path.isdir(d):
                    conds.append((f'{m}/{nt}/{snr_cond}', d))
    return conds


def main():
    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    conditions = build_conditions()
    results = []

    for i, (label, audio_dir) in enumerate(conditions):
        print(f'[{i+1}/{len(conditions)}] {label} ... ', end='', flush=True)
        val = measure_dir(audio_dir, records)
        if val is None:
            print('スキップ'); continue
        print(f'PESQ={val:.4f}')

        parts = label.split('/')
        nt  = parts[1] if 'snr_' in label and len(parts) >= 3 else 'none'
        snr = parts[2].replace('snr_','').replace('dB','') if 'snr_' in label else 'clean'
        results.append({'condition': label, 'noise_type': nt,
                        'snr_db': snr, 'avg_pesq': round(val, 4)})

    out = os.path.join(RESULT_DIR, 'pesq_results.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition','noise_type','snr_db','avg_pesq'])
        writer.writeheader(); writer.writerows(results)

    print(f'\n完了: {out}  ({len(results)}条件)')


if __name__ == '__main__':
    main()
