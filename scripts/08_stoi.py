"""
STOI（Short-Time Objective Intelligibility）計測
PESQはPython3.13+コンパイルエラーのためSTOIのみ計測

参照信号: data/raw/taps/throat/{speaker_id}_{sentence_id}.wav（クリーン喉マイク）
比較対象: 各条件の処理済み音声

出力: results/stoi_results.csv
"""
import os, csv
import numpy as np
import soundfile as sf
from pystoi import stoi

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
META_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
CLEAN_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat')
NOISY_BASE = os.path.join(BASE_DIR, 'data', 'processed', 'noisy')
SE_BASE    = os.path.join(BASE_DIR, 'data', 'processed', 'se')
RESULT_DIR = os.path.join(BASE_DIR, 'results')


def compute_stoi(ref_path, deg_path):
    ref, sr_r = sf.read(ref_path, dtype='float32')
    deg, sr_d = sf.read(deg_path, dtype='float32')
    assert sr_r == sr_d, f"SR mismatch: {sr_r} vs {sr_d}"
    # 長さ揃え
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    return stoi(ref, deg, sr_r, extended=False)


def measure_dir(audio_dir, records):
    scores = []
    for rec in records:
        fname = f"{rec['speaker_id']}_{rec['sentence_id']}.wav"
        deg_path = os.path.join(audio_dir, fname)
        ref_path = os.path.join(CLEAN_DIR, fname)
        if not os.path.exists(deg_path) or not os.path.exists(ref_path):
            continue
        try:
            s = compute_stoi(ref_path, deg_path)
            scores.append(s)
        except Exception as e:
            print(f'  STOI error: {fname} {e}')
    return float(np.mean(scores)) if scores else None


def build_conditions():
    conds = []

    # クリーン比較（参照との同一性確認）
    conds.append(('throat_clean_ref', CLEAN_DIR))

    # SE × clean
    for m in ['dsp_only', 'gtcrn']:
        conds.append((f'{m}/clean',
                      os.path.join(SE_BASE, m, 'clean')))

    # noisy × no_se
    for nt in sorted(os.listdir(NOISY_BASE)):
        snr_base = os.path.join(NOISY_BASE, nt)
        if not os.path.isdir(snr_base):
            continue
        for snr_cond in sorted(os.listdir(snr_base)):
            conds.append((f'no_se/{nt}/{snr_cond}',
                          os.path.join(snr_base, snr_cond)))

    # noisy × SE
    for m in ['dsp_only', 'gtcrn']:
        for nt in sorted(os.listdir(NOISY_BASE)):
            snr_base = os.path.join(NOISY_BASE, nt)
            if not os.path.isdir(snr_base):
                continue
            for snr_cond in sorted(os.listdir(snr_base)):
                audio_dir = os.path.join(SE_BASE, m, nt, snr_cond)
                if os.path.isdir(audio_dir):
                    conds.append((f'{m}/{nt}/{snr_cond}', audio_dir))

    return conds


def main():
    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    conditions = build_conditions()
    results = []

    for i, (label, audio_dir) in enumerate(conditions):
        print(f'[{i+1}/{len(conditions)}] {label} ... ', end='', flush=True)
        avg_stoi = measure_dir(audio_dir, records)
        if avg_stoi is None:
            print('スキップ（ファイルなし）')
            continue
        print(f'STOI={avg_stoi:.4f}')

        # label から noise_type, snr_db を抽出
        parts = label.split('/')
        if 'snr_' in label:
            nt = parts[1] if len(parts) >= 3 else 'unknown'
            snr = parts[2].replace('snr_', '').replace('dB', '') if len(parts) >= 3 else 'unknown'
        else:
            nt = 'none'
            snr = 'clean'

        results.append({
            'condition': label,
            'noise_type': nt,
            'snr_db': snr,
            'avg_stoi': round(avg_stoi, 4),
        })

    out_path = os.path.join(RESULT_DIR, 'stoi_results.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition','noise_type','snr_db','avg_stoi'])
        writer.writeheader()
        writer.writerows(results)

    print(f'\n完了: {out_path}  ({len(results)}条件)')


if __name__ == '__main__':
    main()
