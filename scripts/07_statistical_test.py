"""
サンプルごとのCERを計測し、統計検定を行う
出力:
  results/per_sample_cer.csv  : 全サンプル × 全条件のCER
  results/stat_test.csv       : 各条件ペアのWilcoxon検定結果
"""
import os, csv
import unicodedata, re
import numpy as np
from faster_whisper import WhisperModel
from jiwer import cer
from scipy import stats

BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
META_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
LOG_PATH  = '/tmp/stat_test_progress.log'

MODEL_SIZE = 'small'


def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\s。、！？()（）\[\]【】,，]', '', text)
    return text.strip()


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def measure_per_sample(model, audio_dir, records):
    """サンプルごとのCERリストを返す"""
    scores = []
    for rec in records:
        path = os.path.join(audio_dir, f"{rec['speaker_id']}_{rec['sentence_id']}.wav")
        if not os.path.exists(path):
            continue
        ref = normalize(rec['text'])
        segs, _ = model.transcribe(path, language='ko')
        hyp = normalize(''.join([s.text for s in segs]))
        scores.append({
            'speaker_id': rec['speaker_id'],
            'sentence_id': rec['sentence_id'],
            'cer': cer(ref, hyp),
        })
    return scores


def build_condition_list():
    """全34条件の (label, audio_dir) リストを返す"""
    noisy_base = os.path.join(BASE_DIR, 'data', 'processed', 'noisy')
    conds = []

    # ベースライン
    conds.append(('baseline_acoustic',
                  os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'acoustic')))
    conds.append(('baseline_throat',
                  os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat')))

    # clean × SE
    for m in ['dsp_only', 'gtcrn']:
        conds.append((f'{m}/clean',
                      os.path.join(BASE_DIR, 'data', 'processed', 'se', m, 'clean')))

    # noisy × no_se
    for nt in sorted(os.listdir(noisy_base)):
        snr_base = os.path.join(noisy_base, nt)
        if not os.path.isdir(snr_base):
            continue
        for snr_cond in sorted(os.listdir(snr_base)):
            snr_val = snr_cond.replace('snr_', '').replace('dB', '')
            conds.append((f'no_se/{nt}/{snr_cond}',
                          os.path.join(snr_base, snr_cond)))

    # noisy × SE
    for m in ['dsp_only', 'gtcrn']:
        se_base = os.path.join(BASE_DIR, 'data', 'processed', 'se', m)
        for nt in sorted(os.listdir(noisy_base)):
            snr_base = os.path.join(noisy_base, nt)
            if not os.path.isdir(snr_base):
                continue
            for snr_cond in sorted(os.listdir(snr_base)):
                audio_dir = os.path.join(se_base, nt, snr_cond)
                if not os.path.isdir(audio_dir):
                    continue
                conds.append((f'{m}/{nt}/{snr_cond}', audio_dir))

    return conds


def main():
    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    log(f'Whisperモデル読み込み中: {MODEL_SIZE}')
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')

    conditions = build_condition_list()
    all_rows = []

    for i, (label, audio_dir) in enumerate(conditions):
        log(f'[{i+1}/{len(conditions)}] {label}')
        per_sample = measure_per_sample(model, audio_dir, records)
        for s in per_sample:
            all_rows.append({'condition': label,
                             'speaker_id': s['speaker_id'],
                             'sentence_id': s['sentence_id'],
                             'cer': round(s['cer'], 4)})
        avg = np.mean([s['cer'] for s in per_sample])
        log(f'  avg CER={avg*100:.1f}%  n={len(per_sample)}')

    # per_sample_cer.csv 保存
    out_per = os.path.join(RESULT_DIR, 'per_sample_cer.csv')
    with open(out_per, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition','speaker_id','sentence_id','cer'])
        writer.writeheader()
        writer.writerows(all_rows)
    log(f'\nper_sample_cer.csv 保存完了: {out_per}')

    # ── 統計検定 ──────────────────────────────────────────────
    # SNR条件ごとに no_se vs dsp_only, no_se vs gtcrn の Wilcoxon 符号順位検定
    from collections import defaultdict
    by_cond = defaultdict(dict)
    for row in all_rows:
        sid = row['sentence_id']
        by_cond[row['condition']][sid] = row['cer']

    noise_types = ['white', 'pink']
    snr_vals    = ['-5', '+0', '+5', '+10', '+20']
    comparisons = [('dsp_only', 'no_se'), ('gtcrn', 'no_se'), ('gtcrn', 'dsp_only')]

    stat_rows = []
    for nt in noise_types:
        for snr in snr_vals:
            snr_cond = f'snr_{snr}dB' if not snr.startswith('+') else f'snr_+{snr[1:]}dB'
            for m_a, m_b in comparisons:
                cond_a = f'{m_a}/{nt}/{snr_cond}'
                cond_b = f'{m_b}/{nt}/{snr_cond}'
                if cond_a not in by_cond or cond_b not in by_cond:
                    continue
                shared = sorted(set(by_cond[cond_a]) & set(by_cond[cond_b]))
                if len(shared) < 5:
                    continue
                a_vals = [by_cond[cond_a][s] for s in shared]
                b_vals = [by_cond[cond_b][s] for s in shared]
                stat, p = stats.wilcoxon(a_vals, b_vals, alternative='two-sided')
                diff = np.mean(a_vals) - np.mean(b_vals)
                stat_rows.append({
                    'noise_type': nt,
                    'snr_db': snr,
                    'condition_a': m_a,
                    'condition_b': m_b,
                    'mean_cer_a': round(np.mean(a_vals), 4),
                    'mean_cer_b': round(np.mean(b_vals), 4),
                    'diff_a_minus_b': round(diff, 4),
                    'wilcoxon_stat': round(float(stat), 4),
                    'p_value': round(float(p), 4),
                    'significant_p05': 'yes' if p < 0.05 else 'no',
                })

    out_stat = os.path.join(RESULT_DIR, 'stat_test.csv')
    with open(out_stat, 'w', newline='', encoding='utf-8') as f:
        fields = ['noise_type','snr_db','condition_a','condition_b',
                  'mean_cer_a','mean_cer_b','diff_a_minus_b',
                  'wilcoxon_stat','p_value','significant_p05']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stat_rows)

    sig = sum(1 for r in stat_rows if r['significant_p05'] == 'yes')
    log(f'\nstat_test.csv 保存完了: {out_stat}')
    log(f'検定数: {len(stat_rows)}  有意差あり(p<0.05): {sig}')


if __name__ == '__main__':
    main()
