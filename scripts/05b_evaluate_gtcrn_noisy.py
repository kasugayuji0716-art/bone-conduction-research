"""
GTCRNノイズ条件のCER計測（残り10条件）
完了後に既存のsummary部分と結合してsummary.csvを生成する
"""
import os, csv
import unicodedata, re
from faster_whisper import WhisperModel
from jiwer import cer

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
META_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
LOG_PATH   = '/tmp/eval_gtcrn_progress.log'

MODEL_SIZE = 'small'

EXISTING = [
    ('baseline_acoustic', 'none', 'clean', 0.131),
    ('baseline_throat',   'none', 'clean', 0.269),
    ('dsp_only/clean',    'none', 'clean', 0.416),
    ('gtcrn/clean',       'none', 'clean', 0.296),
    ('no_se/pink/snr_+0dB',   'pink',  '+0',  0.862),
    ('no_se/pink/snr_+10dB',  'pink',  '+10', 0.463),
    ('no_se/pink/snr_+20dB',  'pink',  '+20', 0.305),
    ('no_se/pink/snr_+5dB',   'pink',  '+5',  0.635),
    ('no_se/pink/snr_-5dB',   'pink',  '-5',  1.042),
    ('no_se/white/snr_+0dB',  'white', '+0',  0.882),
    ('no_se/white/snr_+10dB', 'white', '+10', 0.512),
    ('no_se/white/snr_+20dB', 'white', '+20', 0.330),
    ('no_se/white/snr_+5dB',  'white', '+5',  0.662),
    ('no_se/white/snr_-5dB',  'white', '-5',  0.977),
    ('dsp_only/pink/snr_+0dB',   'pink',  '+0',  1.010),
    ('dsp_only/pink/snr_+10dB',  'pink',  '+10', 0.654),
    ('dsp_only/pink/snr_+20dB',  'pink',  '+20', 0.489),
    ('dsp_only/pink/snr_+5dB',   'pink',  '+5',  0.843),
    ('dsp_only/pink/snr_-5dB',   'pink',  '-5',  1.001),
    ('dsp_only/white/snr_+0dB',  'white', '+0',  1.028),
    ('dsp_only/white/snr_+10dB', 'white', '+10', 0.802),
    ('dsp_only/white/snr_+20dB', 'white', '+20', 0.512),
    ('dsp_only/white/snr_+5dB',  'white', '+5',  0.977),
    ('dsp_only/white/snr_-5dB',  'white', '-5',  0.998),
    ('gtcrn/pink/snr_+0dB',   'pink',  '+0',  1.108),
    ('gtcrn/pink/snr_+10dB',  'pink',  '+10', 0.542),
    ('gtcrn/pink/snr_+20dB',  'pink',  '+20', 0.370),
]

def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\s。、！？()（）\[\]【】,，]', '', text)
    return text.strip()

def measure_cer(model, audio_dir, records):
    scores = []
    for rec in records:
        path = os.path.join(audio_dir, f"{rec['speaker_id']}_{rec['sentence_id']}.wav")
        if not os.path.exists(path):
            continue
        ref = normalize(rec['text'])
        segs, _ = model.transcribe(path, language='ko')
        hyp = normalize(''.join([s.text for s in segs]))
        scores.append(cer(ref, hyp))
    return sum(scores) / len(scores) if scores else 1.0

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')

def main():
    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    log(f"Whisperモデル読み込み中: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')

    results = [{'condition': c, 'noise_type': n, 'snr_db': s, 'avg_cer': round(v, 4)}
               for c, n, s, v in EXISTING]

    # 残り条件: gtcrn/pink/snr_+5dB 以降
    remaining = [
        ('gtcrn/pink/snr_+5dB',   'pink',  '+5'),
        ('gtcrn/pink/snr_-5dB',   'pink',  '-5'),
        ('gtcrn/white/snr_+0dB',  'white', '+0'),
        ('gtcrn/white/snr_+10dB', 'white', '+10'),
        ('gtcrn/white/snr_+20dB', 'white', '+20'),
        ('gtcrn/white/snr_+5dB',  'white', '+5'),
        ('gtcrn/white/snr_-5dB',  'white', '-5'),
    ]

    for label, noise_type, snr_val in remaining:
        snr_cond = f'snr_{snr_val}dB' if not snr_val.startswith('+') else f'snr_+{snr_val[1:]}dB'
        # ラベルからパスを組み立て
        parts = label.split('/')
        audio_dir = os.path.join(BASE_DIR, 'data', 'processed', 'se',
                                 parts[0], noise_type, snr_cond)
        log(f"  計測中: {label} ... ")
        avg = measure_cer(model, audio_dir, records)
        results.append({'condition': label, 'noise_type': noise_type,
                        'snr_db': snr_val, 'avg_cer': round(avg, 4)})
        log(f"  完了: {label} CER={avg*100:.1f}%")

    out_path = os.path.join(RESULT_DIR, 'summary.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition','noise_type','snr_db','avg_cer'])
        writer.writeheader()
        writer.writerows(results)

    log(f"\n全条件完了: {out_path}")

if __name__ == '__main__':
    main()
