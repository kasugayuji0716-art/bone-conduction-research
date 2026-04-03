"""
全条件のCERをまとめて計測して results/summary.csv に保存する

条件:
  baseline_throat  : 喉マイク・生音声
  baseline_acoustic: 気導マイク（参照）
  dsp_only/clean   : DSPのみ・クリーン
  dsp_only/noisy   : DSPのみ・ノイズあり（SNR別）
  gtcrn/clean      : GTCRNのみ・クリーン
  gtcrn/noisy      : GTCRNのみ・ノイズあり（SNR別）
"""
import os, csv
import unicodedata, re
from faster_whisper import WhisperModel
from jiwer import cer

# ---- 設定 ----
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
META_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
RESULT_DIR  = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_SIZE  = 'small'


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


def main():
    print(f"Whisperモデル読み込み中: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')

    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    results = []

    def run(label, noise_type, snr, audio_dir):
        print(f"  計測中: {label} ... ", end='', flush=True)
        avg = measure_cer(model, audio_dir, records)
        results.append({
            'condition': label,
            'noise_type': noise_type,
            'snr_db': snr,
            'avg_cer': round(avg, 4),
        })
        print(f"CER={avg*100:.1f}%")

    # ---- ベースライン ----
    run('baseline_acoustic', 'none', 'clean',
        os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'acoustic'))
    run('baseline_throat', 'none', 'clean',
        os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat'))

    # ---- ノイズなし（clean）× SEモデル ----
    for model_name in ['dsp_only', 'gtcrn']:
        audio_dir = os.path.join(BASE_DIR, 'data', 'processed', 'se', model_name, 'clean')
        run(f'{model_name}/clean', 'none', 'clean', audio_dir)

    # ---- ノイズあり × 処理なし ----
    noisy_base = os.path.join(BASE_DIR, 'data', 'processed', 'noisy')
    for noise_type in sorted(os.listdir(noisy_base)):
        snr_base = os.path.join(noisy_base, noise_type)
        for snr_cond in sorted(os.listdir(snr_base)):
            snr_val = snr_cond.replace('snr_', '').replace('dB', '')
            audio_dir = os.path.join(snr_base, snr_cond)
            run(f'no_se/{noise_type}/{snr_cond}', noise_type, snr_val, audio_dir)

    # ---- ノイズあり × SEモデル ----
    for model_name in ['dsp_only', 'gtcrn']:
        se_base = os.path.join(BASE_DIR, 'data', 'processed', 'se', model_name)
        for noise_type in sorted(os.listdir(noisy_base)):
            snr_base = os.path.join(noisy_base, noise_type)
            for snr_cond in sorted(os.listdir(snr_base)):
                snr_val = snr_cond.replace('snr_', '').replace('dB', '')
                audio_dir = os.path.join(se_base, noise_type, snr_cond)
                if not os.path.isdir(audio_dir):
                    continue
                run(f'{model_name}/{noise_type}/{snr_cond}', noise_type, snr_val, audio_dir)

    # CSV保存
    out_path = os.path.join(RESULT_DIR, 'summary.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n結果を保存: {out_path}")
    print(f"計測条件数: {len(results)}")


if __name__ == '__main__':
    main()
