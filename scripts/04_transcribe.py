"""
Whisper で音声ファイルを文字起こしして CER を計測する
使い方: python3 04_transcribe.py --condition baseline
"""
import os, csv, argparse
import unicodedata, re
from faster_whisper import WhisperModel
from jiwer import cer

# ---- 設定 ----
BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
META_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_SIZE = 'small'   # 動作確認用。後で large-v3 に切り替える

def normalize(text):
    """テキスト正規化（全角→半角、記号除去）"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\s。、！？()（）\[\]【】]', '', text)
    return text.strip()

def transcribe_file(model, path):
    segments, _ = model.transcribe(path, language='ko')
    return normalize(''.join([s.text for s in segments]))

def main(condition, audio_dir):
    print(f"モデル読み込み中: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')

    # メタデータ読み込み
    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    results = []
    total_cer = 0.0

    for i, rec in enumerate(records):
        sid    = rec['speaker_id']
        sentid = rec['sentence_id']
        ref    = normalize(rec['text'])

        audio_path = os.path.join(audio_dir, f'{sid}_{sentid}.wav')
        if not os.path.exists(audio_path):
            print(f"  [skip] {audio_path} が見つかりません")
            continue

        hyp   = transcribe_file(model, audio_path)
        score = cer(ref, hyp)
        total_cer += score

        results.append({
            'condition': condition,
            'speaker_id': sid,
            'sentence_id': sentid,
            'reference': ref,
            'hypothesis': hyp,
            'cer': round(score, 4),
        })

        print(f"  [{i+1:02d}] CER={score:.3f} | ref: {ref[:25]}... | hyp: {hyp[:25]}...")

    avg_cer = total_cer / len(results) if results else 0
    print(f"\n平均CER ({condition}): {avg_cer:.4f} ({avg_cer*100:.1f}%)")

    # 結果をCSV保存
    out_path = os.path.join(RESULT_DIR, f'cer_{condition}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"保存: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='baseline_throat')
    parser.add_argument('--audio_dir',
        default=os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat'))
    args = parser.parse_args()
    main(args.condition, args.audio_dir)
