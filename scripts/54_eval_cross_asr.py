"""
スクリプト54: 別ASRモデルでのCER評価（汎用性検証）
CE-aware SE（Whisper-smallで学習）が他のASRモデルでも有効か検証。

使い方（DNN PC）:
    python scripts/54_eval_cross_asr.py
"""

import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel

BASE_DIR = Path(__file__).parent.parent
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)

SE_CONDITIONS = [
    ('no_se',           'No SE',                None),
    ('taps_pretrained', 'TAPS pretrained',      BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
    ('enc_best',        'Enc L1 (λ=5.0)',       BASE_DIR / 'checkpoints' / 'enc_v2_lambda_5.0' / 'best.th'),
    ('ce_best',         'CE (λ=10.0)',          BASE_DIR / 'checkpoints' / 'ce_v2_lambda_10.0' / 'best.th'),
]

def _asr_path(name):
    ct2 = Path(f'/tmp/{name}-ct2')
    return str(ct2) if ct2.exists() else f'openai/{name}'

ASR_MODELS = [
    ('whisper-base',   lambda: _asr_path('whisper-base')),
    ('whisper-small',  lambda: _asr_path('whisper-small')),
    ('whisper-medium', lambda: _asr_path('whisper-medium')),
]


def load_se(ckpt_path):
    from models.seconformer import seconformer
    model = seconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def main():
    conditions = [(k, l, p) for k, l, p in SE_CONDITIONS if p is None or p.exists()]
    print(f'SE conditions: {[l for _, l, _ in conditions]}')
    print(f'ASR models: {[name for name, _ in ASR_MODELS]}')
    print(f'ASR paths: {[fn() for _, fn in ASR_MODELS]}')

    # Load test data
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'path': p, 'text': row['text']})
    print(f'Test: {len(samples)} utterances\n')

    # Load SE models
    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt)
            print(f'  SE {label}: loaded')

    # Pre-compute SE outputs
    print('\nPre-computing SE outputs...')
    se_outputs = {k: [] for k, _, _ in conditions}
    for i, s in enumerate(samples):
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        for key, _, ckpt in conditions:
            if ckpt is None:
                se_outputs[key].append(wav)
            else:
                with torch.no_grad():
                    audio = se_models[key](torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
                se_outputs[key].append(audio)
    print(f'  Done ({len(samples)} utterances)')

    # Free SE models
    del se_models
    torch.cuda.empty_cache()

    # Evaluate with each ASR model
    all_results = []

    for asr_name, asr_fn in ASR_MODELS:
        asr_id = asr_fn()
        print(f'\n=== ASR: {asr_name} ({asr_id}) ===')
        asr = FasterWhisperModel(asr_id, device=DEVICE,
                                 compute_type='float16' if DEVICE == 'cuda' else 'int8')

        for key, label, _ in conditions:
            cers = []
            for i, s in enumerate(samples):
                audio = se_outputs[key][i]
                segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
                hyp = ''.join(seg.text for seg in segs).strip()
                c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
                cers.append(c)
            mean_cer = np.mean(cers)
            print(f'  {label:<25} CER={mean_cer:.4f}')
            all_results.append({
                'asr_model': asr_name, 'se_condition': key,
                'se_label': label, 'cer_mean': mean_cer, 'n_utts': len(cers),
            })

        del asr
        torch.cuda.empty_cache()

    # Summary table
    print(f'\n{"="*70}')
    print(f'{"SE Condition":<25}', end='')
    for asr_name, _ in ASR_MODELS:
        print(f' {asr_name:>14}', end='')
    print()
    print('-' * 70)
    for key, label, _ in conditions:
        print(f'{label:<25}', end='')
        for asr_name, _ in ASR_MODELS:
            r = [x for x in all_results if x['asr_model'] == asr_name and x['se_condition'] == key]
            if r:
                print(f' {r[0]["cer_mean"]:>14.4f}', end='')
        print()

    # CSV
    out = BASE_DIR / 'results' / 'cross_asr_evaluation.csv'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['asr_model', 'se_condition', 'se_label', 'cer_mean', 'n_utts'])
        w.writeheader()
        w.writerows(all_results)
    print(f'\nCSV: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
