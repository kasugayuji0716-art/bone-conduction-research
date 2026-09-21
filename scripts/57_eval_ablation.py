"""
スクリプト57: CE-only ablation評価
L1+STFT再構成損失の有無によるCER比較

使い方（DNN PC）:
    python scripts/57_eval_ablation.py
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

CONDITIONS = [
    ('no_se',       'No SE',             None),
    ('taps',        'TAPS pretrained',   BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
    ('ce_recon',    'CE+recon (λ=10.0)', BASE_DIR / 'checkpoints' / 'ce_v2_lambda_10.0' / 'best.th'),
    ('ce_only',     'CE only (λ=10.0)',  BASE_DIR / 'checkpoints' / 'ce_v2_only_lambda_10.0' / 'best.th'),
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
    conditions = [(k, l, p) for k, l, p in CONDITIONS if p is None or p.exists()]
    print(f'Conditions: {[l for _, l, _ in conditions]}')

    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'path': p, 'text': row['text']})
    print(f'Test: {len(samples)} utterances')

    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')

    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt)
            print(f'  {label}: loaded')

    results = {k: [] for k, _, _ in conditions}
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(samples)}...')
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)

        for key, _, ckpt in conditions:
            if ckpt is None:
                audio = wav
            else:
                with torch.no_grad():
                    audio = se_models[key](torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
            segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
            hyp = ''.join(seg.text for seg in segs).strip()
            c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
            results[key].append(c)

    print(f'\n{"="*55}')
    print(f'{"Condition":<25} {"CER":>8} {"n":>6}')
    print('-' * 55)
    for key, label, _ in conditions:
        m = np.mean(results[key])
        print(f'{label:<25} {m:>8.4f} {len(results[key]):>6}')
    print('Done.')


if __name__ == '__main__':
    main()
