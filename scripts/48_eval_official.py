"""
スクリプト48: TAPS公式SE-Conformerベースの全条件CER評価
script 47で学習したモデルをTAPS公式forwardで評価。

条件:
    A: No SE
    B: TAPS pretrained SE-Conformer（公式forward）
    C: SI-SDRのみ再学習（公式forward）
    D: ASR-aware再学習（公式forward）

使い方（DNN PC）:
    python scripts/48_eval_official.py
"""

import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
CKPT_DIR       = BASE_DIR / 'checkpoints'
RESULT_DIR     = BASE_DIR / 'results'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)


def load_se_official(ckpt_path):
    from models.seconformer import seconformer as TAPSSeconformer
    model = TAPSSeconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def main():
    conditions = [
        ('no_se',               'No SE',                        None),
        ('taps_pretrained',     'TAPS pretrained',              PRETRAINED_DIR / 'seconformer.th'),
        ('finetune_baseline',   'L1+STFT finetune (no ASR)',    CKPT_DIR / 'finetune_baseline' / 'best.th'),
        ('finetune_asr_aware',  'L1+STFT+ASR finetune',        CKPT_DIR / 'finetune_asr_aware' / 'best.th'),
    ]
    conditions = [(k, l, p) for k, l, p in conditions if p is None or Path(p).exists()]
    print(f'Conditions: {[l for _, l, _ in conditions]}')

    # Data
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'path': p, 'text': row['text']})
    print(f'Test utterances: {len(samples)}')

    # ASR
    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')

    # SE models
    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se_official(ckpt)
            print(f'  {label}: loaded')

    # Evaluate
    results = {k: [] for k, _, _ in conditions}
    for i, s in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(samples)}...')
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)

        for key, _, ckpt in conditions:
            if ckpt is None:
                audio = wav
            else:
                with torch.no_grad():
                    out = se_models[key](torch.from_numpy(wav).unsqueeze(0))
                    audio = out.squeeze().numpy()
            segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
            hyp = ''.join(seg.text for seg in segs).strip()
            c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
            results[key].append(c)

    # Results
    print(f'\n{"="*50}')
    print(f'{"Condition":<35} {"CER":>8} {"n":>6}')
    print('-' * 50)
    for key, label, _ in conditions:
        m = np.mean(results[key])
        print(f'{label:<35} {m:>8.4f} {len(results[key]):>6}')

    # CSV
    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / 'official_comparison.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'label', 'cer_mean', 'n_utts'])
        for key, label, _ in conditions:
            w.writerow([key, label, np.mean(results[key]), len(results[key])])
    print(f'\nCSV: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
