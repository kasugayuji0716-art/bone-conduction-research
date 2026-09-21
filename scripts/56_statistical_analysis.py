"""
スクリプト56: 統計検定 + 話者別分析
- Wilcoxon符号順位検定（TAPS pretrained vs CE λ=2.0）
- 話者別CER比較
- per-utterance CER出力

使い方（DNN PC）:
    python scripts/56_statistical_analysis.py
"""

import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel
from scipy.stats import wilcoxon
from collections import defaultdict

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
    ('no_se',           'No SE',                None),
    ('taps_pretrained', 'TAPS pretrained',      BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
    ('enc_best',        'Enc L1 (λ=5.0)',       BASE_DIR / 'checkpoints' / 'enc_v2_lambda_5.0' / 'best.th'),
    ('ce_best',         'CE (λ=10.0)',          BASE_DIR / 'checkpoints' / 'ce_v2_lambda_10.0' / 'best.th'),
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

    # Load test data
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid = row['speaker_id']
            uid = row['sentence_id']
            p = TAPS_DIR / 'throat' / 'test' / f'{sid}_{uid}.wav'
            if p.exists():
                samples.append({'path': p, 'text': row['text'], 'speaker': sid})
    print(f'Test: {len(samples)} utterances')

    # ASR
    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')

    # SE models
    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt)
            print(f'  {label}: loaded')

    # Per-utterance evaluation
    per_utt = []
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(samples)}...')
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)

        row = {'speaker': s['speaker'], 'utterance': Path(s['path']).stem}
        for key, _, ckpt in conditions:
            if ckpt is None:
                audio = wav
            else:
                with torch.no_grad():
                    audio = se_models[key](torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
            segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
            hyp = ''.join(seg.text for seg in segs).strip()
            c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
            row[key] = c
        per_utt.append(row)

    # Save per-utterance CER
    out_per = BASE_DIR / 'results' / 'per_utterance_cer_all.csv'
    out_per.parent.mkdir(exist_ok=True)
    keys = [k for k, _, _ in conditions]
    with open(out_per, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['speaker', 'utterance'] + keys)
        w.writeheader()
        w.writerows(per_utt)
    print(f'\nPer-utterance CER: {out_per}')

    # === 1. Wilcoxon signed-rank tests ===
    print(f'\n{"="*60}')
    print('Wilcoxon Signed-Rank Tests')
    print('='*60)

    pairs = [
        ('taps_pretrained', 'ce_best',  'TAPS vs CE(λ=2.0)'),
        ('taps_pretrained', 'enc_best', 'TAPS vs Enc L1(λ=5.0)'),
        ('enc_best',        'ce_best',  'Enc L1 vs CE(λ=2.0)'),
        ('no_se',           'ce_best',  'No SE vs CE(λ=2.0)'),
    ]

    for k1, k2, desc in pairs:
        if k1 not in keys or k2 not in keys:
            continue
        a = np.array([r[k1] for r in per_utt])
        b = np.array([r[k2] for r in per_utt])
        diff = a - b
        n_better = np.sum(diff > 0)
        n_worse = np.sum(diff < 0)
        n_tie = np.sum(diff == 0)
        stat, p = wilcoxon(a, b, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'\n  {desc}')
        print(f'    {k1} mean={np.mean(a):.4f}, {k2} mean={np.mean(b):.4f}, diff={np.mean(diff):.4f}')
        print(f'    better={n_better}, worse={n_worse}, tie={n_tie}')
        print(f'    W={stat:.1f}, p={p:.2e} {sig}')

    # === 2. Per-speaker analysis ===
    print(f'\n{"="*60}')
    print('Per-Speaker CER')
    print('='*60)

    speakers = sorted(set(r['speaker'] for r in per_utt))
    print(f'\n{"Speaker":<10}', end='')
    for key, label, _ in conditions:
        print(f' {label:>20}', end='')
    print()
    print('-' * (10 + 21 * len(conditions)))

    speaker_data = defaultdict(lambda: defaultdict(list))
    for r in per_utt:
        for key in keys:
            speaker_data[r['speaker']][key].append(r[key])

    for spk in speakers:
        print(f'{spk:<10}', end='')
        for key, _, _ in conditions:
            m = np.mean(speaker_data[spk][key])
            print(f' {m:>20.4f}', end='')
        print()

    # Check consistency: CE < TAPS for all speakers?
    if 'taps_pretrained' in keys and 'ce_best' in keys:
        print(f'\nCE(λ=2.0) < TAPS pretrained for all speakers?')
        all_better = True
        for spk in speakers:
            taps_m = np.mean(speaker_data[spk]['taps_pretrained'])
            ce_m = np.mean(speaker_data[spk]['ce_best'])
            status = 'YES' if ce_m < taps_m else 'NO'
            if ce_m >= taps_m:
                all_better = False
            print(f'  {spk}: TAPS={taps_m:.4f}, CE={ce_m:.4f} → {status}')
        print(f'  All speakers improved: {"YES" if all_better else "NO"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
