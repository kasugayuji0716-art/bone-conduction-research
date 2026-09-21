"""
スクリプト55: SE条件別のSTOI・PESQ評価
CE-aware SEが知覚品質を犠牲にしていないか確認。

使い方（DNN PC）:
    pip install pystoi pesq
    python scripts/55_eval_stoi_pesq.py
"""

import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from pystoi import stoi
from pesq import pesq

BASE_DIR = Path(__file__).parent.parent
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)

SR = 16000

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

    # Load test data (throat + acoustic pairs)
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid, uid = row['speaker_id'], row['sentence_id']
            t = TAPS_DIR / 'throat' / 'test' / f'{sid}_{uid}.wav'
            a = TAPS_DIR / 'acoustic' / 'test' / f'{sid}_{uid}.wav'
            if t.exists() and a.exists():
                samples.append({'throat': t, 'acoustic': a})
    print(f'Test: {len(samples)} pairs')

    # Load SE models
    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt)
            print(f'  {label}: loaded')

    # Evaluate
    results = {k: {'stoi': [], 'pesq': []} for k, _, _ in conditions}

    for i, s in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(samples)}...')

        t_wav, _ = sf.read(s['throat'], dtype='float32')
        a_wav, _ = sf.read(s['acoustic'], dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)
        min_len = min(len(t_wav), len(a_wav))
        a_wav = a_wav[:min_len]

        for key, _, ckpt in conditions:
            if ckpt is None:
                audio = t_wav[:min_len]
            else:
                with torch.no_grad():
                    audio = se_models[key](torch.from_numpy(t_wav).unsqueeze(0)).squeeze().numpy()
                audio = audio[:min_len]

            # STOI
            try:
                s_val = stoi(a_wav, audio, SR, extended=False)
                results[key]['stoi'].append(s_val)
            except Exception:
                pass

            # PESQ (wideband)
            try:
                p_val = pesq(SR, a_wav, audio, 'wb')
                results[key]['pesq'].append(p_val)
            except Exception:
                pass

    # Results
    print(f'\n{"="*65}')
    print(f'{"Condition":<25} {"STOI":>8} {"PESQ":>8} {"n":>6}')
    print('-' * 65)
    for key, label, _ in conditions:
        s_mean = np.mean(results[key]['stoi']) if results[key]['stoi'] else float('nan')
        p_mean = np.mean(results[key]['pesq']) if results[key]['pesq'] else float('nan')
        n = len(results[key]['stoi'])
        print(f'{label:<25} {s_mean:>8.4f} {p_mean:>8.4f} {n:>6}')

    # CSV
    out = BASE_DIR / 'results' / 'se_stoi_pesq.csv'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'label', 'stoi_mean', 'pesq_mean', 'n_utts'])
        for key, label, _ in conditions:
            s_mean = np.mean(results[key]['stoi']) if results[key]['stoi'] else ''
            p_mean = np.mean(results[key]['pesq']) if results[key]['pesq'] else ''
            w.writerow([key, label, s_mean, p_mean, len(results[key]['stoi'])])
    print(f'\nCSV: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
