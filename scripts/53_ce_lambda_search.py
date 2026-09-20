"""
スクリプト53: CE損失のλ探索（v2: devでλ選択→testで最終評価）

修正点（2026-09-20）:
  - λ選択をdev CERで行い、testは最終評価のみに使用

使い方（DNN PC）:
    python scripts/53_ce_lambda_search.py
"""

import csv, sys, subprocess, torch, numpy as np, soundfile as sf
from pathlib import Path
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel

BASE_DIR = Path(__file__).parent.parent
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT_DIR = BASE_DIR / 'checkpoints'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)

LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
TAG_PREFIX = 'ce_v2_lambda_'


def load_se(ckpt_path):
    from models.seconformer import seconformer
    model = seconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def evaluate_split(split, conditions, asr):
    """Evaluate all SE conditions on a given split."""
    samples = []
    with open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / split / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'path': p, 'text': row['text']})
    print(f'  {split}: {len(samples)} utterances')

    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None and key not in se_models:
            se_models[key] = load_se(ckpt)

    results = {k: [] for k, _, _ in conditions}
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{len(samples)}...')
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

    return {k: np.mean(v) for k, v in results.items()}, len(samples)


def main():
    # Step 1: Train missing lambdas
    print('=== Step 1: 各λで学習 ===\n')
    train_script = BASE_DIR / 'scripts' / '51_train_se_ce_loss.py'

    for lam in LAMBDAS:
        tag = f'{TAG_PREFIX}{lam}'
        ckpt = CKPT_DIR / tag / 'best.th'
        if ckpt.exists():
            print(f'λ={lam}: 既にチェックポイント存在 → スキップ')
            continue
        print(f'\nλ={lam}: 学習開始...')
        cmd = [sys.executable, str(train_script),
               '--lambda_asr', str(lam),
               '--tag', tag,
               '--batch_size', '4']
        subprocess.run(cmd, check=True)

    # Build conditions
    conditions = [
        ('no_se', 'No SE', None),
        ('taps_pretrained', 'TAPS pretrained',
         BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
    ]
    for lam in LAMBDAS:
        tag = f'{TAG_PREFIX}{lam}'
        ckpt = CKPT_DIR / tag / 'best.th'
        if ckpt.exists():
            conditions.append((tag, f'CE λ={lam}', ckpt))

    # ASR
    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')

    # Step 2: Select λ on DEV
    print('\n=== Step 2: devでλ選択 ===\n')
    dev_results, n_dev = evaluate_split('dev', conditions, asr)

    print(f'\n{"Condition":<30} {"dev CER":>10}')
    print('-' * 42)
    best_key, best_cer = None, float('inf')
    for key, label, _ in conditions:
        m = dev_results[key]
        marker = ''
        if key.startswith(TAG_PREFIX) and m < best_cer:
            best_cer = m
            best_key = key
            marker = ' *'
        print(f'{label:<30} {m:>10.4f}{marker}')

    best_label = [l for k, l, _ in conditions if k == best_key][0]
    print(f'\nBest λ on dev: {best_label} (CER={best_cer:.4f})')

    # Step 3: Final evaluation on TEST
    print('\n=== Step 3: testで最終評価 ===\n')
    test_results, n_test = evaluate_split('test', conditions, asr)

    print(f'\n{"Condition":<30} {"dev CER":>10} {"test CER":>10}')
    print('-' * 52)
    for key, label, _ in conditions:
        d = dev_results[key]
        t = test_results[key]
        marker = ' <-- best' if key == best_key else ''
        print(f'{label:<30} {d:>10.4f} {t:>10.4f}{marker}')

    # Save CSV
    out = BASE_DIR / 'results' / 'ce_v2_lambda_search.csv'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'label', 'dev_cer', 'test_cer', 'best_on_dev'])
        for key, label, _ in conditions:
            w.writerow([key, label, dev_results[key], test_results[key],
                        'yes' if key == best_key else ''])
    print(f'\nCSV: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
