"""
スクリプト49: λの探索
複数のλ値でASR-aware SEを学習し、最適なλを見つける。
学習後に全条件を一括評価する。

使い方（DNN PC）:
    python scripts/49_lambda_search.py
"""

import csv, sys, subprocess
from pathlib import Path
import numpy as np, soundfile as sf, torch
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

LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def load_se(ckpt_path):
    from models.seconformer import seconformer as T
    model = T(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def main():
    # ── Step 1: 各λで学習 ──
    print('=== Step 1: 各λで学習 ===\n')
    for lam in LAMBDAS:
        tag = f'lambda_{lam}'
        ckpt = CKPT_DIR / tag / 'best.th'
        if ckpt.exists():
            print(f'λ={lam}: 既にチェックポイント存在 → スキップ')
            continue
        # finetune_baseline (λ=0.0) は既に学習済みならコピー
        if lam == 0.0:
            baseline_ckpt = CKPT_DIR / 'finetune_baseline' / 'best.th'
            if baseline_ckpt.exists():
                (CKPT_DIR / tag).mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(baseline_ckpt, ckpt)
                print(f'λ=0.0: finetune_baselineからコピー')
                continue
        print(f'\n--- λ={lam} 学習開始 ---')
        cmd = [sys.executable, str(BASE_DIR / 'scripts' / '47_train_se_official.py'),
               '--lambda_asr', str(lam), '--tag', tag, '--batch_size', '4']
        subprocess.run(cmd, check=True)

    # ── Step 2: 評価 ──
    print('\n=== Step 2: 全λ評価 ===\n')

    # テストデータ
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'path': p, 'text': row['text']})
    print(f'Test: {len(samples)} utterances')

    # ASR
    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')

    # 条件: TAPS pretrained + 各λ
    conditions = [('taps_pretrained', 'TAPS pretrained', PRETRAINED_DIR / 'seconformer.th')]
    for lam in LAMBDAS:
        tag = f'lambda_{lam}'
        ckpt = CKPT_DIR / tag / 'best.th'
        if ckpt.exists():
            conditions.append((tag, f'λ={lam}', ckpt))

    # SEモデルロード
    se_models = {}
    for key, label, ckpt in conditions:
        se_models[key] = load_se(ckpt)
        print(f'  {label}: loaded')

    # 評価
    results = {k: [] for k, _, _ in conditions}
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(samples)}...')
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)

        for key, _, _ in conditions:
            with torch.no_grad():
                out = se_models[key](torch.from_numpy(wav).unsqueeze(0))
                audio = out.squeeze().numpy()
            segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
            hyp = ''.join(seg.text for seg in segs).strip()
            c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
            results[key].append(c)

    # 結果
    print(f'\n{"="*50}')
    print(f'{"Condition":<25} {"CER":>8}')
    print('-' * 35)
    for key, label, _ in conditions:
        m = np.mean(results[key])
        print(f'{label:<25} {m:>8.4f}')

    # CSV
    RESULT_DIR.mkdir(exist_ok=True)
    out_csv = RESULT_DIR / 'lambda_search.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'label', 'lambda', 'cer_mean', 'n_utts'])
        for key, label, _ in conditions:
            lam = key.replace('lambda_', '') if 'lambda_' in key else 'pretrained'
            w.writerow([key, label, lam, np.mean(results[key]), len(results[key])])
    print(f'\nCSV: {out_csv}')
    print('Done.')


if __name__ == '__main__':
    main()
