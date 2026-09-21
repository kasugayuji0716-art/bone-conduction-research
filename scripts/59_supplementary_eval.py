"""
スクリプト59: 補足実験
1. 気導マイクCER（test 1,000発話）
2. Enc L1のdev CER（λ=5.0固定）
3. v2 λ=0（再構成損失のみの追加学習）対照

使い方（DNN PC）:
    # 気導CER + Enc devCER
    python scripts/59_supplementary_eval.py

    # v2 λ=0の学習も行う場合
    python scripts/59_supplementary_eval.py --train_lambda0
"""

import argparse
import csv, sys, os, subprocess, torch, numpy as np, soundfile as sf
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


def load_se(ckpt_path):
    from models.seconformer import seconformer
    model = seconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def eval_cer(samples, asr, se_model=None, label=""):
    cers = []
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{len(samples)}...')
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if se_model is not None:
            with torch.no_grad():
                wav = se_model(torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
        segs, _ = asr.transcribe(wav, language='ko', beam_size=5)
        hyp = ''.join(seg.text for seg in segs).strip()
        c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
        cers.append(c)
    mean_cer = np.mean(cers)
    print(f'  {label}: CER={mean_cer:.4f} (n={len(cers)})')
    return mean_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lambda0', action='store_true')
    args = parser.parse_args()

    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')

    # === 1. Acoustic CER (test 1000) ===
    print('\n=== 1. 気導マイクCER（test 1,000発話） ===')
    acoustic_samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'acoustic' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                acoustic_samples.append({'path': p, 'text': row['text']})
    acoustic_cer = eval_cer(acoustic_samples, asr, label="Acoustic (test)")

    # === 2. Enc L1 dev CER ===
    print('\n=== 2. Enc L1 dev CER ===')
    enc_ckpt = BASE_DIR / 'checkpoints' / 'enc_v2_lambda_5.0' / 'best.th'
    if enc_ckpt.exists():
        enc_model = load_se(enc_ckpt)
        for split in ['dev', 'test']:
            samples = []
            with open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    p = TAPS_DIR / 'throat' / split / f"{row['speaker_id']}_{row['sentence_id']}.wav"
                    if p.exists():
                        samples.append({'path': p, 'text': row['text']})
            eval_cer(samples, asr, enc_model, label=f"Enc L1 ({split})")

        # TAPS pretrained dev for comparison
        taps_ckpt = BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'
        taps_model = load_se(taps_ckpt)
        dev_samples = []
        with open(TAPS_DIR / 'metadata_dev.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                p = TAPS_DIR / 'throat' / 'dev' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
                if p.exists():
                    dev_samples.append({'path': p, 'text': row['text']})
        eval_cer(dev_samples, asr, taps_model, label="TAPS pretrained (dev)")
        eval_cer(dev_samples, asr, label="No SE (dev)")
    else:
        print(f'  Enc L1 checkpoint not found: {enc_ckpt}')

    # === 3. v2 λ=0 (recon-only fine-tuning) ===
    print('\n=== 3. v2 λ=0（再構成損失のみの追加学習） ===')
    lambda0_ckpt = BASE_DIR / 'checkpoints' / 'ce_v2_lambda_0.0' / 'best.th'
    if args.train_lambda0 and not lambda0_ckpt.exists():
        print('  学習開始...')
        train_script = BASE_DIR / 'scripts' / '51_train_se_ce_loss.py'
        cmd = [sys.executable, str(train_script),
               '--lambda_asr', '0.0',
               '--tag', 'ce_v2_lambda_0.0',
               '--batch_size', '4']
        subprocess.run(cmd, check=True)

    if lambda0_ckpt.exists():
        lambda0_model = load_se(lambda0_ckpt)
        for split in ['dev', 'test']:
            samples = []
            with open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    p = TAPS_DIR / 'throat' / split / f"{row['speaker_id']}_{row['sentence_id']}.wav"
                    if p.exists():
                        samples.append({'path': p, 'text': row['text']})
            eval_cer(samples, asr, lambda0_model, label=f"v2 lambda=0 ({split})")
    else:
        print(f'  λ=0 checkpoint not found. Run with --train_lambda0 to train.')

    print('\nDone.')


if __name__ == '__main__':
    main()
