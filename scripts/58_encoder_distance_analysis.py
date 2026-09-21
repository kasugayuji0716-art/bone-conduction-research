"""
スクリプト58: Encoder距離分析
- 各SE条件 × 各ASRモデルのencoder距離を計測
- 「なぜCEがEnc L1より汎用的か」の理論的根拠
- 「SEがASRに有効/逆効果になる条件」の体系的分析

使い方（DNN PC）:
    python scripts/58_encoder_distance_analysis.py
"""

import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from transformers import WhisperModel, WhisperFeatureExtractor
import torch.nn.functional as F

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
    ('no_se',           'No SE',            None),
    ('taps_pretrained', 'TAPS pretrained',  BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
    ('enc_best',        'Enc L1 (λ=5.0)',   BASE_DIR / 'checkpoints' / 'enc_v2_lambda_5.0' / 'best.th'),
    ('ce_best',         'CE (λ=10.0)',      BASE_DIR / 'checkpoints' / 'ce_v2_lambda_10.0' / 'best.th'),
]

ASR_MODELS = [
    ('whisper-base',   'openai/whisper-base'),
    ('whisper-small',  'openai/whisper-small'),
    ('whisper-medium', 'openai/whisper-medium'),
]


def load_se(ckpt_path):
    from models.seconformer import seconformer
    model = seconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def compute_mel(wav, feature_extractor):
    """Compute log-mel spectrogram using Whisper's feature extractor."""
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    return inputs.input_features.to(DEVICE)


def main():
    conditions = [(k, l, p) for k, l, p in SE_CONDITIONS if p is None or p.exists()]
    print(f'SE conditions: {[l for _, l, _ in conditions]}')
    print(f'ASR models: {[n for n, _ in ASR_MODELS]}')

    # Load test data (throat + acoustic pairs)
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid, uid = row['speaker_id'], row['sentence_id']
            t = TAPS_DIR / 'throat' / 'test' / f'{sid}_{uid}.wav'
            a = TAPS_DIR / 'acoustic' / 'test' / f'{sid}_{uid}.wav'
            if t.exists() and a.exists():
                samples.append({'throat': t, 'acoustic': a, 'speaker': sid})
    print(f'Test: {len(samples)} pairs\n')

    # Load SE models
    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt)
            print(f'  SE {label}: loaded')

    # Pre-compute SE outputs
    print('\nPre-computing SE outputs...')
    se_outputs = {k: [] for k, _, _ in conditions}
    acoustic_wavs = []
    for s in samples:
        t_wav, _ = sf.read(s['throat'], dtype='float32')
        a_wav, _ = sf.read(s['acoustic'], dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)
        acoustic_wavs.append(a_wav)

        for key, _, ckpt in conditions:
            if ckpt is None:
                se_outputs[key].append(t_wav)
            else:
                with torch.no_grad():
                    audio = se_models[key](torch.from_numpy(t_wav).unsqueeze(0)).squeeze().numpy()
                se_outputs[key].append(audio)
    print(f'  Done ({len(samples)} utterances)')

    del se_models
    torch.cuda.empty_cache()

    # For each ASR model, compute encoder distances
    all_results = []

    for asr_name, asr_id in ASR_MODELS:
        print(f'\n=== Encoder: {asr_name} ===')

        feature_extractor = WhisperFeatureExtractor.from_pretrained(asr_id)
        whisper = WhisperModel.from_pretrained(asr_id).to(DEVICE).eval()
        encoder = whisper.encoder

        for key, label, _ in conditions:
            l1_dists = []
            cos_dists = []

            for i in range(len(samples)):
                se_wav = se_outputs[key][i]
                ac_wav = acoustic_wavs[i]

                with torch.no_grad():
                    mel_se = compute_mel(se_wav, feature_extractor)
                    mel_ac = compute_mel(ac_wav, feature_extractor)

                    enc_se = encoder(mel_se).last_hidden_state  # (1, T, D)
                    enc_ac = encoder(mel_ac).last_hidden_state

                    # L1 distance (per-frame, then mean)
                    min_t = min(enc_se.shape[1], enc_ac.shape[1])
                    l1 = F.l1_loss(enc_se[:, :min_t], enc_ac[:, :min_t]).item()
                    l1_dists.append(l1)

                    # Cosine distance (1 - cosine_similarity, averaged over frames)
                    cos_sim = F.cosine_similarity(
                        enc_se[:, :min_t].squeeze(0),
                        enc_ac[:, :min_t].squeeze(0),
                        dim=-1
                    ).mean().item()
                    cos_dists.append(1.0 - cos_sim)

            mean_l1 = np.mean(l1_dists)
            mean_cos = np.mean(cos_dists)
            print(f'  {label:<25} L1={mean_l1:.4f}  cos_dist={mean_cos:.6f}')

            all_results.append({
                'asr_model': asr_name,
                'se_condition': key,
                'se_label': label,
                'l1_distance': mean_l1,
                'cosine_distance': mean_cos,
                'n_utts': len(l1_dists),
            })

        del whisper, encoder, feature_extractor
        torch.cuda.empty_cache()

    # Summary table
    print(f'\n{"="*80}')
    print(f'{"SE Condition":<25}', end='')
    for asr_name, _ in ASR_MODELS:
        print(f' {asr_name + " L1":>16} {asr_name + " cos":>16}', end='')
    print()
    print('-' * 80)
    for key, label, _ in conditions:
        print(f'{label:<25}', end='')
        for asr_name, _ in ASR_MODELS:
            r = [x for x in all_results if x['asr_model'] == asr_name and x['se_condition'] == key]
            if r:
                print(f' {r[0]["l1_distance"]:>16.4f} {r[0]["cosine_distance"]:>16.6f}', end='')
        print()

    # CER data for correlation (from cross_asr_evaluation.csv if available)
    cer_file = BASE_DIR / 'results' / 'cross_asr_evaluation.csv'
    if cer_file.exists():
        print(f'\n{"="*80}')
        print('Encoder Distance vs CER Correlation')
        print('='*80)

        cer_data = {}
        with open(cer_file) as f:
            for row in csv.DictReader(f):
                cer_data[(row['asr_model'], row['se_condition'])] = float(row['cer_mean'])

        for asr_name, _ in ASR_MODELS:
            dists = []
            cers = []
            labels = []
            for r in all_results:
                if r['asr_model'] == asr_name:
                    cer_key = (asr_name, r['se_condition'])
                    if cer_key in cer_data:
                        dists.append(r['l1_distance'])
                        cers.append(cer_data[cer_key])
                        labels.append(r['se_label'])

            if len(dists) >= 3:
                corr = np.corrcoef(dists, cers)[0, 1]
                print(f'\n  {asr_name}: r = {corr:.4f} (n={len(dists)})')
                for l, d, c in zip(labels, dists, cers):
                    print(f'    {l:<25} L1={d:.4f}  CER={c:.4f}')

    # Save CSV
    out = BASE_DIR / 'results' / 'encoder_distance_analysis.csv'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'asr_model', 'se_condition', 'se_label',
            'l1_distance', 'cosine_distance', 'n_utts'
        ])
        w.writeheader()
        w.writerows(all_results)
    print(f'\nCSV: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
