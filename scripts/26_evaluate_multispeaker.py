"""
全話者評価（2×2 × 10話者）
testセット全10話者・各100発話 = 1,000件で4条件を評価

条件:
  A: pretrained Whisper small + No SE
  B: pretrained Whisper small + GTCRN
  C: fine-tuned  Whisper small + No SE
  D: fine-tuned  Whisper small + GTCRN

出力:
  results/multispeaker_per_sample.csv   — サンプルごと詳細
  results/multispeaker_per_speaker.csv  — 話者ごとCER
  results/multispeaker_summary.csv      — 条件ごとCER（全体）
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from jiwer import cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR  = Path(__file__).parent.parent
TAPS_DIR  = BASE_DIR / 'data' / 'raw' / 'taps'
GTCRN_DIR = BASE_DIR / 'gtcrn'
FT_CKPT   = BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned'
RESULT_DIR = BASE_DIR / 'results'
TARGET_SR  = 16000

CONDITIONS = ['A_pretrained_no_se', 'B_pretrained_gtcrn',
              'C_finetuned_no_se',  'D_finetuned_gtcrn']


# ── GTCRN ────────────────────────────────────────────────────
def load_gtcrn():
    sys.path.insert(0, str(GTCRN_DIR))
    from gtcrn import GTCRN
    model = GTCRN().eval()
    ckpt = torch.load(
        GTCRN_DIR / 'checkpoints' / 'model_trained_on_dns3.tar',
        map_location='cpu', weights_only=True
    )
    model.load_state_dict(ckpt['model'])
    return model


def apply_gtcrn(model, wav):
    x   = torch.from_numpy(wav.astype('float32'))
    win = torch.hann_window(512).pow(0.5)
    stft_c = torch.stft(x, 512, 256, 512, win, return_complex=True)
    stft_in = torch.view_as_real(stft_c)
    with torch.no_grad():
        stft_out = model(stft_in[None])[0]
    out_c = torch.view_as_complex(stft_out.contiguous())
    enhanced = torch.istft(out_c, 512, 256, 512, win)
    return enhanced.cpu().numpy().astype('float32')


# ── Whisper（pretrained / FT 共通インターフェース）─────────────
def load_whisper(ckpt_path):
    processor = WhisperProcessor.from_pretrained(str(ckpt_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(ckpt_path))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.to(device), device


def transcribe(wav, processor, model, device):
    feats = processor.feature_extractor(
        wav, sampling_rate=TARGET_SR, return_tensors='pt'
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language='korean', task='transcribe',
                             max_new_tokens=225)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


# ── メイン ────────────────────────────────────────────────────
def main():
    # testセット全話者のサンプルを収集
    samples = []
    meta = TAPS_DIR / 'metadata_test.csv'
    if not meta.exists():
        print(f'ERROR: {meta} が見つかりません')
        return

    with open(meta, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid  = row['sentence_id']
            spk  = row['speaker_id']
            p1   = TAPS_DIR / 'throat' / 'test' / f'{spk}_{sid}.wav'
            p2   = TAPS_DIR / 'throat' / f'{spk}_{sid}.wav'
            path = p1 if p1.exists() else (p2 if p2.exists() else None)
            if path:
                samples.append({'spk': spk, 'sid': sid,
                                'path': path, 'text': row['text']})

    speakers = sorted(set(s['spk'] for s in samples))
    print(f'話者数: {len(speakers)},  総サンプル数: {len(samples)}')
    print(f'話者一覧: {speakers}')

    # モデルロード
    print('\nGTCRNロード中...')
    gtcrn = load_gtcrn()

    print('Whisper pretrained ロード中...')
    proc_pre, model_pre, device = load_whisper('openai/whisper-small')

    print('Whisper FT ロード中...')
    proc_ft,  model_ft,  _      = load_whisper(str(FT_CKPT))
    model_ft = model_ft.to(device)

    print(f'\nデバイス: {device}')
    print('評価開始...\n')

    rows = []
    for i, s in enumerate(samples):
        wav, sr = sf.read(str(s['path']), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        wav_se = apply_gtcrn(gtcrn, wav)

        hyp_a = transcribe(wav,    proc_pre, model_pre, device)
        hyp_b = transcribe(wav_se, proc_pre, model_pre, device)
        hyp_c = transcribe(wav,    proc_ft,  model_ft,  device)
        hyp_d = transcribe(wav_se, proc_ft,  model_ft,  device)

        ref = s['text']
        row = {
            'speaker_id':  s['spk'],
            'sentence_id': s['sid'],
            'reference':   ref,
            'A_pretrained_no_se': round(cer(ref, hyp_a), 4),
            'B_pretrained_gtcrn': round(cer(ref, hyp_b), 4),
            'C_finetuned_no_se':  round(cer(ref, hyp_c), 4),
            'D_finetuned_gtcrn':  round(cer(ref, hyp_d), 4),
        }
        rows.append(row)

        if (i + 1) % 50 == 0 or i == 0:
            print(f'[{i+1:4d}/{len(samples)}] {s["spk"]} {s["sid"]}  '
                  f'A={row["A_pretrained_no_se"]:.3f} '
                  f'B={row["B_pretrained_gtcrn"]:.3f} '
                  f'C={row["C_finetuned_no_se"]:.3f} '
                  f'D={row["D_finetuned_gtcrn"]:.3f}')

    # ── per-sample CSV ─────────────────────────────────────
    out_sample = RESULT_DIR / 'multispeaker_per_sample.csv'
    fields = ['speaker_id', 'sentence_id', 'reference'] + CONDITIONS
    with open(out_sample, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # ── per-speaker CSV ────────────────────────────────────
    spk_rows = []
    for spk in speakers:
        sub = [r for r in rows if r['speaker_id'] == spk]
        spk_row = {'speaker_id': spk, 'n': len(sub)}
        for cond in CONDITIONS:
            spk_row[cond] = round(float(np.mean([r[cond] for r in sub])), 4)
        spk_rows.append(spk_row)

    out_spk = RESULT_DIR / 'multispeaker_per_speaker.csv'
    with open(out_spk, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['speaker_id', 'n'] + CONDITIONS)
        w.writeheader(); w.writerows(spk_rows)

    # ── summary CSV ────────────────────────────────────────
    summary = []
    for cond in CONDITIONS:
        vals = [r[cond] for r in rows]
        summary.append({
            'condition': cond,
            'cer_mean':  round(float(np.mean(vals)), 4),
            'cer_std':   round(float(np.std(vals)),  4),
            'n_samples': len(vals),
        })

    out_sum = RESULT_DIR / 'multispeaker_summary.csv'
    with open(out_sum, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'cer_mean', 'cer_std', 'n_samples'])
        w.writeheader(); w.writerows(summary)

    # ── コンソール表示 ─────────────────────────────────────
    print('\n===== 話者別 CER =====')
    print(f'{"話者":<8}', end='')
    for c in CONDITIONS:
        print(f'  {c[:12]:>12}', end='')
    print()
    print('-' * 62)
    for sr in spk_rows:
        print(f'{sr["speaker_id"]:<8}', end='')
        for c in CONDITIONS:
            print(f'  {sr[c]:>12.4f}', end='')
        print()

    print('\n===== 全体平均 =====')
    labels = {'A_pretrained_no_se': 'A: Pretrained + No SE',
              'B_pretrained_gtcrn': 'B: Pretrained + GTCRN',
              'C_finetuned_no_se':  'C: Fine-tuned + No SE',
              'D_finetuned_gtcrn':  'D: Fine-tuned + GTCRN'}
    for s in summary:
        print(f'{labels[s["condition"]]:<28}  CER={s["cer_mean"]:.4f} (±{s["cer_std"]:.4f})')

    print(f'\n保存: {out_sample}')
    print(f'保存: {out_spk}')
    print(f'保存: {out_sum}')


if __name__ == '__main__':
    main()
