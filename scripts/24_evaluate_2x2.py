"""
2×2 比較評価: ASRモデル（未学習 vs FT済み）× SE（なし vs GTCRN）

                | No SE          | With GTCRN SE  |
----------------|----------------|----------------|
Pretrained      | [A] 0.269 *    | [B] 0.296 *    |
Fine-tuned      | [C] 0.095 *    | [D] ???  ← 本スクリプト
                       * フェーズ1・2の既存結果

使い方:
  python scripts/24_evaluate_2x2.py

出力:
  results/phase2_2x2.csv  — 2×2比較表
  results/phase2_2x2_per_sample.csv  — サンプルごと詳細
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from jiwer import cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR   = Path(__file__).parent.parent
TAPS_DIR   = BASE_DIR / 'data' / 'raw' / 'taps'
GTCRN_DIR  = BASE_DIR / 'gtcrn'
FT_CKPT    = BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned'
RESULT_DIR = BASE_DIR / 'results'
TARGET_SR  = 16000

# フェーズ1・2の既存結果
RESULTS_AB_C = {
    'A_pretrained_no_se':   0.269,
    'B_pretrained_gtcrn':   0.296,
    'C_finetuned_no_se':    0.095,
}


# ── GTCRN ────────────────────────────────────────────────────────
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


# ── Whisper FT ───────────────────────────────────────────────────
def load_whisper_ft():
    processor = WhisperProcessor.from_pretrained(str(FT_CKPT))
    model = WhisperForConditionalGeneration.from_pretrained(str(FT_CKPT))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.to(device), device


def transcribe_ft(wav, processor, model, device):
    feats = processor.feature_extractor(
        wav, sampling_rate=TARGET_SR, return_tensors='pt'
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language='korean', task='transcribe',
                             max_new_tokens=225)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


# ── メイン ────────────────────────────────────────────────────────
def main():
    # テストサンプル（話者p00, 50件）を取得
    samples = []
    for meta in [TAPS_DIR / 'metadata_test.csv', TAPS_DIR / 'metadata.csv']:
        if not meta.exists():
            continue
        with open(meta, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['speaker_id'] != 'p00':
                    continue
                sid = row['sentence_id']
                # test split用パス
                p1 = TAPS_DIR / 'throat' / 'test' / f'p00_{sid}.wav'
                # flat構造（フェーズ1）
                p2 = TAPS_DIR / 'throat' / f'p00_{sid}.wav'
                path = p1 if p1.exists() else (p2 if p2.exists() else None)
                if path:
                    samples.append({'sid': sid, 'path': path, 'text': row['text']})
        if samples:
            break

    print(f"評価サンプル数: {len(samples)}")

    print("\nGTCRNロード中...")
    gtcrn = load_gtcrn()

    print("Whisper FTロード中...")
    processor, wft_model, device = load_whisper_ft()

    rows = []
    for i, s in enumerate(samples):
        wav, sr = sf.read(str(s['path']), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # SE適用
        wav_se = apply_gtcrn(gtcrn, wav)

        # FT Whisper で転写
        hyp = transcribe_ft(wav_se, processor, wft_model, device)
        c   = cer(s['text'], hyp)

        rows.append({
            'sentence_id': s['sid'],
            'reference':   s['text'],
            'hypothesis':  hyp,
            'cer':         round(c, 4),
        })
        print(f"[{i+1:2d}/{len(samples)}] {s['sid']}  CER={c:.4f}")

    avg_d = float(np.mean([r['cer'] for r in rows]))
    print(f"\n[D] FT済みWhisper + GTCRN SE: CER = {avg_d:.4f}")

    # ── 2×2 サマリー ──────────────────────────────────────────────
    summary = [
        {'condition': 'A: Pretrained + No SE  (Phase1)',
         'cer': RESULTS_AB_C['A_pretrained_no_se'], 'note': 'Phase1 baseline'},
        {'condition': 'B: Pretrained + GTCRN  (Phase1)',
         'cer': RESULTS_AB_C['B_pretrained_gtcrn'], 'note': 'Phase1 result'},
        {'condition': 'C: Fine-tuned + No SE  (Phase2)',
         'cer': RESULTS_AB_C['C_finetuned_no_se'],  'note': 'Phase2 result'},
        {'condition': 'D: Fine-tuned + GTCRN  (This exp)',
         'cer': round(avg_d, 4), 'note': 'New experiment'},
    ]

    print("\n===== 2×2 比較 =====")
    print(f"{'条件':<40} {'CER':>6}")
    print("-" * 48)
    for r in summary:
        print(f"{r['condition']:<40} {r['cer']:>6.4f}")

    # 保存
    out1 = RESULT_DIR / 'phase2_2x2.csv'
    with open(out1, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'cer', 'note'])
        writer.writeheader()
        writer.writerows(summary)

    out2 = RESULT_DIR / 'phase2_2x2_per_sample.csv'
    with open(out2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence_id', 'reference', 'hypothesis', 'cer'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n保存: {out1}")
    print(f"保存: {out2}")


if __name__ == '__main__':
    main()
