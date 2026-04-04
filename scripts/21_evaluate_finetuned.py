"""
フェーズ2評価: ファインチューニング済みGTCRNのCER計測

- フェーズ1と同じ50サンプル（話者p00, u00〜u49）で評価
- ファインチューニング済みGTCRNをSEとして適用後、Whisperでテキスト化してCERを計測
- フェーズ1のsummary.csvと比較した結果を出力

出力:
  results/phase2_cer.csv      — 本スクリプトの計測結果
  results/phase2_comparison.csv — フェーズ1との比較表
"""

import os, sys, csv, tempfile, unicodedata, re
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import cer

sys.path.insert(0, str(Path(__file__).parent.parent / 'gtcrn'))
from gtcrn import GTCRN

BASE_DIR    = Path(__file__).parent.parent
META_PATH   = BASE_DIR / 'data' / 'raw' / 'taps' / 'metadata_test.csv'
CKPT_PATH   = BASE_DIR / 'checkpoints' / 'gtcrn_taps_finetuned.tar'
PRETRAINED  = BASE_DIR / 'gtcrn' / 'checkpoints' / 'model_trained_on_dns3.tar'
RESULT_DIR  = BASE_DIR / 'results'
RESULT_DIR.mkdir(exist_ok=True)

N_SAMPLES   = 50   # フェーズ1と同じ50件
N_FFT, HOP, WIN = 512, 256, 512


def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\s。、！？()（）\[\]【】,，]', '', text)
    return text.strip()


def load_model(ckpt_path, device):
    model = GTCRN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def apply_se(model, wav_path, device, rms_normalize=False):
    """喉マイク音声にSEを適用して強調音声を返す

    rms_normalize=False: 生音声をそのまま入力（オリジナルGTCRN / フェーズ1と同条件）
    rms_normalize=True:  入力をRMS=1に正規化してSE適用後、出力RMSを入力に揃える
                         （RMS正規化で再学習したFTモデル用）
    """
    wav, sr = sf.read(wav_path, dtype='float32')
    assert sr == 16000, f"SR={sr}"
    wav_t = torch.from_numpy(wav).to(device)
    window = torch.hann_window(WIN).pow(0.5).to(device)

    input_rms = wav_t.pow(2).mean().sqrt().clamp(min=1e-8)
    wav_in = wav_t / input_rms if rms_normalize else wav_t

    with torch.no_grad():
        spec_c = torch.stft(wav_in, N_FFT, HOP, WIN, window, return_complex=True)
        spec = torch.view_as_real(spec_c)                      # (F, T, 2)
        enh_spec = model(spec[None])[0]                        # (F, T, 2)
        enh_c = torch.view_as_complex(enh_spec.contiguous())   # (F, T) complex
        enh = torch.istft(enh_c, N_FFT, HOP, WIN, window)
        if rms_normalize:
            enh_rms = enh.pow(2).mean().sqrt().clamp(min=1e-8)
            enh = enh / enh_rms * input_rms
    return enh.cpu().numpy(), sr


def measure_cer_batch(whisper_model, wav_paths, refs):
    scores = []
    for path, ref in zip(wav_paths, refs):
        segs, _ = whisper_model.transcribe(str(path), language='ko')
        hyp = normalize(''.join([s.text for s in segs]))
        scores.append(cer(normalize(ref), hyp))
    return scores


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ── メタデータ読み込み（p00の最初50件）
    with open(META_PATH, encoding='utf-8') as f:
        all_records = list(csv.DictReader(f))
    records = [r for r in all_records if r['speaker_id'] == 'p00'][:N_SAMPLES]
    print(f"評価サンプル: {len(records)}件（話者p00）")

    # ── Whisperロード
    print("Whisperロード中...")
    whisper = WhisperModel('small', device='cpu', compute_type='int8')

    # ── フェーズ1 baseline_throat のCER（生音声）
    print("\n[1/3] baseline_throat（生喉マイク音声）...")
    throat_paths = [r['throat_path'] for r in records]
    refs = [r['text'] for r in records]
    baseline_cers = measure_cer_batch(whisper, throat_paths, refs)
    baseline_avg = sum(baseline_cers) / len(baseline_cers)
    print(f"  CER = {baseline_avg:.4f}")

    # ── オリジナルGTCRN（DNS3学習済み）
    print("\n[2/3] gtcrn_original（DNS3学習済み）...")
    orig_model = load_model(PRETRAINED, device)
    with tempfile.TemporaryDirectory() as tmpdir:
        enh_paths = []
        for r in records:
            enh, sr = apply_se(orig_model, r['throat_path'], device, rms_normalize=False)
            out = os.path.join(tmpdir, f"{r['speaker_id']}_{r['sentence_id']}.wav")
            sf.write(out, enh, sr)
            enh_paths.append(out)
        orig_cers = measure_cer_batch(whisper, enh_paths, refs)
    orig_avg = sum(orig_cers) / len(orig_cers)
    print(f"  CER = {orig_avg:.4f}")

    # ── ファインチューニング済みGTCRN
    print("\n[3/3] gtcrn_finetuned（TAPS喉マイクFT）...")
    ft_model = load_model(CKPT_PATH, device)
    with tempfile.TemporaryDirectory() as tmpdir:
        enh_paths = []
        for i, r in enumerate(records):
            enh, sr = apply_se(ft_model, r['throat_path'], device, rms_normalize=True)
            out = os.path.join(tmpdir, f"{r['speaker_id']}_{r['sentence_id']}.wav")
            sf.write(out, enh, sr)
            enh_paths.append(out)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_SAMPLES}件処理済み")
        ft_cers = measure_cer_batch(whisper, enh_paths, refs)
    ft_avg = sum(ft_cers) / len(ft_cers)
    print(f"  CER = {ft_avg:.4f}")

    # ── 結果保存
    per_sample = []
    for r, b, o, f in zip(records, baseline_cers, orig_cers, ft_cers):
        per_sample.append({
            'speaker_id':      r['speaker_id'],
            'sentence_id':     r['sentence_id'],
            'cer_baseline':    round(b, 4),
            'cer_gtcrn_orig':  round(o, 4),
            'cer_gtcrn_ft':    round(f, 4),
        })
    cer_path = RESULT_DIR / 'phase2_cer.csv'
    with open(cer_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=per_sample[0].keys())
        writer.writeheader()
        writer.writerows(per_sample)

    # ── フェーズ1 summary.csv との比較表
    phase1 = {}
    summary_path = RESULT_DIR / 'summary.csv'
    if summary_path.exists():
        with open(summary_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                phase1[row['condition']] = float(row['avg_cer'])

    comparison = [
        {'condition': 'baseline_acoustic',   'phase1_cer': phase1.get('baseline_acoustic', '-'),  'phase2_cer': '-'},
        {'condition': 'baseline_throat',      'phase1_cer': phase1.get('baseline_throat',   '-'),  'phase2_cer': round(baseline_avg, 4)},
        {'condition': 'gtcrn/clean (orig)',   'phase1_cer': phase1.get('gtcrn/clean',       '-'),  'phase2_cer': round(orig_avg, 4)},
        {'condition': 'gtcrn/clean (ft)',     'phase1_cer': '-',                                    'phase2_cer': round(ft_avg, 4)},
    ]
    comp_path = RESULT_DIR / 'phase2_comparison.csv'
    with open(comp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'phase1_cer', 'phase2_cer'])
        writer.writeheader()
        writer.writerows(comparison)

    # ── サマリ表示
    print("\n" + "=" * 55)
    print("フェーズ1 vs フェーズ2 CER比較")
    print("=" * 55)
    print(f"{'条件':<28} {'フェーズ1':>10} {'フェーズ2':>10}")
    print("-" * 55)
    for row in comparison:
        print(f"{row['condition']:<28} {str(row['phase1_cer']):>10} {str(row['phase2_cer']):>10}")
    print("=" * 55)

    delta = ft_avg - baseline_avg
    sign = "↑悪化" if delta > 0 else "↓改善"
    print(f"\nbaseline_throat → gtcrn_ft: {delta:+.4f} ({sign})")
    delta2 = ft_avg - orig_avg
    sign2 = "↑悪化" if delta2 > 0 else "↓改善"
    print(f"gtcrn_orig      → gtcrn_ft: {delta2:+.4f} ({sign2})")

    print(f"\n保存先: {cer_path}")
    print(f"        {comp_path}")


if __name__ == '__main__':
    main()
