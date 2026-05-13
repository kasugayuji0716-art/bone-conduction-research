"""
NIR (Noise Immunity Retention) 計測スクリプト
フェーズ3 Step 1: FT済みWhisperのノイズ免疫性を定量評価する

NIR = pretrained_noise_sensitivity / FT_noise_sensitivity
noise_sensitivity = CER_noisy - CER_clean

NIR > 1.0: FTによりノイズ耐性が向上（理想的）
NIR ≈ 1.0: ノイズ免疫性が保持されている
NIR < 1.0: FTによりノイズ免疫性が低下 → Phase 3（Adapter+KD）の動機

出力:
  results/nir_per_sample_{start}_{end}.csv  -- サンプル×条件 CER（部分）
  results/nir_summary.csv                   -- 条件ごとCER + NIR（全体マージ後）
  results/nir_report.txt                    -- 人間が読みやすいサマリー（全体マージ後）

実行例:
  # 全件（1GPU）
  python3 scripts/28_nir_evaluation.py

  # 2GPU並列
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 scripts/28_nir_evaluation.py --start 0   --end 500  2>&1 | tee logs/nir_gpu0.log &
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 scripts/28_nir_evaluation.py --start 500 --end 1000 2>&1 | tee logs/nir_gpu1.log &
  wait
  python3 scripts/28_nir_evaluation.py --merge
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from jiwer import cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR   = Path(__file__).parent.parent
TAPS_DIR   = BASE_DIR / 'data' / 'raw' / 'taps'
FT_CKPT    = BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned'
RESULT_DIR = BASE_DIR / 'results'
TARGET_SR  = 16000

SNR_LEVELS  = [20, 10, 5, 0, -5]
NOISE_TYPES = ['white', 'pink']


# ── ノイズ生成 ────────────────────────────────────────────────
def rms(signal):
    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def generate_white_noise(length):
    return np.random.randn(length).astype(np.float32)


def generate_pink_noise(length):
    white = np.random.randn(length)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1.0
    power = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(np.fft.rfft(white) * power, n=length)
    return (pink / (np.max(np.abs(pink)) + 1e-12)).astype(np.float32)


def add_noise(wav, noise_type, snr_db):
    noise = generate_white_noise(len(wav)) if noise_type == 'white' \
            else generate_pink_noise(len(wav))
    target_rms = rms(wav) / (10 ** (snr_db / 20.0))
    noise = noise * (target_rms / rms(noise))
    mixed = wav + noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


# ── Whisper ───────────────────────────────────────────────────
def load_whisper(ckpt_path):
    processor = WhisperProcessor.from_pretrained(str(ckpt_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(ckpt_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


def transcribe(wav, processor, model, device):
    feats = processor.feature_extractor(
        wav, sampling_rate=TARGET_SR, return_tensors='pt'
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language='korean', task='transcribe',
                             max_new_tokens=225)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


# ── NIRサマリー計算・レポート出力（マージ後に呼ぶ）────────────
def compute_summary_and_report(rows):
    pre_clean_mean = float(np.mean([r['pre_clean'] for r in rows]))
    ft_clean_mean  = float(np.mean([r['ft_clean']  for r in rows]))

    summary_rows = []
    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS:
            key = f'{noise_type}_snr{snr:+d}'
            pre_noisy = float(np.mean([r[f'pre_{key}'] for r in rows]))
            ft_noisy  = float(np.mean([r[f'ft_{key}']  for r in rows]))
            pre_sens  = pre_noisy - pre_clean_mean
            ft_sens   = ft_noisy  - ft_clean_mean
            nir = (pre_sens / ft_sens) if abs(ft_sens) > 1e-6 else float('inf')
            summary_rows.append({
                'noise_type':      noise_type,
                'snr_db':          snr,
                'pre_clean_cer':   round(pre_clean_mean, 4),
                'ft_clean_cer':    round(ft_clean_mean,  4),
                'pre_noisy_cer':   round(pre_noisy, 4),
                'ft_noisy_cer':    round(ft_noisy,  4),
                'pre_sensitivity': round(pre_sens,  4),
                'ft_sensitivity':  round(ft_sens,   4),
                'NIR':             round(nir, 4) if nir != float('inf') else 'inf',
            })

    out_sum = RESULT_DIR / 'nir_summary.csv'
    fields = ['noise_type', 'snr_db', 'pre_clean_cer', 'ft_clean_cer',
              'pre_noisy_cer', 'ft_noisy_cer', 'pre_sensitivity', 'ft_sensitivity', 'NIR']
    with open(out_sum, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    lines = []
    lines.append('=' * 65)
    lines.append('  Noise Immunity Retention (NIR) Report')
    lines.append('=' * 65)
    lines.append(f'\nClean CER:')
    lines.append(f'  Pretrained Whisper : {pre_clean_mean:.4f}')
    lines.append(f'  Fine-tuned Whisper : {ft_clean_mean:.4f}')
    lines.append(f'  改善率             : {(1 - ft_clean_mean / pre_clean_mean) * 100:.1f}%')
    lines.append(f'\nNIR の読み方:')
    lines.append(f'  NIR > 1.0 → FTによりノイズ耐性が向上（理想的）')
    lines.append(f'  NIR ≈ 1.0 → ノイズ免疫性が保持されている')
    lines.append(f'  NIR < 1.0 → FTによりノイズ免疫性が低下 → Phase 3の動機')
    lines.append('')
    lines.append(f'{"ノイズ":>6} {"SNR":>6}  {"pre_noisy":>10} {"ft_noisy":>10}'
                 f'  {"pre_sens":>9} {"ft_sens":>9}  {"NIR":>7}')
    lines.append('-' * 68)
    for r in summary_rows:
        nir_str = f'{r["NIR"]:>7.4f}' if r['NIR'] != 'inf' else '    inf'
        lines.append(f'{r["noise_type"]:>6} {r["snr_db"]:>+5}dB'
                     f'  {r["pre_noisy_cer"]:>10.4f} {r["ft_noisy_cer"]:>10.4f}'
                     f'  {r["pre_sensitivity"]:>9.4f} {r["ft_sensitivity"]:>9.4f}'
                     f'  {nir_str}')

    nir_vals = [r['NIR'] for r in summary_rows if r['NIR'] != 'inf']
    avg_nir = float(np.mean(nir_vals)) if nir_vals else 0.0
    lines.append(f'\n平均NIR（全条件）: {avg_nir:.4f}')
    if avg_nir < 0.8:
        lines.append('→ FTによりノイズ免疫性が大幅に低下。Adapter+KDの導入根拠として強い。')
    elif avg_nir < 1.0:
        lines.append('→ FTにより若干のノイズ免疫性低下。Adapter+KDで改善の余地あり。')
    else:
        lines.append('→ FTによりノイズ免疫性が保持/向上。Adapter+KDでさらなる向上を目標に。')

    report_txt = '\n'.join(lines)
    print('\n' + report_txt)
    out_report = RESULT_DIR / 'nir_report.txt'
    out_report.write_text(report_txt, encoding='utf-8')
    print(f'\n保存: {out_sum}')
    print(f'保存: {out_report}')


# ── マージモード ──────────────────────────────────────────────
def merge_mode():
    parts = sorted(RESULT_DIR.glob('nir_per_sample_*_*.csv'))
    if not parts:
        print('ERROR: nir_per_sample_*_*.csv が見つかりません')
        return
    print(f'マージ対象: {[p.name for p in parts]}')

    rows = []
    fieldnames = None
    for p in parts:
        with open(p, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(list(reader))

    # 数値に変換
    for r in rows:
        for k in r:
            if k not in ('speaker_id', 'sentence_id', 'reference'):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass

    out_all = RESULT_DIR / 'nir_per_sample.csv'
    with open(out_all, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'マージ完了: {len(rows)}件 → {out_all}')

    compute_summary_and_report(rows)


# ── メイン ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0,    help='開始サンプルインデックス')
    parser.add_argument('--end',   type=int, default=None, help='終了サンプルインデックス')
    parser.add_argument('--merge', action='store_true',    help='分割CSVをマージしてNIR計算')
    args = parser.parse_args()

    if args.merge:
        merge_mode()
        return

    np.random.seed(42)  # ノイズ再現性のため固定

    # サンプル収集
    all_samples = []
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
                all_samples.append({'spk': spk, 'sid': sid,
                                    'path': path, 'text': row['text']})

    end = args.end if args.end is not None else len(all_samples)
    samples = all_samples[args.start:end]
    print(f'サンプル数: {len(samples)} ({args.start}〜{end})')

    # モデルロード
    print('Whisper pretrained ロード中...')
    proc_pre, model_pre, device = load_whisper('openai/whisper-small')

    print('Whisper FT ロード中...')
    proc_ft, model_ft, _ = load_whisper(str(FT_CKPT))
    model_ft = model_ft.to(device)

    print(f'デバイス: {device}')
    print(f'評価条件: clean + {len(NOISE_TYPES)}種×{len(SNR_LEVELS)}SNR = '
          f'{1 + len(NOISE_TYPES) * len(SNR_LEVELS)}条件 × 2モデル\n')

    # 評価ループ
    rows = []
    for i, s in enumerate(samples):
        wav, _ = sf.read(str(s['path']), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        ref = s['text']
        row = {'speaker_id': s['spk'], 'sentence_id': s['sid'], 'reference': ref}

        # クリーン条件
        row['pre_clean'] = round(cer(ref, transcribe(wav, proc_pre, model_pre, device)), 4)
        row['ft_clean']  = round(cer(ref, transcribe(wav, proc_ft,  model_ft,  device)), 4)

        # ノイズ条件
        for noise_type in NOISE_TYPES:
            for snr in SNR_LEVELS:
                noisy = add_noise(wav, noise_type, snr)
                key   = f'{noise_type}_snr{snr:+d}'
                row[f'pre_{key}'] = round(cer(ref, transcribe(noisy, proc_pre, model_pre, device)), 4)
                row[f'ft_{key}']  = round(cer(ref, transcribe(noisy, proc_ft,  model_ft,  device)), 4)

        rows.append(row)

        if (i + 1) % 50 == 0 or i == 0:
            print(f'[{i+1:4d}/{len(samples)}] {s["spk"]} {s["sid"]}  '
                  f'pre_clean={row["pre_clean"]:.3f}  ft_clean={row["ft_clean"]:.3f}')

    # ── per-sample CSV（分割ファイルとして保存）──────────────────
    RESULT_DIR.mkdir(exist_ok=True)
    all_keys = list(rows[0].keys())
    out_sample = RESULT_DIR / f'nir_per_sample_{args.start}_{end}.csv'
    with open(out_sample, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f'\n保存: {out_sample}')
    print('全件完了後に --merge オプションでNIRを計算してください。')
    print('  python3 scripts/28_nir_evaluation.py --merge')


if __name__ == '__main__':
    main()
