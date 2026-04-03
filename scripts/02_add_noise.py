"""
喉マイク音声に異なるSNRでノイズを付加する
出力: data/processed/noisy/snr_{XX}dB/

ノイズ種別:
  - white  : ホワイトノイズ（基礎実験用）
  - pink   : ピンクノイズ（工場騒音に近い特性）

使い方:
  python3 02_add_noise.py
"""
import os
import numpy as np
import soundfile as sf
import csv

# ---- 設定 ----
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
THROAT_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat')
META_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'metadata.csv')
OUT_BASE    = os.path.join(BASE_DIR, 'data', 'processed', 'noisy')

# 実験するSNRレベル（dB）
SNR_LEVELS  = [20, 10, 5, 0, -5]
NOISE_TYPES = ['white', 'pink']


def rms(signal):
    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def generate_white_noise(length):
    return np.random.randn(length).astype(np.float32)


def generate_pink_noise(length):
    """1/f ノイズ（ピンクノイズ）の近似生成"""
    white = np.random.randn(length)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1.0  # ゼロ除算回避
    power = 1.0 / np.sqrt(freqs)
    pink_fft = np.fft.rfft(white) * power
    pink = np.fft.irfft(pink_fft, n=length)
    return (pink / (np.max(np.abs(pink)) + 1e-12)).astype(np.float32)


def mix_at_snr(signal, noise, snr_db):
    """指定SNRでノイズを混合する"""
    sig_rms   = rms(signal)
    noise_rms = rms(noise)
    # 目標ノイズRMS = signal_rms / 10^(snr/20)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)
    mixed = signal + noise_scaled
    # クリッピング防止
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def main():
    # メタデータ読み込み
    with open(META_PATH, encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    np.random.seed(42)  # 再現性確保

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            out_dir = os.path.join(OUT_BASE, noise_type, f'snr_{snr_db:+d}dB')
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n[{noise_type}] SNR={snr_db:+d}dB → {out_dir}")

            for rec in records:
                sid    = rec['speaker_id']
                sentid = rec['sentence_id']
                in_path = os.path.join(THROAT_DIR, f'{sid}_{sentid}.wav')

                if not os.path.exists(in_path):
                    continue

                signal, sr = sf.read(in_path)
                signal = signal.astype(np.float32)

                # ノイズ生成（信号と同じ長さ）
                if noise_type == 'white':
                    noise = generate_white_noise(len(signal))
                else:
                    noise = generate_pink_noise(len(signal))

                mixed = mix_at_snr(signal, noise, snr_db)

                out_path = os.path.join(out_dir, f'{sid}_{sentid}.wav')
                sf.write(out_path, mixed, sr)

            print(f"  {len(records)}件を保存完了")

    # サマリー
    print("\n=== 生成完了 ===")
    print(f"ノイズ種別: {NOISE_TYPES}")
    print(f"SNRレベル:  {SNR_LEVELS} dB")
    print(f"合計条件数: {len(NOISE_TYPES) * len(SNR_LEVELS)}")
    total = len(NOISE_TYPES) * len(SNR_LEVELS) * len(records)
    print(f"生成ファイル数: {total}件")
    print(f"保存先: {OUT_BASE}")


if __name__ == '__main__':
    main()
