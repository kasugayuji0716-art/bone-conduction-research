"""
各条件の音声にSEモデルを適用する

対応モデル:
  - dsp_only  : ハイパスフィルタ + プリエンファシス（DSPのみ）
  - gtcrn     : GTCRN（超軽量ニューラルSE）

使い方:
  python3 03_apply_se.py --model dsp_only
  python3 03_apply_se.py --model gtcrn
"""
import os, sys, argparse
import numpy as np
import soundfile as sf
import scipy.signal as sp_signal
import torch

# ---- パス設定 ----
BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
NOISY_BASE = os.path.join(BASE_DIR, 'data', 'processed', 'noisy')
CLEAN_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'taps', 'throat')  # cleanor baseline
SE_BASE    = os.path.join(BASE_DIR, 'data', 'processed', 'se')
GTCRN_DIR  = os.path.join(BASE_DIR, 'gtcrn')


# ============================================================
# DSP処理
# ============================================================
def apply_dsp(signal, sr):
    signal = signal.astype(np.float32)
    # 1. ハイパスフィルタ（300Hz）
    b, a = sp_signal.butter(6, 300, btype='high', fs=sr)
    signal = sp_signal.filtfilt(b, a, signal)
    # 2. プリエンファシス
    pre_emphasis = 0.97
    signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # 3. RMS正規化
    rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    signal = signal * (0.1 / rms)
    signal = np.clip(signal, -1.0, 1.0)
    return signal.astype(np.float32)


# ============================================================
# GTCRNモデル
# ============================================================
def load_gtcrn():
    sys.path.insert(0, GTCRN_DIR)
    from gtcrn import GTCRN
    model = GTCRN().eval()
    ckpt_path = os.path.join(GTCRN_DIR, 'checkpoints', 'model_trained_on_dns3.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    return model


def apply_gtcrn(model, signal, sr):
    assert sr == 16000, f"GTCRNは16kHz必須です（入力: {sr}Hz）"
    signal = signal.astype(np.float32)
    x = torch.from_numpy(signal)
    win = torch.hann_window(512).pow(0.5)
    # 新PyTorch API: complex出力 → real/imag分離してモデルへ
    stft_complex = torch.stft(x, 512, 256, 512, win, return_complex=True)
    stft_in = torch.view_as_real(stft_complex)  # (freq, time, 2)
    with torch.no_grad():
        stft_out = model(stft_in[None])[0]      # (freq, time, 2)
    stft_out_complex = torch.view_as_complex(stft_out.contiguous())
    enhanced = torch.istft(stft_out_complex, 512, 256, 512, win)
    return enhanced.detach().cpu().numpy().astype(np.float32)


# ============================================================
# 処理実行
# ============================================================
def process_dir(in_dir, out_dir, model_fn):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.wav'))
    for fname in files:
        in_path  = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        signal, sr = sf.read(in_path, dtype='float32')
        enhanced = model_fn(signal, sr)
        sf.write(out_path, enhanced, sr)
    return len(files)


def main(model_name):
    print(f"モデル: {model_name}")

    if model_name == 'dsp_only':
        model_fn = apply_dsp
    elif model_name == 'gtcrn':
        print("GTCRNモデル読み込み中...")
        gtcrn_model = load_gtcrn()
        model_fn = lambda sig, sr: apply_gtcrn(gtcrn_model, sig, sr)
    else:
        raise ValueError(f"未知のモデル: {model_name}")

    total = 0

    # 1. クリーン音声（ノイズなし）への適用
    out_dir = os.path.join(SE_BASE, model_name, 'clean')
    n = process_dir(CLEAN_DIR, out_dir, model_fn)
    total += n
    print(f"  [clean] {n}件完了")

    # 2. ノイズ付き音声への適用
    for noise_type in sorted(os.listdir(NOISY_BASE)):
        noise_base = os.path.join(NOISY_BASE, noise_type)
        if not os.path.isdir(noise_base):
            continue
        for snr_cond in sorted(os.listdir(noise_base)):
            in_dir  = os.path.join(noise_base, snr_cond)
            out_dir = os.path.join(SE_BASE, model_name, noise_type, snr_cond)
            if not os.path.isdir(in_dir):
                continue
            n = process_dir(in_dir, out_dir, model_fn)
            total += n
            print(f"  [{noise_type}/{snr_cond}] {n}件完了")

    print(f"\n合計: {total}件処理完了")
    print(f"保存先: {SE_BASE}/{model_name}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dsp_only',
                        choices=['dsp_only', 'gtcrn'])
    args = parser.parse_args()
    main(args.model)
