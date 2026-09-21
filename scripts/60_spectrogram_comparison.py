"""
スクリプト60: スペクトログラム比較可視化
No SE / TAPS pretrained / CE (λ=10.0) の出力スペクトログラムを比較。
気導マイク音声も参照として表示。

使い方（DNN PC）:
    python scripts/60_spectrogram_comparison.py
    # 特定の話者・発話を指定
    python scripts/60_spectrogram_comparison.py --speaker p00 --utterance 5
"""

import argparse
import csv, sys, torch, numpy as np, soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)

SR = 16000


def load_se(ckpt_path):
    from models.seconformer import seconformer
    model = seconformer(**TAPS_SE_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def plot_spectrogram(ax, wav, title, sr=SR, vmin=-80, vmax=0):
    """Plot log-power spectrogram."""
    n_fft = 512
    hop = 160
    S = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(
            np.pad(wav, (n_fft // 2, n_fft // 2)),
            n_fft
        ) * np.hanning(n_fft),
        axis=-1
    )) ** 2
    S_db = 10 * np.log10(np.maximum(S.T, 1e-10))
    # Use hop-based time axis
    times = np.arange(S_db.shape[1]) * hop / sr
    freqs = np.arange(S_db.shape[0]) * sr / n_fft
    ax.pcolormesh(times, freqs / 1000, S_db, vmin=vmin, vmax=vmax,
                  cmap='magma', shading='auto')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_ylim(0, 8)


def plot_difference(ax, wav1, wav2, title, sr=SR):
    """Plot spectral difference (wav2 - wav1) in dB."""
    n_fft = 512
    hop = 160
    def spec(w):
        S = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(w, (n_fft // 2, n_fft // 2)),
                n_fft
            ) * np.hanning(n_fft),
            axis=-1
        )) ** 2
        return 10 * np.log10(np.maximum(S.T, 1e-10))
    S1 = spec(wav1)
    S2 = spec(wav2)
    min_t = min(S1.shape[1], S2.shape[1])
    diff = S2[:, :min_t] - S1[:, :min_t]
    times = np.arange(min_t) * hop / sr
    freqs = np.arange(diff.shape[0]) * sr / n_fft
    im = ax.pcolormesh(times, freqs / 1000, diff, vmin=-20, vmax=20,
                       cmap='RdBu_r', shading='auto')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_ylim(0, 8)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker', type=str, default='p00')
    parser.add_argument('--utterance', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=3)
    args = parser.parse_args()

    # Load SE models
    conditions = [
        ('TAPS pretrained', BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'),
        ('CE (λ=10.0)', BASE_DIR / 'checkpoints' / 'ce_v2_lambda_10.0' / 'best.th'),
    ]
    se_models = {}
    for label, ckpt in conditions:
        if ckpt.exists():
            se_models[label] = load_se(ckpt)
            print(f'  {label}: loaded')

    # Find utterances
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid, uid = row['speaker_id'], row['sentence_id']
            t = TAPS_DIR / 'throat' / 'test' / f'{sid}_{uid}.wav'
            a = TAPS_DIR / 'acoustic' / 'test' / f'{sid}_{uid}.wav'
            if t.exists() and a.exists():
                samples.append({
                    'throat': t, 'acoustic': a,
                    'speaker': sid, 'uid': uid, 'text': row['text']
                })

    # Select examples
    speaker_samples = [s for s in samples if s['speaker'] == args.speaker]
    if not speaker_samples:
        print(f'Speaker {args.speaker} not found. Available: {sorted(set(s["speaker"] for s in samples))}')
        return

    selected = speaker_samples[args.utterance:args.utterance + args.n_examples]
    if not selected:
        selected = speaker_samples[:args.n_examples]

    out_dir = BASE_DIR / 'results' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, s in enumerate(selected):
        print(f'\n--- {s["speaker"]}_{s["uid"]} ---')
        print(f'  Text: {s["text"][:50]}...')

        t_wav, _ = sf.read(s['throat'], dtype='float32')
        a_wav, _ = sf.read(s['acoustic'], dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)

        # SE outputs
        se_wavs = {}
        for label, _ in conditions:
            if label in se_models:
                with torch.no_grad():
                    se_wavs[label] = se_models[label](
                        torch.from_numpy(t_wav).unsqueeze(0)
                    ).squeeze().numpy()

        # === Figure 1: 4-panel spectrogram comparison ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        plot_spectrogram(axes[0, 0], a_wav, 'Acoustic (reference)')
        plot_spectrogram(axes[0, 1], t_wav, 'Throat (No SE)')
        if 'TAPS pretrained' in se_wavs:
            plot_spectrogram(axes[1, 0], se_wavs['TAPS pretrained'], 'TAPS pretrained SE')
        if 'CE (λ=10.0)' in se_wavs:
            plot_spectrogram(axes[1, 1], se_wavs['CE (λ=10.0)'], 'CE-aware SE (λ=10.0)')

        for ax in axes.flat:
            ax.set_xlabel('Time (s)')

        fig.suptitle(f'{s["speaker"]}_{s["uid"]}', fontsize=13, fontweight='bold')
        fig.tight_layout()
        out_path = out_dir / f'spectrogram_{s["speaker"]}_{s["uid"]}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_path}')

        # === Figure 2: Difference spectrograms ===
        if 'TAPS pretrained' in se_wavs and 'CE (λ=10.0)' in se_wavs:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            plot_difference(axes[0], t_wav, se_wavs['TAPS pretrained'],
                          'TAPS - NoSE (dB)')
            plot_difference(axes[1], t_wav, se_wavs['CE (λ=10.0)'],
                          'CE - NoSE (dB)')
            im = plot_difference(axes[2], se_wavs['TAPS pretrained'], se_wavs['CE (λ=10.0)'],
                                'CE - TAPS (dB)')

            for ax in axes:
                ax.set_xlabel('Time (s)')

            fig.colorbar(im, ax=axes, label='dB difference', shrink=0.8)
            fig.suptitle(f'Spectral Differences: {s["speaker"]}_{s["uid"]}',
                        fontsize=13, fontweight='bold')
            fig.tight_layout()
            out_path = out_dir / f'specdiff_{s["speaker"]}_{s["uid"]}.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out_path}')

        # === Figure 3: Band energy comparison ===
        if 'TAPS pretrained' in se_wavs and 'CE (λ=10.0)' in se_wavs:
            n_fft = 512
            def band_energy(wav):
                S = np.abs(np.fft.rfft(
                    np.lib.stride_tricks.sliding_window_view(
                        np.pad(wav, (n_fft // 2, n_fft // 2)), n_fft
                    ) * np.hanning(n_fft), axis=-1
                )) ** 2
                freqs = np.arange(S.shape[1]) * SR / n_fft
                return freqs, S.mean(axis=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            for label, wav, color, ls in [
                ('Acoustic', a_wav, 'green', '-'),
                ('Throat (No SE)', t_wav, 'gray', '--'),
                ('TAPS pretrained', se_wavs['TAPS pretrained'], 'blue', '-'),
                ('CE (λ=10.0)', se_wavs['CE (λ=10.0)'], 'red', '-'),
            ]:
                freqs, energy = band_energy(wav)
                ax.plot(freqs / 1000, 10 * np.log10(np.maximum(energy, 1e-10)),
                       label=label, color=color, linestyle=ls, linewidth=1.5)

            ax.set_xlabel('Frequency (kHz)', fontsize=12)
            ax.set_ylabel('Power (dB)', fontsize=12)
            ax.set_xlim(0, 8)
            ax.legend(fontsize=10)
            ax.set_title(f'Average Spectral Envelope: {s["speaker"]}_{s["uid"]}',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = out_dir / f'spectral_envelope_{s["speaker"]}_{s["uid"]}.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out_path}')

    print(f'\nAll figures saved to {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
