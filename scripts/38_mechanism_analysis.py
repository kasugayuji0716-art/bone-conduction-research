"""
スクリプト38: 機構解明分析
  実験B: SE-Conformer適用前後のスペクトル変化（帯域別エネルギー）
  実験A: FT前後のWhisperエンコーダ表現距離分析
         - pretrained Whisper: 喉マイク vs 気導マイク の encoder出力距離
         - FT済みWhisper: 喉マイクのencoder出力が気導表現に近づくか

DNN PC (dl-box3) で実行すること。
出力:
  results/figures/mechanism_B_spectrum.png   (実験B)
  results/figures/mechanism_A_encoder.png    (実験A)
  results/mechanism_analysis.csv             (数値サマリー)
"""

import csv, sys, json, io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
FIG_DIR        = BASE_DIR / 'results' / 'figures'
RESULT_DIR     = BASE_DIR / 'results'
CKPT_DIR       = BASE_DIR / 'checkpoints'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))

SR      = 16000
N_FFT   = 2048
MODEL_ID = 'openai/whisper-small'
D_MODEL  = 768
N_SAMPLES = 200  # 高速化のため200件（全発話のサブセット）

BLUE   = '#1565C0'
GREEN  = '#2E7D32'
RED    = '#C62828'
ORANGE = '#E65100'
GRAY   = '#546E7A'


# ─────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────

def load_samples(n=N_SAMPLES):
    meta = TAPS_DIR / 'metadata_test.csv'
    wdir_t = TAPS_DIR / 'throat' / 'test'
    wdir_a = TAPS_DIR / 'acoustic' / 'test'
    rows = []
    with open(meta, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sid, spk = r['sentence_id'], r['speaker_id']
            p_t = wdir_t / f'{spk}_{sid}.wav'
            p_a = wdir_a / f'{spk}_{sid}.wav'
            if p_t.exists() and p_a.exists():
                rows.append({
                    'sid': sid, 'spk': spk,
                    'throat': p_t, 'acoustic': p_a,
                    'text': r['text']
                })
            if len(rows) >= n:
                break
    print(f'サンプル数: {len(rows)}')
    return rows


# ─────────────────────────────────────────────────
# SE-Conformer
# ─────────────────────────────────────────────────

def load_seconformer(device):
    from models.seconformer import seconformer
    sec = seconformer(
        hidden=64, depth=4, conformer_dim=512, conformer_ffn_dim=64,
        conformer_num_attention_heads=4, conformer_depth=4,
        depthwise_conv_kernel_size=15, kernel_size=8, stride=4,
        resample=4, growth=2, dropout=0.1, rescale=0.1, normalize=True
    )
    ckpt = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    sec.load_state_dict(state, strict=False)
    return sec.eval().to(device)


def apply_se(model, wav, device):
    t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    rms_in  = np.sqrt(np.mean(wav ** 2) + 1e-12)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-12)
    return (out * rms_in / rms_out).astype(np.float32)


# ─────────────────────────────────────────────────
# スペクトル計算
# ─────────────────────────────────────────────────

def power_spectrum(wav, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    hop = n_fft // 2
    frames = []
    for i in range(0, len(wav) - n_fft, hop):
        frame = wav[i:i + n_fft] * np.hanning(n_fft)
        ps = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(ps)
    avg = np.mean(frames, axis=0)
    return freqs, 10 * np.log10(avg + 1e-10)


def spectrogram(wav, n_fft=512, hop=256):
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, len(wav) - n_fft, hop):
        frame = wav[i:i + n_fft] * win
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    S = np.array(frames).T
    return 10 * np.log10(S + 1e-10)


# ─────────────────────────────────────────────────
# Whisperエンコーダ特徴量抽出
# ─────────────────────────────────────────────────

class AdapterModule(nn.Module):
    def __init__(self, d_model=D_MODEL, r=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model)
    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class WhisperEncoderLayerWithAdapter(nn.Module):
    def __init__(self, original_layer, adapter):
        super().__init__()
        self.layer   = original_layer
        self.adapter = adapter
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = self.layer(hidden_states, attention_mask, **kwargs)
        return self.adapter(hidden_states)


def load_whisper_pretrained(device):
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    return model.eval().to(device)


def load_whisper_ft(device):
    ft_dir = CKPT_DIR / 'whisper_throat_finetuned'
    model = WhisperForConditionalGeneration.from_pretrained(str(ft_dir))
    return model.eval().to(device)


def get_encoder_output(model, processor, wav, device):
    """Whisperエンコーダの出力（最終層）を取得 → (T, D)"""
    feats = processor.feature_extractor(
        wav, sampling_rate=SR, return_tensors='pt'
    ).input_features.to(device)
    with torch.no_grad():
        enc = model.model.encoder(feats)
    # last_hidden_state: (1, T, D) → 時間方向に平均 → (D,)
    return enc.last_hidden_state.squeeze(0).mean(0).cpu().float().numpy()


# ─────────────────────────────────────────────────
# 実験B: SE-Conformerスペクトル分析
# ─────────────────────────────────────────────────

def experiment_B(samples, device):
    print('\n===== 実験B: SE-Conformerスペクトル分析 =====')
    sec = load_seconformer(device)

    ps_before_all, ps_after_all, ps_diff_all = [], [], []
    ref_sample = None

    for i, s in enumerate(samples):
        wav, _ = sf.read(str(s['throat']), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        try:
            wav_se = apply_se(sec, wav, device)
        except Exception as e:
            print(f'  警告 {s["sid"]}: {e}')
            wav_se = wav

        freqs, ps_b = power_spectrum(wav)
        _,     ps_a = power_spectrum(wav_se)
        ps_before_all.append(ps_b)
        ps_after_all.append(ps_a)
        ps_diff_all.append(ps_a - ps_b)

        if ref_sample is None:
            ref_sample = {'wav': wav, 'wav_se': wav_se, 'sid': s['sid']}

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(samples)}')

    del sec
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    ps_before = np.mean(ps_before_all, axis=0)
    ps_after  = np.mean(ps_after_all,  axis=0)
    ps_diff   = np.mean(ps_diff_all,   axis=0)
    freq_khz  = freqs / 1000

    # 帯域別サマリー
    bands = [(0, 0.05), (0.05, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
    band_labels = ['0–50Hz', '50–300Hz', '300Hz–1kHz', '1–2kHz', '2–4kHz', '4–8kHz']
    band_deltas = []
    print('\n帯域別エネルギー変化（SE-Conformer）:')
    for (lo, hi), label in zip(bands, band_labels):
        mask  = (freq_khz >= lo) & (freq_khz < hi)
        delta = float(np.mean(ps_diff[mask]))
        band_deltas.append(delta)
        sign  = '↑追加' if delta > 0 else '↓削除'
        print(f'  {label}: Δ={delta:+.2f} dB  {sign}')

    # 図
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # (A) Before vs After
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(freq_khz, ps_before, color=BLUE,  lw=2.0, label='Before SE (throat mic raw)')
    ax1.plot(freq_khz, ps_after,  color=GREEN, lw=2.0, label='After SE (SE-Conformer)', ls='--')
    ax1.axvspan(4.0, 8.0, alpha=0.10, color=ORANGE)
    ax1.text(5.5, ax1.get_ylim()[0] + 2, '4kHz以上不在\n(8kHz収録)', fontsize=9, color=ORANGE, ha='center')
    ax1.set_xlabel('Frequency (kHz)', fontsize=11)
    ax1.set_ylabel('Power (dB)', fontsize=11)
    ax1.set_title(f'(A)  Average Power Spectrum: Before vs After SE-Conformer  (n={len(samples)})',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 8)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.25)

    # (B) 差分スペクトル
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axhline(0, color='black', lw=0.8)
    ax2.fill_between(freq_khz, ps_diff, 0,
                     where=(ps_diff > 0), color=RED,  alpha=0.6, label='エネルギー追加（アーティファクト）')
    ax2.fill_between(freq_khz, ps_diff, 0,
                     where=(ps_diff < 0), color=BLUE, alpha=0.6, label='エネルギー削除（抑圧）')
    ax2.plot(freq_khz, ps_diff, color='black', lw=0.8, alpha=0.4)
    ax2.set_xlabel('Frequency (kHz)', fontsize=11)
    ax2.set_ylabel('ΔPower (dB)', fontsize=11)
    ax2.set_title('(B)  Spectral Difference: After − Before SE-Conformer',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 8)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.25)

    # (C) 代表スペクトログラム Before
    ax3 = fig.add_subplot(gs[2, 0])
    sg_b = spectrogram(ref_sample['wav'])
    dur  = len(ref_sample['wav']) / SR
    im3  = ax3.imshow(sg_b, origin='lower', aspect='auto',
                      extent=[0, dur, 0, SR / 2 / 1000],
                      cmap='magma', vmin=-80, vmax=20)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Frequency (kHz)', fontsize=10)
    ax3.set_title(f'(C)  Spectrogram — Before SE\n({ref_sample["sid"]})', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='dB')

    # (D) 代表スペクトログラム After
    ax4 = fig.add_subplot(gs[2, 1])
    sg_a = spectrogram(ref_sample['wav_se'])
    im4  = ax4.imshow(sg_a, origin='lower', aspect='auto',
                      extent=[0, dur, 0, SR / 2 / 1000],
                      cmap='magma', vmin=-80, vmax=20)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Frequency (kHz)', fontsize=10)
    ax4.set_title(f'(D)  Spectrogram — After SE-Conformer\n({ref_sample["sid"]})', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='dB')

    fig.suptitle(
        'Experiment B: SE-Conformer Spectral Analysis on Throat Mic Audio\n'
        'Domain-adapted SE transforms signal — does it align with Whisper\'s learned representation?',
        fontsize=13, fontweight='bold', y=1.01
    )

    out = FIG_DIR / 'mechanism_B_spectrum.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n保存: {out}')

    return freqs, ps_diff, band_labels, band_deltas


# ─────────────────────────────────────────────────
# 実験A: エンコーダ表現距離分析
# ─────────────────────────────────────────────────

def experiment_A(samples, device):
    print('\n===== 実験A: Whisperエンコーダ表現距離分析 =====')

    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    print('Pretrained Whisper ロード...')
    model_pre = load_whisper_pretrained(device)
    print('FT済みWhisper ロード...')
    model_ft  = load_whisper_ft(device)

    # 各サンプルについて3種のencoder表現を取得:
    #   e_t_pre : pretrained × 喉マイク
    #   e_a_pre : pretrained × 気導マイク
    #   e_t_ft  : FT済み × 喉マイク
    cos_pre   = []  # cos_sim(e_t_pre, e_a_pre) : pretrained での喉-気導距離
    cos_ft    = []  # cos_sim(e_t_ft,  e_a_pre) : FT後の喉が気導表現に近づくか
    l2_pre    = []  # L2距離 (pretrained)
    l2_ft     = []  # L2距離 (FT)

    print(f'{len(samples)}件処理中...')
    for i, s in enumerate(samples):
        wav_t, _ = sf.read(str(s['throat']),  dtype='float32')
        wav_a, _ = sf.read(str(s['acoustic']), dtype='float32')
        if wav_t.ndim > 1: wav_t = wav_t.mean(1)
        if wav_a.ndim > 1: wav_a = wav_a.mean(1)

        # encoder出力（平均プール済みベクトル）
        e_t_pre = get_encoder_output(model_pre, processor, wav_t, device)
        e_a_pre = get_encoder_output(model_pre, processor, wav_a, device)
        e_t_ft  = get_encoder_output(model_ft,  processor, wav_t, device)

        # コサイン類似度
        cos_pre.append(float(F.cosine_similarity(
            torch.from_numpy(e_t_pre).unsqueeze(0),
            torch.from_numpy(e_a_pre).unsqueeze(0)
        ).item()))
        cos_ft.append(float(F.cosine_similarity(
            torch.from_numpy(e_t_ft).unsqueeze(0),
            torch.from_numpy(e_a_pre).unsqueeze(0)
        ).item()))

        # L2距離
        l2_pre.append(float(np.linalg.norm(e_t_pre - e_a_pre)))
        l2_ft.append(float(np.linalg.norm(e_t_ft  - e_a_pre)))

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(samples)}'
                  f'  cos_pre={np.mean(cos_pre):.4f}'
                  f'  cos_ft={np.mean(cos_ft):.4f}')

    cos_pre = np.array(cos_pre)
    cos_ft  = np.array(cos_ft)
    l2_pre  = np.array(l2_pre)
    l2_ft   = np.array(l2_ft)

    print(f'\n結果サマリー:')
    print(f'  cos_sim(喉×pretrained, 気導×pretrained): {cos_pre.mean():.4f} ± {cos_pre.std():.4f}')
    print(f'  cos_sim(喉×FT,         気導×pretrained): {cos_ft.mean():.4f} ± {cos_ft.std():.4f}')
    print(f'  Δcos_sim (FT改善量):                     {(cos_ft - cos_pre).mean():+.4f}')
    print(f'  L2(喉×pretrained, 気導×pretrained): {l2_pre.mean():.4f} ± {l2_pre.std():.4f}')
    print(f'  L2(喉×FT,         気導×pretrained): {l2_ft.mean():.4f} ± {l2_ft.std():.4f}')
    print(f'  L2削減率: {(1 - l2_ft.mean() / l2_pre.mean()) * 100:.1f}%')

    # 図
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Experiment A: Whisper Encoder Representation Distance Analysis\n'
        'Does fine-tuning move throat mic representations closer to acoustic mic?',
        fontsize=13, fontweight='bold'
    )

    # (A-1) コサイン類似度の分布比較
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(cos_pre, bins=bins, color=BLUE,  alpha=0.65, label=f'Pretrained  μ={cos_pre.mean():.3f}')
    ax.hist(cos_ft,  bins=bins, color=GREEN, alpha=0.65, label=f'FT          μ={cos_ft.mean():.3f}')
    ax.axvline(cos_pre.mean(), color=BLUE,  ls='--', lw=2)
    ax.axvline(cos_ft.mean(),  color=GREEN, ls='--', lw=2)
    ax.set_xlabel('Cosine Similarity  (喉マイク encoder出力 vs 気導マイク encoder出力)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('(A-1)  Cosine Similarity Distribution\n(higher = throat representation aligns with acoustic)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # (A-2) L2距離の分布比較
    ax = axes[1]
    max_l2 = max(l2_pre.max(), l2_ft.max())
    bins2  = np.linspace(0, max_l2, 30)
    ax.hist(l2_pre, bins=bins2, color=BLUE,  alpha=0.65, label=f'Pretrained  μ={l2_pre.mean():.2f}')
    ax.hist(l2_ft,  bins=bins2, color=GREEN, alpha=0.65, label=f'FT          μ={l2_ft.mean():.2f}')
    ax.axvline(l2_pre.mean(), color=BLUE,  ls='--', lw=2)
    ax.axvline(l2_ft.mean(),  color=GREEN, ls='--', lw=2)
    ax.set_xlabel('L2 Distance  (喉マイク encoder出力 vs 気導マイク encoder出力)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'(A-2)  L2 Distance Distribution\n(L2削減率 {(1 - l2_ft.mean() / l2_pre.mean()) * 100:.1f}%  after FT)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out = FIG_DIR / 'mechanism_A_encoder.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'保存: {out}')

    return {
        'cos_pre_mean': float(cos_pre.mean()),
        'cos_pre_std':  float(cos_pre.std()),
        'cos_ft_mean':  float(cos_ft.mean()),
        'cos_ft_std':   float(cos_ft.std()),
        'l2_pre_mean':  float(l2_pre.mean()),
        'l2_pre_std':   float(l2_pre.std()),
        'l2_ft_mean':   float(l2_ft.mean()),
        'l2_ft_std':    float(l2_ft.std()),
        'l2_reduction_pct': float((1 - l2_ft.mean() / l2_pre.mean()) * 100),
    }


# ─────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'デバイス: {device}')

    samples = load_samples(N_SAMPLES)

    # 実験B
    freqs, ps_diff, band_labels, band_deltas = experiment_B(samples, device)

    # 実験A
    enc_stats = experiment_A(samples, device)

    # CSVサマリー保存
    rows = []
    for label, delta in zip(band_labels, band_deltas):
        rows.append({'experiment': 'B_spectrum', 'key': label, 'value': f'{delta:+.4f}'})
    for k, v in enc_stats.items():
        rows.append({'experiment': 'A_encoder', 'key': k, 'value': f'{v:.4f}'})

    out_csv = RESULT_DIR / 'mechanism_analysis.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'key', 'value'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n数値サマリー保存: {out_csv}')

    print('\n===== 完了 =====')
    print('  results/figures/mechanism_B_spectrum.png')
    print('  results/figures/mechanism_A_encoder.png')
    print('  results/mechanism_analysis.csv')


if __name__ == '__main__':
    main()
