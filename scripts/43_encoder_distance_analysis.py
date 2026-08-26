"""
スクリプト43: Whisper encoder距離 vs CER の相関分析
複数SE条件でencoder距離とCERを計測し、相関を可視化する。

目的: 「encoder距離がCERの予測指標として有効」であることを示す
      → 従来指標（STOI・PESQ）がCERと相関しないのと対比

条件:
    1. No SE（生喉マイク）
    2. TAPS pretrained SE-Conformer
    3. SI-SDRのみ再学習（λ=0.0）
    4. ASR-aware（λ=1.0）

出力:
    results/encoder_distance_analysis.csv
    results/figures/encoder_distance_vs_cer.png

使い方（DNN PC）:
    python scripts/43_encoder_distance_analysis.py --n_utts 100 --balanced
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel
from transformers import WhisperModel, WhisperFeatureExtractor

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
CKPT_DIR       = BASE_DIR / 'checkpoints'
RESULT_DIR     = BASE_DIR / 'results'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TARGET_SR = 16000
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SE_DEVICE = 'cpu'


# ══════════════════════════════════════════════════════════════
# SE-Conformer
# ══════════════════════════════════════════════════════════════

class _SEConformerFFN(nn.Module):
    def __init__(self, d=512, r=64):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, r), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(r, d),
        )
    def forward(self, x):
        return self.sequential(x)


class _SEConformerConvModule(nn.Module):
    def __init__(self, d=512, k=15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.sequential = nn.Sequential(
            nn.Conv1d(d, d * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, k, padding=k // 2, groups=d),
            nn.BatchNorm1d(d), nn.SiLU(), nn.Conv1d(d, d, 1),
        )
    def forward(self, x):
        r = x
        x = self.layer_norm(x).transpose(1, 2)
        x = self.sequential(x).transpose(1, 2)
        return x + r


class _SEConformerBlock(nn.Module):
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.ffn1                 = _SEConformerFFN(d)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.self_attn            = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv_module          = _SEConformerConvModule(d)
        self.ffn2                 = _SEConformerFFN(d)
        self.final_layer_norm     = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        r = x
        x = self.self_attn_layer_norm(x)
        a, _ = self.self_attn(x, x, x)
        x = r + a
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)


class SEConformerModel(nn.Module):
    CH = [1, 64, 128, 256, 512]; K = 8; S = 4
    RESAMPLE = 4; FLOOR = 1e-3

    def __init__(self, heads=4, n_conf=4):
        super().__init__()
        enc, dec = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            ic, hc = self.CH[i], self.CH[i + 1]
            enc.append(nn.Sequential(
                nn.Conv1d(ic, hc, self.K, stride=self.S), nn.ReLU(),
                nn.Conv1d(hc, hc * 2, 1), nn.GLU(dim=1),
            ))
        for i in range(3, -1, -1):
            ic, oc = self.CH[i + 1], self.CH[i]
            dec.append(nn.Sequential(
                nn.Conv1d(ic, ic * 2, 1), nn.GLU(dim=1), nn.ReLU(),
                nn.ConvTranspose1d(ic, max(oc, 1), self.K, stride=self.S),
            ))
        self.encoder    = enc
        self.conformers = nn.ModuleList([_SEConformerBlock(self.CH[-1], heads) for _ in range(n_conf)])
        self.decoder    = dec

    def valid_length(self, length: int) -> int:
        length = math.ceil(length * self.RESAMPLE)
        for _ in range(len(self.CH) - 1):
            length = math.ceil((length - self.K) / self.S) + 1
            length = max(length, 1)
        for _ in range(len(self.CH) - 1):
            length = (length - 1) * self.S + self.K
        return int(math.ceil(length / self.RESAMPLE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from models.demucs import upsample2, downsample2
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2: x = x.unsqueeze(1)
        mono   = x.mean(dim=1, keepdim=True)
        std    = mono.std(dim=-1, keepdim=True)
        x      = x / (self.FLOOR + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        x = upsample2(upsample2(x))
        skips = []
        for enc in self.encoder:
            x = enc(x); skips.append(x)
        x = x.permute(0, 2, 1)
        for conf in self.conformers:
            x = conf(x)
        x = x.permute(0, 2, 1)
        for dec in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = dec(x)
        x = downsample2(downsample2(x))
        x = x[..., :length]
        out = std * x
        return out.squeeze(1).squeeze(0)


# ══════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════

def load_se(ckpt_path):
    model = SEConformerModel()
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    return model.eval().to(SE_DEVICE)


def load_asr():
    ct2_path = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2_path) if ct2_path.exists() else 'openai/whisper-small'
    return FasterWhisperModel(model_id, device=DEVICE,
                              compute_type='float16' if DEVICE == 'cuda' else 'int8')


def transcribe(asr, wav):
    segs, _ = asr.transcribe(wav, language='ko', beam_size=5)
    return ''.join(s.text for s in segs).strip()


def get_encoder_features(wav_np, feat_extractor, whisper_enc, device):
    """音声波形 → Whisper encoder特徴量（mean pooling）"""
    inputs = feat_extractor(wav_np, sampling_rate=16000, return_tensors='pt', padding=True)
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        outputs = whisper_enc(input_features)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (D,)


# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_utts',   type=int, default=None)
    parser.add_argument('--balanced', action='store_true')
    args = parser.parse_args()

    # SE条件の定義
    conditions = [
        ('no_se',           'No SE',              None),
        ('taps_pretrained', 'TAPS pretrained',    PRETRAINED_DIR / 'seconformer.th'),
        ('si_sdr_only',     'SI-SDR only (λ=0)',  CKPT_DIR / 'se_si_sdr_only' / 'best.th'),
        ('asr_aware',       'ASR-aware (λ=1.0)',  CKPT_DIR / 'se_asr_aware' / 'best.th'),
    ]
    conditions = [(k, l, p) for k, l, p in conditions if p is None or Path(p).exists()]
    print(f'条件数: {len(conditions)}')

    # テストデータ
    meta_path = TAPS_DIR / 'metadata_test.csv'
    samples = []
    with open(meta_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t_path = TAPS_DIR / 'throat'   / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            a_path = TAPS_DIR / 'acoustic' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if t_path.exists() and a_path.exists():
                samples.append({'throat': t_path, 'acoustic': a_path,
                                'text': row['text'], 'speaker': row['speaker_id']})

    if args.balanced and args.n_utts:
        by_spk = defaultdict(list)
        for s in samples:
            by_spk[s['speaker']].append(s)
        n_per = max(1, args.n_utts // len(by_spk))
        samples = [s for ss in by_spk.values() for s in ss[:n_per]]
    elif args.n_utts:
        samples = samples[:args.n_utts]
    print(f'評価発話数: {len(samples)}')

    # モデルロード
    print('\nモデルロード中...')
    asr = load_asr()
    se_models = {}
    for key, label, ckpt in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt)
            print(f'  SE: {label}')

    # Whisper encoder（特徴量抽出用）
    whisper = WhisperModel.from_pretrained('openai/whisper-small')
    whisper_enc = whisper.encoder.to(DEVICE).eval()
    feat_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
    print('  Whisper encoder: loaded')

    # 評価ループ: 各条件×各発話で (encoder距離, CER) を計測
    # encoder距離 = SE出力と気導音声の encoder cosine距離
    per_utt_results = []

    for i, s in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(samples)}...')

        t_wav, _ = sf.read(s['throat'],   dtype='float32')
        a_wav, _ = sf.read(s['acoustic'], dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)

        # 気導音声のencoder特徴量（全条件共通の参照）
        feat_air = get_encoder_features(a_wav, feat_extractor, whisper_enc, DEVICE)

        for key, label, ckpt in conditions:
            # SE適用
            if ckpt is None:
                audio = t_wav
            else:
                with torch.no_grad():
                    inp = torch.from_numpy(t_wav).to(SE_DEVICE)
                    audio = se_models[key](inp).cpu().numpy()

            # CER
            hyp = transcribe(asr, audio)
            c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0

            # encoder距離（cosine距離 = 1 - cosine_similarity）
            feat_se = get_encoder_features(audio, feat_extractor, whisper_enc, DEVICE)
            cos_sim = F.cosine_similarity(feat_se.unsqueeze(0), feat_air.unsqueeze(0)).item()
            enc_dist = 1.0 - cos_sim

            per_utt_results.append({
                'condition': key, 'label': label,
                'speaker': s['speaker'], 'cer': c,
                'enc_dist': enc_dist, 'cos_sim': cos_sim,
            })

    # CSV保存
    RESULT_DIR.mkdir(exist_ok=True)
    out_csv = RESULT_DIR / 'encoder_distance_analysis.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'label', 'speaker', 'cer', 'enc_dist', 'cos_sim'])
        w.writeheader()
        w.writerows(per_utt_results)
    print(f'\nCSV保存: {out_csv}')

    # 条件別の集計
    print('\n=== 条件別集計 ===')
    print(f'{"条件":<25} {"CER":>8} {"enc_dist":>10} {"cos_sim":>10}')
    print('-' * 58)
    for key, label, _ in conditions:
        rows = [r for r in per_utt_results if r['condition'] == key]
        mean_cer  = np.mean([r['cer'] for r in rows])
        mean_dist = np.mean([r['enc_dist'] for r in rows])
        mean_cos  = np.mean([r['cos_sim'] for r in rows])
        print(f'{label:<25} {mean_cer:>8.4f} {mean_dist:>10.6f} {mean_cos:>10.6f}')

    # プロット: 散布図（全発話）+ 条件別平均
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 発話レベルの散布図
    ax = axes[0]
    colors_map = {'no_se': 'steelblue', 'taps_pretrained': 'orange',
                  'si_sdr_only': 'green', 'asr_aware': 'red'}
    for key, label, _ in conditions:
        rows = [r for r in per_utt_results if r['condition'] == key]
        xs = [r['enc_dist'] for r in rows]
        ys = [r['cer'] for r in rows]
        ax.scatter(xs, ys, alpha=0.4, s=15, c=colors_map.get(key, 'gray'), label=label)

    # 全体の相関係数
    all_dists = [r['enc_dist'] for r in per_utt_results]
    all_cers  = [r['cer'] for r in per_utt_results]
    corr = np.corrcoef(all_dists, all_cers)[0, 1]
    ax.set_xlabel('Encoder distance (1 - cosine sim)', fontsize=11)
    ax.set_ylabel('CER', fontsize=11)
    ax.set_title(f'Per-utterance: encoder dist vs CER (r={corr:.3f})', fontsize=12)
    ax.legend(fontsize=9)

    # (b) 条件別平均の散布図
    ax = axes[1]
    for key, label, _ in conditions:
        rows = [r for r in per_utt_results if r['condition'] == key]
        mean_dist = np.mean([r['enc_dist'] for r in rows])
        mean_cer  = np.mean([r['cer'] for r in rows])
        ax.scatter(mean_dist, mean_cer, s=100, c=colors_map.get(key, 'gray'),
                   edgecolor='black', zorder=5)
        ax.annotate(label, (mean_dist, mean_cer), textcoords='offset points',
                    xytext=(8, 5), fontsize=9)

    cond_dists = []
    cond_cers  = []
    for key, _, _ in conditions:
        rows = [r for r in per_utt_results if r['condition'] == key]
        cond_dists.append(np.mean([r['enc_dist'] for r in rows]))
        cond_cers.append(np.mean([r['cer'] for r in rows]))
    if len(cond_dists) >= 3:
        corr_cond = np.corrcoef(cond_dists, cond_cers)[0, 1]
    else:
        corr_cond = float('nan')
    ax.set_xlabel('Mean encoder distance', fontsize=11)
    ax.set_ylabel('Mean CER', fontsize=11)
    ax.set_title(f'Per-condition mean (r={corr_cond:.3f})', fontsize=12)

    plt.tight_layout()
    fig_path = RESULT_DIR / 'figures' / 'encoder_distance_vs_cer.png'
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'図保存: {fig_path}')
    print(f'\n全体相関係数: r = {corr:.4f}')
    print('完了。')


if __name__ == '__main__':
    main()
