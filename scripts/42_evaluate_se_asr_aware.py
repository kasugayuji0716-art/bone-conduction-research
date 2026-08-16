"""
スクリプト42: ASR-aware SE評価
script 41で学習したモデルのCERをTAPSベースラインと比較する

比較条件:
    A: No SE（生喉マイク音声）
    B: TAPS pretrained SE-Conformer（そのまま）
    C: SI-SDRのみ再学習（λ=0.0）
    D: ASR-aware再学習（λ=0.1）

使い方（DNN PC）:
    python scripts/42_evaluate_se_asr_aware.py
    python scripts/42_evaluate_se_asr_aware.py --n_utts 100 --balanced

出力:
    results/se_asr_aware_comparison.csv
    results/figures/se_asr_aware_comparison.png
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
# SE-Conformer（script 37/40/41と同一）
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
# モデルロード
# ══════════════════════════════════════════════════════════════

def load_se(ckpt_path: Path) -> SEConformerModel:
    model = SEConformerModel()
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    return model.eval().to(SE_DEVICE)


def load_asr() -> FasterWhisperModel:
    ct2_path = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2_path) if ct2_path.exists() else 'openai/whisper-small'
    return FasterWhisperModel(model_id, device=DEVICE,
                              compute_type='float16' if DEVICE == 'cuda' else 'int8')


def transcribe(asr, wav: np.ndarray) -> str:
    segs, _ = asr.transcribe(wav, language='ko', beam_size=5)
    return ''.join(s.text for s in segs).strip()


# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_utts',   type=int, default=None)
    parser.add_argument('--balanced', action='store_true',
                        help='話者ごとに均等サンプリング')
    args = parser.parse_args()

    # 評価するモデルの定義
    conditions = [
        ('A_no_se',      'No SE（生喉マイク）',       None),
        ('B_taps_pretrained', 'TAPS pretrained SE',  PRETRAINED_DIR / 'seconformer.th'),
        ('C_si_sdr_only',    'SI-SDRのみ再学習 (λ=0.0)', CKPT_DIR / 'se_si_sdr_only' / 'best.th'),
        ('D_asr_aware',      'ASR-aware (λ=0.1)',    CKPT_DIR / 'se_asr_aware' / 'best.th'),
    ]
    # 存在するチェックポイントのみ評価
    conditions = [(k, label, p) for k, label, p in conditions
                  if p is None or Path(p).exists()]
    print(f'評価条件: {[label for _, label, _ in conditions]}')

    # テストデータ読み込み
    meta_path = TAPS_DIR / 'metadata_test.csv'
    samples = []
    with open(meta_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            wav_path = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if wav_path.exists():
                samples.append({'path': wav_path, 'text': row['text'],
                                 'speaker': row['speaker_id']})

    if args.balanced and args.n_utts:
        by_spk = defaultdict(list)
        for s in samples:
            by_spk[s['speaker']].append(s)
        n_per = max(1, args.n_utts // len(by_spk))
        samples = [s for ss in by_spk.values() for s in ss[:n_per]]
    elif args.n_utts:
        samples = samples[:args.n_utts]
    print(f'評価発話数: {len(samples)}')

    # ASRロード
    asr = load_asr()

    # SEモデルのロード（条件ごと）
    se_models = {}
    for key, label, ckpt_path in conditions:
        if ckpt_path is not None:
            se_models[key] = load_se(ckpt_path)
            print(f'  {label}: {ckpt_path}')

    # 評価ループ
    results = {key: [] for key, _, _ in conditions}

    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(samples)} 発話処理中...')

        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        ref = s['text']

        for key, label, ckpt_path in conditions:
            if ckpt_path is None:
                audio = wav
            else:
                with torch.no_grad():
                    inp = torch.from_numpy(wav).to(SE_DEVICE)
                    audio = se_models[key](inp).cpu().numpy()

            hyp = transcribe(asr, audio)
            c = cer(ref, hyp) if ref else 0.0
            results[key].append(min(c, 1.0))

    # 集計・表示
    summary = {key: float(np.mean(v)) for key, v in results.items()}
    print('\n=== 結果 ===')
    print(f'{"条件":<30} {"CER":>8}')
    print('-' * 42)
    for key, label, _ in conditions:
        marker = ''
        if key != 'A_no_se':
            diff = summary[key] - summary['A_no_se']
            marker = f'  ({diff:+.4f} vs No SE)'
        print(f'{label:<30} {summary[key]:>8.4f}{marker}')

    # CSV保存
    RESULT_DIR.mkdir(exist_ok=True)
    out_csv = RESULT_DIR / 'se_asr_aware_comparison.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'label', 'cer_mean', 'n_utts'])
        for key, label, _ in conditions:
            w.writerow([key, label, summary[key], len(samples)])
    print(f'\nCSV保存: {out_csv}')

    # プロット
    fig_dir = RESULT_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)
    labels  = [label for _, label, _ in conditions]
    cers    = [summary[key] for key, _, _ in conditions]
    colors  = ['steelblue', 'orange', 'green', 'red'][:len(conditions)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, cers, color=colors, alpha=0.8, edgecolor='black')
    for bar, c in zip(bars, cers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{c:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('CER（低いほど良い）', fontsize=12)
    ax.set_title('SE学習損失の比較: CER on TAPS test', fontsize=13)
    ax.set_ylim(0, max(cers) * 1.2)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fig_path = fig_dir / 'se_asr_aware_comparison.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'図保存: {fig_path}')
    print('\n完了。')


if __name__ == '__main__':
    main()
