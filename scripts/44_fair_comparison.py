"""
スクリプト44: 全SE条件を同一ASRモデルで公平比較
Phase 4のTAPS SEベースラインと本研究のASR-aware SEを
同じCT2 Whisperモデルで評価し、直接比較可能にする。

条件:
    1. No SE
    2. TAPS pretrained SE-Conformer
    3. TAPS pretrained Demucs
    4. SI-SDR only 再学習 (λ=0.0)
    5. ASR-aware 再学習 (L1, λ=1.0)

出力:
    results/fair_comparison.csv

使い方（DNN PC）:
    python scripts/44_fair_comparison.py
"""

import csv
import math
import sys
from pathlib import Path

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

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SE_DEVICE = 'cpu'


# ── SE-Conformer ──────────────────────────────────────────────

class _SEConformerFFN(nn.Module):
    def __init__(self, d=512, r=64):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, r), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(r, d))
    def forward(self, x): return self.sequential(x)

class _SEConformerConvModule(nn.Module):
    def __init__(self, d=512, k=15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.sequential = nn.Sequential(
            nn.Conv1d(d, d*2, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, k, padding=k//2, groups=d),
            nn.BatchNorm1d(d), nn.SiLU(), nn.Conv1d(d, d, 1))
    def forward(self, x):
        r = x
        x = self.layer_norm(x).transpose(1, 2)
        x = self.sequential(x).transpose(1, 2)
        return x + r

class _SEConformerBlock(nn.Module):
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.ffn1 = _SEConformerFFN(d)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv_module = _SEConformerConvModule(d)
        self.ffn2 = _SEConformerFFN(d)
        self.final_layer_norm = nn.LayerNorm(d)
    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        r = x; x = self.self_attn_layer_norm(x)
        a, _ = self.self_attn(x, x, x); x = r + a
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
            ic, hc = self.CH[i], self.CH[i+1]
            enc.append(nn.Sequential(
                nn.Conv1d(ic, hc, self.K, stride=self.S), nn.ReLU(),
                nn.Conv1d(hc, hc*2, 1), nn.GLU(dim=1)))
        for i in range(3, -1, -1):
            ic, oc = self.CH[i+1], self.CH[i]
            dec.append(nn.Sequential(
                nn.Conv1d(ic, ic*2, 1), nn.GLU(dim=1), nn.ReLU(),
                nn.ConvTranspose1d(ic, max(oc,1), self.K, stride=self.S)))
        self.encoder = enc
        self.conformers = nn.ModuleList([_SEConformerBlock(self.CH[-1], heads) for _ in range(n_conf)])
        self.decoder = dec
    def valid_length(self, length):
        length = math.ceil(length * self.RESAMPLE)
        for _ in range(len(self.CH)-1):
            length = math.ceil((length - self.K) / self.S) + 1
            length = max(length, 1)
        for _ in range(len(self.CH)-1):
            length = (length - 1) * self.S + self.K
        return int(math.ceil(length / self.RESAMPLE))
    def forward(self, x):
        from models.demucs import upsample2, downsample2
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2: x = x.unsqueeze(1)
        mono = x.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        x = x / (self.FLOOR + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        x = upsample2(upsample2(x))
        skips = []
        for enc in self.encoder:
            x = enc(x); skips.append(x)
        x = x.permute(0, 2, 1)
        for conf in self.conformers: x = conf(x)
        x = x.permute(0, 2, 1)
        for dec in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = dec(x)
        x = downsample2(downsample2(x))
        x = x[..., :length]
        return (std * x).squeeze(1).squeeze(0)


# ── Demucs ──────────────────────────────────────────────────

class _DemucsLSTM(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden * 2, hidden)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        y, _ = self.lstm(x)
        return self.linear(y).permute(0, 2, 1)

class ConvDemucs(nn.Module):
    CH = [1, 64, 128, 256, 512, 1024]; K = 8; S = 4
    def __init__(self):
        super().__init__()
        enc, dec = nn.ModuleList(), nn.ModuleList()
        for i in range(5):
            ic, hc = self.CH[i], self.CH[i+1]
            enc.append(nn.Sequential(
                nn.Conv1d(ic, hc, self.K, stride=self.S), nn.ReLU(),
                nn.Conv1d(hc, hc*2, 1), nn.GLU(dim=1)))
        for i in range(4, -1, -1):
            ic, oc = self.CH[i+1], self.CH[i]
            dec.append(nn.Sequential(
                nn.Conv1d(ic, ic*2, 1), nn.GLU(dim=1),
                nn.ConvTranspose1d(ic, max(oc,1), self.K, stride=self.S),
                nn.ReLU() if i > 0 else nn.Tanh()))
        self.encoder = enc
        self.lstm = _DemucsLSTM(self.CH[-1])
        self.decoder = dec
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2: x = x.unsqueeze(1)
        T = x.shape[-1]
        h = x
        for enc in self.encoder: h = enc(h)
        h = self.lstm(h)
        for dec in self.decoder: h = dec(h)
        return h[..., :T].squeeze(1).squeeze(0)


# ── ユーティリティ ──────────────────────────────────────────

def load_se(ckpt_path, cls=SEConformerModel):
    model = cls()
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    return model.eval().to(SE_DEVICE)


def main():
    # 条件定義
    conditions = [
        ('no_se',           'No SE',                    None, None),
        ('taps_seconf',     'TAPS SE-Conformer',        PRETRAINED_DIR / 'seconformer.th', SEConformerModel),
        ('taps_demucs',     'TAPS Demucs',              PRETRAINED_DIR / 'demucs.th', ConvDemucs),
        ('si_sdr_only',     'SI-SDR only (λ=0)',        CKPT_DIR / 'se_si_sdr_only' / 'best.th', SEConformerModel),
        ('asr_aware',       'ASR-aware (L1, λ=1.0)',    CKPT_DIR / 'se_asr_aware' / 'best.th', SEConformerModel),
    ]
    # 存在するもののみ
    conditions = [(k, l, p, c) for k, l, p, c in conditions if p is None or Path(p).exists()]

    print(f'評価条件: {[l for _, l, _, _ in conditions]}')

    # テストデータ
    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'path': p, 'text': row['text']})
    print(f'評価発話数: {len(samples)}')

    # ASR
    ct2 = Path('/tmp/whisper-small-ct2')
    model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
    asr = FasterWhisperModel(model_id, device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')
    print(f'ASR: {model_id}')

    # SE models
    se_models = {}
    for key, label, ckpt, cls in conditions:
        if ckpt is not None:
            se_models[key] = load_se(ckpt, cls)
            print(f'  SE: {label}')

    # 評価
    results = {k: [] for k, _, _, _ in conditions}
    for i, s in enumerate(samples):
        if (i+1) % 100 == 0:
            print(f'  {i+1}/{len(samples)}...')
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)

        for key, _, ckpt, _ in conditions:
            if ckpt is None:
                audio = wav
            else:
                with torch.no_grad():
                    audio = se_models[key](torch.from_numpy(wav).to(SE_DEVICE)).cpu().numpy()
            segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
            hyp = ''.join(seg.text for seg in segs).strip()
            c = min(cer(s['text'], hyp), 1.0) if s['text'] else 0.0
            results[key].append(c)

    # 結果表示
    print('\n=== 公平比較結果（同一ASRモデル） ===')
    print(f'{"条件":<30} {"CER":>8} {"n":>6}')
    print('-' * 48)
    for key, label, _, _ in conditions:
        mean_cer = np.mean(results[key])
        print(f'{label:<30} {mean_cer:>8.4f} {len(results[key]):>6}')

    # CSV
    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / 'fair_comparison.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'label', 'cer_mean', 'n_utts'])
        for key, label, _, _ in conditions:
            w.writerow([key, label, np.mean(results[key]), len(results[key])])
    print(f'\nCSV: {out}')
    print('完了。')


if __name__ == '__main__':
    main()
