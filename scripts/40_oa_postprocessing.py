"""
スクリプト40: OA（Observation Adding）後処理の検証
落合ら (TASLP 2024) の改善提案を喉マイクドメインに適用

OA後処理:
    s_OA = (1 - ω) * ŝ + ω * y
    ŝ: SE出力（SE-Conformer適用後）
    y: 元の喉マイク音声（観測信号）
    ω ∈ [0.0, 0.1, ..., 1.0]

仮説: 気導マイクではOAがWER改善（落合ら実証済み）
      喉マイクではyがOOD信号のため、OAが逆効果（CER悪化）になる

使い方（DNN PC）:
    python scripts/40_oa_postprocessing.py
    python scripts/40_oa_postprocessing.py --se_model demucs
    python scripts/40_oa_postprocessing.py --n_utts 100  # 高速確認用

入力:  data/raw/taps/throat/test/
出力:  results/oa_postprocessing.csv
       results/figures/oa_postprocessing.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel

# taps-baselines のモデルコードをパスに追加
BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
RESULT_DIR     = BASE_DIR / 'results'
CKPT_DIR       = BASE_DIR / 'checkpoints'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TARGET_SR = 16000
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
OA_OMEGAS = [round(w * 0.1, 1) for w in range(11)]  # 0.0 ~ 1.0


# ── SE-Conformer アーキテクチャ（script37より流用） ──────────────────

class _ConformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, ff_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model*ff_mult),
                                   nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model*ff_mult, d_model), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_a = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(nn.LayerNorm(d_model),
                                   nn.Conv1d(d_model, d_model*2, 1), nn.GLU(dim=1),
                                   nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel//2, groups=d_model),
                                   nn.BatchNorm1d(d_model), nn.SiLU(),
                                   nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ff2  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model*ff_mult),
                                   nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model*ff_mult, d_model), nn.Dropout(dropout))
        self.ln   = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        r, _ = self.attn(*(self.ln_a(x),)*3)
        x = x + r
        c = x.transpose(1, 2)
        c = self.conv[0](x)
        c = self.conv[1](c.transpose(1, 2)).transpose(1, 2)
        c = self.conv[2](c); c = self.conv[3](c); c = self.conv[4](c)
        c = self.conv[5](c); c = self.conv[6](c)
        x = x + c.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class SEConformer(nn.Module):
    D = 256; K = 8; S = 4; N_CONF = 4
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, self.D, self.K, stride=self.S), nn.PReLU())
        self.conformers = nn.Sequential(*[_ConformerBlock(self.D) for _ in range(self.N_CONF)])
        self.decoder = nn.Sequential(nn.ConvTranspose1d(self.D, 1, self.K, stride=self.S))

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2: x = x.unsqueeze(1)
        skip = self.encoder(x)
        h = self.conformers(skip.transpose(1, 2)).transpose(1, 2)
        out = self.decoder(h + skip)
        return out.squeeze(1).squeeze(0)


# ── Demucs アーキテクチャ（script37より流用） ────────────────────────

class _DemucsLSTM(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.lstm   = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
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
            enc.append(nn.Sequential(nn.Conv1d(ic, hc, self.K, stride=self.S), nn.ReLU(),
                                      nn.Conv1d(hc, hc*2, 1), nn.GLU(dim=1)))
            dec.append(nn.Sequential(nn.Conv1d(self.CH[i+1], self.CH[i+1]*2, 1), nn.GLU(dim=1),
                                      nn.ConvTranspose1d(self.CH[i+1], self.CH[i], self.K, stride=self.S)))
        self.encoder = enc
        self.lstm    = _DemucsLSTM(self.CH[-1])
        self.decoder = dec

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2: x = x.unsqueeze(1)
        skips = []
        for enc in self.encoder:
            x = enc(x); skips.append(x)
        x = self.lstm(x)
        for dec, s in zip(reversed(self.decoder), reversed(skips)):
            x = dec(x + s[:, :, :x.shape[2]])
        return x.squeeze(1).squeeze(0)


# ── SE モデルのロード ─────────────────────────────────────────────────

def load_se_model(name: str):
    name_map = {'seconformer': ('seconformer.th', SEConformer),
                'demucs':      ('demucs.th',      ConvDemucs)}
    fname, cls = name_map[name]
    ckpt_path = PRETRAINED_DIR / fname
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SE model not found: {ckpt_path}\n"
                                f"Run script 37 first to download pretrained models.")
    model = cls()
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    print(f"Loaded SE model: {fname}")
    return model


# ── ASR（faster-whisper） ─────────────────────────────────────────────

def load_asr():
    fw = FasterWhisperModel('openai/whisper-small', device=DEVICE,
                             compute_type='float16' if DEVICE == 'cuda' else 'int8')
    print("Loaded ASR: faster-whisper small")
    return fw


def transcribe(asr_model, wav: np.ndarray) -> str:
    segs, _ = asr_model.transcribe(wav, language='ko', beam_size=5)
    return ''.join(s.text for s in segs).strip()


# ── OA 後処理 ─────────────────────────────────────────────────────────

def apply_oa(se_out: np.ndarray, original: np.ndarray, omega: float) -> np.ndarray:
    """s_OA = (1 - omega) * se_out + omega * original"""
    min_len = min(len(se_out), len(original))
    return (1 - omega) * se_out[:min_len] + omega * original[:min_len]


# ── メイン ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--se_model', default='seconformer',
                        choices=['seconformer', 'demucs'],
                        help='SEモデル名 (default: seconformer)')
    parser.add_argument('--n_utts', type=int, default=None,
                        help='評価発話数（Noneで全件）。高速確認には100を指定')
    args = parser.parse_args()

    print(f"\n=== OA後処理検証 | SEモデル: {args.se_model} | デバイス: {DEVICE} ===\n")

    # モデルロード
    se_model = load_se_model(args.se_model)
    asr      = load_asr()

    # テストデータ読み込み
    meta_path = TAPS_DIR / 'metadata_test.csv'
    samples = []
    with open(meta_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            wav_path = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if wav_path.exists():
                samples.append({'path': wav_path, 'text': row['text']})

    if args.n_utts:
        samples = samples[:args.n_utts]
    print(f"評価発話数: {len(samples)}")

    # ── 評価ループ ──
    # 結果格納: {omega: [cer_values]}
    results = {w: [] for w in OA_OMEGAS}

    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} 発話処理中...")

        wav, sr = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # SE 適用
        with torch.no_grad():
            inp = torch.from_numpy(wav).to(DEVICE)
            se_out = se_model(inp).cpu().numpy()

        ref_text = s['text']

        # 各 omega で OA 適用 → ASR → CER 計算
        for omega in OA_OMEGAS:
            audio_oa = apply_oa(se_out, wav, omega)
            hyp = transcribe(asr, audio_oa)
            c   = cer(ref_text, hyp) if ref_text else 0.0
            results[omega].append(min(c, 1.0))  # cap @ 1.0

    # ── 集計 ──
    summary = {w: float(np.mean(v)) for w, v in results.items()}

    print("\n=== 結果 ===")
    print(f"{'ω_OA':>6}  {'CER':>8}  {'備考'}")
    print("-" * 35)
    for w, c in summary.items():
        note = ''
        if w == 0.0:
            note = '← SE出力のみ（OAなし）'
        elif w == 1.0:
            note = '← 喉マイク生音声のみ'
        print(f"{w:>6.1f}  {c:>8.4f}  {note}")

    baseline_se  = summary[0.0]
    baseline_raw = summary[1.0]
    best_omega   = min(summary, key=summary.get)
    best_cer     = summary[best_omega]

    print(f"\nSE出力のみ (ω=0.0):    CER = {baseline_se:.4f}")
    print(f"喉マイク生音声 (ω=1.0): CER = {baseline_raw:.4f}")
    print(f"最良ω:                  ω = {best_omega:.1f}  (CER = {best_cer:.4f})")
    if best_omega == 0.0:
        print("→ OAは効果なし（SEそのままが最良）")
    elif best_cer < baseline_se:
        print(f"→ OAが改善: CER {baseline_se:.4f} → {best_cer:.4f}")
    else:
        print("→ OAは逆効果")

    # ── CSV 保存 ──
    RESULT_DIR.mkdir(exist_ok=True)
    out_csv = RESULT_DIR / 'oa_postprocessing.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['se_model', 'omega', 'cer_mean', 'n_utts'])
        for omega, c in summary.items():
            w.writerow([args.se_model, omega, c, len(samples)])
    print(f"\nCSV保存: {out_csv}")

    # ── プロット ──
    fig_dir = RESULT_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    omegas = list(summary.keys())
    cers   = list(summary.values())

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, cers, 'o-', color='steelblue', linewidth=2, markersize=7, label=f'OA ({args.se_model})')
    plt.axhline(baseline_se,  color='green',  linestyle='--', alpha=0.7, label=f'SE出力のみ (ω=0.0): CER={baseline_se:.3f}')
    plt.axhline(baseline_raw, color='red',    linestyle='--', alpha=0.7, label=f'喉マイク生音声 (ω=1.0): CER={baseline_raw:.3f}')
    plt.xlabel('ω_OA（喉マイク生音声の混合比）', fontsize=12)
    plt.ylabel('CER（低いほど良い）', fontsize=12)
    plt.title(f'OA後処理の効果 — {args.se_model} × Whisper-small (Korean)', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = fig_dir / f'oa_postprocessing_{args.se_model}.png'
    plt.savefig(fig_path, dpi=150)
    print(f"図保存: {fig_path}")
    plt.close()

    print("\n完了。")


if __name__ == '__main__':
    main()
