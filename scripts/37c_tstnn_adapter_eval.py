"""
スクリプト37c: Adapter × SE:tstnn の1条件のみ評価してCSVに追記
TSTNN実装はスクリプト37と同一（N_FFT=1024, freq_size=512）
"""

import csv, json, sys, io
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from jiwer import cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
RESULT_DIR     = BASE_DIR / 'results'
CKPT_DIR       = BASE_DIR / 'checkpoints'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
MODEL_ID       = 'openai/whisper-small'
ADAPTER_CKPT   = CKPT_DIR / 'whisper_encoder_adapter'
TARGET_SR      = 16000
D_MODEL        = 768


# ── TSTNN (script 37と同一) ────────────────────────────────────

class _DenseBlock(nn.Module):
    def __init__(self, ch=64, n=4, k=(2, 3), freq_size=512):
        super().__init__()
        for i in range(1, n + 1):
            in_ch = ch * i
            setattr(self, f'conv{i}',  nn.Conv2d(in_ch, ch, k, padding=(k[0]-1, k[1]//2)))
            setattr(self, f'norm{i}',  nn.LayerNorm(freq_size))
            setattr(self, f'prelu{i}', nn.PReLU(ch))
        self.n, self.ch = n, ch

    def forward(self, x):
        inp = x
        for i in range(1, self.n + 1):
            y = getattr(self, f'conv{i}')(inp)[..., :x.shape[2], :]
            y = getattr(self, f'prelu{i}')(getattr(self, f'norm{i}')(y))
            inp = torch.cat([inp, y], dim=1)
        return inp[:, -self.ch:]


class _RowColTransLayer(nn.Module):
    def __init__(self, d=32, heads=1, gru_hidden=64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.gru       = nn.GRU(d, gru_hidden, batch_first=True, bidirectional=True)
        self.linear2   = nn.Linear(gru_hidden * 2, d)
        self.norm1     = nn.LayerNorm(d)
        self.norm2     = nn.LayerNorm(d)

    def forward(self, x):
        a, _ = self.self_attn(x, x, x); x = self.norm1(x + a)
        g, _ = self.gru(x); g = self.linear2(g)
        return self.norm2(x + g)


class _DualTransformer(nn.Module):
    def __init__(self, in_ch=64, d=32, n_row=4, n_col=4):
        super().__init__()
        self.input     = nn.Sequential(nn.Conv2d(in_ch, d, 1), nn.PReLU(1))
        self.row_trans = nn.ModuleList([_RowColTransLayer(d) for _ in range(n_row)])
        self.col_trans = nn.ModuleList([_RowColTransLayer(d) for _ in range(n_col)])
        self.row_norm  = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_row)])
        self.col_norm  = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_col)])
        self.output    = nn.Sequential(nn.PReLU(1), nn.Conv2d(d, in_ch, 1))

    def forward(self, x):
        h = self.input(x)
        B, d, T, F = h.shape
        for layer in self.row_trans:
            r = h.permute(0, 2, 3, 1).reshape(B * T, F, d)
            r = layer(r).reshape(B, T, F, d).permute(0, 3, 1, 2)
            h = h + r
        for layer in self.col_trans:
            c = h.permute(0, 3, 2, 1).reshape(B * F, T, d)
            c = layer(c).reshape(B, F, T, d).permute(0, 3, 2, 1)
            h = h + c
        return self.output(h)


class _DecConv(nn.Module):
    def __init__(self, in_ch=64, out_ch=128, k=(1, 3)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=(0, k[1]//2))

    def forward(self, x):
        y = self.conv(x)
        B, C, T, F = y.shape
        r = C // x.shape[1]
        return y.view(B, x.shape[1], r, T, F).permute(0, 1, 3, 4, 2).reshape(B, x.shape[1], T, F * r)


class TSTNNModel(nn.Module):
    N_FFT = 1024
    HOP   = 256

    def __init__(self):
        super().__init__()
        self.inp_conv         = nn.Conv2d(1, 64, (1, 1))
        self.inp_norm         = nn.LayerNorm(512)
        self.inp_prelu        = nn.PReLU(64)
        self.enc_dense1       = _DenseBlock(64, 4, (2, 3), freq_size=512)
        self.enc_conv1        = nn.Conv2d(64, 64, (1, 3), stride=(1, 2), padding=(0, 1))
        self.enc_norm1        = nn.LayerNorm(256)
        self.enc_prelu1       = nn.PReLU(64)
        self.dual_transformer = _DualTransformer(64, 32, 4, 4)
        self.output1          = nn.Sequential(nn.Conv2d(64, 64, (1, 1)))
        self.output2          = nn.Sequential(nn.Conv2d(64, 64, (1, 1)))
        self.maskconv         = nn.Conv2d(64, 64, (1, 1))
        self.dec_dense1       = _DenseBlock(64, 4, (2, 3), freq_size=256)
        self.dec_conv1        = _DecConv(64, 128, (1, 3))
        self.dec_norm1        = nn.LayerNorm(512)
        self.dec_prelu1       = nn.PReLU(64)
        self.out_conv         = nn.Conv2d(64, 1, (1, 1))

    def forward(self, x):
        T   = x.shape[-1]
        win = torch.hann_window(self.N_FFT, device=x.device)
        X   = torch.stft(x.squeeze(1), self.N_FFT, self.HOP, window=win, return_complex=True)
        mag = X.abs().unsqueeze(1).permute(0, 1, 3, 2)[..., :512]   # (B,1,T_s,512)

        h = self.inp_prelu(self.inp_norm(self.inp_conv(mag)))
        h = self.enc_dense1(h)
        h = self.enc_prelu1(self.enc_norm1(self.enc_conv1(h)))
        h = self.dual_transformer(h)

        m1 = torch.sigmoid(self.output1[0](h))
        m2 = torch.sigmoid(self.output2[0](h))
        h  = self.maskconv(h * m1) * m2

        h    = self.dec_dense1(h)
        h    = self.dec_prelu1(self.dec_norm1(self.dec_conv1(h)))
        mask = torch.sigmoid(self.out_conv(h))                        # (B,1,T_s,512)

        mask_t    = mask.squeeze(1).permute(0, 2, 1)                  # (B,512,T_s)
        mask_full = torch.cat([mask_t, mask_t[:, -1:]], dim=1)        # (B,513,T_s)
        X_enh     = X * mask_full
        return torch.istft(X_enh, self.N_FFT, self.HOP, window=win, length=T).unsqueeze(1)


# ── Adapter ────────────────────────────────────────────────────

class Adapter(nn.Module):
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


# ── utils ──────────────────────────────────────────────────────

def apply_se_tstnn(model, wav, device):
    t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    rms_in  = np.sqrt(np.mean(wav ** 2) + 1e-12)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-12)
    return (out * rms_in / rms_out).astype(np.float32)


def load_samples():
    meta = TAPS_DIR / 'metadata_test.csv'
    wdir = TAPS_DIR / 'throat' / 'test'
    rows = []
    with open(meta, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sid, spk = r['sentence_id'], r['speaker_id']
            p = wdir / f'{spk}_{sid}.wav'
            if not p.exists():
                p = wdir / f'{sid}.wav'
            if p.exists():
                rows.append({'sid': sid, 'spk': spk, 'path': p, 'text': r['text']})
    return rows


def transcribe_adapter(wav, proc, model, device):
    feats = proc.feature_extractor(wav, sampling_rate=TARGET_SR,
                                   return_tensors='pt').input_features
    if model.dtype == torch.float16:
        feats = feats.half()
    feats = feats.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language='ko', task='transcribe', max_new_tokens=225)
    return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def compute_cer(refs, hyps):
    if not refs or sum(len(r) for r in refs) == 0:
        return float('nan')
    return min(cer(refs, hyps), 1.0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'デバイス: {device}')

    # ── TSTNN ──────────────────────────────────────────────────
    tstnn_ckpt = PRETRAINED_DIR / 'tstnn.th'
    print('TSTNNロード...')
    tstnn = TSTNNModel()
    raw   = torch.load(tstnn_ckpt, map_location='cpu', weights_only=False)
    state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    miss, unexp = tstnn.load_state_dict(state, strict=False)
    n_miss = len([k for k in miss if 'num_batches_tracked' not in k])
    print(f'  TSTNN: missing={n_miss}, unexpected={len(unexp)}')
    tstnn = tstnn.eval().to(device)

    # ── データ + SE適用 ────────────────────────────────────────
    samples = load_samples()
    print(f'データ: {len(samples)} 件')
    print('TSTNN SE適用中...')
    cache = []
    for i, s in enumerate(samples):
        wav, _ = sf.read(str(s['path']), dtype='float32')
        try:
            enh = apply_se_tstnn(tstnn, wav, device)
        except Exception as e:
            print(f'  警告 {s["sid"]}: {e} → 元音声使用')
            enh = wav
        cache.append((enh, s['text']))
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(samples)}', flush=True)

    del tstnn
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Adapterモデル ────────────────────────────────────────────
    print('Adapterモデルロード...')
    with open(ADAPTER_CKPT / 'adapter_config.json') as f:
        cfg = json.load(f)
    r_val, d_val = cfg['r'], cfg.get('d_model', D_MODEL)
    proc  = WhisperProcessor.from_pretrained(str(ADAPTER_CKPT / 'processor'))
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    lays  = model.model.encoder.layers
    for i in range(len(lays)):
        lays[i] = WhisperEncoderLayerWithAdapter(lays[i], Adapter(d_val, r_val))
    state = torch.load(ADAPTER_CKPT / 'adapter_weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = model.to(dtype).eval().to(device)
    print('  Adapter: ロード完了')

    # ── CER計測 ──────────────────────────────────────────────────
    print('[Adapter × SE:tstnn]', flush=True)
    refs, hyps = [], []
    for i, (wav, ref) in enumerate(cache):
        hyp = transcribe_adapter(wav, proc, model, device)
        refs.append(ref)
        hyps.append(hyp)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(samples)}, CER={compute_cer(refs, hyps):.4f}', flush=True)
    c = compute_cer(refs, hyps)
    print(f'  → CER = {c:.4f}', flush=True)

    # ── CSVに追記 ────────────────────────────────────────────────
    out_csv = RESULT_DIR / 'taps_se_comparison.csv'
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['asr', 'se', 'cer', 'n'])
        writer.writerow({'asr': 'Adapter', 'se': 'SE:tstnn', 'cer': c, 'n': len(refs)})
    print(f'CSVに追記完了: {out_csv}')

    # ── 全結果表示 ────────────────────────────────────────────────
    print('\n=== 全結果 ===')
    with open(out_csv, encoding='utf-8') as f:
        for line in f:
            print(line.rstrip())


if __name__ == '__main__':
    main()
