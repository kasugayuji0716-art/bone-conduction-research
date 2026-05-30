"""
スクリプト37b: 37の途中再開 — Adapter × {SE:demucs, SE:seconformer, SE:tstnn} のみ実行
既知の9件の結果と合わせてtaps_se_comparison.csvを完成させる
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

BASE_DIR      = Path(__file__).parent.parent
TAPS_DIR      = BASE_DIR / 'data' / 'raw' / 'taps'
RESULT_DIR    = BASE_DIR / 'results'
CKPT_DIR      = BASE_DIR / 'checkpoints'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'

MODEL_ID     = 'openai/whisper-small'
ADAPTER_CKPT = CKPT_DIR / 'whisper_encoder_adapter'
TARGET_SR    = 16000
D_MODEL      = 768

# ── 既知の9件（37のログから） ──────────────────────────────────
KNOWN_RESULTS = [
    {'asr': 'Pretrained', 'se': 'No SE',         'cer': 0.4712, 'n': 1000},
    {'asr': 'Pretrained', 'se': 'SE:demucs',      'cer': 1.0000, 'n': 1000},
    {'asr': 'Pretrained', 'se': 'SE:seconformer', 'cer': 1.0000, 'n': 1000},
    {'asr': 'Pretrained', 'se': 'SE:tstnn',       'cer': 0.4728, 'n': 1000},
    {'asr': 'FT',         'se': 'No SE',         'cer': 0.1364, 'n': 1000},
    {'asr': 'FT',         'se': 'SE:demucs',      'cer': 1.0000, 'n': 1000},
    {'asr': 'FT',         'se': 'SE:seconformer', 'cer': 1.0000, 'n': 1000},
    {'asr': 'FT',         'se': 'SE:tstnn',       'cer': 0.1367, 'n': 1000},
    {'asr': 'Adapter',    'se': 'No SE',         'cer': 0.1490, 'n': 1000},
]


# ── Adapter モデル定義 ─────────────────────────────────────────

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


# ── SE モデル定義（必要な3つのみ） ────────────────────────────

class _DemucsLSTM(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.lstm   = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y, _ = self.lstm(x)
        y = self.linear(y)
        return y.permute(0, 2, 1)


class ConvDemucs(nn.Module):
    CH = [1, 64, 128, 256, 512, 1024]
    K  = 8
    S  = 4

    def __init__(self):
        super().__init__()
        enc, dec = nn.ModuleList(), nn.ModuleList()
        for i in range(5):
            ic, hc = self.CH[i], self.CH[i + 1]
            enc.append(nn.Sequential(
                nn.Conv1d(ic, hc, self.K, stride=self.S),
                nn.ReLU(),
                nn.Conv1d(hc, hc * 2, 1),
                nn.GLU(dim=1),
            ))
        for i in range(4, -1, -1):
            ic, oc = self.CH[i + 1], self.CH[i]
            dec.append(nn.Sequential(
                nn.Conv1d(ic, ic * 2, 1),
                nn.GLU(dim=1),
                nn.ConvTranspose1d(ic, max(oc, 1), self.K, stride=self.S),
                nn.ReLU() if i > 0 else nn.Tanh(),
            ))
        self.encoder = enc
        self.lstm    = _DemucsLSTM(self.CH[-1])
        self.decoder = dec

    def forward(self, x):
        T = x.shape[-1]
        for enc in self.encoder:
            x = enc(x)
        x = self.lstm(x)
        for dec in self.decoder:
            x = dec(x)
        return x[..., :T]


class _SEConformerFFN(nn.Module):
    def __init__(self, d=512, r=64):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, r), nn.SiLU(), nn.Dropout(0.1), nn.Linear(r, d),
        )
    def forward(self, x): return self.sequential(x)


class _SEConformerConvModule(nn.Module):
    def __init__(self, d=512, k=15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.sequential = nn.Sequential(
            nn.Conv1d(d, d * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, k, padding=k//2, groups=d),
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
        r = x; x = self.self_attn_layer_norm(x)
        a, _ = self.self_attn(x, x, x); x = r + a
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)


class SEConformerModel(nn.Module):
    CH = [1, 64, 128, 256, 512]
    K  = 8
    S  = 4
    def __init__(self, heads=8, n_conf=4):
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
    def forward(self, x):
        T = x.shape[-1]
        for enc in self.encoder: x = enc(x)
        x = x.permute(0, 2, 1)
        for conf in self.conformers: x = conf(x)
        x = x.permute(0, 2, 1)
        for dec in self.decoder: x = dec(x)
        return x[..., :T]


# TSTNN（37と同一実装）
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
        self.proj  = nn.Conv2d(in_ch, d, 1)
        self.rows  = nn.ModuleList([_RowColTransLayer(d) for _ in range(n_row)])
        self.cols  = nn.ModuleList([_RowColTransLayer(d) for _ in range(n_col)])
        self.unproj = nn.Conv2d(d, in_ch, 1)
    def forward(self, x):
        B, C, T, F = x.shape
        r = x
        x = self.proj(x)
        for row in self.rows:
            x = x.permute(0, 3, 2, 1).reshape(B * F, T, -1)
            x = row(x).reshape(B, F, T, -1).permute(0, 3, 2, 1)
        for col in self.cols:
            x = x.permute(0, 2, 3, 1).reshape(B * T, F, -1)
            x = col(x).reshape(B, T, F, -1).permute(0, 3, 1, 2)
        return self.unproj(x) + r


class TSTNNModel(nn.Module):
    N_FFT = 512; HOP = 256; WIN = 512
    def __init__(self):
        super().__init__()
        F = self.N_FFT // 2 + 1
        self.inp_norm  = nn.LayerNorm(F)
        self.inp_conv  = nn.Conv2d(2, 64, (1, 1))
        self.enc_dense1 = _DenseBlock(ch=64, n=4, freq_size=F)
        self.enc_norm1  = nn.LayerNorm(F // 2)
        self.enc_conv1  = nn.Conv2d(64, 64, (1, 3), stride=(1, 2), padding=(0, 1))
        self.dual_trans = _DualTransformer(in_ch=64, d=32, n_row=4, n_col=4)
        self.dec_norm1  = nn.LayerNorm(F)
        self.dec_conv1  = nn.Sequential(nn.Conv2d(64, 128, 1), nn.PixelShuffle(2) if False else nn.Identity())
        self.dec_dense1 = _DenseBlock(ch=64, n=4, freq_size=F)
        self.out_conv   = nn.Conv2d(64, 2, (1, 1))
        self._F = F
    def forward(self, x):
        win = torch.hann_window(self.WIN, device=x.device)
        spec = torch.stft(x.squeeze(1), self.N_FFT, self.HOP, self.WIN, win,
                          return_complex=True)
        mag   = spec.abs().unsqueeze(1)
        phase = torch.view_as_real(spec / (spec.abs() + 1e-8)).permute(0, 3, 2, 1)
        h = self.inp_norm(mag.squeeze(1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1)
        h = torch.cat([h, phase[:, :1]], dim=1)
        h = self.inp_conv(h)
        h = self.enc_dense1(h)
        h_e1 = h
        h = self.enc_conv1(h)
        h = self.enc_norm1(h.permute(0, 1, 3, 2).reshape(-1, h.shape[2], h.shape[-1]))
        h = h.reshape(h_e1.shape[0], 64, h_e1.shape[2], -1).permute(0, 1, 3, 2)
        h = self.dual_trans(h)
        dec_up = torch.nn.functional.interpolate(h, size=(h_e1.shape[2], h_e1.shape[3]),
                                                  mode='nearest')
        dec_conv = nn.Conv2d(64, 64, 1).to(x.device)
        with torch.no_grad():
            dec_conv.weight.fill_(0); dec_conv.bias.fill_(0)
            for i in range(64): dec_conv.weight[i, i] = 1.0
        h = dec_conv(dec_up) + h_e1
        h = self.dec_norm1(h.permute(0, 1, 3, 2).reshape(-1, h.shape[2], self._F))
        h = h.reshape(h_e1.shape[0], 64, h_e1.shape[3], self._F).permute(0, 1, 3, 2)
        h = self.dec_dense1(h)
        mask = torch.sigmoid(self.out_conv(h))
        mask_r, mask_i = mask[:, 0], mask[:, 1]
        spec_r = spec.real * mask_r - spec.imag * mask_i
        spec_i = spec.real * mask_i + spec.imag * mask_r
        out_spec = torch.complex(spec_r, spec_i)
        out = torch.istft(out_spec, self.N_FFT, self.HOP, self.WIN, win,
                          length=x.shape[-1])
        return out.unsqueeze(1)


def _load_ckpt(model, path, label):
    try:
        raw = torch.load(path, map_location='cpu', weights_only=True)
        state = raw.get('state_dict', raw.get('model', raw))
        miss, unexp = model.load_state_dict(state, strict=False)
        n_miss = len([k for k in miss if 'num_batches_tracked' not in k])
        print(f'  {label}: OK (missing={n_miss}, unexpected={len(unexp)})')
        return True
    except Exception as e:
        print(f'  [スキップ] {label}: {e}')
        return False


def apply_se(tag, model, wav, sr, device):
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
    import csv as csv_mod

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'デバイス: {device}')

    # ── データ ──────────────────────────────────────────────────
    print('データ読み込み...')
    samples = load_samples()
    print(f'  {len(samples)} 件')

    # ── SEモデル ─────────────────────────────────────────────────
    print('SEモデルロード...')
    se_models = {}
    for name, cls in [('demucs', ConvDemucs), ('seconformer', SEConformerModel), ('tstnn', TSTNNModel)]:
        ckpt = PRETRAINED_DIR / f'{name}.th'
        if ckpt.exists():
            m = cls()
            if _load_ckpt(m, ckpt, name):
                se_models[name] = m.eval().to(device)

    # ── SE適用キャッシュ ─────────────────────────────────────────
    print('SE適用中...')
    cache = {}
    for key, model in se_models.items():
        cache[key] = []
        print(f'  [{key}] ...')
        for i, s in enumerate(samples):
            wav, sr = sf.read(str(s['path']), dtype='float32')
            try:
                enh = apply_se(key, model, wav, sr, device)
            except Exception as e:
                print(f'    警告 {s["sid"]}: {e} → 元音声使用')
                enh = wav
            cache[key].append((enh, s['text']))
            if (i + 1) % 200 == 0:
                print(f'    {i+1}/{len(samples)}', flush=True)
        print(f'  {key}: キャッシュ完了')

    del se_models
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
    print('CER計測中...')
    se_labels = {'demucs': 'SE:demucs', 'seconformer': 'SE:seconformer', 'tstnn': 'SE:tstnn'}
    new_results = []
    for se_key, se_wav_list in cache.items():
        sl = se_labels[se_key]
        print(f'  [Adapter × {sl}]', flush=True)
        refs, hyps = [], []
        for i, (wav, ref) in enumerate(se_wav_list):
            hyp = transcribe_adapter(wav, proc, model, device)
            refs.append(ref)
            hyps.append(hyp)
            if (i + 1) % 100 == 0:
                print(f'    {i+1}/{len(samples)}, CER={compute_cer(refs, hyps):.4f}', flush=True)
        c = compute_cer(refs, hyps)
        new_results.append({'asr': 'Adapter', 'se': sl, 'cer': c, 'n': len(refs)})
        print(f'    → CER = {c:.4f}', flush=True)

    # ── 保存 ────────────────────────────────────────────────────
    all_results = KNOWN_RESULTS + new_results
    out_csv = RESULT_DIR / 'taps_se_comparison.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv_mod.DictWriter(f, fieldnames=['asr', 'se', 'cer', 'n'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f'\n結果保存: {out_csv}')

    print('\n=== 最終結果 ===')
    print(f'{"ASR":<12} {"SE":<18} {"CER":>6}  n')
    for r in all_results:
        print(f'{r["asr"]:<12} {r["se"]:<18} {r["cer"]:6.4f}  {r["n"]}')


if __name__ == '__main__':
    main()
