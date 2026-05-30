"""
スクリプト37: TAPSベースラインSEモデルの評価
チェックポイント構造を精査して実装:
  demucs.th     → Conv1d encoder + BiLSTM + ConvTranspose1d decoder (no skip)
  seconformer.th → Conv1d encoder + 4-layer Conformer + ConvTranspose1d decoder (additive skip)
  tstnn.th       → STFT域 DenseBlock + DualTransformer (row/col Attn+GRU)

ASR: Pretrained / FT / Adapter Whisper-small (Korean)
出力: results/taps_se_comparison.csv
"""

import csv, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ─── パス ────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
RESULT_DIR     = BASE_DIR / 'results'
CKPT_DIR       = BASE_DIR / 'checkpoints'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
GDRIVE_ID      = '133hBcBob8wJ-WaV7qLNj9G3eqdzyd2vX'

MODEL_ID     = 'openai/whisper-small'
FT_CKPT      = CKPT_DIR / 'whisper_throat_finetuned'
ADAPTER_CKPT = CKPT_DIR / 'whisper_encoder_adapter'
TARGET_SR    = 16000
D_MODEL      = 768


# ══════════════════════════════════════════════════════════════
# Demucs  (Conv1d GLU encoder → BiLSTM → ConvTranspose1d decoder)
# encoder.[i].{0,2}  /  lstm.{lstm,linear}  /  decoder.[i].{0,2}
# ══════════════════════════════════════════════════════════════

class _DemucsLSTM(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.lstm   = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden * 2, hidden)

    def forward(self, x):           # x: (B, C, T)
        x = x.permute(0, 2, 1)     # (B, T, C)
        y, _ = self.lstm(x)
        y = self.linear(y)
        return y.permute(0, 2, 1)  # (B, C, T)


class ConvDemucs(nn.Module):
    """
    Demucs-style Conv+LSTM+ConvTranspose SE model (no U-Net skip connections).
    Encoder: 5 levels  64→128→256→512→1024
    Bottleneck: 2-layer BiLSTM
    Decoder: 5 levels  1024→512→256→128→64→1
    """
    CH   = [1, 64, 128, 256, 512, 1024]
    K    = 8
    S    = 4

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

    def forward(self, x):           # x: (B, 1, T)
        T = x.shape[-1]
        skips, h = [], x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
        h = self.lstm(h)
        for dec in self.decoder:
            h = dec(h)
        return h[..., :T]           # trim padding

    def named_parameters_flat(self):
        return list(self.named_parameters())


# ══════════════════════════════════════════════════════════════
# SE-Conformer  (Conv1d encoder → 4 Conformer blocks → ConvTranspose1d decoder)
# encoder.[i].{0,2}  /  conformers.[i].*  /  decoder.[i].{0,3}
# ══════════════════════════════════════════════════════════════

class _SEConformerFFN(nn.Module):
    def __init__(self, d=512, r=64):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, r),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(r, d),
        )

    def forward(self, x):
        return self.sequential(x)


class _SEConformerConvModule(nn.Module):
    def __init__(self, d=512, k=15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.sequential = nn.Sequential(
            nn.Conv1d(d, d * 2, 1),          # [0] pointwise (for GLU)
            nn.GLU(dim=1),                    # [1]
            nn.Conv1d(d, d, k, padding=k//2, groups=d),  # [2] depthwise
            nn.BatchNorm1d(d),               # [3]
            nn.SiLU(),                        # [4]
            nn.Conv1d(d, d, 1),              # [5] pointwise
        )

    def forward(self, x):           # x: (B, T, d)
        r = x
        x = self.layer_norm(x).transpose(1, 2)  # (B, d, T)
        x = self.sequential(x).transpose(1, 2)  # (B, T, d)
        return x + r


class _SEConformerBlock(nn.Module):
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.ffn1                = _SEConformerFFN(d)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.self_attn           = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv_module         = _SEConformerConvModule(d)
        self.ffn2                = _SEConformerFFN(d)
        self.final_layer_norm    = nn.LayerNorm(d)

    def forward(self, x):           # x: (B, T, d)
        x = x + 0.5 * self.ffn1(x)
        r = x
        x = self.self_attn_layer_norm(x)
        a, _ = self.self_attn(x, x, x)
        x = r + a
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)


class SEConformerModel(nn.Module):
    """
    4-level Conv1d GLU encoder  +  4 Conformer blocks  +  4-level ConvTranspose1d decoder.
    Additive skip connections between encoder and decoder.
    Encoder channels: 1→64→128→256→512
    Conformer d_model = 512
    Decoder channels: 512→256→128→64→1
    """
    CH = [1, 64, 128, 256, 512]
    K  = 8
    S  = 4

    def __init__(self, heads=8, n_conf=4):
        super().__init__()
        enc, dec = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            ic, hc = self.CH[i], self.CH[i + 1]
            enc.append(nn.Sequential(
                nn.Conv1d(ic, hc, self.K, stride=self.S),
                nn.ReLU(),
                nn.Conv1d(hc, hc * 2, 1),
                nn.GLU(dim=1),
            ))
        for i in range(3, -1, -1):
            ic, oc = self.CH[i + 1], self.CH[i]
            dec.append(nn.Sequential(
                nn.Conv1d(ic, ic * 2, 1),
                nn.GLU(dim=1),
                nn.ReLU(),
                nn.ConvTranspose1d(ic, max(oc, 1), self.K, stride=self.S),
            ))
        self.encoder   = enc
        self.conformers = nn.ModuleList([_SEConformerBlock(self.CH[-1], heads) for _ in range(n_conf)])
        self.decoder   = dec

    def forward(self, x):           # x: (B, 1, T)
        T = x.shape[-1]
        skips, h = [], x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)

        # Conformers on flattened time axis
        h = h.permute(0, 2, 1)     # (B, T', 512)
        for conf in self.conformers:
            h = conf(h)
        h = h.permute(0, 2, 1)     # (B, 512, T')

        for i, dec in enumerate(self.decoder):
            h = dec(h)

        return h[..., :T]


# ══════════════════════════════════════════════════════════════
# TSTNN  (STFT-domain 2D DenseBlock + DualTransformer)
# Checkpoint key analysis:
#   - inp_norm / enc_dense1.norm{i}: size=512 → LayerNorm(512) on freq dim
#   - enc_norm1: size=256 (after 2x freq downsample)
#   - dec_norm1: size=512 (after 2x freq upsample)
#   - dec_conv1.conv: Conv2d(64→128) + freq pixel_shuffle(2) → (B,64,T,512)
# ══════════════════════════════════════════════════════════════

class _DenseBlock(nn.Module):
    """
    4-layer dense conv block. Each conv layer sees all previous outputs.
    Norm = LayerNorm on the frequency dimension (applied to new output only).
    Output = last ch channels (from the final conv layer).
    """
    def __init__(self, ch=64, n=4, k=(2, 3), freq_size=512):
        super().__init__()
        for i in range(1, n + 1):
            in_ch = ch * i
            setattr(self, f'conv{i}', nn.Conv2d(in_ch, ch, k, padding=(k[0]-1, k[1]//2)))
            setattr(self, f'norm{i}', nn.LayerNorm(freq_size))
            setattr(self, f'prelu{i}', nn.PReLU(ch))
        self.n  = n
        self.ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ch, T, F)
        inp = x
        for i in range(1, self.n + 1):
            conv  = getattr(self, f'conv{i}')
            norm  = getattr(self, f'norm{i}')
            prelu = getattr(self, f'prelu{i}')
            y = conv(inp)[..., :x.shape[2], :]   # (B, ch, T, F), trim causal padding
            y = prelu(norm(y))                    # LayerNorm on freq dim, then PReLU
            inp = torch.cat([inp, y], dim=1)
        return inp[:, -self.ch:]                  # return last ch channels


class _RowColTransLayer(nn.Module):
    """Attention + BiGRU transformer layer (row or col)."""
    def __init__(self, d=32, heads=1, gru_hidden=64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.gru       = nn.GRU(d, gru_hidden, batch_first=True, bidirectional=True)
        self.linear2   = nn.Linear(gru_hidden * 2, d)
        self.norm1     = nn.LayerNorm(d)
        self.norm2     = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.self_attn(x, x, x)
        x    = self.norm1(x + a)
        g, _ = self.gru(x)
        g    = self.linear2(g)
        return self.norm2(x + g)


class _DualTransformer(nn.Module):
    """Row (freq) + Col (time) transformer. d_model=32, 4 blocks each."""
    def __init__(self, in_ch=64, d=32, n_row=4, n_col=4):
        super().__init__()
        self.input     = nn.Sequential(nn.Conv2d(in_ch, d, 1), nn.PReLU(1))
        self.row_trans = nn.ModuleList([_RowColTransLayer(d) for _ in range(n_row)])
        self.col_trans = nn.ModuleList([_RowColTransLayer(d) for _ in range(n_col)])
        self.row_norm  = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_row)])
        self.col_norm  = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_col)])
        self.output    = nn.Sequential(nn.PReLU(1), nn.Conv2d(d, in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)             # (B, d, T, F)
        B, d, T, F = h.shape

        for layer in self.row_trans:
            r = h.permute(0, 2, 3, 1).reshape(B * T, F, d)
            r = layer(r).reshape(B, T, F, d).permute(0, 3, 1, 2)
            h = h + r

        for layer in self.col_trans:
            c = h.permute(0, 3, 2, 1).reshape(B * F, T, d)
            c = layer(c).reshape(B, F, T, d).permute(0, 3, 2, 1)
            h = h + c

        return self.output(h)         # (B, in_ch, T, F)


class _DecConv(nn.Module):
    """Conv2d(64→128) + channel-to-freq pixel_shuffle(r=2) → (B,64,T,F*2)."""
    def __init__(self, in_ch=64, out_ch=128, k=(1, 3)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=(0, k[1]//2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)              # (B, 128, T, F)
        B, C, T, F = y.shape
        r = C // x.shape[1]           # upsample factor = 2 (128/64)
        return y.view(B, x.shape[1], r, T, F).permute(0, 1, 3, 4, 2).reshape(B, x.shape[1], T, F * r)


class TSTNNModel(nn.Module):
    """
    STFT-domain TSTNN (frequency-time 2D processing).
    waveform → STFT(n_fft=1024) → magnitude (B,1,T,512) →
    Dense encoder → DualTransformer → mask → complex spectrogram → ISTFT.
    """
    N_FFT = 1024
    HOP   = 256

    def __init__(self):
        super().__init__()
        self.inp_conv      = nn.Conv2d(1, 64, (1, 1))
        self.inp_norm      = nn.LayerNorm(512)
        self.inp_prelu     = nn.PReLU(64)
        self.enc_dense1    = _DenseBlock(64, 4, (2, 3), freq_size=512)
        self.enc_conv1     = nn.Conv2d(64, 64, (1, 3), stride=(1, 2), padding=(0, 1))
        self.enc_norm1     = nn.LayerNorm(256)
        self.enc_prelu1    = nn.PReLU(64)
        self.dual_transformer = _DualTransformer(64, 32, 4, 4)
        self.output1       = nn.Sequential(nn.Conv2d(64, 64, (1, 1)))
        self.output2       = nn.Sequential(nn.Conv2d(64, 64, (1, 1)))
        self.maskconv      = nn.Conv2d(64, 64, (1, 1))
        self.dec_dense1    = _DenseBlock(64, 4, (2, 3), freq_size=256)
        self.dec_conv1     = _DecConv(64, 128, (1, 3))
        self.dec_norm1     = nn.LayerNorm(512)
        self.dec_prelu1    = nn.PReLU(64)
        self.out_conv      = nn.Conv2d(64, 1, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.shape[-1]
        win = torch.hann_window(self.N_FFT, device=x.device)
        X   = torch.stft(x.squeeze(1), self.N_FFT, self.HOP, window=win, return_complex=True)
        # X: (B, 513, T_stft)
        mag = X.abs().unsqueeze(1).permute(0, 1, 3, 2)[..., :512]  # (B, 1, T_stft, 512)

        h = self.inp_prelu(self.inp_norm(self.inp_conv(mag)))   # (B, 64, T_s, 512)
        h = self.enc_dense1(h)                                  # (B, 64, T_s, 512)
        h = self.enc_prelu1(self.enc_norm1(self.enc_conv1(h)))  # (B, 64, T_s, 256)
        h = self.dual_transformer(h)                            # (B, 64, T_s, 256)

        m1 = torch.sigmoid(self.output1[0](h))
        m2 = torch.sigmoid(self.output2[0](h))
        h  = self.maskconv(h * m1) * m2                         # (B, 64, T_s, 256)

        h = self.dec_dense1(h)                                  # (B, 64, T_s, 256)
        h = self.dec_prelu1(self.dec_norm1(self.dec_conv1(h)))  # (B, 64, T_s, 512)
        mask = torch.sigmoid(self.out_conv(h))                  # (B, 1, T_s, 512)

        # Apply mask to complex STFT (512 bins; pad Nyquist bin with last value)
        mask_t = mask.squeeze(1).permute(0, 2, 1)              # (B, 512, T_s)
        mask_full = torch.cat([mask_t, mask_t[:, -1:]], dim=1) # (B, 513, T_s)
        X_enh    = X * mask_full
        return torch.istft(X_enh, self.N_FFT, self.HOP, window=win, length=T).unsqueeze(1)


# ══════════════════════════════════════════════════════════════
# モデルロード
# ══════════════════════════════════════════════════════════════

def _load_ckpt(model: nn.Module, path: Path, label: str) -> bool:
    try:
        raw   = torch.load(path, map_location='cpu', weights_only=False)
        state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
        miss, unexp = model.load_state_dict(state, strict=False)
        n_miss = len([k for k in miss if 'num_batches_tracked' not in k])
        print(f'  {label}: ロード完了 (missing={n_miss}, unexpected={len(unexp)})')
        return True
    except Exception as e:
        print(f'  [スキップ] {label}: {e}')
        return False


def load_se_models(device: torch.device, no_custom_ckpt=False) -> dict:
    models = {}

    # Demucs
    print('  Demucs ...')
    demucs_ckpt = PRETRAINED_DIR / 'demucs.th'
    if demucs_ckpt.exists() and not no_custom_ckpt:
        dm = ConvDemucs()
        if _load_ckpt(dm, demucs_ckpt, 'Demucs'):
            models['demucs'] = ('demucs', dm.eval().to(device))
    if 'demucs' not in models:
        print('  Demucs: htdemucs (デフォルト事前学習済み) にフォールバック')
        try:
            from demucs.pretrained import get_model
            models['demucs'] = ('htdemucs', get_model('htdemucs').eval().to(device))
        except Exception as e:
            print(f'  [スキップ] Demucs htdemucs: {e}')

    # SE-Conformer
    print('  SE-Conformer ...')
    sec_ckpt = PRETRAINED_DIR / 'seconformer.th'
    if sec_ckpt.exists():
        sec = SEConformerModel()
        if _load_ckpt(sec, sec_ckpt, 'SE-Conformer'):
            models['seconformer'] = ('seconformer', sec.eval().to(device))
    else:
        print(f'  [スキップ] SE-Conformer: {sec_ckpt} なし')

    # TSTNN
    print('  TSTNN ...')
    tstnn_ckpt = PRETRAINED_DIR / 'tstnn.th'
    if tstnn_ckpt.exists():
        tstnn = TSTNNModel()
        if _load_ckpt(tstnn, tstnn_ckpt, 'TSTNN'):
            models['tstnn'] = ('tstnn', tstnn.eval().to(device))
    else:
        print(f'  [スキップ] TSTNN: {tstnn_ckpt} なし')

    return models


def apply_se(tag: str, model, wav: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    if tag == 'htdemucs':
        from demucs.apply import apply_model
        t = torch.from_numpy(wav).float()
        t = t.unsqueeze(0).repeat(2, 1).unsqueeze(0).to(device)  # (1,2,T)
        with torch.no_grad():
            src = apply_model(model, t, overlap=0.25, shifts=0)
        idx = model.sources.index('vocals') if 'vocals' in model.sources else 0
        out = src[0, idx].mean(0).cpu().numpy()
    else:
        t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t).squeeze().cpu().numpy()

    rms_in  = np.sqrt(np.mean(wav ** 2) + 1e-12)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-12)
    return (out * rms_in / rms_out).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# ASR モデル（script 32と同一）
# ══════════════════════════════════════════════════════════════

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


def load_asr_models(device: torch.device) -> dict:
    """
    Returns dict: key → ('faster_whisper', FasterWhisperModel)
                    or  ('transformers',   (WhisperProcessor, WhisperForConditionalGeneration))
    """
    asr = {}
    fw_device = 'cuda' if device.type == 'cuda' else 'cpu'
    fw_compute = 'float16' if fw_device == 'cuda' else 'float32'

    CT2_PRETRAINED = Path('/tmp/whisper_small_ct2')
    CT2_FT         = Path('/tmp/whisper_ft_ct2')

    print(f'  A: Pretrained Whisper-small (faster-whisper, {fw_compute}) ...')
    fw = FasterWhisperModel(str(CT2_PRETRAINED), device=fw_device, compute_type=fw_compute)
    asr['pretrained'] = ('faster_whisper', fw)

    print(f'  B: FT Whisper-small (faster-whisper, {fw_compute}) ...')
    if CT2_FT.exists():
        fw_ft = FasterWhisperModel(str(CT2_FT), device=fw_device, compute_type=fw_compute)
        asr['ft'] = ('faster_whisper', fw_ft)
    elif FT_CKPT.exists():
        print(f'     CT2形式なし、HuggingFace形式から変換 ...')
        import subprocess
        subprocess.run(['ct2-transformers-converter', '--model', str(FT_CKPT),
                        '--output_dir', str(CT2_FT), '--quantization', 'float16', '--force'],
                       check=True)
        fw_ft = FasterWhisperModel(str(CT2_FT), device=fw_device, compute_type=fw_compute)
        asr['ft'] = ('faster_whisper', fw_ft)
    else:
        print(f'     スキップ: {FT_CKPT} なし')

    print('  C: Encoder Adapter (transformers, fp16) ...')
    if ADAPTER_CKPT.exists():
        try:
            with open(ADAPTER_CKPT / 'adapter_config.json') as f:
                cfg = json.load(f)
            r, d = cfg['r'], cfg.get('d_model', D_MODEL)
            proc  = WhisperProcessor.from_pretrained(str(ADAPTER_CKPT / 'processor'))
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
            lays  = model.model.encoder.layers
            for i in range(len(lays)):
                lays[i] = WhisperEncoderLayerWithAdapter(lays[i], Adapter(d, r))
            state = torch.load(ADAPTER_CKPT / 'adapter_weights.pt',
                               map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=False)
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            asr['adapter'] = ('transformers', (proc, model.to(dtype).eval().to(device)))
        except Exception as e:
            print(f'     スキップ: {e}')
    else:
        print(f'     スキップ: {ADAPTER_CKPT} なし')
    return asr


def transcribe(wav: np.ndarray, model_info: tuple, device: torch.device, language='ko') -> str:
    mtype, mdata = model_info
    if mtype == 'faster_whisper':
        segments, _ = mdata.transcribe(wav, language=language, beam_size=5,
                                        without_timestamps=True)
        return ''.join(s.text for s in segments)
    else:
        proc, model = mdata
        feats = proc.feature_extractor(wav, sampling_rate=TARGET_SR,
                                        return_tensors='pt').input_features
        if model.dtype == torch.float16:
            feats = feats.half()
        feats = feats.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language=language, task='transcribe', max_new_tokens=225)
        return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def compute_cer(refs, hyps) -> float:
    if not refs or sum(len(r) for r in refs) == 0:
        return float('nan')
    return min(cer(refs, hyps), 1.0)


# ══════════════════════════════════════════════════════════════
# データ読み込み
# ══════════════════════════════════════════════════════════════

def load_samples(max_samples=None):
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
    return rows[:max_samples] if max_samples else rows


# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

def main():
    import argparse, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument('--max_samples',    type=int, default=None)
    p.add_argument('--device',         default='auto', choices=['auto','cuda','cpu'])
    p.add_argument('--no_download',    action='store_true')
    p.add_argument('--no_custom_ckpt', action='store_true')
    p.add_argument('--skip_se',        nargs='*', default=[])
    args = p.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f'デバイス: {device}')
    RESULT_DIR.mkdir(exist_ok=True)

    # ── ダウンロード ────────────────────────────────────────
    if not args.no_download:
        sentinel = PRETRAINED_DIR / 'demucs.th'
        if not sentinel.exists():
            PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
            print('Google Driveからダウンロード中...')
            try:
                import gdown
                gdown.download_folder(id=GDRIVE_ID, output=str(PRETRAINED_DIR), quiet=False)
            except Exception as e:
                print(f'[警告] ダウンロード失敗: {e}')
        else:
            print('チェックポイント既存。スキップ。')

    # ── データ ─────────────────────────────────────────────
    print('\nデータ読み込み中...')
    samples = load_samples(args.max_samples)
    print(f'  テストサンプル: {len(samples)} 件')

    # ── SEモデル ────────────────────────────────────────────
    print('\nSEモデルロード中...')
    se_models = load_se_models(device, args.no_custom_ckpt)
    for tag in args.skip_se:
        se_models.pop(tag, None)
    se_keys = ['no_se'] + [k for k in ('demucs','seconformer','tstnn') if k in se_models]
    print(f'  有効SE: {se_keys}')

    # ── ASRモデル ───────────────────────────────────────────
    print('\nASRモデルロード中...')
    asr_models = load_asr_models(device)
    print(f'  有効ASR: {list(asr_models.keys())}')

    # ── SE事前適用キャッシュ ─────────────────────────────────
    print('\nSE適用中...')
    cache: dict[str, list] = {}
    for key in se_keys:
        cache[key] = []
        if key == 'no_se':
            for s in samples:
                wav, _ = sf.read(str(s['path']), dtype='float32')
                cache[key].append((wav, s['text']))
            continue
        tag, model = se_models[key]
        print(f'  [{key}] {len(samples)}件...')
        for i, s in enumerate(samples):
            wav, sr = sf.read(str(s['path']), dtype='float32')
            try:
                enh = apply_se(tag, model, wav, sr, device)
            except Exception as e:
                print(f'    警告 {s["sid"]}: {e} → 元音声使用')
                enh = wav
            cache[key].append((enh, s['text']))
            if (i + 1) % 200 == 0:
                print(f'    {i+1}/{len(samples)}')

    # ── CER計測 ─────────────────────────────────────────────
    print('\nCER計測中...')
    results = []
    asr_labels = {'pretrained': 'Pretrained', 'ft': 'FT', 'adapter': 'Adapter'}
    se_labels  = {'no_se': 'No SE', 'demucs': 'SE:demucs',
                  'seconformer': 'SE:seconformer', 'tstnn': 'SE:tstnn'}

    for asr_key, model_info in asr_models.items():
        al = asr_labels.get(asr_key, asr_key)
        for se_key in se_keys:
            sl = se_labels.get(se_key, se_key)
            print(f'  [{al} × {sl}]', flush=True)
            refs, hyps = [], []
            for i, (wav, ref) in enumerate(cache[se_key]):
                hyp = transcribe(wav, model_info, device)
                refs.append(ref)
                hyps.append(hyp)
                if (i + 1) % 100 == 0:
                    print(f'    {i+1}/{len(samples)}, CER={compute_cer(refs, hyps):.4f}', flush=True)
            c = compute_cer(refs, hyps)
            results.append({'asr': al, 'se': sl, 'cer': c, 'n': len(refs)})
            print(f'    → CER = {c:.4f}', flush=True)

    # ── 保存・表示 ──────────────────────────────────────────
    out_csv = RESULT_DIR / 'taps_se_comparison.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['asr', 'se', 'cer', 'n'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\n結果保存: {out_csv}')

    print('\n' + '=' * 60)
    print(f'{"ASRモデル":<18} {"条件":<22} {"CER":>6}')
    print('-' * 60)
    for r in results:
        print(f'{r["asr"]:<18} {r["se"]:<22} {r["cer"]:>6.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
