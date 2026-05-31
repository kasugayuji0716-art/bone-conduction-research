"""
スクリプト37d: Demucs・SEConformerの再実行（アーキテクチャ修正後）

修正内容:
  Demucs    : 公式コード demucs(hidden=64, causal=False, stride=2, resample=2)
  SEConformer: forward にスキップ接続・正規化・valid_length padding・resample=4 を追加
               heads=4 (config準拠)

全3 ASRモデル × 2 SEモデル = 6条件を実行し、
既存の taps_se_comparison.csv の Demucs・SEConformer行を上書きする。
"""

import csv, json, sys, io, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from jiwer import cer
from faster_whisper import WhisperModel as FasterWhisperModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
RESULT_DIR     = BASE_DIR / 'results'
CKPT_DIR       = BASE_DIR / 'checkpoints'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
MODEL_ID       = 'openai/whisper-small'
FT_CKPT        = CKPT_DIR / 'whisper_throat_finetuned'
ADAPTER_CKPT   = CKPT_DIR / 'whisper_encoder_adapter'
TARGET_SR      = 16000
D_MODEL        = 768

sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))


# ── SEConformerModel（修正版）──────────────────────────────────

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
            nn.Conv1d(d, d, k, padding=k // 2, groups=d),
            nn.BatchNorm1d(d), nn.SiLU(), nn.Conv1d(d, d, 1),
        )
    def forward(self, x):
        r = x
        x = self.layer_norm(x).transpose(1, 2)
        x = self.sequential(x).transpose(1, 2)
        return x + r


class _SEConformerBlock(nn.Module):
    def __init__(self, d=512, heads=4):
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
    CH       = [1, 64, 128, 256, 512]
    K        = 8
    S        = 4
    RESAMPLE = 4
    FLOOR    = 1e-3

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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        mono = x.mean(dim=1, keepdim=True)
        std  = mono.std(dim=-1, keepdim=True)
        x    = x / (self.FLOOR + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.RESAMPLE >= 2:
            x = upsample2(x)
        if self.RESAMPLE == 4:
            x = upsample2(x)
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        x = x.permute(0, 2, 1)
        for conf in self.conformers:
            x = conf(x)
        x = x.permute(0, 2, 1)
        for dec in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = dec(x)
        if self.RESAMPLE == 4:
            x = downsample2(x)
        if self.RESAMPLE >= 2:
            x = downsample2(x)
        x = x[..., :length]
        return std * x


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


# ── utils ──────────────────────────────────────────────────────

def apply_se(model, wav: np.ndarray, device: torch.device) -> np.ndarray:
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


def compute_cer(refs, hyps) -> float:
    if not refs or sum(len(r) for r in refs) == 0:
        return float('nan')
    return min(cer(refs, hyps), 1.0)


def transcribe_fw(wav, model):
    segments, _ = model.transcribe(wav, language='ko', beam_size=5, without_timestamps=True)
    return ''.join(s.text for s in segments)


def transcribe_adapter(wav, proc, model, device):
    feats = proc.feature_extractor(wav, sampling_rate=TARGET_SR,
                                   return_tensors='pt').input_features
    if model.dtype == torch.float16:
        feats = feats.half()
    feats = feats.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language='ko', task='transcribe', max_new_tokens=225)
    return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'デバイス: {device}')

    # ── データ ──────────────────────────────────────────────────
    samples = load_samples()
    print(f'データ: {len(samples)} 件')

    # ── SEモデル ─────────────────────────────────────────────────
    print('SEモデルロード...')

    from models.demucs import demucs as DemucsModel
    dm = DemucsModel(hidden=64, causal=False, stride=2, resample=2)
    raw = torch.load(PRETRAINED_DIR / 'demucs.th', map_location='cpu', weights_only=False)
    dm.load_state_dict(raw['model'])
    dm = dm.eval().to(device)
    print('  Demucs: OK')

    sec = SEConformerModel()
    raw = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location='cpu', weights_only=False)
    state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    miss, unexp = sec.load_state_dict(state, strict=False)
    n_miss = len([k for k in miss if 'num_batches_tracked' not in k])
    print(f'  SEConformer: missing={n_miss}, unexpected={len(unexp)}')
    sec = sec.eval().to(device)

    # ── SE適用キャッシュ ─────────────────────────────────────────
    print('SE適用中...')
    se_cache = {}
    for se_key, se_model in [('demucs', dm), ('seconformer', sec)]:
        se_cache[se_key] = []
        print(f'  [{se_key}] 1000件...')
        for i, s in enumerate(samples):
            wav, _ = sf.read(str(s['path']), dtype='float32')
            try:
                enh = apply_se(se_model, wav, device)
            except Exception as e:
                print(f'    警告 {s["sid"]}: {e} → 元音声')
                enh = wav
            se_cache[se_key].append((enh, s['text']))
            if (i + 1) % 200 == 0:
                print(f'    {i+1}/1000', flush=True)
        print(f'  {se_key}: 完了')

    del dm, sec
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── ASRモデル ────────────────────────────────────────────────
    print('ASRモデルロード...')
    fw_device  = 'cuda' if device.type == 'cuda' else 'cpu'
    fw_compute = 'float16' if fw_device == 'cuda' else 'float32'

    pretrained_fw = FasterWhisperModel(str(Path('/tmp/whisper_small_ct2')),
                                       device=fw_device, compute_type=fw_compute)
    ft_fw = FasterWhisperModel(str(Path('/tmp/whisper_ft_ct2')),
                               device=fw_device, compute_type=fw_compute)

    with open(ADAPTER_CKPT / 'adapter_config.json') as f:
        cfg = json.load(f)
    r_val, d_val = cfg['r'], cfg.get('d_model', D_MODEL)
    proc  = WhisperProcessor.from_pretrained(str(ADAPTER_CKPT / 'processor'))
    amodel = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    for i in range(len(amodel.model.encoder.layers)):
        amodel.model.encoder.layers[i] = WhisperEncoderLayerWithAdapter(
            amodel.model.encoder.layers[i], Adapter(d_val, r_val))
    astate = torch.load(ADAPTER_CKPT / 'adapter_weights.pt', map_location='cpu', weights_only=True)
    amodel.load_state_dict(astate, strict=False)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    amodel = amodel.to(dtype).eval().to(device)
    print('  Pretrained/FT/Adapter: ロード完了')

    asr_models = [
        ('Pretrained', 'fw', pretrained_fw),
        ('FT',         'fw', ft_fw),
        ('Adapter',    'adapter', (proc, amodel)),
    ]

    # ── CER計測 ──────────────────────────────────────────────────
    se_labels = {'demucs': 'SE:demucs', 'seconformer': 'SE:seconformer'}
    new_results = []

    for asr_label, asr_type, asr_model in asr_models:
        for se_key, se_wav_list in se_cache.items():
            sl = se_labels[se_key]
            print(f'  [{asr_label} × {sl}]', flush=True)
            refs, hyps = [], []
            for i, (wav, ref) in enumerate(se_wav_list):
                if asr_type == 'fw':
                    hyp = transcribe_fw(wav, asr_model)
                else:
                    hyp = transcribe_adapter(wav, asr_model[0], asr_model[1], device)
                refs.append(ref)
                hyps.append(hyp)
                if (i + 1) % 100 == 0:
                    print(f'    {i+1}/1000, CER={compute_cer(refs, hyps):.4f}', flush=True)
            c = compute_cer(refs, hyps)
            new_results.append({'asr': asr_label, 'se': sl, 'cer': c, 'n': len(refs)})
            print(f'    → CER = {c:.4f}', flush=True)

    # ── CSV更新（既存のDemucs・SEConformer行を置き換え） ──────────
    out_csv = RESULT_DIR / 'taps_se_comparison.csv'
    with open(out_csv, encoding='utf-8') as f:
        existing = list(csv.DictReader(f))

    replace_keys = {(r['asr'], r['se']) for r in new_results}
    kept = [r for r in existing if (r['asr'], r['se']) not in replace_keys]

    all_rows = kept + new_results
    order = ['Pretrained', 'FT', 'Adapter']
    se_order = ['No SE', 'SE:demucs', 'SE:seconformer', 'SE:tstnn']
    all_rows.sort(key=lambda r: (order.index(r['asr']), se_order.index(r['se'])))

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['asr', 'se', 'cer', 'n'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\n結果保存: {out_csv}')

    print('\n=== 全結果 ===')
    print(f'{"ASR":<12} {"SE":<18} {"CER":>6}  n')
    for r in all_rows:
        print(f'{r["asr"]:<12} {r["se"]:<18} {float(r["cer"]):6.4f}  {r["n"]}')


if __name__ == '__main__':
    main()
