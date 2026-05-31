"""
スクリプト37e: 公式 seconformer クラスで SE-Conformer × 3 ASR を再評価
taps-baselines/models/seconformer.py の seconformer を使用（strict=False ロード）
results/taps_se_comparison.csv の SE:seconformer 行を上書き
"""

import csv, json, sys, io
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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
ADAPTER_CKPT   = CKPT_DIR / 'whisper_encoder_adapter'
TARGET_SR      = 16000
D_MODEL        = 768

sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))


# ── Adapter 定義 ───────────────────────────────────────────────

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

    # ── 公式 SEConformer ─────────────────────────────────────────
    print('SEConformer (公式) ロード...')
    from models.seconformer import seconformer
    sec = seconformer(
        hidden=64, depth=4, conformer_dim=512, conformer_ffn_dim=64,
        conformer_num_attention_heads=4, conformer_depth=4,
        depthwise_conv_kernel_size=15, kernel_size=8, stride=4,
        resample=4, growth=2, dropout=0.1, rescale=0.1, normalize=True
    )
    ckpt = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    miss, unexp = sec.load_state_dict(state, strict=False)
    n_miss = len([k for k in miss if 'num_batches_tracked' not in k])
    print(f'  missing={n_miss}, unexpected={len(unexp)}')
    sec = sec.eval().to(device)

    # ── SE適用キャッシュ ─────────────────────────────────────────
    samples = load_samples()
    print(f'データ: {len(samples)} 件')
    print('SE適用中 (SEConformer × 1000件)...')
    cache = []
    for i, s in enumerate(samples):
        wav, _ = sf.read(str(s['path']), dtype='float32')
        try:
            enh = apply_se(sec, wav, device)
        except Exception as e:
            print(f'  警告 {s["sid"]}: {e} → 元音声')
            enh = wav
        cache.append((enh, s['text']))
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/1000', flush=True)
    print('SE適用完了')

    del sec
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
    print('  ロード完了')

    asr_models = [
        ('Pretrained', 'fw',      pretrained_fw),
        ('FT',         'fw',      ft_fw),
        ('Adapter',    'adapter', (proc, amodel)),
    ]

    # ── CER計測 ──────────────────────────────────────────────────
    print('CER計測中...')
    new_results = []
    for asr_label, asr_type, asr_model in asr_models:
        print(f'  [{asr_label} × SE:seconformer]', flush=True)
        refs, hyps = [], []
        for i, (wav, ref) in enumerate(cache):
            if asr_type == 'fw':
                hyp = transcribe_fw(wav, asr_model)
            else:
                hyp = transcribe_adapter(wav, asr_model[0], asr_model[1], device)
            refs.append(ref)
            hyps.append(hyp)
            if (i + 1) % 100 == 0:
                print(f'    {i+1}/1000, CER={compute_cer(refs, hyps):.4f}', flush=True)
        c = compute_cer(refs, hyps)
        new_results.append({'asr': asr_label, 'se': 'SE:seconformer', 'cer': c, 'n': len(refs)})
        print(f'    → CER = {c:.4f}', flush=True)

    # ── CSV更新 ────────────────────────────────────────────────
    out_csv = RESULT_DIR / 'taps_se_comparison.csv'
    with open(out_csv, encoding='utf-8') as f:
        existing = list(csv.DictReader(f))

    replace_keys = {(r['asr'], r['se']) for r in new_results}
    kept = [r for r in existing if (r['asr'], r['se']) not in replace_keys]
    all_rows = kept + new_results

    order    = ['Pretrained', 'FT', 'Adapter']
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
