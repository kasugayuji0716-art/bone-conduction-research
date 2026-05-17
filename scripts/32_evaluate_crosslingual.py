"""
フェーズ3: Cross-lingual 評価スクリプト

韓国語 TAPS で学習した各モデルを、
韓国語（in-domain）とフランス語 VibraVox（zero-shot）で評価する。

評価モデル（4条件）:
  A: Pretrained Whisper small        （ベースライン）
  B: Full FT Whisper                 （フェーズ2）
  C: Encoder-only Adapter            （29番で学習）
  D: Encoder+Decoder LoRA            （30番で学習）

評価データ:
  Korean : data/raw/taps/throat/test/  + metadata_test.csv
  French : data/raw/vibravox/throat/test/ + metadata_test.csv

仮説:
  C（Encoder-only）は B・D より French CER が低い（Decoder を多言語のまま保つため）

出力:
  results/crosslingual_per_sample.csv  -- サンプルごと詳細
  results/crosslingual_summary.csv     -- モデル×言語 CER サマリー
  results/crosslingual_report.txt      -- 人間が読みやすいレポート

実行:
  python3 scripts/32_evaluate_crosslingual.py
  python3 scripts/32_evaluate_crosslingual.py --max_french 500  # French を 500 件に制限
"""

import csv
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from jiwer import cer

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch.nn as nn

BASE_DIR    = Path(__file__).parent.parent
TAPS_DIR    = BASE_DIR / 'data' / 'raw' / 'taps'
VIBRAVOX_DIR = BASE_DIR / 'data' / 'raw' / 'vibravox'
RESULT_DIR  = BASE_DIR / 'results'
TARGET_SR   = 16000

MODEL_ID         = 'openai/whisper-small'
FT_CKPT          = BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned'
ADAPTER_CKPT     = BASE_DIR / 'checkpoints' / 'whisper_encoder_adapter'
LORA_CKPT        = BASE_DIR / 'checkpoints' / 'whisper_encoder_decoder_lora'
KD_CKPT          = BASE_DIR / 'checkpoints' / 'whisper_kd_adapter'

D_MODEL = 768  # Whisper small（adapter_config.json からも読み込む）


# ── Adapter モジュール（29番と同一定義）────────────────────────
class Adapter(nn.Module):
    def __init__(self, d_model: int = D_MODEL, r: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class WhisperEncoderLayerWithAdapter(nn.Module):
    def __init__(self, original_layer: nn.Module, adapter: Adapter):
        super().__init__()
        self.layer   = original_layer
        self.adapter = adapter

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = self.layer(hidden_states, attention_mask, **kwargs)
        return self.adapter(hidden_states)


# ── モデルロード ───────────────────────────────────────────────
def load_pretrained():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


def load_full_ft():
    if not FT_CKPT.exists():
        raise FileNotFoundError(f'Full FT checkpoint が見つかりません: {FT_CKPT}')
    processor = WhisperProcessor.from_pretrained(str(FT_CKPT))
    model     = WhisperForConditionalGeneration.from_pretrained(str(FT_CKPT))
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


def load_adapter():
    if not ADAPTER_CKPT.exists():
        raise FileNotFoundError(f'Adapter checkpoint が見つかりません: {ADAPTER_CKPT}')

    config_path = ADAPTER_CKPT / 'adapter_config.json'
    with open(config_path) as f:
        config = json.load(f)
    r       = config['r']
    d_model = config.get('d_model', D_MODEL)

    processor = WhisperProcessor.from_pretrained(str(ADAPTER_CKPT / 'processor'))
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # Adapter を挿入
    layers = model.model.encoder.layers
    for i in range(len(layers)):
        adapter  = Adapter(d_model=d_model, r=r)
        layers[i] = WhisperEncoderLayerWithAdapter(layers[i], adapter)

    # Adapter 重みをロード
    adapter_state = torch.load(
        ADAPTER_CKPT / 'adapter_weights.pt',
        map_location='cpu', weights_only=True
    )
    model.load_state_dict(adapter_state, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


def load_lora():
    if not LORA_CKPT.exists():
        raise FileNotFoundError(f'LoRA checkpoint が見つかりません: {LORA_CKPT}')
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError('peft が必要です: pip install peft')

    processor  = WhisperProcessor.from_pretrained(str(LORA_CKPT / 'processor'))
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model      = PeftModel.from_pretrained(base_model, str(LORA_CKPT))
    model      = model.merge_and_unload()  # LoRA をマージして推論高速化

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


def load_kd_adapter():
    if not KD_CKPT.exists():
        raise FileNotFoundError(f'KD Adapter checkpoint が見つかりません: {KD_CKPT}')

    config_path = KD_CKPT / 'adapter_config.json'
    with open(config_path) as f:
        config = json.load(f)
    r       = config['r']
    d_model = config.get('d_model', D_MODEL)

    processor = WhisperProcessor.from_pretrained(str(KD_CKPT / 'processor'))
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    layers = model.model.encoder.layers
    for i in range(len(layers)):
        adapter  = Adapter(d_model=d_model, r=r)
        layers[i] = WhisperEncoderLayerWithAdapter(layers[i], adapter)

    adapter_state = torch.load(
        KD_CKPT / 'adapter_weights_kd.pt',
        map_location='cpu', weights_only=True
    )
    model.load_state_dict(adapter_state, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return processor, model.eval().to(device), device


# ── 推論 ─────────────────────────────────────────────────────
def transcribe(wav: np.ndarray, processor, model, device, language: str) -> str:
    feats = processor.feature_extractor(
        wav, sampling_rate=TARGET_SR, return_tensors='pt'
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language=language, task='transcribe',
                              max_new_tokens=225)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


# ── データ読み込み ─────────────────────────────────────────────
def load_samples(data_dir: Path, meta_csv: Path, max_samples=None):
    samples = []
    with open(meta_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid  = row['sentence_id']
            spk  = row['speaker_id']
            path = data_dir / f'{spk}_{sid}.wav'
            if not path.exists():
                path = data_dir / f'{sid}.wav'
            if path.exists():
                samples.append({'sid': sid, 'spk': spk,
                                 'path': path, 'text': row['text']})
    if max_samples:
        samples = samples[:max_samples]
    return samples


# ── メイン ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_korean', type=int, default=None,
                        help='Korean 評価サンプル数上限')
    parser.add_argument('--max_french', type=int, default=None,
                        help='French 評価サンプル数上限')
    parser.add_argument('--skip_lora',       action='store_true',
                        help='LoRA モデルをスキップ（未学習の場合）')
    parser.add_argument('--skip_adapter',    action='store_true',
                        help='Adapter モデルをスキップ（未学習の場合）')
    parser.add_argument('--skip_kd_adapter', action='store_true',
                        help='KD Adapter モデルをスキップ（未学習の場合）')
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)

    # ── データ収集 ──────────────────────────────────────────
    print('データ収集中...')

    ko_samples = load_samples(
        TAPS_DIR / 'throat' / 'test',
        TAPS_DIR / 'metadata_test.csv',
        max_samples=args.max_korean,
    )
    print(f'  Korean : {len(ko_samples)} サンプル')

    fr_meta = VIBRAVOX_DIR / 'metadata_test.csv'
    fr_wav_dir = VIBRAVOX_DIR / 'throat' / 'test'
    if not fr_meta.exists():
        print('  French : VibraVox 未取得。31_download_vibravox.py を先に実行してください')
        fr_samples = []
    else:
        fr_samples = load_samples(fr_wav_dir, fr_meta, max_samples=args.max_french)
        print(f'  French : {len(fr_samples)} サンプル')

    if not ko_samples and not fr_samples:
        print('ERROR: 評価データが見つかりません')
        return

    # ── モデルロード ────────────────────────────────────────
    models = {}

    print('\nモデルロード中...')
    print('  A: Pretrained Whisper...')
    models['A_pretrained'] = load_pretrained()

    print('  B: Full FT Whisper...')
    try:
        models['B_full_ft'] = load_full_ft()
    except FileNotFoundError as e:
        print(f'     スキップ: {e}')

    if not args.skip_adapter:
        print('  C: Encoder-only Adapter...')
        try:
            models['C_adapter'] = load_adapter()
        except FileNotFoundError as e:
            print(f'     スキップ: {e}')

    if not args.skip_lora:
        print('  D: Encoder+Decoder LoRA...')
        try:
            models['D_lora'] = load_lora()
        except (FileNotFoundError, ImportError) as e:
            print(f'     スキップ: {e}')

    if not args.skip_kd_adapter:
        print('  E: KD Adapter...')
        try:
            models['E_kd_adapter'] = load_kd_adapter()
        except FileNotFoundError as e:
            print(f'     スキップ: {e}')

    device = next(iter(models.values()))[2]
    print(f'\nデバイス: {device}')
    print(f'評価モデル: {list(models.keys())}')

    # ── 評価ループ ──────────────────────────────────────────
    all_rows = []

    datasets = []
    if ko_samples:
        datasets.append(('korean', 'ko', ko_samples))
    if fr_samples:
        datasets.append(('french', 'fr', fr_samples))

    for lang_name, lang_code, samples in datasets:
        print(f'\n===== {lang_name.upper()} ({len(samples)} サンプル) =====')

        for i, s in enumerate(samples):
            wav, _ = sf.read(str(s['path']), dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            ref = s['text']
            row = {
                'language':    lang_name,
                'speaker_id':  s['spk'],
                'sentence_id': s['sid'],
                'reference':   ref,
            }

            for model_name, (proc, model, dev) in models.items():
                hyp = transcribe(wav, proc, model, dev, language=lang_code)
                row[model_name] = round(cer(ref, hyp), 4)

            all_rows.append(row)

            if (i + 1) % 100 == 0 or i == 0:
                cer_strs = '  '.join(
                    f'{k.split("_")[0]}={row[k]:.3f}'
                    for k in models.keys()
                )
                print(f'  [{i+1:4d}/{len(samples)}] {s["spk"]} {s["sid"]}  {cer_strs}')

    # ── CSV 保存 ────────────────────────────────────────────
    model_keys = list(models.keys())
    fields = ['language', 'speaker_id', 'sentence_id', 'reference'] + model_keys

    out_sample = RESULT_DIR / 'crosslingual_per_sample.csv'
    with open(out_sample, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # ── サマリー計算 ────────────────────────────────────────
    summary_rows = []
    for lang_name in ['korean', 'french']:
        lang_rows = [r for r in all_rows if r['language'] == lang_name]
        if not lang_rows:
            continue
        for mk in model_keys:
            vals = [r[mk] for r in lang_rows]
            summary_rows.append({
                'language':  lang_name,
                'model':     mk,
                'cer_mean':  round(float(np.mean(vals)), 4),
                'cer_std':   round(float(np.std(vals)),  4),
                'n_samples': len(vals),
            })

    out_summary = RESULT_DIR / 'crosslingual_summary.csv'
    with open(out_summary, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['language', 'model', 'cer_mean', 'cer_std', 'n_samples'])
        w.writeheader()
        w.writerows(summary_rows)

    # ── レポート生成 ────────────────────────────────────────
    lines = []
    lines.append('=' * 70)
    lines.append('  Cross-lingual Evaluation Report')
    lines.append('  Training: Korean TAPS throat mic')
    lines.append('  Hypothesis: Encoder-only Adapter transfers better to French')
    lines.append('=' * 70)

    model_labels = {
        'A_pretrained':  'A: Pretrained Whisper',
        'B_full_ft':     'B: Full FT (Korean)',
        'C_adapter':     'C: Encoder-only Adapter',
        'D_lora':        'D: Enc+Dec LoRA (Korean)',
        'E_kd_adapter':  'E: KD Adapter (proposed)',
    }

    for lang_name in ['korean', 'french']:
        lang_rows = [r for r in summary_rows if r['language'] == lang_name]
        if not lang_rows:
            continue
        lines.append(f'\n--- {lang_name.upper()} ---')
        lines.append(f'  {"モデル":<30}  {"CER":>7}  {"±std":>7}')
        lines.append('  ' + '-' * 45)
        for r in lang_rows:
            label = model_labels.get(r['model'], r['model'])
            lines.append(f'  {label:<30}  {r["cer_mean"]:>7.4f}  {r["cer_std"]:>7.4f}')

    # 仮説検証
    ko_rows = {r['model']: r for r in summary_rows if r['language'] == 'korean'}
    fr_rows = {r['model']: r for r in summary_rows if r['language'] == 'french'}

    if 'C_adapter' in fr_rows and 'B_full_ft' in fr_rows:
        lines.append('\n--- 仮説検証 ---')
        cer_adapter_fr = fr_rows['C_adapter']['cer_mean']
        cer_ft_fr      = fr_rows['B_full_ft']['cer_mean']
        if cer_adapter_fr < cer_ft_fr:
            diff = cer_ft_fr - cer_adapter_fr
            lines.append(f'  ✓ 仮説支持: Adapter の French CER ({cer_adapter_fr:.4f}) < Full FT ({cer_ft_fr:.4f})')
            lines.append(f'    Adapter が {diff:.4f} ({diff/cer_ft_fr*100:.1f}%) 良い')
        else:
            diff = cer_adapter_fr - cer_ft_fr
            lines.append(f'  ✗ 仮説棄却: Adapter の French CER ({cer_adapter_fr:.4f}) >= Full FT ({cer_ft_fr:.4f})')
            lines.append(f'    差: {diff:.4f}')

    report = '\n'.join(lines)
    print('\n' + report)

    out_report = RESULT_DIR / 'crosslingual_report.txt'
    out_report.write_text(report, encoding='utf-8')

    print(f'\n保存:')
    print(f'  {out_sample}')
    print(f'  {out_summary}')
    print(f'  {out_report}')


if __name__ == '__main__':
    main()
