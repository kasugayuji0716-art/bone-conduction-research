"""
フェーズ3追加実験: TAPSベースラインSEモデルの音声強調 + ASR評価

喉マイク向けに設計された3つのSEモデル（Demucs・SE-conformer・TSTNN）を
TAPSテストセットに適用し、CERを計測する。

比較対象:
  A: Pretrained Whisper (No SE)         ← 既存結果 (0.4896)
  B: Full FT Whisper  (No SE)           ← 既存結果 (0.1498)
  C: Encoder Adapter  (No SE)           ← 既存結果 (0.1469)
  D: Pretrained Whisper + Demucs
  E: Pretrained Whisper + SE-conformer
  F: Pretrained Whisper + TSTNN
  G: Full FT Whisper + Demucs
  H: Full FT Whisper + SE-conformer
  I: Full FT Whisper + TSTNN

前提:
  - DNN PC (CUDA) 推奨
  - taps-baselines を自動クローン
  - 事前学習済みチェックポイントは Google Drive から gdown でダウンロード

実行:
  python3 scripts/37_evaluate_taps_se_baselines.py
  python3 scripts/37_evaluate_taps_se_baselines.py --skip_enhance   # SE済みWAVが既にある場合
  python3 scripts/37_evaluate_taps_se_baselines.py --ft_only        # FT/Adapter評価のみ
"""

import csv
import sys
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

BASE_DIR    = Path(__file__).parent.parent
TAPS_DIR    = BASE_DIR / 'data' / 'raw' / 'taps'
SE_OUT_DIR  = BASE_DIR / 'data' / 'processed' / 'taps_se_baselines'
CKPT_FT     = BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned'
CKPT_ADP    = BASE_DIR / 'checkpoints' / 'whisper_encoder_adapter'
RESULTS_DIR = BASE_DIR / 'results'
TAPS_BL_DIR = BASE_DIR / 'taps-baselines'

GDRIVE_FOLDER_ID = '133hBcBob8wJ-WaV7qLNj9G3eqdzyd2vX'
TARGET_SR = 16000
SE_MODELS = ['demucs', 'seconformer', 'tstnn']

# ── Adapter モジュール（29_finetune_whisper_adapter.py と同一） ───────────────
import torch.nn as nn

D_MODEL = 768


class Adapter(nn.Module):
    def __init__(self, d_model=D_MODEL, r=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

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


# ── Step 1: taps-baselines セットアップ ──────────────────────────────────────

def setup_taps_baselines():
    if not TAPS_BL_DIR.exists():
        print('taps-baselines をクローン中...')
        subprocess.run(
            ['git', 'clone', 'https://github.com/yskim3271/taps-baselines.git',
             str(TAPS_BL_DIR)],
            check=True
        )
        print('クローン完了')
    else:
        print(f'taps-baselines 既存: {TAPS_BL_DIR}')

    # 必要ライブラリをインストール（torch 系は除外）
    req_file = TAPS_BL_DIR / 'requirements.txt'
    if req_file.exists():
        reqs = [l.strip() for l in req_file.read_text().splitlines()
                if l.strip() and not l.startswith('#')
                and 'torch' not in l.lower()]
        if reqs:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + reqs,
                           check=False)

    sys.path.insert(0, str(TAPS_BL_DIR))


def download_checkpoints() -> Path:
    ckpt_root = TAPS_BL_DIR / 'pretrained'
    if ckpt_root.exists() and list(ckpt_root.rglob('*.th')):
        print(f'チェックポイント既存: {ckpt_root}')
        return ckpt_root

    print('チェックポイントを Google Drive からダウンロード中...')
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'],
                       check=True)
        import gdown

    ckpt_root.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(ckpt_root), quiet=False)
        print('ダウンロード完了')
    except Exception as e:
        print(f'\n自動ダウンロード失敗: {e}')
        print('=' * 60)
        print('手動ダウンロード手順:')
        print(f'  1. https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}')
        print(f'  2. 全ファイルを {ckpt_root} に配置')
        print('  3. 期待する構造:')
        for m in SE_MODELS:
            print(f'       {ckpt_root}/{m}/checkpoint.th')
        print('  4. スクリプトを再実行')
        print('=' * 60)
        raise RuntimeError('チェックポイントのダウンロードに失敗')

    return ckpt_root


def find_checkpoint(ckpt_root: Path, model_name: str) -> Path:
    candidates = [
        ckpt_root / model_name / 'checkpoint.th',
        ckpt_root / model_name / 'best.th',
        ckpt_root / f'{model_name}.th',
        ckpt_root / f'{model_name}_checkpoint.th',
    ]
    for c in candidates:
        if c.exists():
            return c
    # フォールバック: rglob
    for f in ckpt_root.rglob('*.th'):
        if model_name in str(f).lower():
            return f
    raise FileNotFoundError(
        f'{model_name} のチェックポイントが {ckpt_root} に見つかりません'
    )


# ── Step 2: SE 強調 ───────────────────────────────────────────────────────────

def build_se_model(model_name: str):
    import importlib
    from omegaconf import OmegaConf

    conf = OmegaConf.load(TAPS_BL_DIR / 'conf' / f'config_{model_name}.yaml')
    model_args = conf.model
    module = importlib.import_module(f'models.{model_args.model_name}')
    model_class = getattr(module, model_args.model_name)
    return model_class(**model_args.param)


def enhance_test_set(model_name: str, ckpt_root: Path, device: str):
    out_dir = SE_OUT_DIR / model_name / 'test'
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob('*.wav'))
    if len(existing) >= 1000:
        print(f'  [{model_name}] 強調済み {len(existing)} 件 → スキップ')
        return

    print(f'  [{model_name}] モデルロード中...')
    model = build_se_model(model_name).to(device)
    ckpt_path = find_checkpoint(ckpt_root, model_name)
    print(f'  [{model_name}] チェックポイント: {ckpt_path}')
    chkpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.eval()

    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / \
                f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'id': f"{row['speaker_id']}_{row['sentence_id']}",
                                 'path': p})

    print(f'  [{model_name}] {len(samples)} 件を強調中...')
    with torch.no_grad():
        for i, s in enumerate(samples):
            out_path = out_dir / f"{s['id']}.wav"
            if out_path.exists():
                continue

            wav, _ = sf.read(str(s['path']), dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device)
            enhanced = model(x)
            if enhanced.dim() == 3:
                enhanced = enhanced.squeeze(1)
            enhanced = enhanced.squeeze(0).cpu().numpy()

            peak = np.abs(enhanced).max()
            if peak > 1.0:
                enhanced = enhanced / peak

            sf.write(str(out_path), enhanced, TARGET_SR)

            if (i + 1) % 200 == 0:
                print(f'    {i+1}/{len(samples)}')

    print(f'  [{model_name}] 完了 → {out_dir}')


# ── Step 3: ASR 評価 ──────────────────────────────────────────────────────────

def load_meta(split='test'):
    rows = []
    with open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_cer_faster_whisper(fw_model, audio_dir: Path, meta_rows: list) -> float:
    import jiwer
    preds, refs = [], []
    for row in meta_rows:
        p = audio_dir / f"{row['speaker_id']}_{row['sentence_id']}.wav"
        if not p.exists():
            continue
        segs, _ = fw_model.transcribe(str(p), language='ko', beam_size=5)
        preds.append(''.join(s.text for s in segs).strip())
        refs.append(row['text'].strip())
    return jiwer.cer(refs, preds)


def compute_cer_hf(model, processor, audio_dir: Path, meta_rows: list,
                   device: str) -> float:
    import jiwer
    preds, refs = [], []
    for row in meta_rows:
        p = audio_dir / f"{row['speaker_id']}_{row['sentence_id']}.wav"
        if not p.exists():
            continue
        wav, _ = sf.read(str(p), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        inputs = processor.feature_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors='pt'
        ).input_features
        if device == 'cuda':
            inputs = inputs.half().to('cuda')

        with torch.no_grad():
            ids = model.generate(inputs, language='ko', task='transcribe',
                                 max_new_tokens=225)
        pred = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        preds.append(pred.strip())
        refs.append(row['text'].strip())

    return jiwer.cer(refs, preds)


def load_ft_model(device):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc  = WhisperProcessor.from_pretrained(str(CKPT_FT))
    model = WhisperForConditionalGeneration.from_pretrained(str(CKPT_FT))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    model.eval()
    if device == 'cuda':
        model = model.half().to('cuda')
    return model, proc


def load_adapter_model(device):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    with open(CKPT_ADP / 'adapter_config.json') as f:
        adp_conf = json.load(f)
    r = adp_conf['r']

    proc  = WhisperProcessor.from_pretrained(str(CKPT_ADP / 'processor'))
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    layers = model.model.encoder.layers
    for i in range(len(layers)):
        layers[i] = WhisperEncoderLayerWithAdapter(layers[i], Adapter(D_MODEL, r))

    model.load_state_dict(
        torch.load(CKPT_ADP / 'adapter_weights.pt', map_location='cpu'),
        strict=False
    )
    model.eval()
    if device == 'cuda':
        model = model.half().to('cuda')
    return model, proc


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_enhance', action='store_true')
    parser.add_argument('--ft_only', action='store_true',
                        help='FT/Adapter Whisper のみ評価（faster-whisper 不要）')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    print(f'デバイス: {device}')

    meta_rows = load_meta('test')
    throat_test = TAPS_DIR / 'throat' / 'test'

    # ── Setup & SE 強調 ───────────────────────────────────────────────────────
    setup_taps_baselines()
    ckpt_root = download_checkpoints()

    if not args.skip_enhance:
        print('\n== SE 強調 ==')
        for m in SE_MODELS:
            try:
                enhance_test_set(m, ckpt_root, device)
            except Exception as e:
                print(f'  [{m}] エラー: {e}')

    # ── 評価対象の列挙 ────────────────────────────────────────────────────────
    eval_dirs = [('No SE', throat_test)]
    for m in SE_MODELS:
        d = SE_OUT_DIR / m / 'test'
        if d.exists() and list(d.glob('*.wav')):
            eval_dirs.append((f'SE:{m}', d))
        else:
            print(f'  [{m}] 強調済みWAVなし → スキップ')

    results = []

    def record(cond, asr, cer):
        results.append({'condition': cond, 'asr_model': asr, 'cer': round(cer, 4)})
        print(f'    CER={cer:.4f}  ({cond} | {asr})')

    # Pretrained Whisper (faster-whisper)
    if not args.ft_only:
        print('\n-- Pretrained Whisper-small --')
        try:
            from faster_whisper import WhisperModel
            fw = WhisperModel('small', device=device,
                              compute_type='float16' if device == 'cuda' else 'float32')
            for cond, d in eval_dirs:
                cer = compute_cer_faster_whisper(fw, d, meta_rows)
                record(cond, 'Pretrained', cer)
            del fw
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'  エラー: {e}')

    # FT Whisper
    if CKPT_FT.exists():
        print('\n-- FT Whisper-small --')
        try:
            m_ft, p_ft = load_ft_model(device)
            for cond, d in eval_dirs:
                cer = compute_cer_hf(m_ft, p_ft, d, meta_rows, device)
                record(cond, 'FT', cer)
            del m_ft, p_ft
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'  エラー: {e}')
    else:
        print(f'FT チェックポイントなし: {CKPT_FT}')

    # Adapter Whisper
    if CKPT_ADP.exists() and (CKPT_ADP / 'adapter_weights.pt').exists():
        print('\n-- Encoder Adapter Whisper-small --')
        try:
            m_adp, p_adp = load_adapter_model(device)
            for cond, d in eval_dirs:
                cer = compute_cer_hf(m_adp, p_adp, d, meta_rows, device)
                record(cond, 'Adapter', cer)
            del m_adp, p_adp
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'  エラー: {e}')
    else:
        print(f'Adapter チェックポイントなし: {CKPT_ADP}')

    # ── 結果保存 ──────────────────────────────────────────────────────────────
    if not results:
        print('\n評価結果なし')
        return

    out_csv = RESULTS_DIR / 'taps_se_comparison.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'asr_model', 'cer'])
        w.writeheader()
        w.writerows(results)
    print(f'\n結果保存: {out_csv}')

    # サマリー
    print('\n' + '=' * 60)
    print(f'{"ASRモデル":<18} {"条件":<20} {"CER":>6}')
    print('-' * 60)
    for asr in ['Pretrained', 'FT', 'Adapter']:
        for r in results:
            if r['asr_model'] == asr:
                print(f'{r["asr_model"]:<18} {r["condition"]:<20} {r["cer"]:>6.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
