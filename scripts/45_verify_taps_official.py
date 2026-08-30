"""
スクリプト45: TAPS公式コードと我々の実装の出力を比較
taps-baselines/models/seconformer.py の公式実装でロード・推論し、
script 37/40の実装と出力が一致するか確認する。
"""

import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
sys.path.insert(0, str(_TAPS_BASELINES))

PRETRAINED = BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'

# テスト音声
test_dir = Path(BASE_DIR / 'data' / 'raw' / 'taps' / 'throat' / 'test')
wav_path = sorted(test_dir.glob('*.wav'))[0]
print(f'テスト音声: {wav_path}')
wav, sr = sf.read(wav_path, dtype='float32')
inp = torch.from_numpy(wav)

# ── チェックポイントの中身を確認 ──
print('\n=== チェックポイント分析 ===')
state = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'model' in state:
    state = state['model']

# パラメータ名とshapeを表示
print(f'パラメータ数: {len(state)}')
for i, (k, v) in enumerate(state.items()):
    if i < 20 or 'conformer' in k:
        print(f'  {k}: {tuple(v.shape)}')
    elif i == 20:
        print(f'  ... ({len(state) - 20} more)')

# encoder最終層の出力チャネル数を推定
encoder_keys = [k for k in state if k.startswith('encoder')]
last_enc = sorted([k for k in encoder_keys if '.0.weight' in k])[-1]
print(f'\n最終encoder conv weight: {last_enc} → shape {tuple(state[last_enc].shape)}')
print(f'  → 出力チャネル数: {state[last_enc].shape[0]}')

# conformerの次元を推定
conf_keys = [k for k in state if 'conformer' in k]
for k in conf_keys[:5]:
    print(f'  {k}: {tuple(state[k].shape)}')

# ── TAPS公式コードでロード ──
print('\n=== TAPS公式コードでロード ===')
try:
    from models.seconformer import seconformer as TAPSSeconformer

    # チェックポイントから次元を推測してモデル作成
    # encoder.3.0.weight の shape[0] が最終hidden → conformer_dim
    final_hidden = state[last_enc].shape[0]
    # conformer FFN dimを推測
    ffn_keys = [k for k in state if 'ffn' in k and 'weight' in k and 'conformer' in k]
    if ffn_keys:
        ffn_shape = state[ffn_keys[0]].shape
        print(f'  FFN weight: {ffn_keys[0]} → shape {tuple(ffn_shape)}')

    # デフォルトパラメータで試行
    model_official = TAPSSeconformer()
    result = model_official.load_state_dict(state, strict=False)
    print(f'  デフォルト設定: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')
    if result.missing_keys:
        print(f'  missing: {result.missing_keys[:5]}')
    if result.unexpected_keys:
        print(f'  unexpected: {result.unexpected_keys[:5]}')

    # デフォルトで失敗したら、hidden=64で試行
    if result.missing_keys or result.unexpected_keys:
        print('\n  hidden=64, conformer_dim=512で再試行...')
        model_official = TAPSSeconformer(hidden=64, conformer_dim=512,
                                          conformer_ffn_dim=512,
                                          conformer_depth=4)
        result = model_official.load_state_dict(state, strict=False)
        print(f'  結果: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')

    # 推論
    if not result.missing_keys and not result.unexpected_keys:
        model_official.eval()
        with torch.no_grad():
            out_official = model_official(inp.unsqueeze(0)).squeeze()
        print(f'  出力: mean={out_official.mean():.4f}, std={out_official.std():.4f}, shape={tuple(out_official.shape)}')
    else:
        out_official = None
        print('  ロード失敗 → 出力比較スキップ')

except Exception as e:
    print(f'  エラー: {e}')
    out_official = None

# ── 我々の実装でロード ──
print('\n=== 我々の実装（script 40）でロード ===')
import importlib.util
spec = importlib.util.spec_from_file_location('s40', str(BASE_DIR / 'scripts' / '40_oa_postprocessing.py'))
s40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s40)

model_ours = s40.SEConformerModel()
result_ours = model_ours.load_state_dict(state, strict=False)
print(f'  missing={len(result_ours.missing_keys)}, unexpected={len(result_ours.unexpected_keys)}')
if result_ours.missing_keys:
    print(f'  missing: {result_ours.missing_keys[:5]}')

model_ours.eval()
with torch.no_grad():
    out_ours = model_ours(inp)
print(f'  出力: mean={out_ours.mean():.4f}, std={out_ours.std():.4f}, shape={tuple(out_ours.shape)}')

# ── 比較 ──
if out_official is not None:
    min_len = min(len(out_official), len(out_ours))
    diff = (out_official[:min_len] - out_ours[:min_len]).abs()
    print(f'\n=== 出力比較 ===')
    print(f'max diff: {diff.max():.6f}')
    print(f'mean diff: {diff.mean():.6f}')
    if diff.max() < 1e-4:
        print('→ 一致（実装は正しい）')
    else:
        print('→ 不一致（実装に差異あり）')
else:
    print('\n公式コードのロードに失敗したため比較不能')
