"""
script 37 と script 40 の SEConformer 出力を比較するデバッグスクリプト
"""
import sys, torch, soundfile as sf
sys.path.insert(0, 'taps-baselines')

import importlib.util

spec = importlib.util.spec_from_file_location('s37', 'scripts/37_evaluate_taps_se_baselines.py')
s37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s37)

spec2 = importlib.util.spec_from_file_location('s40', 'scripts/40_oa_postprocessing.py')
s40 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(s40)

wav, _ = sf.read('data/raw/taps/throat/test/p00_s001.wav', dtype='float32')
inp = torch.from_numpy(wav)

PRETRAINED = 'taps-baselines/pretrained/seconformer.th'
state = torch.load(PRETRAINED, map_location='cpu')
if isinstance(state, dict) and 'model' in state:
    state = state['model']

m37 = s37.SEConformerModel()
m40 = s40.SEConformerModel()
r37 = m37.load_state_dict(state, strict=False)
r40 = m40.load_state_dict(state, strict=False)

print('=== 重みロード結果 ===')
print(f's37 missing: {r37.missing_keys}')
print(f's37 unexpected: {r37.unexpected_keys}')
print(f's40 missing: {r40.missing_keys}')
print(f's40 unexpected: {r40.unexpected_keys}')

m37.eval()
m40.eval()
with torch.no_grad():
    o37 = m37(inp.unsqueeze(0).unsqueeze(1)).squeeze()
    o40 = m40(inp)

print('\n=== 出力比較 ===')
print(f's37: mean={o37.mean():.4f}  std={o37.std():.4f}  shape={tuple(o37.shape)}')
print(f's40: mean={o40.mean():.4f}  std={o40.std():.4f}  shape={tuple(o40.shape)}')
min_len = min(len(o37), len(o40))
print(f'max diff: {(o37[:min_len] - o40[:min_len]).abs().max():.6f}')
