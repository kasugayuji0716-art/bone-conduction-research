"""
スクリプト45: TAPS公式コードと我々の実装の出力を比較
"""
import sys, torch, soundfile as sf
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))

PRETRAINED = BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'

test_dir = BASE_DIR / 'data' / 'raw' / 'taps' / 'throat' / 'test'
wav_path = sorted(test_dir.glob('*.wav'))[0]
print(f'テスト音声: {wav_path}')
wav, _ = sf.read(wav_path, dtype='float32')
inp = torch.from_numpy(wav)

state = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'model' in state:
    state = state['model']

# ── TAPS公式コードで正しいハイパーパラメータを試す ──
from models.seconformer import seconformer as TAPSSeconformer

configs = [
    dict(hidden=64, conformer_dim=512, conformer_ffn_dim=64,
         conformer_depth=4, depthwise_conv_kernel_size=15),
    dict(hidden=64, conformer_dim=512, conformer_ffn_dim=64,
         conformer_num_attention_heads=4,
         conformer_depth=4, depthwise_conv_kernel_size=15),
    dict(hidden=64, conformer_dim=512, conformer_ffn_dim=64,
         conformer_num_attention_heads=8,
         conformer_depth=4, depthwise_conv_kernel_size=15),
]

out_official = None
for cfg in configs:
    print(f'\n=== TAPS公式コード: {cfg} ===')
    try:
        model_official = TAPSSeconformer(**cfg)
        result = model_official.load_state_dict(state, strict=True)
        print(f'ロード成功: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')
        model_official.eval()
        with torch.no_grad():
            out_official = model_official(inp.unsqueeze(0)).squeeze()
        print(f'出力: mean={out_official.mean():.4f}, std={out_official.std():.4f}')
        break
    except Exception as e:
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + '...'
        print(f'失敗: {msg}')

# ── 我々の実装 ──
print('\n=== 我々の実装 ===')
import importlib.util
spec = importlib.util.spec_from_file_location('s40', str(BASE_DIR / 'scripts' / '40_oa_postprocessing.py'))
s40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s40)

model_ours = s40.SEConformerModel()
model_ours.load_state_dict(state, strict=True)
model_ours.eval()
with torch.no_grad():
    out_ours = model_ours(inp)
print(f'出力: mean={out_ours.mean():.4f}, std={out_ours.std():.4f}')

# ── 比較 ──
if out_official is not None:
    min_len = min(len(out_official), len(out_ours))
    diff = (out_official[:min_len] - out_ours[:min_len]).abs()
    print(f'\n=== 比較 ===')
    print(f'max diff: {diff.max():.6f}')
    print(f'mean diff: {diff.mean():.6f}')
    if diff.max() < 1e-4:
        print('→ 一致')
    else:
        print('→ 不一致')
