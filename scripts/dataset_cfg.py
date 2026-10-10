"""
評価に使うデータセットの切り替え（環境変数 BCR_DATASET、既定 taps）

  BCR_DATASET=vibravox python scripts/66_lambda_cross_asr.py asr --asr whisper-small --set ...
とすると、喉マイク音声・正解は data/raw/vibravox（script 31 の出力と同じ形）から読み、
SE の出力は data/processed/se_wav_vibravox、認識結果は results/lambda_asr_vibravox に分けて書く。
認識の言語はフランス語にし、CER の正規化で小文字にそろえる（TAPS の結果には影響しない）。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASET = os.environ.get('BCR_DATASET', 'taps')
assert DATASET == 'taps' or DATASET.startswith('vibravox'), DATASET   # vibravox_lvl: 音量・帯域を TAPS にそろえた版（script 86）
RAW_DIR = BASE_DIR / 'data' / 'raw' / DATASET
SUFFIX = '' if DATASET == 'taps' else f'_{DATASET}'
SE_WAV = BASE_DIR / 'data' / 'processed' / f'se_wav{SUFFIX}'
LANG, LANG_NAME = ('ko', 'Korean') if DATASET == 'taps' else ('fr', 'French')
LOWER = DATASET != 'taps'
