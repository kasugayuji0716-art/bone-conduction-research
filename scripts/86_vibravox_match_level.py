"""
スクリプト86: VibraVox の喉マイク音声を TAPS の喉マイク音声の条件にそろえる（入力の不一致の切り分け）
  - 音量: 発話の RMS を TAPS の喉マイク test の中央値（0.0545）にそろえる（script 31 はピーク 0.9 に正規化していて約 +8 dB）
  - 帯域: 4 kHz で低域通過（TAPS の喉マイクは 8 kHz 収録で 4 kHz 以上が無い）
出力: data/raw/vibravox_lvl/{throat/test, metadata_test.csv}（BCR_DATASET=vibravox_lvl で評価）
"""
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

BASE_DIR = Path(__file__).parent.parent
SRC = BASE_DIR / 'data' / 'raw' / 'vibravox'
DST = BASE_DIR / 'data' / 'raw' / 'vibravox_lvl'
TARGET_RMS = 0.0545
SOS = butter(8, 4000, btype='low', fs=16000, output='sos')

out = DST / 'throat' / 'test'
out.mkdir(parents=True, exist_ok=True)
shutil.copy(SRC / 'metadata_test.csv', DST / 'metadata_test.csv')
for f in sorted((SRC / 'throat' / 'test').glob('*.wav')):
    x, sr = sf.read(f, dtype='float32')
    y = sosfiltfilt(SOS, x)
    y = y * TARGET_RMS / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    sf.write(out / f.name, y.astype(np.float32), sr)
print('done →', DST)
