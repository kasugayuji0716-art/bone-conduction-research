"""
スクリプト83: 論文用の図 — 認識器ごとの TAPS SE 比の CER 相対変化（生成1回・生成4平均・融合）
数値は results/GENERATIVE_RESULTS_2026-09-30.md §1.1・§6.1・§7.2（test 1000発話、句読点除去 CER）
出力: results/figures/fusion_relative.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Hiragino Sans'
OUT = Path(__file__).parent.parent / 'results' / 'figures' / 'fusion_relative.png'

# 認識器: (TAPS, 生成1回, 生成4平均, 融合)
CER = {
    'Whisper-small*': (0.2297, 0.1796, 0.1685, 0.1633),
    'Whisper-base': (0.2726, 0.2178, 0.2048, 0.2011),
    'Whisper-medium': (0.1962, 0.1586, 0.1505, 0.1436),
    'large-v3-turbo': (0.1697, 0.1370, 0.1312, 0.1248),
    'FT Whisper': (0.1462, 0.1104, 0.1000, 0.0920),
    'Qwen3-ASR': (0.1359, 0.1035, 0.0968, 0.0936),
    'XLS-R': (0.2216, 0.1689, 0.1613, 0.1632),
    'MMS-1B': (0.3498, 0.2973, 0.2842, 0.2847),
    'Zipformer': (0.5422, 0.4507, 0.4328, 0.4833),
}
names = list(CER)
rel = np.array([[100 * (v / c[0] - 1) for v in c[1:]] for c in CER.values()])
labels = ['生成（1回）', '生成（4回平均）', '融合']
colors = ['#BBBBBB', '#2E75B6', '#1F4E79']

fig, ax = plt.subplots(figsize=(3.3, 3.0), dpi=300)
y = np.arange(len(names))
h = 0.26
for k in range(3):
    ax.barh(y + (k - 1) * h, rel[:, k], h, color=colors[k], label=labels[k])
ax.axhline(5.5, color='#888888', lw=0.6, ls='--')
ax.set_yticks(y, names, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('TAPS SE 比の CER 変化 [%]', fontsize=7)
ax.tick_params(axis='x', labelsize=7)
ax.set_xlim(-40, 0)
ax.legend(fontsize=6.5, loc='lower left', frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print('saved', OUT)
