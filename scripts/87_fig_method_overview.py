"""
スクリプト87: 論文用の図 — 手法の全体像（ブロック図）
出力: results/figures/method_overview.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = 'Hiragino Sans'
import sys
SHORT = '--short' in sys.argv   # 4ページ版（内挿を載せない）
OUT = Path(__file__).parent.parent / 'results' / 'figures' / 'method_overview.png'
if SHORT:
    OUT = OUT.with_name('method_overview_4p.png')
NAVY, BLUE, GREY = '#1F4E79', '#2E75B6', '#555555'

fig, ax = plt.subplots(figsize=(3.3, 2.0), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')


def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc='white', ec=color, lw=1.0))
    ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=7.5, color=color, weight='bold')
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=6.0, color=GREY, linespacing=1.2)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6, lw=0.8, color=GREY))


ax.text(4.5, 26, '喉マイク\n音声', ha='center', va='center', fontsize=6.5, weight='bold', linespacing=1.2)
box(12, 33, 45, 16, '生成モデル（2.1節）', '事前学習済みモデルを TAPS で適応\n4 回生成して振幅を平均', NAVY)
box(12, 3, 45, 16, 'Whisper 用 SE（2.2節）', 'Whisper-small の CE 損失で学習' + ('' if SHORT else '\n（重みの内挿は 4.3 節）'), BLUE)
box(66, 18, 18, 16, '振幅融合', ('（2.2節）' if SHORT else '（2.3節）') + '\n位相は生成側', NAVY)
ax.text(94.5, 26, 'Whisper系\n認識器', ha='center', va='center', fontsize=6, weight='bold', linespacing=1.2)
arrow(9, 29, 12, 40); arrow(9, 23, 12, 12)
arrow(57.8, 41, 65.5, 30); arrow(57.8, 11, 65.5, 22)
ax.text(61.5, 40, '$|G|, \\angle G$', fontsize=5.6, color=GREY, ha='left')
ax.text(62.5, 10, '$|C|$', fontsize=5.6, color=GREY, ha='left')
arrow(84.8, 26, 89, 26)
fig.tight_layout(pad=0.1)
fig.savefig(OUT)
print('saved', OUT)
