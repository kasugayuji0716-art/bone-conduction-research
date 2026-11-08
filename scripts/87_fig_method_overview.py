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
LARGE = '--large' in sys.argv   # paper_asj7: 文字を大きくした版（短い説明文、内挿なし）
OUT = Path(__file__).parent.parent / 'results' / 'figures' / 'method_overview.png'
if SHORT:
    OUT = OUT.with_name('method_overview_4p.png')
if LARGE:
    SHORT = True
    OUT = OUT.with_name('method_overview_large.png')
FS = (8.0, 7.8, 8.5, 7.5) if LARGE else (7.5, 6.0, 6.5, 5.6)   # 箱の題・箱の説明・端の語・記号
NAVY, BLUE, GREY = '#1F4E79', '#2E75B6', '#555555'

if LARGE:
    # paper_asj7 用の配置: 図を縦に広げ、最小の文字を 9 pt 前後にする（1 段幅で約 0.9 倍に縮小されて約 8 pt）
    FS = (9.0, 9.0, 9.5, 9.0)
    fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc='white', ec=color, lw=1.1))
        ax.text(x + w / 2, y + h * 0.72, title, ha='center', va='center', fontsize=FS[0], color=color, weight='bold', linespacing=1.15)
        ax.text(x + w / 2, y + h * 0.30, sub, ha='center', va='center', fontsize=FS[1], color=GREY, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        # 箱の角丸の pad（0.4）の外側で止め、矢印が枠に重ならないようにする
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=1.0, color=GREY,
                                     shrinkA=0, shrinkB=0))

    ax.text(5, 35, '喉マイク\n音声', ha='center', va='center', fontsize=FS[2], weight='bold', linespacing=1.2)
    box(12.5, 43, 49, 24, '生成モデル（2.1節）', 'TAPSで適応\n4回生成し振幅を平均', NAVY)
    box(12.5, 3, 49, 24, 'Whisper用SE（2.2節）', 'Whisper-smallの\nCE損失で学習', BLUE)
    box(67, 17, 18, 36, '振幅の\n融合', '位相は\n生成側', NAVY)
    ax.text(95, 35, '認識器\n9種', ha='center', va='center', fontsize=FS[2], weight='bold', linespacing=1.2)
    arrow(9.5, 39, 11.0, 50); arrow(9.5, 31, 11.0, 20)
    arrow(63.0, 55, 65.4, 45); arrow(63.0, 15, 65.4, 25)
    ax.text(62.5, 61, '$|G|, \\angle G$', fontsize=FS[3], color=GREY, ha='left')
    ax.text(62.5, 6, '$|C|$', fontsize=FS[3], color=GREY, ha='left')
    arrow(86.6, 35, 89.5, 35)
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT)
    print('saved', OUT)
    sys.exit()

fig, ax = plt.subplots(figsize=(3.3, 2.0), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')


def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc='white', ec=color, lw=1.0))
    ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=FS[0], color=color, weight='bold')
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=FS[1], color=GREY, linespacing=1.2)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6, lw=0.8, color=GREY))


ax.text(4.5, 26, '喉マイク\n音声', ha='center', va='center', fontsize=FS[2], weight='bold', linespacing=1.2)
box(12, 33, 45, 16, '生成モデル（2.1節）', ('TAPS で適応\n4 回生成し振幅を平均' if LARGE else '事前学習済みモデルを TAPS で適応\n4 回生成して振幅を平均'), NAVY)
box(12, 3, 45, 16, 'Whisper用SE（2.2節）' if LARGE else 'Whisper 用 SE（2.2節）', ('Whisper-small の\nCE 損失で学習' if LARGE else 'Whisper-small の CE 損失で学習' + ('' if SHORT else '\n（重みの内挿は 4.3 節）')), BLUE)
if LARGE:
    box(63, 12, 22, 28, '振幅の\n融合', '位相は\n生成側', NAVY)
else:
  box(66, 18, 18, 16, '振幅融合', ('（2.2節）' if SHORT else '（2.3節）') + '\n位相は生成側', NAVY)
ax.text(94.5, 26, '認識器\n（9種）', ha='center', va='center', fontsize=FS[2] if LARGE else 6, weight='bold', linespacing=1.2)
arrow(9, 29, 12, 40); arrow(9, 23, 12, 12)
arrow(57.8, 41, 62.5 if LARGE else 65.5, 30); arrow(57.8, 11, 62.5 if LARGE else 65.5, 22)
ax.text(58.5 if LARGE else 61.5, 44 if LARGE else 40, '$|G|, \\angle G$', fontsize=FS[3], color=GREY, ha='left')
ax.text(58.5 if LARGE else 62.5, 6 if LARGE else 10, '$|C|$', fontsize=FS[3], color=GREY, ha='left')
arrow(85.8 if LARGE else 84.8, 26, 89, 26)
fig.tight_layout(pad=0.1)
fig.savefig(OUT)
print('saved', OUT)
