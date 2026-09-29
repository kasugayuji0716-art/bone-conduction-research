"""
スクリプト66b: results/lambda_asr/hyp_*.csv の整理（2026-09-29）
- 同じ (cond, utt) の重複行を、最初の1行だけ残して削除（Qwen3-ASR で CE 評価と蒸留評価が同時に同じファイルへ書いたため）
- cond=distill_m4_med の行を削除（train 教師の蒸留は dev 損失が一度も下がらず、保存されたのは TAPS そのもの。評価も途中で停止）
集計（script 66 summary）はもともと (utt, spk) ごとに1値にまとめるため、既報の数値への影響はない
"""
import csv, glob, sys
from pathlib import Path
base = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent)
for p in sorted(glob.glob(str(base / 'results' / 'lambda_asr' / 'hyp_*.csv'))):
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    seen, keep = set(), []
    for r in rows:
        k = (r['cond'], r['utt'])
        if r['cond'] == 'distill_m4_med' or k in seen:
            continue
        seen.add(k); keep.append(r)
    if len(keep) != len(rows):
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(keep)
        print(f'{Path(p).name}: {len(rows)} → {len(keep)}')
