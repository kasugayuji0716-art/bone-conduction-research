"""
スクリプト90: 再査読（results/reviews/asj_manuscript_rereview_2026-10-19.md）指摘6・14・15 のための確認記録

  - 論文で使った重み（SE-Conformer と生成モデル）のパス・大きさ・更新時刻・SHA-256（先頭16桁）
  - 生成モデルの各学習ログの最終行（最終ステップまで学習したか）
  - TAPS / VibraVox の test・dev の発話数と最長の音声長（15秒超の扱いの確認）
  - 論文の主な条件の評価件数（summary.csv の n）

使い方（GPU PC、GPU 不要）
    venv/bin/python scripts/90_provenance.py
出力: results/review_response/provenance.md
"""

import csv
import hashlib
import time
from pathlib import Path

import soundfile as sf

BASE_DIR = Path(__file__).parent.parent
CK = BASE_DIR / 'checkpoints'
OUT = BASE_DIR / 'results' / 'review_response' / 'provenance.md'

SE = {'TAPS SE': BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th',
      'CE-SE（元）': CK / 'ce_v2_lambda_10.0' / 'best.th',
      'λ=0 SE': CK / 'ce_v2_lambda_0.0' / 'best.th',
      'SSL-MSE': CK / 'ssl_wavlm_0.3' / 'best.th',
      '内挿 α=0.7': CK / 'wise_a0.7' / 'best.th',
      'CE-SE（再学習 s2）': CK / 'ce_s2' / 'best.th',
      'CE-SE（再学習 s3）': CK / 'ce_s3' / 'best.th',
      'VibraVox SE（再構成のみ）': CK / 'vbx_se0' / 'best.th',
      'VibraVox CE-SE': CK / 'vbx_ce10' / 'best.th',
      'VibraVox 内挿': CK / 'vbx_wise' / 'best.th'}
GEN = {'生成（元、TAPS）': 'fm_taps', '生成（再学習 s2）': 'fm_s2', '生成（再学習 s3）': 'fm_s3', '生成（VibraVox）': 'fm_vbx'}
SUMMARIES = {'TAPS': ('results/lambda_asr/summary.csv',
                      ['taps', 'ce10.0', 'ssl_wavlm_0.3', 'fm_taps', 'fm_taps_avg4', 'fm_avg4_mavg_ce10', 'wise_a0.7',
                       'fm_avg4_lmavg_wise', 'fm_avg4_mavg_taps', 'fm_avg4_mavg_ce0', 'fm_avg4_mavg_ssl',
                       'ce_s2', 'ce_s3', 'fm_s2_avg4', 'fm_s3_avg4', 'fm_s2_avg4_mavg_ce_s2', 'fm_s3_avg4_mavg_ce_s3']),
             'VibraVox': ('results/lambda_asr_vibravox/summary.csv',
                          ['no_se', 'taps', 'fm_taps_avg4', 'fm_avg4_lmavg_wise', 'vbx_se0', 'vbx_ce10', 'vbx_wise',
                           'fm_vbx_avg4', 'fm_vbx_avg4_mavg_ce10', 'fm_vbx_avg4_lmavg_wise'])}


def info(p):
    if not p.exists():
        return '（なし）', '', ''
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return f'{p.stat().st_size / 1e6:.1f} MB', time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime)), h


def rel(p):
    return str(p.relative_to(BASE_DIR)) if p.is_relative_to(BASE_DIR) else p.name


def durations(d):
    ds = [sf.info(f).duration for f in sorted(d.glob('*.wav'))]
    return len(ds), max(ds) if ds else 0.0, sum(x > 15 for x in ds)


def main():
    L = ['# 確認記録（scripts/90_provenance.py が生成）', '', f'生成日時: {time.strftime("%Y-%m-%d %H:%M")}', '',
         '## SE-Conformer の重み', '', '| 名前 | パス | 大きさ | 更新時刻 | SHA-256（先頭16桁） |', '|---|---|---|---|---|']
    for k, p in SE.items():
        L.append(f'| {k} | `{rel(p)}` | ' + ' | '.join(info(p)) + ' |')
    L += ['', '## 生成モデル（NeMo）の重み', '',
          '評価には各 tag の下で最も新しい .nemo（script 75 `find_nemo`）を用いる。TRAIN_DONE は学習コマンドが正常終了した印。', '',
          '| 名前 | .nemo | 大きさ | 更新時刻 | SHA-256 | TRAIN_DONE | 最終 ckpt |', '|---|---|---|---|---|---|---|']
    for k, tag in GEN.items():
        ns = sorted((CK / tag).rglob('*.nemo'), key=lambda p: p.stat().st_mtime)
        last = sorted((CK / tag).rglob('*-last.ckpt'), key=lambda p: p.stat().st_mtime)
        n = ns[-1] if ns else CK / tag / '(none)'
        L.append(f'| {k} | `{rel(n)}` | ' + ' | '.join(info(n)) +
                 f" | {'あり' if (CK / tag / 'TRAIN_DONE').exists() else 'なし'} | {last[-1].name if last else '—'} |")
    L += ['', '## 音声の件数と長さ（喉マイク）', '', '| データ | split | 発話数 | 最長（秒） | 15秒超 |', '|---|---|---|---|---|']
    for name, root in [('TAPS', BASE_DIR / 'data/raw/taps/throat'), ('VibraVox', BASE_DIR / 'data/raw/vibravox/throat')]:
        for sp in ['dev', 'test']:
            n, mx, over = durations(root / sp)
            L.append(f'| {name} | {sp} | {n} | {mx:.1f} | {over} |')
    L += ['', '## 主な条件の評価件数（summary.csv の n）', '']
    for name, (path, conds) in SUMMARIES.items():
        rows = list(csv.DictReader(open(BASE_DIR / path, encoding='utf-8')))
        asrs = sorted({r['asr'] for r in rows})
        L += [f'### {name}（`{path}`）', '', '| cond | ' + ' | '.join(asrs) + ' |', '|---|' + '---|' * len(asrs)]
        for c in conds:
            m = {r['asr']: r['n'] for r in rows if r['cond'] == c}
            L.append(f'| {c} | ' + ' | '.join(m.get(a, '—') for a in asrs) + ' |')
        L.append('')
    OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('saved →', OUT)


if __name__ == '__main__':
    main()
