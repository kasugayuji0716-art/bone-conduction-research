"""
CER cap@1.0 後処理スクリプト

crosslingual_per_sample.csv の CER 値を上限1.0でキャップし、
サマリーとレポートを再計算して上書きする。

背景:
  Whisper がハルシネーションを起こすと CER > 1.0 になる（出力文字数 > 参照文字数）。
  これは統計的外れ値として扱い、CER = 1.0 にクリップして平均・標準偏差を再計算。

実行:
  python3 scripts/33_postprocess_cap_cer.py
"""

import csv
import json
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).parent.parent
RESULT_DIR = BASE_DIR / 'results'

IN_SAMPLE   = RESULT_DIR / 'crosslingual_per_sample.csv'
OUT_SAMPLE  = RESULT_DIR / 'crosslingual_per_sample_capped.csv'
OUT_SUMMARY = RESULT_DIR / 'crosslingual_summary.csv'
OUT_REPORT  = RESULT_DIR / 'crosslingual_report.txt'

CAP = 1.0

MODEL_LABELS = {
    'A_pretrained': 'A: Pretrained Whisper',
    'B_full_ft':    'B: Full FT (Korean)',
    'C_adapter':    'C: Encoder-only Adapter',
    'D_lora':       'D: Enc+Dec LoRA (Korean)',
}


def main():
    # ── per-sample 読み込み ──────────────────────────────────────
    with open(IN_SAMPLE, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    model_keys = [k for k in fieldnames
                  if k not in ('language', 'speaker_id', 'sentence_id', 'reference')]

    print(f'サンプル数: {len(rows)}')
    print(f'モデル: {model_keys}')

    # ── CER > 1.0 の件数を報告 ──────────────────────────────────
    print('\n--- CER > 1.0 件数 (上限クリップ前) ---')
    for lang in ['korean', 'french']:
        lang_rows = [r for r in rows if r['language'] == lang]
        for mk in model_keys:
            over = sum(1 for r in lang_rows if float(r[mk]) > CAP)
            pct  = over / len(lang_rows) * 100 if lang_rows else 0
            print(f'  {lang:7s}  {mk:<15s}  {over:4d} / {len(lang_rows)} ({pct:.1f}%)')

    # ── CER をキャップ ───────────────────────────────────────────
    capped_rows = []
    for r in rows:
        new_r = dict(r)
        for mk in model_keys:
            val = min(float(r[mk]), CAP)
            new_r[mk] = round(val, 4)
        capped_rows.append(new_r)

    # ── capped per-sample を保存 ─────────────────────────────────
    with open(OUT_SAMPLE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(capped_rows)
    print(f'\ncapped per-sample -> {OUT_SAMPLE}')

    # ── サマリー再計算 ───────────────────────────────────────────
    summary_rows = []
    for lang in ['korean', 'french']:
        lang_rows = [r for r in capped_rows if r['language'] == lang]
        if not lang_rows:
            continue
        for mk in model_keys:
            vals = [r[mk] for r in lang_rows]
            summary_rows.append({
                'language':  lang,
                'model':     mk,
                'cer_mean':  round(float(np.mean(vals)), 4),
                'cer_std':   round(float(np.std(vals)),  4),
                'cer_median':round(float(np.median(vals)), 4),
                'n_samples': len(vals),
            })

    with open(OUT_SUMMARY, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f, fieldnames=['language','model','cer_mean','cer_std','cer_median','n_samples'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f'summary (capped) -> {OUT_SUMMARY}')

    # ── レポート生成 ─────────────────────────────────────────────
    lines = []
    lines.append('=' * 70)
    lines.append('  Cross-lingual Evaluation Report  [CER capped @ 1.0]')
    lines.append('  Training: Korean TAPS throat mic')
    lines.append('  Evaluation: Korean TAPS test (1,000) + French VibraVox test (3,064)')
    lines.append('  Note: CER values capped at 1.0 to remove Whisper hallucination outliers')
    lines.append('=' * 70)

    for lang in ['korean', 'french']:
        lang_rows = [r for r in summary_rows if r['language'] == lang]
        if not lang_rows:
            continue
        lines.append(f'\n--- {lang.upper()} ---')
        lines.append(f'  {"モデル":<32}  {"CER":>7}  {"±std":>7}  {"median":>7}')
        lines.append('  ' + '-' * 55)
        for r in lang_rows:
            label = MODEL_LABELS.get(r['model'], r['model'])
            lines.append(
                f'  {label:<32}  {r["cer_mean"]:>7.4f}  {r["cer_std"]:>7.4f}  {r["cer_median"]:>7.4f}'
            )

    # 仮説検証
    ko_rows = {r['model']: r for r in summary_rows if r['language'] == 'korean'}
    fr_rows = {r['model']: r for r in summary_rows if r['language'] == 'french'}

    lines.append('\n--- 仮説検証 (cap@1.0) ---')
    if 'C_adapter' in fr_rows and 'B_full_ft' in fr_rows:
        ca_fr = fr_rows['C_adapter']['cer_mean']
        ft_fr = fr_rows['B_full_ft']['cer_mean']
        if ca_fr < ft_fr:
            diff = ft_fr - ca_fr
            lines.append(f'  ✓ 仮説支持: Adapter ({ca_fr:.4f}) < Full FT ({ft_fr:.4f}), 差={diff:.4f}')
        else:
            diff = ca_fr - ft_fr
            lines.append(f'  ✗ 仮説棄却: Adapter ({ca_fr:.4f}) >= Full FT ({ft_fr:.4f}), 差={diff:.4f}')

    if 'D_lora' in fr_rows and 'B_full_ft' in fr_rows:
        lo_fr = fr_rows['D_lora']['cer_mean']
        ft_fr = fr_rows['B_full_ft']['cer_mean']
        if lo_fr < ft_fr:
            lines.append(f'  LoRA ({lo_fr:.4f}) < Full FT ({ft_fr:.4f})')
        else:
            lines.append(f'  LoRA ({lo_fr:.4f}) >= Full FT ({ft_fr:.4f})')

    # 主要な発見をまとめ
    lines.append('\n--- 主要発見 ---')
    lines.append('  In-domain (Korean):')
    for mk in model_keys:
        if mk in ko_rows:
            r = ko_rows[mk]
            label = MODEL_LABELS.get(mk, mk)
            lines.append(f'    {label:<32}: CER={r["cer_mean"]:.4f} ±{r["cer_std"]:.4f}')

    lines.append('  Zero-shot transfer (French):')
    for mk in model_keys:
        if mk in fr_rows:
            r = fr_rows[mk]
            label = MODEL_LABELS.get(mk, mk)
            lines.append(f'    {label:<32}: CER={r["cer_mean"]:.4f} ±{r["cer_std"]:.4f}  median={r["cer_median"]:.4f}')

    report = '\n'.join(lines)
    print('\n' + report)
    OUT_REPORT.write_text(report, encoding='utf-8')
    print(f'\nreport -> {OUT_REPORT}')


if __name__ == '__main__':
    main()
