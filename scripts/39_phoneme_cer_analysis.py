"""
スクリプト39: 音素別CER分析（実験C）
韓国語テキストをハングル字母（ジャモ）に分解し、
音素カテゴリ別の誤り率を条件間で比較する。

条件: {No SE, SE-Conformer} × {Pretrained Whisper, FT Whisper}
出力:
  results/figures/mechanism_C_phoneme.png
  results/phoneme_cer_analysis.csv

DNN PC (dl-box3) で実行すること。
"""

import csv, sys, io
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import soundfile as sf
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from faster_whisper import WhisperModel as FasterWhisperModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
FIG_DIR        = BASE_DIR / 'results' / 'figures'
RESULT_DIR     = BASE_DIR / 'results'
CKPT_DIR       = BASE_DIR / 'checkpoints'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))

SR       = 16000
MODEL_ID = 'openai/whisper-small'
D_MODEL  = 768
N_SAMPLES = 500  # 500件で集計

BLUE   = '#1565C0'
GREEN  = '#2E7D32'
RED    = '#C62828'
ORANGE = '#E65100'


# ─────────────────────────────────────────────────
# ハングル字母分解
# ─────────────────────────────────────────────────

CHO  = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')   # 19種
JUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')  # 21種
JONG = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ',
        'ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ',
        'ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']  # 28種（空=終声なし）

def decompose(text):
    """韓国語テキストをジャモのリストに変換（非ハングル文字は除外）"""
    jamos = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            jong_i = offset % 28
            jung_i = (offset // 28) % 21
            cho_i  = offset // 28 // 21
            jamos.append(CHO[cho_i])
            jamos.append(JUNG[jung_i])
            if jong_i > 0:
                jamos.append(JONG[jong_i])
    return jamos


# ─────────────────────────────────────────────────
# 音素カテゴリ定義
# ─────────────────────────────────────────────────

# 摩擦音・破擦音（高域3-8kHz依存）
FRICATIVE = set('ㅅㅆㅈㅉㅊㅎ')
# 閉鎖音（破裂音）
STOP      = set('ㄱㄲㄷㄸㅂㅃㅋㅌㅍ')
# 鼻音
NASAL     = set('ㄴㅁㅇ')
# 流音
LIQUID    = set('ㄹ')
# 母音（全21種）
VOWEL     = set(JUNG)

def jamo_category(j):
    if j in FRICATIVE: return '摩擦音/破擦音\n(ㅅㅆㅈㅊㅎ)\n[高域3-8kHz依存]'
    if j in STOP:      return '閉鎖音\n(ㄱㄷㅂ等)'
    if j in NASAL:     return '鼻音\n(ㄴㅁㅇ)'
    if j in LIQUID:    return '流音\n(ㄹ)'
    if j in VOWEL:     return '母音\n(ㅏㅓㅗ等)\n[フォルマント依存]'
    return 'その他'

CATEGORY_ORDER = [
    '摩擦音/破擦音\n(ㅅㅆㅈㅊㅎ)\n[高域3-8kHz依存]',
    '閉鎖音\n(ㄱㄷㅂ等)',
    '鼻音\n(ㄴㅁㅇ)',
    '流音\n(ㄹ)',
    '母音\n(ㅏㅓㅗ等)\n[フォルマント依存]',
]

CATEGORY_SHORT = {
    '摩擦音/破擦音\n(ㅅㅆㅈㅊㅎ)\n[高域3-8kHz依存]': '摩擦音/破擦音\n(ㅅ ㅆ ㅈ ㅊ ㅎ)',
    '閉鎖音\n(ㄱㄷㅂ等)':                           '閉鎖音\n(ㄱ ㄷ ㅂ等)',
    '鼻音\n(ㄴㅁㅇ)':                               '鼻音\n(ㄴ ㅁ ㅇ)',
    '流音\n(ㄹ)':                                    '流音\n(ㄹ)',
    '母音\n(ㅏㅓㅗ等)\n[フォルマント依存]':           '母音\n(ㅏ ㅓ ㅗ等)',
}


# ─────────────────────────────────────────────────
# Levenshtein アライメント
# ─────────────────────────────────────────────────

def levenshtein_ops(ref, hyp):
    """
    ref, hyp: リスト（ジャモ列）
    戻り値: list of (op, ref_char, hyp_char)
      op in {'match', 'sub', 'del', 'ins'}
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = min edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])

    # バックトレース
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ops.append(('match', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('sub', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('del', ref[i-1], None))
            i -= 1
        else:
            ops.append(('ins', None, hyp[j-1]))
            j -= 1
    ops.reverse()
    return ops


def phoneme_error_rate(ops):
    """カテゴリ別の（エラー数, 全数）を集計"""
    counts   = defaultdict(int)  # カテゴリ → 参照ジャモ数
    errors   = defaultdict(int)  # カテゴリ → エラー数（sub+del）
    for op, ref_j, hyp_j in ops:
        if ref_j is not None:
            cat = jamo_category(ref_j)
            if cat == 'その他':
                continue
            counts[cat] += 1
            if op in ('sub', 'del'):
                errors[cat] += 1
    return counts, errors


# ─────────────────────────────────────────────────
# モデル・データ
# ─────────────────────────────────────────────────

def load_samples(n=N_SAMPLES):
    meta   = TAPS_DIR / 'metadata_test.csv'
    wdir_t = TAPS_DIR / 'throat' / 'test'
    rows   = []
    with open(meta, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sid, spk = r['sentence_id'], r['speaker_id']
            p = wdir_t / f'{spk}_{sid}.wav'
            if p.exists():
                rows.append({'sid': sid, 'spk': spk, 'path': p, 'text': r['text']})
            if len(rows) >= n:
                break
    print(f'サンプル数: {len(rows)}')
    return rows


def load_seconformer(device):
    from models.seconformer import seconformer
    sec = seconformer(
        hidden=64, depth=4, conformer_dim=512, conformer_ffn_dim=64,
        conformer_num_attention_heads=4, conformer_depth=4,
        depthwise_conv_kernel_size=15, kernel_size=8, stride=4,
        resample=4, growth=2, dropout=0.1, rescale=0.1, normalize=True
    )
    ckpt  = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    sec.load_state_dict(state, strict=False)
    return sec.eval().to(device)


def apply_se(model, wav, device):
    t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    rms_in  = np.sqrt(np.mean(wav ** 2) + 1e-12)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-12)
    return (out * rms_in / rms_out).astype(np.float32)


def transcribe_fw(wav, model):
    segs, _ = model.transcribe(wav, language='ko', beam_size=5, without_timestamps=True)
    return ''.join(s.text for s in segs)


# ─────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────

def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fw_device = 'cuda' if device.type == 'cuda' else 'cpu'
    fw_ctype  = 'float16' if fw_device == 'cuda' else 'float32'
    print(f'デバイス: {device}')

    samples = load_samples()

    # ── SE-Conformerでキャッシュ ──────────────────────────────────
    print('SE-Conformer ロード・適用中...')
    sec = load_seconformer(device)
    wavs_raw = []
    wavs_se  = []
    for i, s in enumerate(samples):
        wav, _ = sf.read(str(s['path']), dtype='float32')
        if wav.ndim > 1: wav = wav.mean(1)
        wavs_raw.append(wav)
        try:
            wavs_se.append(apply_se(sec, wav, device))
        except Exception as e:
            print(f'  警告 {s["sid"]}: {e}')
            wavs_se.append(wav)
        if (i + 1) % 100 == 0:
            print(f'  SE適用 {i+1}/{len(samples)}')
    del sec
    if device.type == 'cuda': torch.cuda.empty_cache()
    print('SE適用完了')

    # ── ASRモデルロード ───────────────────────────────────────────
    print('Pretrained Whisper ロード...')
    fw_pre = FasterWhisperModel('/tmp/whisper_small_ct2',  device=fw_device, compute_type=fw_ctype)
    print('FT済みWhisper ロード...')
    fw_ft  = FasterWhisperModel('/tmp/whisper_ft_ct2',     device=fw_device, compute_type=fw_ctype)

    # ── 4条件で推論 ───────────────────────────────────────────────
    conditions = [
        ('Pretrained + No SE',      fw_pre, wavs_raw),
        ('Pretrained + SE-Conformer', fw_pre, wavs_se),
        ('FT + No SE',              fw_ft,  wavs_raw),
        ('FT + SE-Conformer',       fw_ft,  wavs_se),
    ]

    # カテゴリ別集計
    # { condition_label: { category: (error_count, total_count) } }
    results = {}

    for label, asr_model, wavlist in conditions:
        print(f'\n推論中: {label}')
        all_counts = defaultdict(int)
        all_errors = defaultdict(int)
        for i, (wav, s) in enumerate(zip(wavlist, samples)):
            hyp = transcribe_fw(wav, asr_model)
            ref = s['text']
            ref_j = decompose(ref)
            hyp_j = decompose(hyp)
            ops   = levenshtein_ops(ref_j, hyp_j)
            cnts, errs = phoneme_error_rate(ops)
            for cat in cnts:
                all_counts[cat] += cnts[cat]
                all_errors[cat] += errs[cat]
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(samples)}')

        results[label] = {cat: (all_errors[cat], all_counts[cat]) for cat in all_counts}
        print(f'  完了: {label}')
        for cat in CATEGORY_ORDER:
            if cat in results[label]:
                err, tot = results[label][cat]
                print(f'    {CATEGORY_SHORT[cat].replace(chr(10)," ")}: {err}/{tot} = {err/tot:.3f}' if tot else f'    {cat}: N/A')

    # ── CSV保存 ────────────────────────────────────────────────
    out_csv = RESULT_DIR / 'phoneme_cer_analysis.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'category', 'errors', 'total', 'error_rate'])
        for label, cat_data in results.items():
            for cat in CATEGORY_ORDER:
                if cat in cat_data:
                    err, tot = cat_data[cat]
                    er = err / tot if tot else float('nan')
                    writer.writerow([label, CATEGORY_SHORT[cat].replace('\n', ' '), err, tot, f'{er:.4f}'])
    print(f'\n保存: {out_csv}')

    # ── 図作成 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    fig.suptitle(
        'Experiment C: Phoneme-level Error Rate Analysis\n'
        'Does SE-Conformer specifically help/hurt high-frequency phonemes (fricatives)?',
        fontsize=13, fontweight='bold'
    )

    cond_pairs = [
        ('Pretrained Whisper', 'Pretrained + No SE', 'Pretrained + SE-Conformer', BLUE, RED),
        ('FT済みWhisper',       'FT + No SE',         'FT + SE-Conformer',         GREEN, ORANGE),
    ]

    for ax, (title, cond_base, cond_se, col_base, col_se) in zip(axes, cond_pairs):
        x     = np.arange(len(CATEGORY_ORDER))
        width = 0.35

        er_base, er_se = [], []
        for cat in CATEGORY_ORDER:
            def get_er(label):
                if label in results and cat in results[label]:
                    err, tot = results[label][cat]
                    return err / tot if tot > 0 else 0
                return 0
            er_base.append(get_er(cond_base))
            er_se.append(get_er(cond_se))

        bars1 = ax.bar(x - width/2, er_base, width, color=col_base, alpha=0.8,
                       label='No SE',        edgecolor='white')
        bars2 = ax.bar(x + width/2, er_se,   width, color=col_se,   alpha=0.8,
                       label='SE-Conformer', edgecolor='white')

        # 差分アノテーション
        for xi, (b, s) in enumerate(zip(er_base, er_se)):
            diff = s - b
            sign = '+' if diff >= 0 else ''
            col  = RED if diff > 0.01 else GREEN if diff < -0.01 else 'gray'
            ax.text(xi, max(b, s) + 0.01, f'{sign}{diff:.3f}',
                    ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')

        x_labels = [CATEGORY_SHORT[c] for c in CATEGORY_ORDER]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_ylabel('Error Rate (errors / total jamo)', fontsize=10)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(max(er_base), max(er_se)) * 1.25 + 0.05)
        ax.grid(axis='y', alpha=0.3)

        # 摩擦音の列を強調
        ax.axvspan(-0.5, 0.5, alpha=0.06, color=ORANGE)
        ax.text(0, ax.get_ylim()[1] * 0.98, '← 高域依存', fontsize=8,
                ha='center', va='top', color=ORANGE, style='italic')

    plt.tight_layout()
    out_fig = FIG_DIR / 'mechanism_C_phoneme.png'
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'保存: {out_fig}')

    print('\n===== 実験C 完了 =====')
    print('  results/phoneme_cer_analysis.csv')
    print('  results/figures/mechanism_C_phoneme.png')


if __name__ == '__main__':
    main()
