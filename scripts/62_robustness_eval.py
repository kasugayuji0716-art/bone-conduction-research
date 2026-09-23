"""
スクリプト62: CE-aware SEの弱点検証（査読対策）

Part A — 非Whisper系ASRでの評価（「Whisper系列でしか確認していない」への対策）
  Whisperと構造・学習データの異なる韓国語CTCモデルで同じSE出力を評価
    - facebook/mms-1b-all (target_lang=kor)          : wav2vec2系 1B, 多言語
    - kresnik/wav2vec2-large-xlsr-korean              : XLS-R 300M, Zeroth Korean FT
  SE条件: No SE / TAPS / Enc L1 λ=5 / CE λ=0 / CE λ=10

Part B — 改善がどこから来るかの切り分け（whisper-small）
  「櫛状ピーク（約250 Hz間隔）がCER改善を担っているのか」への対策
    notch   : 250 Hzの倍数（2.75–7.75 kHz）+1/2 kHz に幅20 Hzのノッチ
    lpf4k   : 4 kHz以上を除去（喉マイクに元々ない帯域を捨てる）
    hybrid  : 4 kHzで分割し、低域/高域をTAPSとCEで入れ替え
  TAPSにも同じ処理を施し、処理そのものの影響を対照にする

出力:
  results/robustness_per_utt.csv   発話ごとのCER（途中再開可能）
  results/robustness_summary.csv   条件別平均・Wilcoxon

使い方（DNN PC）:
    python scripts/62_robustness_eval.py --limit 20     # 動作確認（出力文字の確認）
    python scripts/62_robustness_eval.py                # 本番 1000発話
    python scripts/62_robustness_eval.py --summary_only # 集計だけやり直す
"""

import argparse
import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from jiwer import cer
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SR = 16000
TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)
CKPT = BASE_DIR / 'checkpoints'
SE_CKPTS = {
    'taps': BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th',
    'enc5': CKPT / 'enc_v2_lambda_5.0' / 'best.th',
    'ce0':  CKPT / 'ce_v2_lambda_0.0' / 'best.th',
    'ce10': CKPT / 'ce_v2_lambda_10.0' / 'best.th',
}
CTC_MODELS = {
    'mms-1b-all':    'facebook/mms-1b-all',
    'xlsr-korean':   'kresnik/wav2vec2-large-xlsr-korean',
}
PART_A_CONDS = ['no_se', 'taps', 'enc5', 'ce0', 'ce10']
PART_B_CONDS = ['taps', 'ce10',
                'taps_notch', 'ce10_notch',
                'taps_lpf4k', 'ce10_lpf4k',
                'hyb_tapsLow_ceHigh', 'hyb_ceLow_tapsHigh']

OUT_UTT = BASE_DIR / 'results' / 'robustness_per_utt.csv'
OUT_SUM = BASE_DIR / 'results' / 'robustness_summary.csv'


# ---------------- 信号処理 ----------------
_SOS_LP = butter(8, 4000, btype='low', fs=SR, output='sos')
# script 61: ピークは 3.0–7.75 kHz の250 Hz格子 + 1 kHz / 2 kHz
_NOTCH_F0 = [1000, 2000] + list(np.arange(2750, 7751, 250))
_NOTCHES = [iirnotch(f0, Q=f0 / 20.0, fs=SR) for f0 in _NOTCH_F0]


def lowpass4k(x):
    return sosfiltfilt(_SOS_LP, x).astype(np.float32)


def split4k(x):
    low = lowpass4k(x)
    return low, (x - low).astype(np.float32)   # ゼロ位相なので相補的


def comb_notch(x):
    y = x.astype(np.float64)
    for b, a in _NOTCHES:
        y = filtfilt(b, a, y)
    return y.astype(np.float32)


def make_variants(taps, ce):
    n = min(len(taps), len(ce))
    taps, ce = taps[:n], ce[:n]
    tl, th = split4k(taps)
    cl, ch = split4k(ce)
    return {
        'taps_notch': comb_notch(taps), 'ce10_notch': comb_notch(ce),
        'taps_lpf4k': tl, 'ce10_lpf4k': cl,
        'hyb_tapsLow_ceHigh': tl + ch, 'hyb_ceLow_tapsHigh': cl + th,
    }


# ---------------- モデル ----------------
def load_se(path):
    from models.seconformer import seconformer
    m = seconformer(**TAPS_SE_CONFIG)
    st = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(st, dict) and 'model' in st:
        st = st['model']
    m.load_state_dict(st)
    return m.to(DEVICE).eval()


def load_ctc(name, repo):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    if name.startswith('mms'):
        proc = AutoProcessor.from_pretrained(repo, target_lang='kor')
        model = Wav2Vec2ForCTC.from_pretrained(repo, target_lang='kor',
                                               ignore_mismatched_sizes=True)
    else:
        proc = AutoProcessor.from_pretrained(repo)
        model = Wav2Vec2ForCTC.from_pretrained(repo)
    return proc, model.to(DEVICE).eval()


def ctc_transcribe(proc, model, wav):
    inp = proc(wav, sampling_rate=SR, return_tensors='pt')
    with torch.no_grad():
        logits = model(inp.input_values.to(DEVICE)).logits
    ids = logits.argmax(-1)[0].cpu()
    return proc.decode(ids).strip()


def load_whisper_small():
    from faster_whisper import WhisperModel
    ct2 = Path('/tmp/whisper-small-ct2')
    src = str(ct2) if ct2.exists() else 'small'
    print(f'  whisper-small: {src}')
    return WhisperModel(src, device=DEVICE,
                        compute_type='float16' if DEVICE == 'cuda' else 'int8')


def whisper_transcribe(asr, wav):
    segs, _ = asr.transcribe(wav, language='ko', beam_size=5)
    return ''.join(s.text for s in segs).strip()


def capped_cer(ref, hyp):
    return min(cer(ref, hyp), 1.0) if ref else 0.0


def hangul_ratio(s):
    chars = [c for c in s if not c.isspace()]
    return sum('가' <= c <= '힣' for c in chars) / max(len(chars), 1)


# ---------------- 集計 ----------------
def summarize(path=OUT_UTT):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    by = {}
    for r in rows:
        by.setdefault((r['asr'], r['cond']), {})[r['utt']] = float(r['cer'])
    out = []
    print(f'\n{"ASR":<14} {"condition":<22} {"n":>5} {"CER":>8}')
    for (asr, cond), d in sorted(by.items()):
        out.append(dict(asr=asr, cond=cond, n=len(d), cer=round(np.mean(list(d.values())), 4),
                        vs='', delta='', p=''))
        print(f'{asr:<14} {cond:<22} {len(d):>5} {np.mean(list(d.values())):>8.4f}')

    pairs = [(a, 'taps', 'ce10') for a in CTC_MODELS] + [(a, 'ce0', 'ce10') for a in CTC_MODELS] + [
        ('whisper-small', 'ce10', 'ce10_notch'), ('whisper-small', 'taps', 'taps_notch'),
        ('whisper-small', 'ce10', 'ce10_lpf4k'), ('whisper-small', 'taps', 'taps_lpf4k'),
        ('whisper-small', 'taps_lpf4k', 'ce10_lpf4k'),
        ('whisper-small', 'taps', 'hyb_tapsLow_ceHigh'), ('whisper-small', 'taps', 'hyb_ceLow_tapsHigh'),
    ]
    print(f'\n{"ASR":<14} {"A":<20} {"B":<22} {"CER A":>7} {"CER B":>7} {"B-A":>8} {"p":>10}')
    for asr, a, b in pairs:
        da, db = by.get((asr, a)), by.get((asr, b))
        if not da or not db:
            continue
        keys = sorted(set(da) & set(db))
        xa = np.array([da[k] for k in keys]); xb = np.array([db[k] for k in keys])
        p = wilcoxon(xa, xb).pvalue if np.any(xa != xb) else 1.0
        out.append(dict(asr=asr, cond=b, n=len(keys), cer=round(xb.mean(), 4),
                        vs=a, delta=round(xb.mean() - xa.mean(), 4), p=f'{p:.2e}'))
        print(f'{asr:<14} {a:<20} {b:<22} {xa.mean():>7.4f} {xb.mean():>7.4f} '
              f'{xb.mean() - xa.mean():>+8.4f} {p:>10.2e}')

    with open(OUT_SUM, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f'\nSaved: {OUT_SUM}')


# ---------------- メイン ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--summary_only', action='store_true')
    ap.add_argument('--skip_a', action='store_true')
    ap.add_argument('--skip_b', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        return summarize()

    samples = []
    with open(TAPS_DIR / 'metadata_test.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / 'test' / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                samples.append({'utt': f"{row['speaker_id']}_{row['sentence_id']}",
                                'spk': row['speaker_id'], 'path': p, 'text': row['text']})
    if args.limit:
        samples = samples[:args.limit]
    print(f'Test: {len(samples)} utterances')

    done = set()
    if OUT_UTT.exists() and not args.limit:
        for r in csv.DictReader(open(OUT_UTT, encoding='utf-8')):
            done.add((r['utt'], r['asr'], r['cond']))
        print(f'Resume: {len(done)} rows already done')

    se = {k: load_se(p) for k, p in SE_CKPTS.items()}
    print(f'  SE loaded: {list(se)}')
    ctc = {} if args.skip_a else {n: load_ctc(n, r) for n, r in CTC_MODELS.items()}
    print(f'  CTC loaded: {list(ctc)}')
    wsm = None if args.skip_b else load_whisper_small()

    out_path = OUT_UTT if not args.limit else OUT_UTT.with_name('robustness_per_utt_trial.csv')
    new_file = not out_path.exists() or args.limit
    fout = open(out_path, 'w' if args.limit else 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(fout, fieldnames=['utt', 'spk', 'asr', 'cond', 'cer', 'hyp'])
    if new_file:
        writer.writeheader()

    for i, s in enumerate(samples):
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outs = {k: m(x).squeeze().cpu().numpy().astype(np.float32) for k, m in se.items()}
        outs['no_se'] = wav

        jobs = []
        for name in ctc:
            jobs += [(name, c) for c in PART_A_CONDS]
        if wsm is not None:
            outs.update(make_variants(outs['taps'], outs['ce10']))
            jobs += [('whisper-small', c) for c in PART_B_CONDS]

        for asr, cond in jobs:
            if (s['utt'], asr, cond) in done:
                continue
            audio = outs[cond]
            if asr == 'whisper-small':
                hyp = whisper_transcribe(wsm, audio)
            else:
                hyp = ctc_transcribe(*ctc[asr], audio)
            c = capped_cer(s['text'], hyp)
            writer.writerow(dict(utt=s['utt'], spk=s['spk'], asr=asr, cond=cond,
                                 cer=f'{c:.4f}', hyp=hyp))
            if args.limit and i < 3:
                print(f'  [{s["utt"]}] {asr:<13} {cond:<20} CER={c:.3f} '
                      f'hangul={hangul_ratio(hyp):.2f} | {hyp[:40]}')
        fout.flush()
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(samples)}')
    fout.close()

    if args.limit:
        print(f'\nREF[0]: {samples[0]["text"][:60]}')
        print('確認: hangul が 0.9 前後なら出力はハングル。0 付近ならローマ字等 → CER無効')
    summarize(out_path)


if __name__ == '__main__':
    main()
