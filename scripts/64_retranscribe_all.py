"""
スクリプト64: 論文の全CERを書き起こし文付きで再計算（句読点正規化対応）

背景: 従来のCERは句読点未正規化。TAPS正解は句読点なし、Whisperは句読点を付ける
      → 句読点の有無がCERに混入（script 63で確認: whisper-small改善の約4割）。
      従来スクリプトはhypを保存していなかったため、全条件を再推論する。
      以後、CERはhypから script 63/本スクリプトの summarize で算出する。

ステージ（優先度順・1発話ずつ追記・再開可能）:
  1. whisper-small × test × 全SE条件       … 表1(test)・アブレーション
  2. whisper-small × dev  × 全SE条件       … 表1(dev)・λ再選択
  3. whisper-ft    × test × no_se/taps/ce  … フェーズ2 FT比較（要チェックポイント）
  4. whisper-base  × test × 4条件          … 表2
  5. whisper-medium× test × 4条件          … 表2

出力:
  results/retranscribe_hyp.csv      split, utt, spk, asr, cond, hyp
  results/retranscribe_summary.csv  条件別 raw / nopunct / nospace CER、文末句点率

使い方（DNN PC）:
    python scripts/64_retranscribe_all.py                # 全ステージ（途中から再開）
    python scripts/64_retranscribe_all.py --stages 1 2   # 一部だけ
    python scripts/64_retranscribe_all.py --summary_only
"""

import argparse, csv, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
from importlib import import_module
_r63 = import_module('63_rescore_normalized')
norm, capped = _r63.norm, _r63.capped

TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
CKPT = BASE_DIR / 'checkpoints'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TAPS_SE_CONFIG = dict(hidden=64, conformer_dim=512, conformer_ffn_dim=64,
                      conformer_depth=4, depthwise_conv_kernel_size=15)

SE_CKPTS = {
    'no_se': None,
    'taps': BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th',
    **{f'ce{l}': CKPT / f'ce_v2_lambda_{l}' / 'best.th'
       for l in ['0.0', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']},
    'ce_only10.0': CKPT / 'ce_v2_only_lambda_10.0' / 'best.th',
    'enc5.0': CKPT / 'enc_v2_lambda_5.0' / 'best.th',
}
ALL_SE = list(SE_CKPTS)
CORE4 = ['no_se', 'taps', 'enc5.0', 'ce10.0']
FT_HF = CKPT / 'whisper_throat_finetuned'

STAGES = {
    1: ('whisper-small', 'test', ALL_SE),
    2: ('whisper-small', 'dev', ALL_SE),
    3: ('whisper-ft', 'test', ['no_se', 'taps', 'ce10.0']),
    4: ('whisper-base', 'test', CORE4),
    5: ('whisper-medium', 'test', CORE4),
}

OUT_HYP = BASE_DIR / 'results' / 'retranscribe_hyp.csv'
OUT_SUM = BASE_DIR / 'results' / 'retranscribe_summary.csv'
FIELDS = ['split', 'utt', 'spk', 'asr', 'cond', 'hyp']


def load_se(path):
    from models.seconformer import seconformer
    m = seconformer(**TAPS_SE_CONFIG)
    st = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(st, dict) and 'model' in st:
        st = st['model']
    m.load_state_dict(st)
    return m.to(DEVICE).eval()


def asr_source(name):
    """faster-whisper に渡すモデル。無ければ None"""
    if name == 'whisper-ft':
        ct2 = Path('/tmp/whisper_ft_ct2')
        if not ct2.exists():
            if not FT_HF.exists():
                return None
            print(f'  FT: {FT_HF} を CT2 に変換 ...')
            subprocess.run(['ct2-transformers-converter', '--model', str(FT_HF),
                            '--output_dir', str(ct2), '--quantization', 'float16', '--force'],
                           check=True)
        return str(ct2)
    size = name.split('-')[1]
    for c in (Path(f'/tmp/{name}-ct2'), Path(f'/tmp/whisper_{size}_ct2')):
        if c.exists():
            return str(c)
    return size   # faster-whisper が Systran/faster-whisper-<size> を取得


def load_samples(split):
    out = []
    with open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = TAPS_DIR / 'throat' / split / f"{row['speaker_id']}_{row['sentence_id']}.wav"
            if p.exists():
                out.append(dict(utt=f"{row['speaker_id']}_{row['sentence_id']}",
                                spk=row['speaker_id'], path=p, text=row['text']))
    return out


def read_rows(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('hyp') is not None and r.get('cond'):
                rows.append(r)
    return rows


def run_stage(sid, done, writer, fout):
    asr_name, split, conds = STAGES[sid]
    conds = [c for c in conds if SE_CKPTS[c] is None or SE_CKPTS[c].exists()]
    missing = [c for c in STAGES[sid][2] if c not in conds]
    samples = load_samples(split)
    todo = [(s, c) for s in samples for c in conds if (split, s['utt'], asr_name, c) not in done]
    print(f'\n=== Stage {sid}: {asr_name} / {split} / {len(conds)} conds '
          f'({len(todo)} jobs left){"  missing: " + str(missing) if missing else ""}')
    if not todo:
        return
    src = asr_source(asr_name)
    if src is None:
        print(f'  スキップ: FTチェックポイントなし（{FT_HF}）')
        return
    from faster_whisper import WhisperModel
    asr = WhisperModel(src, device=DEVICE, compute_type='float16' if DEVICE == 'cuda' else 'int8')
    se = {c: load_se(SE_CKPTS[c]) for c in conds if SE_CKPTS[c] is not None}

    t0, n = time.time(), 0
    by_utt = defaultdict(list)
    for s, c in todo:
        by_utt[s['utt']].append((s, c))
    for i, (utt, jobs) in enumerate(by_utt.items()):
        s = jobs[0][0]
        wav, _ = sf.read(s['path'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        for _, c in jobs:
            if c == 'no_se':
                audio = wav
            else:
                with torch.no_grad():
                    audio = se[c](x).squeeze().cpu().numpy().astype(np.float32)
            segs, _ = asr.transcribe(audio, language='ko', beam_size=5)
            hyp = ''.join(g.text for g in segs).strip()
            writer.writerow(dict(split=split, utt=utt, spk=s['spk'], asr=asr_name, cond=c, hyp=hyp))
            n += 1
        fout.flush()
        if (i + 1) % 50 == 0:
            rate = n / (time.time() - t0)
            print(f'  {i + 1}/{len(by_utt)} utts, {rate:.1f} jobs/s, '
                  f'残り約{(len(todo) - n) / rate / 60:.0f}分')
    del asr, se
    torch.cuda.empty_cache()


def summarize():
    refs = {}   # wav の有無に依存しない（Mac でも集計できるように）
    for split in ('dev', 'test'):
        with open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                refs[(split, f"{row['speaker_id']}_{row['sentence_id']}")] = row['text']
    sc = defaultdict(dict)      # (asr, split, cond) -> utt -> (raw, nopunct, nospace)
    spk_of, pend = {}, defaultdict(list)
    for r in read_rows(OUT_HYP):
        ref = refs.get((r['split'], r['utt']))
        if ref is None:
            continue
        h = r['hyp']
        sc[(r['asr'], r['split'], r['cond'])][r['utt']] = (
            capped(ref, h), capped(norm(ref), norm(h)), capped(norm(ref, 1), norm(h, 1)))
        spk_of[r['utt']] = r['spk']
        pend[(r['asr'], r['split'], r['cond'])].append(h.rstrip()[-1:] in '.?!。' if h.strip() else False)

    out = []
    print(f'\n{"ASR":<15}{"split":<6}{"cond":<13}{"n":>5}{"raw":>8}{"nopunct":>9}{"nospace":>9}{"句点":>6}')
    for k in sorted(sc):
        m = np.array(list(sc[k].values())).mean(axis=0)
        pe = np.mean(pend[k])
        out.append(dict(asr=k[0], split=k[1], cond=k[2], n=len(sc[k]), cer_raw=round(m[0], 4),
                        cer_nopunct=round(m[1], 4), cer_nospace=round(m[2], 4),
                        final_punct_rate=round(pe, 3)))
        print(f'{k[0]:<15}{k[1]:<6}{k[2]:<13}{len(sc[k]):>5}{m[0]:>8.4f}{m[1]:>9.4f}{m[2]:>9.4f}{pe:>6.0%}')
    if not out:
        print('no rows yet'); return
    with open(OUT_SUM, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'\nSaved: {OUT_SUM}')

    # λ選択（dev, whisper-small）を指標ごとに
    lam = [c for c in ALL_SE if c.startswith('ce') and not c.startswith('ce_only')]
    for mi, mname in enumerate(['raw', 'nopunct', 'nospace']):
        dev = {c: np.mean([v[mi] for v in sc[('whisper-small', 'dev', c)].values()])
               for c in lam if len(sc.get(('whisper-small', 'dev', c), {})) > 0}
        if dev:
            best = min(dev, key=dev.get)
            print(f'dev最良λ [{mname}]: {best} (dev {dev[best]:.4f})  ' +
                  ' '.join(f'{c}={v:.4f}' for c, v in dev.items()))

    # 話者単位の検定: 各ASR/split で taps → 各条件
    print(f'\n{"ASR":<15}{"split":<6}{"taps→":<13}{"metric":<9}{"taps":>7}{"B":>7}{"B-A":>8}'
          f'{"p(spk)":>8}{"改善話者":>8}')
    for (asr, split, cond) in sorted(sc):
        if cond in ('taps', 'no_se') or (asr, split, 'taps') not in sc:
            continue
        da, db = sc[(asr, split, 'taps')], sc[(asr, split, cond)]
        keys = sorted(set(da) & set(db))
        spks = sorted({spk_of[u] for u in keys})
        for mi, mname in [(0, 'raw'), (1, 'nopunct')]:
            sa = np.array([np.mean([da[u][mi] for u in keys if spk_of[u] == s]) for s in spks])
            sb = np.array([np.mean([db[u][mi] for u in keys if spk_of[u] == s]) for s in spks])
            p = wilcoxon(sa, sb).pvalue if len(spks) >= 5 and np.any(sa != sb) else float('nan')
            xa = np.mean([da[u][mi] for u in keys]); xb = np.mean([db[u][mi] for u in keys])
            print(f'{asr:<15}{split:<6}{cond:<13}{mname:<9}{xa:>7.4f}{xb:>7.4f}{xb - xa:>+8.4f}'
                  f'{p:>8.3f}{int((sb < sa).sum()):>5}/{len(spks)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', type=int, nargs='*', default=list(STAGES))
    ap.add_argument('--summary_only', action='store_true')
    args = ap.parse_args()
    if not args.summary_only:
        done = {(r['split'], r['utt'], r['asr'], r['cond']) for r in read_rows(OUT_HYP)}
        print(f'Resume: {len(done)} rows already done')
        new = not OUT_HYP.exists()
        if not new:   # 強制終了で最終行が改行なしで途切れていたら改行を補う
            data = OUT_HYP.read_bytes()
            if data and not data.endswith(b'\n'):
                with open(OUT_HYP, 'ab') as f:
                    f.write(b'\n')
        with open(OUT_HYP, 'a', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            for sid in args.stages:
                run_stage(sid, done, writer, fout)
    summarize()


if __name__ == '__main__':
    main()
