"""
スクリプト82: 比較対象 — Dissen et al.（Interspeech 2024 / TASLP 2025）型の mel 領域フロントエンド

Whisper の log-mel（80本、Whisper の正規化済み特徴）を入力・出力する U-Net（残差）を、
凍結 Whisper-small の CE ＋ 気導音の log-mel との L1 で学習する。出力の mel を Whisper に直接入れて認識する。
→ 入力が 80本 mel の Whisper（base/small/medium/FT）でしか使えない（large-v3-turbo・Qwen3 は 128本）。

評価は transformers の Whisper（貪欲デコード）で行い、デコードの違いをそろえるため、
TAPS・融合など波形の条件も同じ経路（波形 → Whisper の log-mel → 認識）で認識し直す。

使い方（GPU PC、venv）
    python scripts/82_dissen_mel_frontend.py train [--epochs 10]
    python scripts/82_dissen_mel_frontend.py eval --asr whisper-small [--conds ...]
    python scripts/82_dissen_mel_frontend.py summary
出力: checkpoints/dissen_mel/best.pt、results/dissen/hyp_<asr>.csv, summary.csv, pairs.csv
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
SE_WAV = BASE_DIR / 'data' / 'processed' / 'se_wav'
OUT = BASE_DIR / 'results' / 'dissen'
CK = BASE_DIR / 'checkpoints' / 'dissen_mel'
DEVICE = 'cuda'
s51 = import_module('51_train_se_ce_loss')
s64 = import_module('64_retranscribe_all')

HF = {'whisper-small': 'openai/whisper-small', 'whisper-base': 'openai/whisper-base',
      'whisper-medium': 'openai/whisper-medium', 'whisper-ft': str(BASE_DIR / 'checkpoints' / 'whisper_throat_finetuned')}
WAV_CONDS = ['no_se', 'taps', 'ce10.0', 'fm_taps_avg4', 'fm_avg4_mavg_ce10']


class UNet(nn.Module):
    """log-mel (B, 80, T) → 残差を足した log-mel。2段の down/up、約 7M パラメータ"""
    def __init__(self, ch=64):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.e1, self.e2, self.e3 = blk(1, ch), blk(ch, 2 * ch), blk(2 * ch, 4 * ch)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * ch, 2 * ch, 2, 2), blk(4 * ch, 2 * ch)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * ch, ch, 2, 2), blk(2 * ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # 最初は「何もしない」

    def forward(self, m):
        x = m[:, None]
        h1 = self.e1(x); h2 = self.e2(F.max_pool2d(h1, 2)); h3 = self.e3(F.max_pool2d(h2, 2))
        g2 = self.d2(torch.cat([self.u2(h3), h2], 1)); g1 = self.d1(torch.cat([self.u1(g2), h1], 1))
        return m + self.out(g1)[:, 0]


def pairs(split, max_sec):
    rows = []
    for r in csv.DictReader(open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8')):
        if float(r.get('duration', 0)) > max_sec:
            continue
        n = f"{r['speaker_id']}_{r['sentence_id']}.wav"
        rows.append((TAPS_DIR / 'throat' / split / n, TAPS_DIR / 'acoustic' / split / n, r['text']))
    return rows


def train(args):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    global CK
    CK = BASE_DIR / 'checkpoints' / args.tag
    torch.manual_seed(0); random.seed(0); CK.mkdir(parents=True, exist_ok=True)
    proc = WhisperProcessor.from_pretrained('openai/whisper-small')
    proc.tokenizer.set_prefix_tokens(language='korean', task='transcribe')  # 学習ラベルの先頭に <|ko|><|transcribe|> を入れる（2026-10-05 まではこれが抜けていた）
    wm = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(DEVICE).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    tok, start = proc.tokenizer, wm.config.decoder_start_token_id
    logmel = s51.WhisperLogMel().to(DEVICE)
    net = UNet().to(DEVICE)
    print(f'U-Net params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M', flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    tr, dv = pairs('train', args.max_sec), pairs('dev', args.max_sec)[::5]

    def batch(rows):
        xs, ys, texts = [], [], []
        for t, a, txt in rows:
            x, _ = sf.read(t, dtype='float32'); y, _ = sf.read(a, dtype='float32')
            n = min(len(x), len(y)); xs.append(x[:n]); ys.append(y[:n]); texts.append(txt)
        L = max(len(v) for v in xs)
        pad = lambda v: torch.stack([F.pad(torch.from_numpy(z), (0, L - len(z))) for z in v]).to(DEVICE)
        lab = s51.collate_fn([(torch.zeros(1), torch.zeros(1), t) for t in texts], tokenizer=tok, decoder_start_token_id=start)[2].to(DEVICE)
        with torch.no_grad():
            return logmel(pad(xs)), logmel(pad(ys)), lab

    def loss_of(mx, my, lab):
        mo = net(mx)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ce = wm(input_features=mo, labels=lab).loss.float()
        return ce + args.alpha * F.l1_loss(mo, my), ce

    def evaluate():
        net.eval(); tot = []
        with torch.no_grad():
            for i in range(0, len(dv), args.batch_size):
                tot.append(loss_of(*batch(dv[i:i + args.batch_size]))[1].item())
        net.train()
        return float(np.mean(tot))

    best = evaluate(); print(f'dev CE（学習前＝喉マイクの mel そのまま）={best:.4f}', flush=True)
    torch.save(net.state_dict(), CK / 'best.pt')
    for ep in range(1, args.epochs + 1):
        random.shuffle(tr); ls = []
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            loss, ce = loss_of(*batch(tr[i:i + args.batch_size]))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            ls.append(ce.item())
        v = evaluate(); mark = ' <- best' if v < best else ''
        print(f'epoch {ep:2d}  train CE={np.mean(ls):.4f}  dev CE={v:.4f}{mark}', flush=True)
        if v < best:
            best = v; torch.save(net.state_dict(), CK / 'best.pt')
    print(f'done. best dev CE={best:.4f}')


def evaluate_asr(args):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    OUT.mkdir(parents=True, exist_ok=True)
    proc = WhisperProcessor.from_pretrained('openai/whisper-small')            # 語彙は全サイズ共通（FT も同じ）
    wm = WhisperForConditionalGeneration.from_pretrained(HF[args.asr]).to(DEVICE).eval()
    logmel = s51.WhisperLogMel().to(DEVICE)
    net = UNet().to(DEVICE).eval(); net.load_state_dict(torch.load(BASE_DIR / 'checkpoints' / args.tag / 'best.pt', map_location=DEVICE))
    conds = args.conds or ([args.tag] + WAV_CONDS)
    path = OUT / f'hyp_{args.asr}.csv'
    done = {(r['utt'], r['cond']) for r in csv.DictReader(open(path, encoding='utf-8'))} if path.exists() else set()
    f = open(path, 'a', newline='', encoding='utf-8'); w = csv.DictWriter(f, fieldnames=['utt', 'spk', 'cond', 'hyp'])
    if not done:
        w.writeheader()
    for i, s in enumerate(s64.load_samples('test')):
        for c in conds:
            if (s['utt'], c) in done:
                continue
            if c in (args.tag, 'no_se'):
                wav, _ = sf.read(s['path'], dtype='float32')
            elif c in s64.SE_CKPTS:
                wav = None
            else:
                wav, _ = sf.read(SE_WAV / c / 'test' / f"{s['utt']}.wav", dtype='float32')
            with torch.no_grad():
                if wav is None:   # TAPS / CE-SE は波形を作ってから
                    if c not in evaluate_asr.se:   # SE は一度だけ読み込む
                        evaluate_asr.se[c] = s64.load_se(s64.SE_CKPTS[c])
                    se = evaluate_asr.se[c]
                    x, _ = sf.read(s['path'], dtype='float32')
                    wav = se(torch.from_numpy(x)[None].to(DEVICE)).reshape(-1).cpu().numpy()
                m = logmel(torch.from_numpy(np.asarray(wav, dtype=np.float32))[None].to(DEVICE))
                if c == args.tag:
                    m = net(m)
                ids = wm.generate(input_features=m.to(wm.dtype), language='ko', task='transcribe', max_new_tokens=225)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            w.writerow(dict(utt=s['utt'], spk=s['spk'], cond=c, hyp=hyp))
        f.flush()
        if (i + 1) % 100 == 0:
            print(f'  {args.asr} {i + 1}/1000', flush=True)
    f.close(); print(f'{args.asr} done → {path}')


evaluate_asr.se = {}


def summary(args):
    from scipy.stats import wilcoxon
    ref = {s['utt']: s['text'] for s in s64.load_samples('test')}
    rows, pr = [], []
    for asr in HF:
        p = OUT / f'hyp_{asr}.csv'
        if not p.exists():
            continue
        d = defaultdict(dict)
        for r in csv.DictReader(open(p, encoding='utf-8')):
            d[r['cond']][(r['utt'], r['spk'])] = s64.capped(s64.norm(ref[r['utt']]), s64.norm(r['hyp']))
        spk = lambda c: np.array([np.mean([v for (u, sp), v in d[c].items() if sp == k]) for k in sorted({sp for _, sp in d[c]})])
        for c, v in d.items():
            rows.append(dict(asr=asr, cond=c, n=len(v), cer=round(float(np.mean(list(v.values()))), 4)))
        if 'taps' in d:
            for c in d:
                if c != 'taps' and len(d[c]) == len(d['taps']):
                    a, b = spk('taps'), spk(c)
                    pr.append(dict(asr=asr, A='taps', B=c, cer_A=round(a.mean(), 4), cer_B=round(b.mean(), 4),
                                   rel=round(100 * (b.mean() / a.mean() - 1), 1), spk_B_better=int((b < a).sum()),
                                   p=round(wilcoxon(a, b).pvalue, 4) if np.any(a != b) else 1.0))
    for name, data in (('summary.csv', rows), ('pairs.csv', pr)):
        if data:
            with open(OUT / name, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    for r in pr:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'eval', 'summary'])
    ap.add_argument('--asr', default='whisper-small', choices=list(HF))
    ap.add_argument('--conds', nargs='+')
    ap.add_argument('--tag', default='dissen_mel', help='チェックポイント名＝評価での条件名')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--alpha', type=float, default=1.0, help='気導音の log-mel との L1 の重み')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_sec', type=float, default=15.0)
    args = ap.parse_args()
    {'train': train, 'eval': evaluate_asr, 'summary': summary}[args.stage](args)


if __name__ == '__main__':
    main()
