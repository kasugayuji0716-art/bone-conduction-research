"""
スクリプト77: Whisper 系に特化した喉マイク SE — 生成モデル（script 75 の fm_taps）＋ Whisper の CE 損失

目標: Whisper 系（Whisper の log-mel 入力）の認識器なら、損失に使っていないものでも誤りが減る SE。
損失は Whisper-small だけ。評価は Whisper-base/medium/large-v3-turbo/FT・Qwen3-ASR（Whisper と同じ入力）で行う。

stage
  chain : 推論だけの確認。生成モデルの出力（既定 fm_taps_avg4）に、既存の Whisper 用 SE（CE λ=10）をかける
  train : fm_taps（TAPS で追加学習済みの生成モデル）を、生成モデル本来の損失 ＋ λ × Whisper-small の CE で追加学習
          CE は「途中の状態から1段で見積もった完成形」 x̂1 = x_t + (1 − t)·v̂ を波形に戻して計算する
          （正解テキストは学習時だけ使う。推論は script 75 と同じ）

使い方（GPU PC、venv-nemo に transformers が入っていること）
    python scripts/77_flow_whisper_ce.py chain [--src fm_taps_avg4] [--split test]
    python scripts/77_flow_whisper_ce.py train --lam 0.1 --tag fmce_0.1 [--steps 4000]
    python scripts/75_flowmatching_se.py infer --tag fm_taps --weights checkpoints/fmce_0.1/step4000.pt --out fmce_0.1
"""

import argparse
import csv
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
SE_WAV = BASE_DIR / 'data' / 'processed' / 'se_wav'
DEVICE = 'cuda'


def chain(args):
    s66 = import_module('66_lambda_cross_asr')
    se = s66.load_any_se(args.se)
    src = SE_WAV / args.src / args.split
    out = SE_WAV / f'{args.src}_{args.se.replace(".", "")}' / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('*.wav'))
    for i, p in enumerate(files):
        dst = out / p.name
        if dst.exists():
            continue
        wav, sr = sf.read(p, dtype='float32')
        with torch.no_grad():
            y = se(torch.from_numpy(wav)[None].to(DEVICE)).reshape(-1).cpu().numpy()
        sf.write(dst, y[:len(wav)].astype(np.float32), sr)
    print(f'{len(files)} files → {out}')


def load_pairs(split, max_sec):
    rows = []
    for r in csv.DictReader(open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8')):
        if float(r.get('duration', 0)) > max_sec:
            continue
        name = f"{r['speaker_id']}_{r['sentence_id']}.wav"
        t, a = TAPS_DIR / 'throat' / split / name, TAPS_DIR / 'acoustic' / split / name
        if t.exists() and a.exists():
            rows.append((t, a, r['text']))
    return rows


def train(args):
    from nemo.collections.audio.models import AudioToAudioModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    s51 = import_module('51_train_se_ce_loss')

    torch.manual_seed(0); random.seed(0)
    ck = BASE_DIR / 'checkpoints' / args.tag
    ck.mkdir(parents=True, exist_ok=True)
    nemo_path = sorted((BASE_DIR / 'checkpoints' / args.init).rglob('*.nemo'), key=lambda p: p.stat().st_mtime)[-1]
    model = AudioToAudioModel.restore_from(str(nemo_path), map_location=DEVICE)
    model.train()
    flow = model.flow

    proc = WhisperProcessor.from_pretrained('openai/whisper-small')
    whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(DEVICE).eval()
    for p in whisper.parameters():
        p.requires_grad_(False)
    whisper.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language='ko', task='transcribe')
    tok, start_id = proc.tokenizer, whisper.config.decoder_start_token_id
    logmel = s51.WhisperLogMel().to(DEVICE)

    def labels(texts):
        batch = [(torch.zeros(1), torch.zeros(1), t) for t in texts]
        return s51.collate_fn(batch, tokenizer=tok, decoder_start_token_id=start_id)[2].to(DEVICE)

    train_rows, dev_rows = load_pairs('train', args.max_sec), load_pairs('dev', args.max_sec)[::10][:100]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def batch_of(rows):
        ts, as_, texts = [], [], []
        for t, a, txt in rows:
            x, _ = sf.read(t, dtype='float32'); y, _ = sf.read(a, dtype='float32')
            n = min(len(x), len(y)); ts.append(x[:n]); as_.append(y[:n]); texts.append(txt)
        L = max(len(x) for x in ts)
        pad = lambda v: torch.stack([F.pad(torch.from_numpy(x), (0, L - len(x))) for x in v])[:, None].to(DEVICE)
        return pad(ts), pad(as_), torch.tensor([len(x) for x in ts], device=DEVICE), texts

    def losses(x, y, lens, texts, ce_only=False):
        """生成モデル本来の損失（ベクトル場の MSE）と、1段で見積もった完成形の Whisper CE"""
        if model.normalize_input:
            x, scale = model._normalize(x)
            y = y / (scale + model.eps)
        xe, xe_len = model.encoder(input=x, input_length=lens)
        ye, _ = model.encoder(input=y, input_length=lens)
        x0 = torch.zeros_like(xe)
        out = {}
        if not ce_only:
            t = flow.generate_time(batch_size=x.size(0)).to(DEVICE)
            pt = flow.sample(time=t, x_start=x0, x_end=ye)
            v, _ = model.estimator(input=torch.cat([pt, xe], dim=-3), input_length=xe_len, condition=t)
            out['flow'] = model.loss(estimate=v, target=flow.vector_field(time=t, x_start=x0, x_end=ye, point=pt), input_length=xe_len)
        # CE: 完成に近い時刻から1段で完成形を見積もる（早い時刻の見積もりはぼやけていて CE の意味が薄い）
        tc = torch.empty(x.size(0), device=DEVICE).uniform_(args.t_min, 1.0)
        pc = flow.sample(time=tc, x_start=x0, x_end=ye)
        vc, _ = model.estimator(input=torch.cat([pc, xe], dim=-3), input_length=xe_len, condition=tc)
        x1 = pc + (1 - tc).view(-1, 1, 1, 1) * vc
        wav, _ = model.decoder(input=x1, input_length=xe_len)
        if model.normalize_input:
            wav = model._denormalize(wav, scale)
        wav = wav[:, 0, :lens.max()]
        enc = whisper.model.encoder(logmel(wav))
        out['ce'] = whisper(encoder_outputs=(enc,), labels=labels(texts)).loss
        return out

    log = open(ck / 'log.csv', 'a')
    print(f'init={nemo_path.name} lam={args.lam} steps={args.steps} bs={args.batch_size} lr={args.lr}', flush=True)
    step = 0
    while step < args.steps:
        random.shuffle(train_rows)
        for i in range(0, len(train_rows) - args.batch_size + 1, args.batch_size):
            x, y, lens, texts = batch_of(train_rows[i:i + args.batch_size])
            l = losses(x, y, lens, texts)
            loss = l['flow'] + args.lam * l['ce']
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)   # NeMo の設定と同じ
            opt.step(); step += 1
            if step % 50 == 0:
                print(f'step {step}  flow={l["flow"].item():.4f}  ce={l["ce"].item():.3f}', flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    ces = [losses(*batch_of(dev_rows[j:j + 4]), ce_only=True)['ce'].item() for j in range(0, len(dev_rows), 4)]
                model.train()
                print(f'[eval] step {step}  dev CE(1段見積もり)={np.mean(ces):.4f}', flush=True)
                log.write(f'{step},{l["flow"].item():.5f},{l["ce"].item():.4f},{np.mean(ces):.4f}\n'); log.flush()
                torch.save(model.state_dict(), ck / f'step{step}.pt')
            if step >= args.steps:
                break
    print(f'done → {ck}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['chain', 'train'])
    ap.add_argument('--src', default='fm_taps_avg4')
    ap.add_argument('--se', default='ce10.0')
    ap.add_argument('--split', default='test')
    ap.add_argument('--init', default='fm_taps')
    ap.add_argument('--tag', default='fmce')
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--t_min', type=float, default=0.5)
    ap.add_argument('--max_sec', type=float, default=15.0)
    ap.add_argument('--eval_every', type=int, default=1000)
    args = ap.parse_args()
    {'chain': chain, 'train': train}[args.stage](args)


if __name__ == '__main__':
    main()
