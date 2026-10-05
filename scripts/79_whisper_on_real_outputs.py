"""
スクリプト79: 「実際の生成出力」に Whisper の損失をかける（script 77 の失敗＝1段見積もりの効果が多段生成に持ち越されない、への対処）

  post  : (a) 生成モデルの実際の出力（20段生成、保存済み）を、小さな後段モジュールで直す。
          振幅スペクトルに掛けるゲインを CNN（約0.1M）で出し、位相はそのまま。初期値は「何もしない」。
          損失 = Whisper-small CE ＋ β × |log振幅の変化|（生成出力から離れすぎない）
  post_apply : 学習した後段を、保存済みの生成出力（test など）にかける → se_wav/<out>
  draft : (b) DRaFT-K。生成の最後の K 段だけ勾配を通して Whisper-small CE を逆伝播（それ以前の段は勾配なし）。
          生成モデル本来の損失も正則化として足す。選択は「実際に20段生成した出力」の Whisper CE（推論と同じ）

学習データ: 生成モデル（fm_taps）は TAPS train で学習済みで、train の出力はほぼ正解の丸暗記になるため、
dev を使う（10話者のうち8話者で学習、残り2話者で選択）。評価は test（Whisper-small 以外の Whisper 系は未使用の評価用）。

使い方（GPU PC、venv-nemo）
    python scripts/79_whisper_on_real_outputs.py post --tag post_ce
    python scripts/79_whisper_on_real_outputs.py post_apply --tag post_ce --src fm_taps --out fm_post_ce
    python scripts/79_whisper_on_real_outputs.py draft --tag draft_k1 --K 1
    python scripts/75_flowmatching_se.py infer --tag fm_taps --weights checkpoints/draft_k1/best.pt --out fm_draft_k1
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
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
TAPS_DIR = BASE_DIR / 'data' / 'raw' / 'taps'
SE_WAV = BASE_DIR / 'data' / 'processed' / 'se_wav'
DEVICE = 'cuda'
N_FFT, HOP = 512, 128


def dev_split(max_sec):
    rows = [r for r in csv.DictReader(open(TAPS_DIR / 'metadata_dev.csv', encoding='utf-8'))
            if float(r.get('duration', 0)) <= max_sec]
    spk = sorted({r['speaker_id'] for r in rows})
    val_spk = set(spk[-2:])
    tr = [r for r in rows if r['speaker_id'] not in val_spk]
    va = [r for r in rows if r['speaker_id'] in val_spk]
    return tr, va


def whisper_ce():
    """凍結 Whisper-small の CE（正解テキスト）を返す関数"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    s51 = import_module('51_train_se_ce_loss')
    proc = WhisperProcessor.from_pretrained('openai/whisper-small')
    proc.tokenizer.set_prefix_tokens(language='korean', task='transcribe')  # 学習ラベルの先頭に <|ko|><|transcribe|> を入れる（2026-10-05 まではこれが抜けていた）
    wm = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(DEVICE).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    wm.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language='ko', task='transcribe')
    tok, start = proc.tokenizer, wm.config.decoder_start_token_id
    logmel = s51.WhisperLogMel().to(DEVICE)

    def ce(wav, texts):   # wav: (B, T)
        lab = s51.collate_fn([(torch.zeros(1), torch.zeros(1), t) for t in texts], tokenizer=tok, decoder_start_token_id=start)[2].to(DEVICE)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            enc = wm.model.encoder(logmel(wav))
            return wm(encoder_outputs=(enc,), labels=lab).loss.float()
    return ce


def pad_batch(wavs):
    L = max(len(w) for w in wavs)
    return torch.stack([F.pad(torch.from_numpy(w), (0, L - len(w))) for w in wavs]).to(DEVICE), [len(w) for w in wavs]


# ---------------- (a) 後段モジュール ----------------
class PostGain(nn.Module):
    """log 振幅 → ゲイン（e^±gmax の範囲）。最後の層を0で初期化して「何もしない」から始める"""
    def __init__(self, ch=32, gmax=1.0):
        super().__init__()
        self.gmax = gmax
        layers, c_in = [], 1
        for d in (1, 2, 4, 8):
            layers += [nn.Conv2d(c_in, ch, 3, padding=(1, d), dilation=(1, d)), nn.GELU()]
            c_in = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.win = None

    def forward(self, wav):   # (B, T) → (B, T), 変化量（log ゲイン）も返す
        if self.win is None:
            self.win = torch.hann_window(N_FFT, device=wav.device)
        X = torch.stft(wav, N_FFT, HOP, window=self.win, return_complex=True)
        lm = torch.log(X.abs() + 1e-5)
        g = self.gmax * torch.tanh(self.head(self.body(lm[:, None])))[:, 0]
        Y = X * torch.exp(g)
        return torch.istft(Y, N_FFT, HOP, window=self.win, length=wav.shape[-1]), g


def post(args):
    torch.manual_seed(0); random.seed(0)
    ck = BASE_DIR / 'checkpoints' / args.tag; ck.mkdir(parents=True, exist_ok=True)
    ce = whisper_ce()
    net = PostGain().to(DEVICE)
    print(f'PostGain params: {sum(p.numel() for p in net.parameters()) / 1e6:.3f}M', flush=True)
    tr, va = dev_split(args.max_sec)
    src = SE_WAV / args.src / 'dev'
    load = lambda rows: ([sf.read(src / f"{r['speaker_id']}_{r['sentence_id']}.wav", dtype='float32')[0] for r in rows], [r['text'] for r in rows])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def run(rows, train):
        net.train(train); tot, n = 0.0, 0
        order = list(range(0, len(rows), args.batch_size))
        if train:
            random.shuffle(rows)
        for i in order:
            wavs, texts = load(rows[i:i + args.batch_size])
            x, lens = pad_batch(wavs)
            with torch.set_grad_enabled(train):
                y, g = net(x)
                l_ce = ce(y, texts)
                l_reg = g.abs().mean()
                loss = l_ce + args.beta * l_reg
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += l_ce.item() * len(texts); n += len(texts)
        return tot / n

    with torch.no_grad():
        base = run(va, False)
    print(f'val CE（後段なし＝生成出力そのまま）={base:.4f}', flush=True)
    best = base
    for ep in range(1, args.epochs + 1):
        trl = run(tr, True)
        with torch.no_grad():
            vl = run(va, False)
        mark = ' <- best' if vl < best else ''
        print(f'epoch {ep:2d}  train CE={trl:.4f}  val CE={vl:.4f}{mark}', flush=True)
        if vl < best:
            best = vl; torch.save(net.state_dict(), ck / 'best.pt')
    print(f'done. best val CE={best:.4f}（後段なし {base:.4f}）')


def post_apply(args):
    net = PostGain().to(DEVICE).eval()
    net.load_state_dict(torch.load(BASE_DIR / 'checkpoints' / args.tag / 'best.pt', map_location=DEVICE))
    src, out = SE_WAV / args.src / args.split, SE_WAV / args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob('*.wav')):
        if (out / p.name).exists():
            continue
        w, sr = sf.read(p, dtype='float32')
        with torch.no_grad():
            y, _ = net(torch.from_numpy(w)[None].to(DEVICE))
        sf.write(out / p.name, y[0].cpu().numpy().astype(np.float32), sr)
    print(f'done → {out}')


# ---------------- (b) DRaFT-K ----------------
def draft(args):
    from nemo.collections.audio.models import AudioToAudioModel
    torch.manual_seed(0); random.seed(0)
    ck = BASE_DIR / 'checkpoints' / args.tag; ck.mkdir(parents=True, exist_ok=True)
    nemo_path = sorted((BASE_DIR / 'checkpoints' / 'fm_taps').rglob('*.nemo'), key=lambda p: p.stat().st_mtime)[-1]
    model = AudioToAudioModel.restore_from(str(nemo_path), map_location=DEVICE)
    model.train()
    flow, smp = model.flow, model.sampler
    ce = whisper_ce()
    tr, va = dev_split(args.max_sec)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def load(rows):
        xs, ys = [], []
        for r in rows:
            name = f"{r['speaker_id']}_{r['sentence_id']}.wav"
            x, _ = sf.read(TAPS_DIR / 'throat' / 'dev' / name, dtype='float32')
            y, _ = sf.read(TAPS_DIR / 'acoustic' / 'dev' / name, dtype='float32')
            n = min(len(x), len(y)); xs.append(x[:n]); ys.append(y[:n])
        x, lens = pad_batch(xs); y, _ = pad_batch(ys)
        return x[:, None], y[:, None], torch.tensor(lens, device=DEVICE), [r['text'] for r in rows]

    def generate(x, lens, K):
        """20段の Euler 生成。最後の K 段だけ勾配を通す。波形 (B, T) を返す"""
        xn, scale = model._normalize(x) if model.normalize_input else (x, None)
        cond, clen = model.encoder(input=xn, input_length=lens)
        state = torch.randn_like(cond) * flow.sigma_start
        ts = torch.linspace(smp.time_min, smp.time_max, smp.num_steps + 1)
        dt = (smp.time_max - smp.time_min) / smp.num_steps
        for k, t in enumerate(ts[:-1]):
            with torch.set_grad_enabled(k >= smp.num_steps - K):
                v, _ = model.estimator(input=torch.cat([state, cond], dim=1), input_length=clen,
                                       condition=t * torch.ones(state.shape[0], device=DEVICE))
                state = state + v * dt
        wav, _ = model.decoder(input=state, input_length=clen)
        if scale is not None:
            wav = model._denormalize(wav, scale)
        return wav[:, 0, :lens.max()]

    def flow_loss(x, y, lens):
        if model.normalize_input:
            x, scale = model._normalize(x); y = y / (scale + model.eps)
        xe, xl = model.encoder(input=x, input_length=lens); ye, _ = model.encoder(input=y, input_length=lens)
        x0 = torch.zeros_like(xe)
        t = flow.generate_time(batch_size=x.size(0)).to(DEVICE)
        pt = flow.sample(time=t, x_start=x0, x_end=ye)
        v, _ = model.estimator(input=torch.cat([pt, xe], dim=-3), input_length=xl, condition=t)
        return model.loss(estimate=v, target=flow.vector_field(time=t, x_start=x0, x_end=ye, point=pt), input_length=xl)

    def val_ce():
        model.eval(); torch.manual_seed(1); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), 4):
                x, _, lens, texts = load(va[i:i + 4])
                tot += ce(generate(x, lens, 0), texts).item() * len(texts); n += len(texts)
        model.train()
        return tot / n

    best = val_ce()
    print(f'val CE（20段生成、追加学習前）={best:.4f}', flush=True)
    torch.save(model.state_dict(), ck / 'best.pt')
    step, micro = 0, 0
    while step < args.steps:
        random.shuffle(tr)
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            x, y, lens, texts = load(tr[i:i + args.batch_size])
            l_ce = ce(generate(x, lens, args.K), texts)
            l_fl = flow_loss(x, y, lens)
            ((args.lam * l_ce + l_fl) / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            opt.step(); opt.zero_grad(); step += 1
            if step % 25 == 0:
                print(f'step {step}  ce={l_ce.item():.3f}  flow={l_fl.item():.4f}', flush=True)
            if step % args.eval_every == 0 or step >= args.steps:
                vl = val_ce()
                mark = ' <- best' if vl < best else ''
                print(f'[eval] step {step}  val CE（20段生成）={vl:.4f}{mark}', flush=True)
                if vl < best:
                    best = vl; torch.save(model.state_dict(), ck / 'best.pt')
            if step >= args.steps:
                break
    print(f'done. best val CE={best:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['post', 'post_apply', 'draft'])
    ap.add_argument('--tag', default='post_ce')
    ap.add_argument('--src', default='fm_taps')
    ap.add_argument('--out', default='')
    ap.add_argument('--split', default='test')
    ap.add_argument('--beta', type=float, default=1.0, help='post: 生成出力から離れすぎない正則化の重み')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--K', type=int, default=1, help='draft: 勾配を通す最後の段数')
    ap.add_argument('--lam', type=float, default=0.1, help='draft: Whisper CE の重み（生成モデル本来の損失に対して）')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--eval_every', type=int, default=50)
    ap.add_argument('--max_sec', type=float, default=12.0)
    args = ap.parse_args()
    {'post': post, 'post_apply': post_apply, 'draft': draft}[args.stage](args)


if __name__ == '__main__':
    main()
