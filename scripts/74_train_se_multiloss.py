"""
スクリプト74: 複数損失によるSE学習（SSL-MSE・複数ASR損失・勾配合成）

script 51（Whisper CE）を一般化したもの。TAPS SE-Conformer（pretrained）から追加学習する。
損失 = 再構成（L1 + 多重解像度STFT）
     + w_ce  × Whisper-small CE（log-mel入力）
     + w_ctc × wav2vec2 CTC（波形入力、--ctc_model）
     + w_ssl × SSL表現MSE（WavLM等、SE出力と気導音声の全層平均MSE。Sato et al. 2025 型）

--combine andmask: ASR損失（ce, ctc）の勾配をSEパラメータ上で別々に求め、
符号が一致する成分だけを平均して使う（AND-mask, Parascandolo et al. ICLR 2021）。
再構成・SSLの勾配はそのまま足す。

使い方（GPU PC）:
    python scripts/74_train_se_multiloss.py --w_ssl 1.0 --tag ssl_wavlm_1
    python scripts/74_train_se_multiloss.py --w_ce 10 --w_ctc 10 --tag ce_ctc_sum
    python scripts/74_train_se_multiloss.py --w_ce 10 --w_ctc 10 --combine andmask --tag ce_ctc_and
"""

import argparse
import csv
import importlib.util
import os
import re
import sys
import unicodedata
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location('s51', BASE_DIR / 'scripts' / '51_train_se_ce_loss.py')
s51 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s51)

DEVICE = s51.DEVICE
CTC_REPOS = {
    'xlsr-korean': 'kresnik/wav2vec2-large-xlsr-korean',
    'mms-1b-all':  'facebook/mms-1b-all',
}


def norm_text(s):
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    return re.sub(r'\s+', ' ', s).strip()


def collate(batch, wtok=None, wstart=None, ctok=None):
    t_wavs, a_wavs, texts = zip(*batch)
    lens = torch.tensor([t.shape[0] for t in t_wavs])
    max_len = int(lens.max())
    pad = lambda ws: torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in ws])
    out = {'t': pad(t_wavs), 'a': pad(a_wavs), 'len': lens}
    if wtok is not None:
        out['w_labels'] = s51.collate_fn(batch, tokenizer=wtok, decoder_start_token_id=wstart)[2]
    if ctok is not None:
        ids = [ctok(norm_text(x)).input_ids for x in texts]
        L = max(len(i) for i in ids)
        out['c_labels'] = torch.tensor([i + [-100] * (L - len(i)) for i in ids])
    return out


def utt_normalize(x, mask):
    """Wav2Vec2FeatureExtractor の do_normalize と同じ発話単位の平均0・分散1（有効区間のみ）。"""
    n = mask.sum(-1, keepdim=True)
    mu = (x * mask).sum(-1, keepdim=True) / n
    var = (((x - mu) * mask) ** 2).sum(-1, keepdim=True) / n
    return (x - mu) / torch.sqrt(var + 1e-7) * mask


class CTCLoss:
    def __init__(self, name):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        repo = CTC_REPOS[name]
        kw = dict(target_lang='kor') if name.startswith('mms') else {}
        self.proc = AutoProcessor.from_pretrained(repo, **kw)
        if name.startswith('mms'):
            kw['ignore_mismatched_sizes'] = True
        self.model = Wav2Vec2ForCTC.from_pretrained(repo, **kw).to(DEVICE).eval()
        self.model.config.ctc_zero_infinity = True
        self.model.config.ctc_loss_reduction = 'mean'
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.do_norm = getattr(self.proc.feature_extractor, 'do_normalize', True)

    def __call__(self, x, mask, labels):
        inp = utt_normalize(x, mask) if self.do_norm else x
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.model(inp, attention_mask=mask.long(), labels=labels)
        return out.loss.float()


class SSLLoss:
    """SE出力と気導音声のSSL表現（全Transformer層）のMSEの層平均。"""
    def __init__(self, repo):
        from transformers import AutoFeatureExtractor, AutoModel
        fe = AutoFeatureExtractor.from_pretrained(repo)
        self.do_norm = getattr(fe, 'do_normalize', False)
        self.model = AutoModel.from_pretrained(repo).to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _hs(self, x, mask):
        if self.do_norm:
            x = utt_normalize(x, mask)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.model(x, attention_mask=mask.long(), output_hidden_states=True)
        return out.hidden_states[1:]

    def __call__(self, x, a, mask):
        hs = self._hs(x, mask)
        with torch.no_grad():
            ht = self._hs(a, mask)
        flen = self.model._get_feat_extract_output_lengths(mask.sum(-1).long())
        T = hs[0].shape[1]
        fm = (torch.arange(T, device=x.device)[None] < flen[:, None]).float()[..., None]
        loss = 0.
        for h, g in zip(hs, ht):
            loss = loss + (((h.float() - g.float()) ** 2) * fm).sum() / (fm.sum() * h.shape[-1])
        return loss / len(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w_ce', type=float, default=0.0)
    ap.add_argument('--w_ctc', type=float, default=0.0)
    ap.add_argument('--ctc_model', default='xlsr-korean', choices=list(CTC_REPOS))
    ap.add_argument('--w_ssl', type=float, default=0.0)
    ap.add_argument('--ssl_model', default='microsoft/wavlm-large')
    ap.add_argument('--combine', default='sum', choices=['sum', 'andmask'])
    ap.add_argument('--no_recon', action='store_true')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='学習・dev発話数の上限（動作確認用）')
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    ckpt_dir = BASE_DIR / 'checkpoints' / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f'=== script 74 | {vars(args)} ===', flush=True)

    from models.seconformer import seconformer
    se = seconformer(**s51.TAPS_SE_CONFIG).to(DEVICE)
    st = torch.load(s51.PRETRAINED_DIR / 'seconformer.th', map_location=DEVICE, weights_only=False)
    se.load_state_dict(st['model'] if isinstance(st, dict) and 'model' in st else st)
    params = [p for p in se.parameters() if p.requires_grad]
    stft = s51.MultiResolutionSTFTLoss().to(DEVICE)

    wtok = wstart = whisper = logmel = None
    if args.w_ce > 0:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained('openai/whisper-small')
        whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(DEVICE).eval()
        for p in whisper.parameters():
            p.requires_grad_(False)
        whisper.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language='ko', task='transcribe')
        wtok, wstart = proc.tokenizer, whisper.config.decoder_start_token_id
        logmel = s51.WhisperLogMel().to(DEVICE)
    ctc = CTCLoss(args.ctc_model) if args.w_ctc > 0 else None
    ssl = SSLLoss(args.ssl_model) if args.w_ssl > 0 else None

    train_ds, dev_ds = s51.TAPSPairDataset('train'), s51.TAPSPairDataset('dev')
    if args.limit:
        train_ds.samples = train_ds.samples[:args.limit]
        dev_ds.samples = dev_ds.samples[:max(args.limit // 4, 8)]
    col = partial(collate, wtok=wtok, wstart=wstart, ctok=ctc.proc.tokenizer if ctc else None)
    nw = min(2, os.cpu_count() or 0)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=col, num_workers=nw)
    dev_dl = DataLoader(dev_ds, args.batch_size, shuffle=False, collate_fn=col, num_workers=nw)
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))

    def losses(b):
        t, a = b['t'].to(DEVICE), b['a'].to(DEVICE)
        y = se(t).squeeze(1)
        n = min(y.shape[-1], a.shape[-1])
        y, a = y[..., :n], a[..., :n]
        mask = (torch.arange(n, device=DEVICE)[None] < b['len'].to(DEVICE)[:, None]).float()
        free, task = {}, {}
        if not args.no_recon:
            free['recon'] = F.l1_loss(y, a) + stft(y, a)
        if ssl:
            free['ssl'] = args.w_ssl * ssl(y, a, mask)
        if whisper is not None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                enc = whisper.model.encoder(logmel(y))
                ce = whisper(encoder_outputs=(enc,), labels=b['w_labels'].to(DEVICE)).loss
            task['ce'] = args.w_ce * ce.float()
        if ctc:
            task['ctc'] = args.w_ctc * ctc(y, mask, b['c_labels'].to(DEVICE))
        return free, task

    log_path = ckpt_dir / 'training_log.csv'
    keys = (['recon'] if not args.no_recon else []) + (['ssl'] if ssl else []) + \
           (['ce'] if whisper is not None else []) + (['ctc'] if ctc else [])
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'] + [f'val_{k}' for k in keys] + ['agree'])

    best, bad = float('inf'), 0
    for ep in range(1, args.epochs + 1):
        se.train()
        tr, agree = [], []
        for b in train_dl:
            free, task = losses(b)
            opt.zero_grad()
            if args.combine == 'andmask' and len(task) >= 2:
                gs = [torch.autograd.grad(v, params, retain_graph=True, allow_unused=True) for v in task.values()]
                gs = [[g if g is not None else torch.zeros_like(p) for g, p in zip(gl, params)] for gl in gs]
                fsum = sum(free.values()) if free else None
                gf = torch.autograd.grad(fsum, params, allow_unused=True) if fsum is not None else [None] * len(params)
                tot_agree, tot_n = 0, 0
                for i, p in enumerate(params):
                    stack = torch.stack([g[i] for g in gs])
                    sgn = torch.sign(stack)
                    m = (sgn == sgn[0:1]).all(0) & (sgn[0] != 0)
                    tot_agree += m.sum().item(); tot_n += m.numel()
                    g = stack.mean(0) * m
                    if gf[i] is not None:
                        g = g + gf[i]
                    p.grad = g
                agree.append(tot_agree / tot_n)
                loss = (fsum if fsum is not None else 0) + sum(task.values())
            else:
                loss = sum(free.values()) + sum(task.values())
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tr.append(float(loss))

        se.eval()
        vals = {k: [] for k in keys}
        with torch.no_grad():
            for b in dev_dl:
                free, task = losses(b)
                for k, v in {**free, **task}.items():
                    vals[k].append(float(v))
        vm = {k: float(np.mean(v)) for k, v in vals.items()}
        vl = sum(vm.values())
        ag = float(np.mean(agree)) if agree else float('nan')
        mark = ' <- best' if vl < best else ''
        print(f'Epoch {ep:2d} | train={np.mean(tr):.4f} | val={vl:.4f} '
              + ' '.join(f'{k}={v:.4f}' for k, v in vm.items())
              + (f' | agree={ag:.3f}' if agree else '') + mark, flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, np.mean(tr), vl] + [vm[k] for k in keys] + [ag])
        torch.save(se.state_dict(), ckpt_dir / f'ep{ep:02d}.th')
        if vl < best:
            best, bad = vl, 0
            torch.save(se.state_dict(), ckpt_dir / 'best.th')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'Early stopping (patience={args.patience})')
                break
    print(f'Done. best val={best:.4f} → {ckpt_dir / "best.th"}')


if __name__ == '__main__':
    main()
