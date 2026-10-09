"""
スクリプト51: Cross-entropy損失によるASR-aware SE学習（v2: レビュー修正版）

修正点（2026-09-20）:
  1. WhisperLogMel → Whisper標準前処理に一致（Slaney mel, 波形パディング）
  2. CE教師ラベル: EOS保持、BOS除去（HF公式に準拠）
  3. 15秒超の発話を除外（音声切り出しとテキスト不一致の回避）
  4. STFT損失パラメータをTAPS公式設定に合わせる

使い方（DNN PC）:
    python scripts/51_train_se_ce_loss.py --lambda_asr 2.0 --tag ce_v2_lambda_2.0
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as Ta
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR       = Path(__file__).parent.parent
TAPS_DIR       = BASE_DIR / 'data' / 'raw' / 'taps'
PRETRAINED_DIR = BASE_DIR / 'taps-baselines' / 'pretrained'

_TAPS_BASELINES = BASE_DIR / 'taps-baselines'
if str(_TAPS_BASELINES) not in sys.path:
    sys.path.insert(0, str(_TAPS_BASELINES))

TARGET_SR = 16000
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

TAPS_SE_CONFIG = dict(
    hidden=64, conformer_dim=512, conformer_ffn_dim=64,
    conformer_depth=4, depthwise_conv_kernel_size=15,
)


# ══════════════════════════════════════════════════════════════
# Fix 1: Whisper-compatible log-mel (Slaney scale, waveform padding)
# ══════════════════════════════════════════════════════════════

class WhisperLogMel(nn.Module):
    """Differentiable log-mel matching Whisper's WhisperFeatureExtractor."""
    N_SAMPLES = 480000  # 30 seconds at 16kHz
    N_FRAMES  = 3000

    def __init__(self, sr=16000):
        super().__init__()
        self.mel = Ta.MelSpectrogram(
            sample_rate=sr, n_fft=400, win_length=400, hop_length=160,
            n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
            window_fn=torch.hann_window, normalized=False,
            mel_scale="slaney", norm="slaney",
        )

    def forward(self, x):
        # Pad waveform to 30s (Whisper standard), not mel output
        if x.shape[-1] < self.N_SAMPLES:
            x = F.pad(x, (0, self.N_SAMPLES - x.shape[-1]))
        else:
            x = x[..., :self.N_SAMPLES]
        mel = self.mel(x)
        log_mel = torch.log10(mel.clamp(min=1e-10))
        max_val = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, max_val - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel[..., :self.N_FRAMES]


# ══════════════════════════════════════════════════════════════
# Fix 4: STFT loss matching TAPS official config
# ══════════════════════════════════════════════════════════════

class MultiResolutionSTFTLoss(nn.Module):
    """TAPS official: (n_fft, hop, win) with 0.5 coefficients for SC and Mag."""
    def __init__(self, resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))):
        super().__init__()
        self.resolutions = resolutions

    def _stft_loss(self, est, target, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length, device=est.device)
        est_stft = torch.stft(est, n_fft, hop_length, win_length, window, return_complex=True)
        tgt_stft = torch.stft(target, n_fft, hop_length, win_length, window, return_complex=True)
        est_mag = est_stft.abs()
        tgt_mag = tgt_stft.abs()
        sc = torch.norm(tgt_mag - est_mag, p='fro') / (torch.norm(tgt_mag, p='fro') + 1e-8)
        mag = F.l1_loss(torch.log(est_mag + 1e-8), torch.log(tgt_mag + 1e-8))
        return 0.5 * sc + 0.5 * mag  # TAPS official coefficients

    def forward(self, est, target):
        loss = 0
        for n_fft, hop, win in self.resolutions:
            loss += self._stft_loss(est, target, n_fft, hop, win)
        return loss / len(self.resolutions)


# ══════════════════════════════════════════════════════════════
# Fix 3: Dataset - exclude utterances > max_sec (don't truncate)
# ══════════════════════════════════════════════════════════════

class TAPSPairDataset(Dataset):
    def __init__(self, split, max_sec=15.0):
        self.samples = []
        n_excluded = 0
        meta = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sid, uid = row['speaker_id'], row['sentence_id']
                dur = float(row.get('duration', 0))
                if dur > max_sec:
                    n_excluded += 1
                    continue
                t = TAPS_DIR / 'throat'   / split / f'{sid}_{uid}.wav'
                a = TAPS_DIR / 'acoustic' / split / f'{sid}_{uid}.wav'
                if t.exists() and a.exists():
                    self.samples.append((t, a, row['text']))
        print(f'  [{split}] {len(self.samples)} pairs (excluded {n_excluded} > {max_sec}s)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_path, a_path, text = self.samples[idx]
        t_wav, _ = sf.read(t_path, dtype='float32')
        a_wav, _ = sf.read(a_path, dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)
        min_len = min(len(t_wav), len(a_wav))
        return torch.from_numpy(t_wav[:min_len]), torch.from_numpy(a_wav[:min_len]), text


# ══════════════════════════════════════════════════════════════
# Fix 2: Collate - proper label handling (keep EOS, remove BOS)
# ══════════════════════════════════════════════════════════════

def collate_fn(batch, tokenizer=None, decoder_start_token_id=None):
    t_wavs, a_wavs, texts = zip(*batch)
    max_len = max(t.shape[0] for t in t_wavs)
    def pad(wavs):
        return torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in wavs])

    labels = tokenizer(texts, return_tensors='pt', padding=True)
    label_ids = labels['input_ids']

    # Remove BOS (decoder_start_token_id) from start if present
    # WhisperForConditionalGeneration.shift_tokens_right adds it internally
    if decoder_start_token_id is not None and label_ids.shape[1] > 0:
        if (label_ids[:, 0] == decoder_start_token_id).all():
            label_ids = label_ids[:, 1:]

    # Mask padding with -100, but keep EOS (even if same token ID)
    # EOS is the last non-pad token - don't mask it
    attention_mask = labels.get('attention_mask')
    if attention_mask is not None:
        # Trim attention_mask to match label_ids after BOS removal
        if attention_mask.shape[1] > label_ids.shape[1]:
            attention_mask = attention_mask[:, 1:]
        label_ids = label_ids.masked_fill(attention_mask == 0, -100)
    else:
        label_ids[label_ids == tokenizer.pad_token_id] = -100

    return pad(t_wavs), pad(a_wavs), label_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_asr', type=float, default=2.0)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--patience',   type=int,   default=5)
    parser.add_argument('--tag',        type=str,   default='ce_v2_lambda_2.0')
    parser.add_argument('--lang_prefix', action='store_true',
                        help='学習ラベルの言語・タスクのトークンを明示的に設定する（get_decoder_prompt_ids でも入るので結果は同じ）')
    parser.add_argument('--amp', choices=['bf16', 'fp16'], default='bf16',
                        help='混合精度。新PCでは fp16 だと勾配が NaN になり学習が進まない')
    parser.add_argument('--no_recon',   action='store_true',
                        help='CE loss only (no L1+STFT reconstruction loss)')
    # 2026-10-08 転移しやすい ASR 損失（results/CROSS_DOMAIN_IDEAS_2026-10-08.md 案1）
    parser.add_argument('--init', type=str, default='',
                        help='初期値の SE の重み（既定: TAPS 公開の SE-Conformer）。例: checkpoints/ce_v2_lambda_10.0/best.th')
    parser.add_argument('--ghost', type=float, default=0.0,
                        help='学習時だけ Whisper の各層の出力に dropout（Ghost Networks, Li et al. AAAI 2020）')
    parser.add_argument('--layerdrop', type=float, default=0.0,
                        help='学習時だけ Whisper の各層を確率的に飛ばす（残差だけ通す）')
    parser.add_argument('--resid_scale', type=float, default=0.0,
                        help='学習時だけ Whisper の各層が残差に足す分を U(1-g, 1+g) 倍（Ghost Networks の skip connection erosion）')
    parser.add_argument('--branch_dropout', type=float, default=0.0,
                        help='学習時だけ Whisper 本来の位置（注意機構・FFN の出力側、残差の手前）に dropout。--ghost より穏やか')
    parser.add_argument('--branch_enc_only', action='store_true', help='--branch_dropout をエンコーダだけにかける')
    parser.add_argument('--input_div', action='store_true',
                        help='学習時だけ Whisper に入れる前の波形を変形（Input Diversity, Xie et al. CVPR 2019）: ±3 dB のゲイン、0〜10 ms の時間シフト')
    args = parser.parse_args()

    ckpt_dir = BASE_DIR / 'checkpoints' / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== SE-Conformer ASR-aware CE (v2 fixed) ===')
    print(f'lambda={args.lambda_asr} | tag={args.tag} | device={DEVICE}\n')

    # SE model
    from models.seconformer import seconformer as TAPSSeconformer
    se_model = TAPSSeconformer(**TAPS_SE_CONFIG).to(DEVICE)
    init_path = Path(args.init) if args.init else PRETRAINED_DIR / 'seconformer.th'
    if not init_path.is_absolute():
        init_path = BASE_DIR / init_path
    state = torch.load(init_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    se_model.load_state_dict(state)
    n_params = sum(p.numel() for p in se_model.parameters() if p.requires_grad)
    print(f'SE-Conformer: {init_path} loaded ({n_params/1e6:.1f}M params)')

    # STFT loss (TAPS official params)
    stft_loss_fn = MultiResolutionSTFTLoss().to(DEVICE)

    # Whisper (full model, frozen)
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    if args.lang_prefix:  # 念のため明示（下の get_decoder_prompt_ids が同じ設定をするので、無くてもラベルには <|ko|><|transcribe|> が入る）
        processor.tokenizer.set_prefix_tokens(language='korean', task='transcribe')
    whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(DEVICE)
    whisper.eval()
    for p in whisper.parameters():
        p.requires_grad_(False)
    print('Whisper (full): frozen')

    tokenizer = processor.tokenizer
    decoder_start_id = whisper.config.decoder_start_token_id

    # Force Korean
    forced_decoder_ids = processor.get_decoder_prompt_ids(language='ko', task='transcribe')
    whisper.config.forced_decoder_ids = forced_decoder_ids

    # log-mel (Whisper-compatible)
    log_mel_fn = WhisperLogMel().to(DEVICE)

    # 学習時だけ Whisper を揺らす（凍結したまま、擬似的に別のモデルを毎ステップ作る）。検証・選択は揺らさない
    ghost = {'on': False}
    if args.ghost > 0 or args.layerdrop > 0 or args.resid_scale > 0:
        def perturb(module, inputs, kwargs, output):
            if not ghost['on']:
                return output
            h_in = inputs[0] if inputs else kwargs['hidden_states']
            h = output[0] if isinstance(output, tuple) else output
            if args.layerdrop > 0 and torch.rand(()) < args.layerdrop:
                h = h_in
            elif args.ghost > 0:
                h = F.dropout(h, p=args.ghost, training=True)
            if args.resid_scale > 0:
                h = h_in + (1 + (torch.rand((), device=h.device) * 2 - 1) * args.resid_scale) * (h - h_in)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        for layer in list(whisper.model.encoder.layers) + list(whisper.model.decoder.layers):
            layer.register_forward_hook(perturb, with_kwargs=True)
        print(f'Ghost Whisper: dropout={args.ghost}, layerdrop={args.layerdrop}, resid_scale={args.resid_scale}（学習時のみ）')

    if args.branch_dropout > 0:
        for layer in list(whisper.model.encoder.layers) + ([] if args.branch_enc_only else list(whisper.model.decoder.layers)):
            q = args.branch_dropout
            layer.dropout = q; layer.activation_dropout = q; layer.self_attn.dropout = q
            if hasattr(layer, 'encoder_attn'):
                layer.encoder_attn.dropout = q
        print(f'Branch dropout: {args.branch_dropout}（{"エンコーダのみ" if args.branch_enc_only else "全層"}、学習時のみ）')

    def diversify(wav):
        if not (args.input_div and ghost['on']):
            return wav
        B = wav.shape[0]
        gain = 10 ** ((torch.rand(B, 1, device=wav.device) * 6 - 3) / 20)
        shift = int(torch.randint(0, 161, ()))
        return F.pad(wav * gain, (shift, 0))[..., :wav.shape[-1]]

    # Data
    print('\nData:')
    train_ds = TAPSPairDataset('train')
    dev_ds   = TAPSPairDataset('dev')

    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, decoder_start_token_id=decoder_start_id)

    n_workers = min(2, os.cpu_count() or 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=n_workers, pin_memory=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=n_workers, pin_memory=True)

    optimizer = torch.optim.Adam(se_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # fp16: 旧PC（DL-Box5）の CE v2 はこれで学習。新PC（Blackwell）では fp16 の逆伝播が NaN になり、
    # GradScaler が毎回更新を飛ばして学習が進まない（2026-10-05 確認）→ 既定は bf16（スケーラ不要）
    amp_dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 'fp16'))

    log_path = ckpt_dir / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_recon', 'val_ce'])

    best_val, patience_cnt = float('inf'), 0

    for epoch in range(1, args.epochs + 1):

        # Train
        se_model.train()
        train_losses = []

        for t_wav, a_wav, label_ids in train_dl:
            t_wav = t_wav.to(DEVICE)
            a_wav = a_wav.to(DEVICE)
            label_ids = label_ids.to(DEVICE)

            se_out = se_model(t_wav).squeeze(1)
            min_len = min(se_out.shape[-1], a_wav.shape[-1])
            se_t = se_out[..., :min_len]
            a_t  = a_wav[..., :min_len]

            if not args.no_recon:
                l_l1   = F.l1_loss(se_t, a_t)
                l_stft = stft_loss_fn(se_t, a_t)
                l_recon = l_l1 + l_stft
            else:
                l_recon = torch.zeros(1, device=DEVICE)

            if args.lambda_asr > 0:
                ghost['on'] = True
                if args.branch_dropout > 0:
                    whisper.train()
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    mel_se = log_mel_fn(diversify(se_t))
                    encoder_out = whisper.model.encoder(mel_se)
                    decoder_out = whisper(
                        encoder_outputs=(encoder_out,),
                        labels=label_ids,
                    )
                    l_ce = decoder_out.loss
                ghost['on'] = False
                whisper.eval()
            else:
                l_ce = torch.zeros(1, device=DEVICE)

            loss = l_recon + args.lambda_asr * l_ce

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(se_model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        # Val
        se_model.eval()
        val_losses, val_recons, val_ces = [], [], []

        with torch.no_grad():
            for t_wav, a_wav, label_ids in dev_dl:
                t_wav = t_wav.to(DEVICE)
                a_wav = a_wav.to(DEVICE)
                label_ids = label_ids.to(DEVICE)

                se_out = se_model(t_wav).squeeze(1)
                min_len = min(se_out.shape[-1], a_wav.shape[-1])
                se_t = se_out[..., :min_len]
                a_t  = a_wav[..., :min_len]

                if not args.no_recon:
                    l_l1   = F.l1_loss(se_t, a_t)
                    l_stft = stft_loss_fn(se_t, a_t)
                    l_recon = l_l1 + l_stft
                else:
                    l_recon = torch.zeros(1, device=DEVICE)

                if args.lambda_asr > 0:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        mel_se = log_mel_fn(se_t)
                        encoder_out = whisper.model.encoder(mel_se)
                        decoder_out = whisper(
                            encoder_outputs=(encoder_out,),
                            labels=label_ids,
                        )
                        l_ce = decoder_out.loss
                else:
                    l_ce = torch.zeros(1, device=DEVICE)

                val_losses.append((l_recon + args.lambda_asr * l_ce).item())
                val_recons.append(l_recon.item())
                val_ces.append(l_ce.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_recon  = float(np.mean(val_recons))
        val_ce     = float(np.mean(val_ces))

        marker = ' <- best' if val_loss < best_val else ''
        print(f'Epoch {epoch:2d} | train={train_loss:.4f} | val={val_loss:.4f} '
              f'(recon={val_recon:.4f}, ce={val_ce:.4f}){marker}')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_recon, val_ce])

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(se_model.state_dict(), ckpt_dir / 'best.th')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'Early stopping (patience={args.patience})')
                break

    print(f'\nDone. Best val loss: {best_val:.4f}')
    print(f'Model: {ckpt_dir / "best.th"}')


if __name__ == '__main__':
    main()
