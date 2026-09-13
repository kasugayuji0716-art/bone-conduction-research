"""
スクリプト51: Cross-entropy損失によるASR-aware SE学習
encoder距離（L1）の代わりにWhisper decoder出力のCE損失を使用。
より直接的なASR最適化。

損失 = L1+STFT（波形品質） + λ × CE(Whisper(SE出力), テキスト)

使い方（DNN PC）:
    python scripts/51_train_se_ce_loss.py --lambda_asr 5.0 --tag ce_lambda_5.0
    python scripts/51_train_se_ce_loss.py --lambda_asr 1.0 --tag ce_lambda_1.0
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


class WhisperLogMel(nn.Module):
    N_FRAMES = 3000

    def __init__(self, sr=16000):
        super().__init__()
        self.mel = Ta.MelSpectrogram(
            sample_rate=sr, n_fft=400, win_length=400, hop_length=160,
            n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
            window_fn=torch.hann_window, normalized=False,
        )

    def forward(self, x):
        mel = self.mel(x)
        log_mel = torch.log10(mel.clamp(min=1e-10))
        max_val = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, max_val - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        if log_mel.shape[-1] < self.N_FRAMES:
            log_mel = F.pad(log_mel, (0, self.N_FRAMES - log_mel.shape[-1]))
        else:
            log_mel = log_mel[..., :self.N_FRAMES]
        return log_mel


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, resolutions=((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))):
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
        return sc + mag

    def forward(self, est, target):
        loss = 0
        for n_fft, hop, win in self.resolutions:
            loss += self._stft_loss(est, target, n_fft, hop, win)
        return loss / len(self.resolutions)


class TAPSPairDataset(Dataset):
    def __init__(self, split, max_sec=15.0):
        self.samples = []
        meta = TAPS_DIR / f'metadata_{split}.csv'
        with open(meta, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sid, uid = row['speaker_id'], row['sentence_id']
                t = TAPS_DIR / 'throat'   / split / f'{sid}_{uid}.wav'
                a = TAPS_DIR / 'acoustic' / split / f'{sid}_{uid}.wav'
                if t.exists() and a.exists():
                    self.samples.append((t, a, row['text']))
        self.max_len = int(max_sec * TARGET_SR)
        print(f'  [{split}] {len(self.samples)} pairs')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_path, a_path, text = self.samples[idx]
        t_wav, _ = sf.read(t_path, dtype='float32')
        a_wav, _ = sf.read(a_path, dtype='float32')
        if t_wav.ndim > 1: t_wav = t_wav.mean(axis=1)
        if a_wav.ndim > 1: a_wav = a_wav.mean(axis=1)
        min_len = min(len(t_wav), len(a_wav), self.max_len)
        return torch.from_numpy(t_wav[:min_len]), torch.from_numpy(a_wav[:min_len]), text


def collate_fn(batch, tokenizer=None):
    t_wavs, a_wavs, texts = zip(*batch)
    max_len = max(t.shape[0] for t in t_wavs)
    def pad(wavs):
        return torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in wavs])

    labels = tokenizer(texts, return_tensors='pt', padding=True)
    label_ids = labels['input_ids']
    # -100 for padding tokens (ignore in CE loss)
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    return pad(t_wavs), pad(a_wavs), label_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_asr', type=float, default=5.0)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=4,
                        help='Full Whisper forward requires more VRAM, use smaller batch')
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--patience',   type=int,   default=5)
    parser.add_argument('--tag',        type=str,   default='ce_lambda_5.0')
    args = parser.parse_args()

    ckpt_dir = BASE_DIR / 'checkpoints' / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== SE-Conformer ASR-aware (Cross-Entropy Loss) ===')
    print(f'lambda={args.lambda_asr} | tag={args.tag} | device={DEVICE}\n')

    # SE model
    from models.seconformer import seconformer as TAPSSeconformer
    se_model = TAPSSeconformer(**TAPS_SE_CONFIG).to(DEVICE)
    state = torch.load(PRETRAINED_DIR / 'seconformer.th', map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    se_model.load_state_dict(state)
    n_params = sum(p.numel() for p in se_model.parameters() if p.requires_grad)
    print(f'SE-Conformer: pretrained loaded ({n_params/1e6:.1f}M params)')

    # STFT loss
    stft_loss_fn = MultiResolutionSTFTLoss().to(DEVICE)

    # Whisper (full model, frozen)
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(DEVICE)
    whisper.eval()
    for p in whisper.parameters():
        p.requires_grad_(False)
    print('Whisper (full): frozen')

    tokenizer = processor.tokenizer
    # Force Korean
    forced_decoder_ids = processor.get_decoder_prompt_ids(language='ko', task='transcribe')
    whisper.config.forced_decoder_ids = forced_decoder_ids

    # log-mel
    log_mel_fn = WhisperLogMel().to(DEVICE)

    # Data
    print('\nData:')
    train_ds = TAPSPairDataset('train')
    dev_ds   = TAPSPairDataset('dev')

    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer)

    n_workers = min(2, os.cpu_count() or 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=n_workers, pin_memory=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=n_workers, pin_memory=True)

    optimizer = torch.optim.Adam(se_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler('cuda')

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

            with torch.amp.autocast('cuda'):
                se_out = se_model(t_wav).squeeze(1)
                min_len = min(se_out.shape[-1], a_wav.shape[-1])
                se_t = se_out[..., :min_len]
                a_t  = a_wav[..., :min_len]

                l_l1   = F.l1_loss(se_t, a_t)
                l_stft = stft_loss_fn(se_t, a_t)
                l_recon = l_l1 + l_stft

                if args.lambda_asr > 0:
                    mel_se = log_mel_fn(se_t)
                    encoder_out = whisper.model.encoder(mel_se)
                    decoder_out = whisper(
                        encoder_outputs=(encoder_out,),
                        labels=label_ids,
                    )
                    l_ce = decoder_out.loss
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

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for t_wav, a_wav, label_ids in dev_dl:
                t_wav = t_wav.to(DEVICE)
                a_wav = a_wav.to(DEVICE)
                label_ids = label_ids.to(DEVICE)

                se_out = se_model(t_wav).squeeze(1)
                min_len = min(se_out.shape[-1], a_wav.shape[-1])
                se_t = se_out[..., :min_len]
                a_t  = a_wav[..., :min_len]

                l_l1   = F.l1_loss(se_t, a_t)
                l_stft = stft_loss_fn(se_t, a_t)
                l_recon = l_l1 + l_stft

                if args.lambda_asr > 0:
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
