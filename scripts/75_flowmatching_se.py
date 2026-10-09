"""
スクリプト75: 事前学習済み生成モデル（NVIDIA NeMo flow matching, sr_ssl_flowmatching_16k_430m）を
TAPS 喉マイク→気導マイクで追加学習し、強調音声を保存する。

モデル: https://huggingface.co/nvidia/sr_ssl_flowmatching_16k_430m （Ku et al., arXiv 2409.16117、CC-BY-NC-SA-4.0）
事前学習はマスク復元なので、帯域拡張には追加学習が必要。ASR損失は使わない（ASR非依存）。

venv-nemo（nemo_toolkit[audio]）で実行する。
    python scripts/75_flowmatching_se.py manifest
    python scripts/75_flowmatching_se.py train --tag fm_taps --max_steps 20000
    python scripts/75_flowmatching_se.py infer --tag fm_taps --split test [--steps 20] [--limit N]
強調音声: data/processed/se_wav/<tag>/<split>/<utt>.wav（script 66 がそのまま読む）
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from dataset_cfg import RAW_DIR as TAPS_DIR, SE_WAV as _SE_WAV   # BCR_DATASET で切り替え（学習は TAPS のみ）
WORK = BASE_DIR / 'data' / 'processed' / 'nemo75'
SE_WAV = _SE_WAV
EXAMPLES = BASE_DIR / 'third_party' / 'nemo_examples'
RAW = 'https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/examples/audio/'
PRETRAINED = 'nvidia/sr_ssl_flowmatching_16k_430m'


def manifest():
    import soundfile as sf
    WORK.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'dev']:
        n = 0
        with open(WORK / f'{split}.json', 'w') as out, \
                open(TAPS_DIR / f'metadata_{split}.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name = f"{row['speaker_id']}_{row['sentence_id']}.wav"
                t, a = TAPS_DIR / 'throat' / split / name, TAPS_DIR / 'acoustic' / split / name
                if t.exists() and a.exists():
                    dur = min(sf.info(t).duration, sf.info(a).duration)
                    out.write(json.dumps(dict(noisy_filepath=str(t), clean_filepath=str(a), duration=dur)) + '\n')
                    n += 1
        print(f'{split}: {n} pairs → {WORK / (split + ".json")}')
    # 学習中の検証用: dev から話者ごとに均等に100発話（生成は発話全体を20ステップで解くので全1000発話だと遅い）
    lines = open(WORK / 'dev.json').read().splitlines()
    open(WORK / 'dev100.json', 'w').write('\n'.join(lines[::10][:100]) + '\n')


def fetch_examples():
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for rel in ['audio_to_audio_train.py', 'conf/flow_matching_generative_finetuning.yaml']:
        dst = EXAMPLES / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RAW + rel, dst)
            print(f'fetched {rel}')
    # 既定の init_from_nemo_model: null があると初期化指定が2つとみなされるので、事前学習モデルに置き換える
    conf = EXAMPLES / 'conf' / 'flow_matching_generative_finetuning.yaml'
    txt = conf.read_text()
    txt = txt.replace('init_from_nemo_model: null', f'init_from_pretrained_model: {PRETRAINED}')
    conf.write_text(txt)
    (EXAMPLES / 'conf' / 'flow_matching_scratch.yaml').write_text(   # 対照: 同じ構造を初期値ランダムで学習
        txt.replace(f'init_from_pretrained_model: {PRETRAINED}', 'init_from_nemo_model: null'))


def train(args):
    fetch_examples()
    exp = BASE_DIR / 'checkpoints' / args.tag
    cmd = [sys.executable, str(EXAMPLES / 'audio_to_audio_train.py'),
           f'--config-path={EXAMPLES / "conf"}',
           '--config-name=' + ('flow_matching_scratch' if args.scratch else 'flow_matching_generative_finetuning'),
           f'model.train_ds.manifest_filepath={WORK / "train.json"}',
           f'model.validation_ds.manifest_filepath={WORK / "dev100.json"}',
           '~model.log_config',
           f'model.train_ds.batch_size={args.batch_size}',
           'model.validation_ds.batch_size=1',
           'model.max_utts_evaluation_metrics=200',
           f'+model.optim.sched.max_steps={args.max_steps}',
           f'model.optim.sched.warmup_steps={args.warmup}',
           f'trainer.max_steps={args.max_steps}',
           f'trainer.val_check_interval={args.val_every}',
           'trainer.check_val_every_n_epoch=null',
           'trainer.devices=1', 'trainer.strategy=auto', 'trainer.precision=32',
           'trainer.sync_batchnorm=false',
           f'exp_manager.exp_dir={exp}', f'exp_manager.name={args.tag}',
           'exp_manager.early_stopping_callback_params.patience=1000',
           'exp_manager.checkpoint_callback_params.save_top_k=1']   # 1つ約5GB（最適化状態込み）
    if args.resume:   # 中断した学習をチェックポイント（*-last.ckpt）から続ける
        cmd.append(f"exp_manager.resume_from_checkpoint='{Path(args.resume).resolve()}'")   # ファイル名の = を Hydra が誤解しないよう引用
    print(' '.join(cmd), flush=True)
    env = dict(os.environ)
    if args.resume:   # 自前のチェックポイント。NeMo の EMA コールバックが weights_only=True で読めないため
        env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    subprocess.run(cmd, check=True, env=env)


def find_nemo(tag):
    cands = sorted((BASE_DIR / 'checkpoints' / tag).rglob('*.nemo'), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f'no .nemo under checkpoints/{tag}')
    return cands[-1]


def infer(args):
    import numpy as np
    import soundfile as sf
    import torch
    from nemo.collections.audio.models import AudioToAudioModel
    path = Path(args.nemo) if args.nemo else find_nemo(args.tag)
    print(f'model: {path}', flush=True)
    model = AudioToAudioModel.restore_from(str(path), map_location='cuda').eval()
    if args.weights:   # script 77 で追加学習した重み（state_dict）を上書き
        model.load_state_dict(torch.load(args.weights, map_location='cuda'))
        print(f'weights: {args.weights}', flush=True)
    model.sampler.num_steps = args.steps
    out_dir = SE_WAV / (args.out or args.tag) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(TAPS_DIR / f'metadata_{args.split}.csv', encoding='utf-8')))
    rows = rows[:args.limit or None]
    win = torch.hann_window(512)
    for i, row in enumerate(rows):
        utt = f"{row['speaker_id']}_{row['sentence_id']}"
        dst = out_dir / f'{utt}.wav'
        if dst.exists():
            continue
        wav, sr = sf.read(TAPS_DIR / 'throat' / args.split / f'{utt}.wav', dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav)[None, None].cuda()          # (B, C, T)
        ys = []
        for k in range(args.n_avg):                            # 発話ごとに乱数を固定（再開しても同じ出力）
            torch.manual_seed(args.seed * 1000003 + k * 7919 + i)
            with torch.no_grad():
                y, _ = model.forward(input_signal=x, input_length=torch.tensor([x.shape[-1]], device='cuda'))
            ys.append(y.float().squeeze().cpu()[:len(wav)])
        if len(ys) == 1:
            y = ys[0].numpy()
        else:   # N個のサンプルの振幅を平均し、位相は1つ目のサンプルのもの（script 67 の振幅統合と同じ）
            S = [torch.stft(v, 512, 128, window=win, return_complex=True) for v in ys]
            M = torch.stack([z.abs() for z in S]).mean(0)
            y = torch.istft(M * torch.exp(1j * S[0].angle()), 512, 128, window=win, length=len(wav)).numpy()
        sf.write(dst, y.astype(np.float32), sr)
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(rows)}', flush=True)
    print(f'done → {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['manifest', 'train', 'infer'])
    ap.add_argument('--tag', default='fm_taps')
    ap.add_argument('--max_steps', type=int, default=20000)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--split', default='test')
    ap.add_argument('--steps', type=int, default=20, help='サンプラーのステップ数')
    ap.add_argument('--nemo', default='', help='.nemo のパス（省略時は checkpoints/<tag> の最新）')
    ap.add_argument('--out', default='', help='出力名（省略時は tag）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_avg', type=int, default=1, help='N個のサンプルを振幅平均')
    ap.add_argument('--weights', default='', help='script 77 などで追加学習した state_dict（.pt）')
    ap.add_argument('--resume', default='', help='続きから学習するチェックポイント（*-last.ckpt）')
    ap.add_argument('--scratch', action='store_true', help='事前学習なし（初期値ランダム）で学習する対照')
    args = ap.parse_args()
    {'manifest': lambda: manifest(), 'train': lambda: train(args), 'infer': lambda: infer(args)}[args.stage]()


if __name__ == '__main__':
    main()
