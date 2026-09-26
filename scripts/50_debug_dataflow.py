"""
スクリプト50: SE→Whisperのデータフロー可視化
1発話分のデータが各ステップでどう変換されるかを表示する。

使い方（DNN PC）:
    python scripts/50_debug_dataflow.py
"""

import sys, torch, soundfile as sf, numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'taps-baselines'))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── テスト音声を1つ読む ──
test_dir = BASE_DIR / 'data' / 'raw' / 'taps' / 'throat' / 'test'
if not test_dir.exists():
    test_dir = BASE_DIR / 'data' / 'raw' / 'taps' / 'throat'  # Mac用
wav_path = sorted(test_dir.glob('*.wav'))[0]
wav, sr = sf.read(wav_path, dtype='float32')

print('=' * 60)
print('SEモデル → Whisper のデータフロー')
print('=' * 60)

# ==============================
# STEP 0: 入力音声
# ==============================
print(f'\n--- STEP 0: 入力音声 ---')
print(f'ファイル: {wav_path.name}')
print(f'サンプリングレート: {sr} Hz')
print(f'長さ: {len(wav)} samples = {len(wav)/sr:.2f} 秒')
print(f'shape: {wav.shape}')
print(f'値の範囲: [{wav.min():.4f}, {wav.max():.4f}]')
print(f'mean: {wav.mean():.4f}, std: {wav.std():.4f}')

# ==============================
# STEP 1: SE-Conformer
# ==============================
print(f'\n{"="*60}')
print(f'--- STEP 1: SE-Conformer ---')
print(f'入力 → encoder → conformer → decoder → 出力')

try:
    from models.seconformer import seconformer
    from models.demucs import upsample2, downsample2

    se = seconformer(hidden=64, conformer_dim=512, conformer_ffn_dim=64,
                     conformer_depth=4, depthwise_conv_kernel_size=15)
    ckpt = BASE_DIR / 'taps-baselines' / 'pretrained' / 'seconformer.th'
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'model' in state: state = state['model']
    se.load_state_dict(state)
    se.eval()

    # SE内部を手動でトレース
    x = torch.from_numpy(wav).unsqueeze(0)
    print(f'\n[入力] shape: {tuple(x.shape)} → (batch=1, time={x.shape[1]})')
    x = x.unsqueeze(1)
    print(f'[unsqueeze] shape: {tuple(x.shape)} → (batch, channels=1, time)')

    mono = x.mean(dim=1, keepdim=True)
    std = mono.std(dim=-1, keepdim=True)
    x_norm = x / (1e-3 + std)
    print(f'[正規化] std={std.item():.4f}, 正規化後 range: [{x_norm.min():.2f}, {x_norm.max():.2f}]')

    length = x_norm.shape[-1]
    valid_len = se.valid_length(length)
    x_pad = torch.nn.functional.pad(x_norm, (0, valid_len - length))
    print(f'[padding] {length} → {valid_len} samples (+ {valid_len - length} padding)')

    x_up = upsample2(upsample2(x_pad))
    print(f'[upsample ×4] shape: {tuple(x_up.shape)} → 時間が4倍')

    print(f'\n[Encoder: Conv1d × 4層]')
    skips = []
    h = x_up
    for i, enc in enumerate(se.encoder):
        h = enc(h); skips.append(h)
        print(f'  層{i}: shape {tuple(h.shape)} → channels={h.shape[1]}, time={h.shape[2]}')

    print(f'\n[Conformer blocks × {len(se.conformers)}]')
    h = h.permute(2, 0, 1)
    print(f'  permute: → {tuple(h.shape)} (time, batch, channels)')
    for i, conf in enumerate(se.conformers):
        h = conf(h, None)
        print(f'  block {i}: shape {tuple(h.shape)}')
    h = h.permute(1, 2, 0)
    print(f'  permute戻し: {tuple(h.shape)}')

    print(f'\n[Decoder: ConvTranspose1d × 4層 + skip接続]')
    for i, dec in enumerate(se.decoder):
        skip = skips.pop(-1)
        h = h + skip[..., :h.shape[-1]]
        h = dec(h)
        print(f'  層{i}: + skip → shape {tuple(h.shape)}')

    h = downsample2(downsample2(h))
    h = h[..., :length]
    se_output = (std * h).squeeze()
    print(f'[downsample + trim + 逆正規化] → shape: {tuple(se_output.shape)}')
    print(f'  出力 range: [{se_output.min():.4f}, {se_output.max():.4f}]')
    se_np = se_output.detach().numpy()

except FileNotFoundError:
    print('(SEチェックポイントなし → 生音声をそのまま使用)')
    se_np = wav

# ==============================
# STEP 2: Whisper Feature Extractor (log-mel)
# ==============================
print(f'\n{"="*60}')
print(f'--- STEP 2: Log-Mel Spectrogram ---')
print(f'音声波形 → STFT → メル周波数変換 → 対数 → 正規化')

from transformers import WhisperFeatureExtractor
feat_ext = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')

inputs = feat_ext(se_np, sampling_rate=16000, return_tensors='pt',
                  padding='max_length', max_length=480000)
mel = inputs.input_features  # (1, 80, 3000)
print(f'入力音声: {len(se_np)} samples ({len(se_np)/16000:.2f}秒)')
print(f'出力mel: shape {tuple(mel.shape)}')
print(f'  → batch=1, mel_bins=80, time_frames=3000')
print(f'  → 3000 frames = 30秒分（短い音声は0でpadding）')
print(f'  → 80 mel bins = 0〜8000Hzを80のメル周波数帯域に分割')
print(f'  値の範囲: [{mel.min():.2f}, {mel.max():.2f}]')

# ==============================
# STEP 3: Whisper Encoder
# ==============================
print(f'\n{"="*60}')
print(f'--- STEP 3: Whisper Encoder ---')
print(f'log-mel → Conv1d×2 → Transformer × 12層 → encoder出力')

from transformers import WhisperForConditionalGeneration
whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
whisper.eval()

with torch.no_grad():
    encoder_out = whisper.model.encoder(mel)

hidden = encoder_out.last_hidden_state
print(f'入力: {tuple(mel.shape)} (batch, mel_bins=80, frames=3000)')
print(f'出力: {tuple(hidden.shape)}')
print(f'  → batch=1, time_frames=1500, hidden_dim=768')
print(f'  → 3000 frames → 1500 frames（Conv1dで半分に圧縮）')
print(f'  → 各フレームが768次元のベクトル')
print(f'  → この768次元に「この時刻の音声の意味」が圧縮されている')
print(f'  値の範囲: [{hidden.min():.2f}, {hidden.max():.2f}]')

# ==============================
# STEP 4: Whisper Decoder
# ==============================
print(f'\n{"="*60}')
print(f'--- STEP 4: Whisper Decoder ---')
print(f'encoder出力 + これまでの文字 → 次の文字の確率分布')

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained('openai/whisper-small')

# 言語トークンを設定
forced_decoder_ids = processor.get_decoder_prompt_ids(language='ko', task='transcribe')
# 最初のdecoder入力: <|startoftranscript|><|ko|><|transcribe|><|notimestamps|>
decoder_input = torch.tensor([[50258, 50264, 50359, 50363]])  # Whisper韓国語設定

with torch.no_grad():
    out = whisper(encoder_outputs=(hidden,), decoder_input_ids=decoder_input)

logits = out.logits  # WhisperForConditionalGenerationはlogitsを持つ
print(f'decoder入力: {tuple(decoder_input.shape)} → 4つの特殊トークン')
print(f'decoder出力 (logits): {tuple(logits.shape)}')
print(f'  → batch=1, steps=4, vocab_size={logits.shape[-1]}')
print(f'  → 各stepで{logits.shape[-1]}語の確率を出力')

# 最後のstepの予測（= 最初に生成する文字）
last_logits = logits[0, -1, :]  # (vocab_size,)
probs = torch.softmax(last_logits, dim=-1)
top5 = torch.topk(probs, 5)
print(f'\n最初に生成する文字のtop5:')
for prob, idx in zip(top5.values, top5.indices):
    token = processor.tokenizer.decode([idx.item()])
    print(f'  "{token}" ({idx.item()}): {prob.item()*100:.1f}%')

# ==============================
# STEP 5: Cross-entropy損失の計算例
# ==============================
print(f'\n{"="*60}')
print(f'--- STEP 5: Cross-entropy損失 ---')

# 正解テキストをトークン化
import csv
meta = list(csv.DictReader(open(BASE_DIR / 'data' / 'raw' / 'taps' / 'metadata_test.csv')))
ref_text = meta[0]['text']
print(f'正解テキスト: {ref_text}')

labels = processor.tokenizer(ref_text, return_tensors='pt').input_ids
print(f'正解トークン: shape {tuple(labels.shape)} → {labels.shape[1]}トークン')
print(f'  tokens: {labels[0][:10].tolist()}... (最初の10個)')

# decoder全体を通す
with torch.no_grad():
    out = whisper(encoder_outputs=(hidden,), decoder_input_ids=labels[:, :-1])
    logits = out.logits  # (1, seq_len, vocab_size)

# cross-entropy計算
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, logits.shape[-1]),
    labels[:, 1:].reshape(-1)  # 1つずらす（次の文字を予測）
)
print(f'\nlogits: {tuple(logits.shape)}')
print(f'labels: {tuple(labels[:, 1:].shape)}')
print(f'cross-entropy loss = {loss.item():.4f}')
print(f'  → この値が小さいほど、SE出力からWhisperが正しく転写できている')

# ==============================
# まとめ
# ==============================
print(f'\n{"="*60}')
print(f'--- まとめ: データの流れ ---')
print(f'''
喉マイク音声   ({len(wav)} samples, {len(wav)/sr:.1f}秒)
  ↓ SE-Conformer
SE出力         ({len(se_np)} samples)
  ↓ log-mel spectrogram
mel特徴量      (80 bins × 3000 frames)
  ↓ Whisper Encoder (Conv×2 + Transformer×12)
encoder表現    (1500 frames × 768 dim)
  ↓ Whisper Decoder (Transformer×12)
logits         (テキスト長 × {logits.shape[-1]} vocab)
  ↓ cross-entropy with 正解テキスト
損失値         {loss.item():.4f}
  ↓ backward()
SEモデルの重みを更新
''')
