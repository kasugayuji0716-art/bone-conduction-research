"""
スクリプト46: TAPS公式SE-Conformerの正しいCERを測定
公式コード（torchaudio.ConformerLayer）で推論し、正しいベースラインを取得する。
"""
import csv, sys, torch, numpy as np, soundfile as sf
from pathlib import Path
from jiwer import cer
from faster_whisper import WhisperModel

sys.path.insert(0, 'taps-baselines')
from models.seconformer import seconformer

# TAPS公式SE-Conformer（チェックポイントに合わせたハイパーパラメータ）
model = seconformer(hidden=64, conformer_dim=512, conformer_ffn_dim=64,
                    conformer_depth=4, depthwise_conv_kernel_size=15)
st = torch.load('taps-baselines/pretrained/seconformer.th',
                map_location='cpu', weights_only=False)
if isinstance(st, dict) and 'model' in st:
    st = st['model']
model.load_state_dict(st)
model.eval()
print('TAPS SE-Conformer (official) loaded')

ct2 = Path('/tmp/whisper-small-ct2')
model_id = str(ct2) if ct2.exists() else 'openai/whisper-small'
asr = WhisperModel(model_id, device='cuda', compute_type='float16')
print(f'ASR: {model_id}')

cers = []
for row in csv.DictReader(open('data/raw/taps/metadata_test.csv')):
    p = Path(f"data/raw/taps/throat/test/{row['speaker_id']}_{row['sentence_id']}.wav")
    if not p.exists():
        continue
    w, _ = sf.read(p, dtype='float32')
    with torch.no_grad():
        o = model(torch.from_numpy(w).unsqueeze(0)).squeeze().numpy()
    segs, _ = asr.transcribe(o, language='ko', beam_size=5)
    hyp = ''.join(s.text for s in segs).strip()
    cers.append(min(cer(row['text'], hyp), 1.0))
    if len(cers) % 100 == 0:
        print(f'  {len(cers)} utterances done...')

print(f'\nTAPS SE-Conformer (official): CER = {np.mean(cers):.4f} (n={len(cers)})')
