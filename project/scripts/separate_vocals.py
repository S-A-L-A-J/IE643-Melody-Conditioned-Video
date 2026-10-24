"""Separate indexed audio with bounded GPU segments and atomic resumable outputs."""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT/'models/torch'))
import numpy as np
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--segment', type=float, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    model = get_model('htdemucs').eval()
    records = [json.loads(x) for x in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines()]
    records = [r for r in records if r['eligible']]
    if args.limit:
        records = records[:args.limit]
    for i, row in enumerate(records):
        dest = ROOT/'data/stems'/row['id']
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest/'complete.json'
        if marker.exists() and (dest/'vocals.wav').exists() and (dest/'instrumental.wav').exists():
            continue
        start = time.monotonic()
        audio, sr = sf.read(ROOT/row['path'], always_2d=True, dtype='float32')
        wav = convert_audio(torch.from_numpy(audio.T.copy()), sr, model.samplerate, model.audio_channels)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std()
        if std < 1e-8:
            raise ValueError(f"Silent input: {row['id']}")
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            sources = apply_model(model, ((wav-mean)/std)[None], device=args.device,
                                  segment=args.segment, shifts=0, split=True,
                                  overlap=0.25, progress=False, num_workers=0)[0].cpu()
        sources = sources * std + mean
        vocals = sources[model.sources.index('vocals')]
        instrumental = sum(sources[j] for j,s in enumerate(model.sources) if s != 'vocals')
        for name, signal in [('vocals',vocals),('instrumental',instrumental)]:
            if not torch.isfinite(signal).all():
                raise ValueError(f'Non-finite {name}')
            temp = dest/(name+'.partial.wav')
            # Float WAV avoids independent stem normalization or clipping.
            stored = signal.mean(0).numpy() if audio.shape[1] == 1 else signal.T.numpy()
            sf.write(temp, stored, model.samplerate, subtype='FLOAT')
            temp.replace(dest/(name+'.wav'))
        result = dict(track_id=row['id'], model='htdemucs', sample_rate=model.samplerate,
                      samples=instrumental.shape[-1], segment=args.segment, shifts=0,
                      seconds=round(time.monotonic()-start,2),
                      peak_vram_mb=round(torch.cuda.max_memory_allocated()/2**20,1) if args.device=='cuda' else None,
                      quality_status='needs_listening_review; vocals may leak; instrumental melody may be affected')
        marker.write_text(json.dumps(result, indent=2))
        print(f"{i+1}/{len(records)} {row['id']} {result}", flush=True)

if __name__ == '__main__':
    main()
