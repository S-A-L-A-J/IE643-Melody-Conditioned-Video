"""New-audio inference using the preserved beat-aware supervised latent model.

This is direct learned latent prediction plus a pretrained VAE decoder, not diffusion.
Source project is read-only; each inference writes a unique assessment output folder.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[2] / 'project')))
PRETRAINED = Path(os.environ.get('IE643_PRETRAINED_ROOT', str(ROOT / 'models')))
OUT = Path(__file__).resolve().parents[1] / 'demo_outputs'
os.environ.setdefault('TORCH_HOME', str(PRETRAINED / 'torch'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT.parent / '.cache/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT.parent / '.cache/matplotlib'))
Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from extract_features import extract
from train_audio_latent_prior import AudioLatentPrior, sample_rhythm
from train_contrastive import normalize_features
from evaluate_e0 import video_events
from metrics import score


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def separate(path, output, device):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    model = get_model('htdemucs').eval()
    wav = convert_audio(torch.from_numpy(audio.T.copy()), sr, model.samplerate, model.audio_channels)
    mean, std = wav.mean(0).mean(), wav.mean(0).std()
    if std < 1e-8:
        raise ValueError('The selected audio is silent.')
    with torch.inference_mode():
        stems = apply_model(model, ((wav-mean)/std)[None], device=device,
                            segment=4, shifts=0, split=True, overlap=.25,
                            progress=False, num_workers=0)[0].cpu()
    stems = stems * std + mean
    instrumental = sum(stems[i] for i, name in enumerate(model.sources) if name != 'vocals')
    sf.write(output, instrumental.mean(0).numpy(), model.samplerate, subtype='FLOAT')
    del model, stems, instrumental
    release()


def context_embedding(path, device):
    from transformers import AutoProcessor, ClapModel
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    audio = librosa.resample(audio.mean(1), orig_sr=sr, target_sr=48000)
    processor = AutoProcessor.from_pretrained(PRETRAINED/'clap', local_files_only=True)
    model = ClapModel.from_pretrained(PRETRAINED/'clap', local_files_only=True).to(device).eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(audio), 480000):
            chunk = audio[start:start+480000]
            if len(chunk) < 48000:
                continue
            data = processor(audios=chunk, sampling_rate=48000, return_tensors='pt')
            vector = model.get_audio_features(**{k:v.to(device) for k,v in data.items()})
            embeddings.append(torch.nn.functional.normalize(vector.float(), dim=-1)[0].cpu())
    vector = torch.nn.functional.normalize(torch.stack(embeddings).mean(0), dim=0)
    del model, processor
    release()
    return vector


def conditioning(arrays, metadata, start_frame, frames=32):
    positions = start_frame + np.arange(frames)*2
    if positions[-1] >= len(arrays['times']):
        raise ValueError('The selected segment must contain at least four seconds of audio.')
    onset = sample_rhythm(arrays['onset_strength'], positions, start_frame, 2, 'causal_peak')
    beat = sample_rhythm(arrays['beat_impulse'], positions, start_frame, 2, 'causal_peak')
    item = {
        'rhythm': torch.tensor(np.stack([onset, beat, np.full(frames,metadata['tempo_bpm'])], -1), dtype=torch.float32),
        'melody': torch.tensor(np.concatenate([arrays['chroma'][positions], arrays['dominant_pitch_hz'][positions,None], arrays['pitch_confidence'][positions,None]], -1)),
        'dynamics': torch.tensor(np.concatenate([arrays['rms'][positions,None], arrays['band_energy'][positions]], -1)),
    }
    return torch.cat(normalize_features(item, torch.arange(frames)), -1)


def generate(audio_path, start=0., remove_vocals=False, device=None, progress=None,
             model_variant='temporal', visual_phase=0., duration=4):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(643)
    np.random.seed(643)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError('Upload an audio file first.')
    start = float(start)
    if not np.isfinite(start) or start < 0:
        raise ValueError('Start time must be a nonnegative number.')
    if model_variant not in ['temporal','baseline']:
        raise ValueError('Choose the temporal model or baseline.')
    duration=float(duration)
    if duration not in (4.,8.):
        raise ValueError('Choose a duration of 4 or 8 seconds.')
    duration=int(duration)
    if model_variant=='baseline' and duration!=4:
        raise ValueError('The earlier baseline supports 4 seconds. Select the improved temporal model for 8 seconds.')
    visual_phase=float(visual_phase)
    if not np.isfinite(visual_phase):
        raise ValueError('Visual phase must be a finite angle.')
    # The original training data uses 30-second excerpts. Analyse up to 30 seconds
    # around the selected clip to retain tempo/context rather than only four seconds.
    window_start = max(0., start-13.)
    offset = start-window_start
    directory = OUT / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    def stage(label):
        if progress:
            progress(label)
    stage('Reading audio')
    decoded = directory/'input.wav'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(window_start),'-i',str(Path(audio_path).resolve()),
                    '-t','30','-vn','-ac','1','-ar','44100','-c:a','pcm_f32le',str(decoded)],check=True,capture_output=True)
    samples, sr = sf.read(decoded, dtype='float32')
    if len(samples)/sr < offset+duration-1e-4:
        raise ValueError(f'Choose a start time with at least {duration} seconds of audio remaining.')
    if not np.isfinite(samples).all() or np.sqrt(np.mean(samples**2)) < 1e-7:
        raise ValueError('The selected audio is silent or invalid.')
    instrumental = decoded
    if remove_vocals:
        stage('Suppressing vocals')
        instrumental = directory/'instrumental.wav'
        separate(decoded, instrumental, device)
    stage('Extracting musical features')
    arrays, metadata = extract(instrumental)
    # Align audio cropping and feature indexing to the same 16-Hz grid.
    first = int(round(offset*16))
    offset = first/16
    if len(samples)/sr < offset+duration-1e-4:
        raise ValueError(f'Move the start time slightly earlier to fit a full {duration}-second segment.')
    music = conditioning(arrays, metadata, first, frames=duration*8)
    np.savez_compressed(directory/'features.npz', **arrays)
    stage('Encoding audio context')
    context = context_embedding(instrumental, device)
    stage('Generating video')
    if model_variant=='temporal':
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        from train_temporal_prior import TemporalPrior
        checkpoint=Path(__file__).resolve().parents[1]/'experiments/temporal_phase/model.safetensors'
        model=TemporalPrior().to(device).eval()
    else:
        checkpoint = ROOT/'outputs/prior_32f_causal_peak/model.safetensors'
        model = AudioLatentPrior().to(device).eval()
    model.load_state_dict(load_file(checkpoint))
    with torch.inference_mode():
        if model_variant=='temporal':
            angle=np.deg2rad(visual_phase)
            phase=torch.tensor([[np.sin(angle),np.cos(angle)]],dtype=torch.float32,device=device)
            x=music[None].to(device);c=context[None].to(device)
            # Preserve the trained 32-frame context. Beyond it, emit only the
            # last frame of each rolling window, as in the saved long demos.
            initial=model(x[:,:32],c,phase).cpu()
            pieces=[initial]
            for window_start_frame in range(32,x.shape[1],16):
                indices=range(window_start_frame,min(window_start_frame+16,x.shape[1]))
                batch=torch.cat([x[:,int(i)-31:int(i)+1] for i in indices],dim=0)
                output=model(batch,c.expand(len(indices),-1),phase.expand(len(indices),-1))
                pieces.append(output[:,:,-1].permute(1,0,2,3)[None].cpu())
            latents=torch.cat(pieces,dim=2)
        else:
            latents = model(music[None].to(device),context[None].to(device)).cpu()
    del model
    release()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(PRETRAINED/'diffusion/sd15', subfolder='vae', variant='fp16',
                                       torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    silent = directory/'silent.mp4'
    writer = cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),8,(128,128))
    if not writer.isOpened():
        raise RuntimeError('Could not open video encoder.')
    try:
        with torch.inference_mode():
            for chunk in latents[0].permute(1,0,2,3).split(4):
                frames = ((vae.decode(chunk.to(device)/vae.config.scaling_factor).sample/2+.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
                for frame in frames:
                    writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    del vae
    release()
    video = directory/'generated.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(silent),'-ss',str(offset),'-i',str(instrumental),
                    '-map','0:v:0','-map','1:a:0','-t',str(duration),'-c:v','libx264','-pix_fmt','yuv420p',
                    '-c:a','aac','-movflags','+faststart',str(video)],check=True,capture_output=True)
    events, changes = video_events(video,fps=8)
    beats = arrays['beat_times']-offset
    beats = beats[(beats>=0)&(beats<duration)]
    metrics = score(beats,events)
    report = {'model':f'{model_variant} supervised latent predictor + pretrained VAE (not diffusion)',
              'visual_phase_degrees':visual_phase if model_variant=='temporal' else None,
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'input_name':Path(audio_path).name,'requested_start_seconds':start,
              'actual_start_seconds':window_start+offset,'duration_seconds':duration,'fps':8,'resolution':[128,128],
              'temporal_method':'trained four-second context; rolling last-frame prediction for eight-second output',
              'vocal_suppression':bool(remove_vocals),'device':device,
              'tempo_bpm':metadata['tempo_bpm'],'audio_beats':len(beats),'visual_events':len(events),
              'timing':metrics if len(beats) else None,
              'timing_method':'one-to-one event matching within 125 ms of estimated audio beats',
              'shifted_reference_200ms':score(beats+.2,events) if len(beats) else None,
              'runtime_seconds':round(time.monotonic()-started,2),
              'limitations':['Exploratory frame-change event metric, not a human quality score.',
                            'Synthetic abstract style; mood accuracy and human preference are unvalidated.',
                            'Evaluated on validation and a small external film-music sample; broader song generalization remains unverified.',
                            'New upload is not automatically an independent test sample.']}
    (directory/'result.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,1,figsize=(9,4),sharex=True)
    indices = slice(first,first+duration*16)
    times = arrays['times'][indices]-offset
    ax[0].plot(times,arrays['onset_strength'][indices],label='Audio onset strength')
    for beat in beats:ax[0].axvline(beat,color='tab:orange',alpha=.6)
    ax[0].legend();ax[0].set_ylabel('Audio')
    ax[1].plot(np.arange(len(changes))/8,changes,label='Visual frame change')
    for event in events:ax[1].axvline(event,color='tab:green',alpha=.6)
    ax[1].legend();ax[1].set_ylabel('Video');ax[1].set_xlabel('Time (seconds)')
    fig.tight_layout();fig.savefig(directory/'alignment.png',dpi=120);plt.close(fig)
    return str(video),report,str(directory/'alignment.png'),str(directory/'result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--duration',type=int,choices=[4,8],default=4)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--remove-vocals',action='store_true')
    parser.add_argument('--device',choices=['cuda','cpu'])
    parser.add_argument('--model',choices=['temporal','baseline'],default='temporal')
    parser.add_argument('--visual-phase',type=float,default=0.)
    args = parser.parse_args()
    video,report,_,_ = generate(args.audio,args.start,args.remove_vocals,args.device,lambda s: print(s,flush=True),args.model,args.visual_phase,args.duration)
    print(json.dumps({'video':video,**report},indent=2))
