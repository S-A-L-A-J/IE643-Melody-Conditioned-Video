import os
"""Extract aligned instrumental rhythm, melody proxies and dynamics at 16 Hz."""
import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/features'
SR = 22050
HOP = 512
N_FFT = 2048
FEATURE_FPS = 16

def interp(source_times, values, target_times):
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(target_times, source_times, values).astype(np.float32)
    return np.stack([np.interp(target_times, source_times, channel)
                     for channel in values], axis=-1).astype(np.float32)

def extract(audio_path):
    audio, original_sr = sf.read(audio_path, dtype='float32', always_2d=True)
    y = librosa.to_mono(audio.T)
    if original_sr != SR:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=SR)
    duration = len(audio) / original_sr
    times = np.arange(int(np.floor(duration * FEATURE_FPS + 1e-8)), dtype=np.float32) / FEATURE_FPS
    if not len(times):
        raise ValueError('Audio shorter than one feature frame')

    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    harmonic, percussive = librosa.decompose.hpss(stft)
    native_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=SR, hop_length=HOP)

    # Rhythm is measured on the percussive spectrogram; beats remain separate events.
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(percussive**2 + 1e-10),
                                         sr=SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP,
                                                 trim=False)
    onset_times = librosa.frames_to_time(np.arange(len(onset)), sr=SR, hop_length=HOP)
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    beat_times = beat_times[beat_times < duration]
    onset_aligned = interp(onset_times, onset, times)
    beat_impulse = np.zeros(len(times), dtype=np.float32)
    if len(beat_times):
        indices = np.rint(beat_times * FEATURE_FPS).astype(int)
        beat_impulse[np.clip(indices, 0, len(times)-1)] = 1

    # Chroma tracks harmonic pitch-class activity; the strongest spectral peak is
    # only a pitch proxy in polyphonic music, not a verified lead melody.
    chroma = librosa.feature.chroma_stft(S=harmonic**2, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         tuning=None)
    chroma = interp(native_times[:chroma.shape[1]], chroma, times)
    pitches, magnitudes = librosa.piptrack(S=harmonic, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                           fmin=80, fmax=1500, threshold=0.15)
    peak_bins = magnitudes.argmax(axis=0)
    columns = np.arange(magnitudes.shape[1])
    pitch_hz = pitches[peak_bins, columns]
    peak_magnitude = magnitudes[peak_bins, columns]
    harmonic_energy = harmonic.sum(axis=0)
    pitch_confidence = peak_magnitude / np.maximum(harmonic_energy, 1e-8)
    pitch_hz = np.where(harmonic_energy > 1e-5, pitch_hz, 0)

    rms = librosa.feature.rms(S=stft, frame_length=N_FFT, hop_length=HOP)[0]
    hz = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    band_edges = [0, 200, 800, 3000, SR / 2 + 1]
    bands = np.stack([np.sqrt(np.mean(stft[(hz >= lo) & (hz < hi)]**2, axis=0))
                      for lo, hi in zip(band_edges[:-1], band_edges[1:])], axis=0)

    arrays = dict(times=times, onset_strength=onset_aligned,
                  beat_impulse=beat_impulse, beat_times=beat_times.astype(np.float32),
                  chroma=chroma, dominant_pitch_hz=interp(native_times, pitch_hz, times),
                  pitch_confidence=interp(native_times, pitch_confidence, times),
                  rms=interp(native_times, rms, times),
                  band_energy=interp(native_times, bands, times))
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite feature: {name}')
    metadata = dict(duration_seconds=duration, original_sample_rate=original_sr,
                    analysis_sample_rate=SR, feature_fps=FEATURE_FPS,
                    n_fft=N_FFT, hop_length=HOP, tempo_bpm=float(np.asarray(tempo).item()),
                    beats=len(beat_times), frames=len(times),
                    melody_note='Chroma and dominant spectral peak are polyphonic melody proxies.',
                    mood_status='pending_frozen_encoder')
    return arrays, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = [json.loads(x) for x in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines()]
    tracks = [row for row in tracks if row['eligible']]
    if args.limit:
        tracks = tracks[:args.limit]
    for i, row in enumerate(tracks):
        path = OUT / f"{row['id']}.npz"
        metadata_path = OUT / f"{row['id']}.json"
        if path.exists() and metadata_path.exists():
            continue
        start = time.monotonic()
        arrays, metadata = extract(ROOT/'data/stems'/row['id']/'instrumental.wav')
        metadata.update(track_id=row['id'], audio_path=f"data/stems/{row['id']}/instrumental.wav",
                        seconds_to_extract=round(time.monotonic()-start, 2))
        temp = path.with_suffix('.partial')
        with temp.open('wb') as file:
            np.savez_compressed(file, **arrays)
        temp.replace(path)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"{i+1}/{len(tracks)} {row['id']}: {metadata['frames']} frames, "
              f"{metadata['beats']} beats, {metadata['seconds_to_extract']} s", flush=True)

if __name__ == '__main__':
    main()
