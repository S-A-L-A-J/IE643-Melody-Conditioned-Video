"""Extract frozen CLAP context from instrumental audio.

The 512-dimensional embedding supplies global context to the temporal model.
Mood similarities are exploratory descriptors, not validated emotion labels.
"""

import os
import argparse
import json
import time
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, ClapModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/clap"
OUT = ROOT / "data/global_audio_context"
REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
MOODS = ["joyful", "melancholic", "calm", "energetic", "romantic", "tense"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    # Load the pretrained processor and freeze CLAP.
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = (
        ClapModel.from_pretrained(MODEL, local_files_only=True).to(args.device).eval()
    )
    model.requires_grad_(False)
    # Optional descriptive mood scores; these are not verified emotion labels.
    prompts = [f"An instrumental music recording with a {mood} mood." for mood in MOODS]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        text_features = model.get_text_features(
            **{key: value.to(args.device) for key, value in text_inputs.items()}
        )
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    tracks = [
        json.loads(line)
        for line in (ROOT / "data/manifests/tracks.jsonl").read_text().splitlines()
    ]
    tracks = [row for row in tracks if row["eligible"]]
    if args.limit:
        tracks = tracks[: args.limit]
    completed = 0
    for i, row in enumerate(tracks):
        path = OUT / f"{row['id']}.npz"
        metadata_path = path.with_suffix(".json")
        if path.exists() and metadata_path.exists():
            completed += 1
            continue
        start = time.monotonic()
        # CLAP expects mono audio sampled at 48 kHz.
        samples, sr = sf.read(
            ROOT / "data/stems" / row["id"] / "instrumental.wav",
            dtype="float32",
            always_2d=True,
        )
        mono_audio = samples.mean(axis=1)
        if sr != 48000:
            mono_audio = librosa.resample(mono_audio, orig_sr=sr, target_sr=48000)
        # Encode ten-second chunks, then pool them into one global vector.
        chunk_length = 480000
        embeddings = []
        for chunk_start in range(0, len(mono_audio), chunk_length):
            chunk = mono_audio[chunk_start : chunk_start + chunk_length]
            if len(chunk) < 48000:
                continue
            inputs = processor(audios=chunk, sampling_rate=48000, return_tensors="pt")
            with torch.inference_mode():
                features = model.get_audio_features(
                    **{key: value.to(args.device) for key, value in inputs.items()}
                )
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            embeddings.append(features[0].cpu())
        if not embeddings:
            raise ValueError(f"No audio chunks for {row['id']}")
        # Average chunk embeddings and normalize the pooled representation.
        embedding = torch.nn.functional.normalize(
            torch.stack(embeddings).mean(dim=0), dim=0
        )
        similarities = (
            (embedding.to(args.device) @ text_features.T)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        probabilities = (
            torch.softmax(torch.from_numpy(similarities) * 10, dim=0)
            .numpy()
            .astype(np.float32)
        )
        audio_embedding = embedding.numpy().astype(np.float32)
        if (
            not np.isfinite(audio_embedding).all()
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError(f"Invalid embedding for {row['id']}")
        # Save the embedding and its provenance for downstream training.
        with path.with_suffix(".partial").open("wb") as file:
            np.savez_compressed(
                file,
                embedding=audio_embedding,
                mood_similarity=similarities,
                mood_softmax=probabilities,
            )
        path.with_suffix(".partial").replace(path)
        metadata_path.write_text(
            json.dumps(
                {
                    "track_id": row["id"],
                    "model": "laion/clap-htsat-unfused",
                    "revision": REVISION,
                    "source": "instrumental",
                    "sample_rate": 48000,
                    "chunks": len(embeddings),
                    "chunk_seconds": 10,
                    "embedding_dim": len(audio_embedding),
                    "mood_labels": MOODS,
                    "mood_note": "Exploratory audio-text similarities, not validated emotion labels.",
                    "seconds_to_extract": round(time.monotonic() - start, 2),
                },
                indent=2,
            )
        )
        completed += 1
        print(
            f"{completed}/{len(tracks)} {row['id']}: {len(embeddings)} chunks, {round(time.monotonic() - start, 1)} s",
            flush=True,
        )
    report = {
        "status": "complete" if completed == len(tracks) else "partial",
        "eligible_tracks": len(tracks),
        "embeddings_created_or_existing": completed,
        "model": "laion/clap-htsat-unfused",
        "revision": REVISION,
        "mood_label_status": "unvalidated_zero_shot_proxy",
    }
    (ROOT / "reports/global_audio_context.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
