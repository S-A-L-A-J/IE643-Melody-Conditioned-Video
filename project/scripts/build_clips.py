import os

"""Index virtual four-second clips without duplicating instrumental audio on disk."""
import json
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    rows = []
    eligible = {
        r["id"]
        for r in (
            json.loads(x)
            for x in (ROOT / "data/manifests/tracks.jsonl").read_text().splitlines()
        )
        if r["eligible"]
    }
    for meta in sorted((ROOT / "data/stems").glob("*/complete.json")):
        track = meta.parent.name
        if track not in eligible:
            continue
        audio, sr = sf.read(
            meta.parent / "instrumental.wav", always_2d=True, dtype="float32"
        )
        window = round(args.duration * sr)
        if window < 1:
            parser.error("--duration must span at least one sample")
        for start in range(0, len(audio) - window + 1, window):
            clip = audio[start : start + window]
            rms = float(np.sqrt(np.mean(clip**2)))
            if rms < 1e-4:
                continue
            path = meta.parent / "instrumental.wav"
            rows.append(
                dict(
                    track_id=track,
                    path=str(path.relative_to(ROOT)),
                    start_seconds=start / sr,
                    duration=window / sr,
                    sample_rate=sr,
                    rms=rms,
                    peak=float(np.abs(clip).max()),
                    split="development",
                )
            )
    (ROOT / "data/manifests/clips.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    print(
        f"Indexed {len(rows)} virtual instrumental development clips; no synthetic videos yet."
    )


if __name__ == "__main__":
    main()
