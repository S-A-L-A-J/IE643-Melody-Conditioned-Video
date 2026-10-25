import os

"""Verify original plus two new visual styles in a provisional grouped manifest."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("geometric", "rings", "bars")


def main():
    base = [
        json.loads(line)
        for line in (ROOT / "data/manifests/synthetic_pairs_provisional_split.jsonl")
        .read_text()
        .splitlines()
    ]
    rows = []
    for pair in base:
        context = ROOT / "data/global_audio_context" / f"{pair['track_id']}.npz"
        if not context.exists():
            raise FileNotFoundError(context)
        with np.load(context) as data:
            if (
                data["embedding"].shape != (512,)
                or not np.isfinite(data["embedding"]).all()
            ):
                raise ValueError(f"Bad audio-context embedding: {context}")
        for style in STYLES:
            row = dict(pair)
            if style == "geometric":
                video = ROOT / pair["video_path"]
                parameters = ROOT / pair["renderer_parameters_path"]
            else:
                video = ROOT / "data/synthetic_videos_v2" / style / f"{pair['id']}.mp4"
                parameters = video.with_suffix(".json")
            if not video.exists() or not parameters.exists():
                raise FileNotFoundError(video)
            capture = cv2.VideoCapture(str(video))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            if (frames, width, height) != (64, 256, 256) or abs(fps - 16) > 0.01:
                raise ValueError(f"Bad shape/fps: {video}")
            row.update(
                id=f"{pair['id']}__{style}",
                source_pair_id=pair["id"],
                style=style,
                video_path=str(video.relative_to(ROOT)),
                renderer_parameters_path=str(parameters.relative_to(ROOT)),
                global_audio_context_path=str(context.relative_to(ROOT)),
                mood_embedding="frozen_clap_audio_context_512",
                mood_label_status="unvalidated_zero_shot_proxy",
            )
            rows.append(row)
    output = ROOT / "data/manifests/synthetic_pairs_v2.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {
        "status": "passed",
        "pairs": len(rows),
        "source_clips": len(base),
        "styles": dict(Counter(row["style"] for row in rows)),
        "split_pairs": dict(Counter(row["split"] for row in rows)),
        "frames": 64,
        "fps": 16,
        "resolution": [256, 256],
        "note": "Track splits are provisional; song identity and mood labels remain unverified.",
    }
    (ROOT / "reports/pair_verification_v2.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
