import os

"""Render two additional persistent, music-reactive visual families.

This augments, rather than overwrites, the original geometric E0 targets.
Whole-track CLAP context changes the palette; frame-level music changes motion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/synthetic_videos_v2"
SIZE = 256
FPS = 16
STYLES = ("rings", "bars")
MOODS = ("joyful", "melancholic", "calm", "energetic", "romantic", "tense")


def pigment(hue, saturation=170, value=220):
    hsv = np.uint8(
        [
            [
                [
                    int(hue) % 180,
                    int(np.clip(saturation, 0, 255)),
                    int(np.clip(value, 0, 255)),
                ]
            ]
        ]
    )
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


# Render feature-driven synthetic training visuals


def render(clip, feature, mood, style, output):
    seed = int(
        hashlib.sha256(
            (clip["track_id"] + f":{clip['start_seconds']:.3f}:{style}").encode()
        ).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    start = round(clip["start_seconds"] * FPS)
    n = round(clip["duration"] * FPS)
    chroma = feature["chroma"][start : start + n]
    onset = feature["onset_strength"][start : start + n]
    beats = feature["beat_impulse"][start : start + n]
    rms = feature["rms"][start : start + n]
    bands = feature["band_energy"][start : start + n]
    if any(len(x) != n for x in (chroma, onset, beats, rms, bands)):
        raise ValueError("Features do not cover the clip")
    mood_probs = mood["mood_softmax"]
    top_mood = MOODS[int(mood_probs.argmax())]
    # Palette offset is deliberately a weak global influence; local chroma
    # still supplies the frame-to-frame colour variation.
    mood_hues = np.array([18, 108, 100, 4, 156, 174], dtype=np.float32)
    hue_bias = float(mood_probs @ mood_hues)
    energy_bias = float(mood_probs[3] + mood_probs[0] - mood_probs[2])
    onset_scale = max(float(np.percentile(onset, 95)), 1e-4)
    rms_scale = max(float(np.percentile(rms, 90)), 1e-4)
    band_scale = np.maximum(np.percentile(bands, 90, axis=0), 1e-4)
    painter = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE)
    )
    if not painter.isOpened():
        raise RuntimeError("Could not open video writer")
    phase = np.exp(2j * np.pi * np.arange(12) / 12)
    hue_vector = 1 + 0j
    pulse = 0.0
    brightness = 0.0
    band_levels = np.zeros(4, dtype=np.float32)
    bar_levels = np.zeros(12, dtype=np.float32)
    spokes = 9
    angular_offset = float(rng.uniform(0, 2 * math.pi))
    try:
        for i in range(n):
            t = i / FPS
            target = np.sum(chroma[i] * phase)
            if abs(target) > 1e-6:
                hue_vector = 0.9 * hue_vector + 0.1 * target / abs(target)
            chroma_hue = (np.angle(hue_vector) % (2 * math.pi)) * 180 / (2 * math.pi)
            hue = (0.72 * chroma_hue + 0.28 * hue_bias) % 180
            hit = float(np.clip(onset[i] / onset_scale, 0, 1.5))
            pulse = max(pulse * 0.78, float(beats[i]), hit * 0.55)
            brightness = 0.82 * brightness + 0.18 * float(
                np.clip(rms[i] / rms_scale, 0, 1.5)
            )
            band_levels = 0.78 * band_levels + 0.22 * np.clip(
                bands[i] / band_scale, 0, 1.7
            )
            frame = np.full((SIZE, SIZE, 3), (11, 9, 17), dtype=np.uint8)
            if style == "rings":
                cx = 128 + int(14 * math.sin(t * 0.7 + angular_offset))
                cy = 128 + int(13 * math.cos(t * 0.5 + angular_offset))
                for j in range(4):
                    radius = int(24 + j * 23 + 10 * pulse + 8 * band_levels[j])
                    cv2.circle(
                        frame,
                        (cx, cy),
                        radius,
                        pigment(hue + j * 19, 160, 115 + 65 * brightness + 35 * pulse),
                        max(2, int(4 + 2 * band_levels[j])),
                        cv2.LINE_AA,
                    )
                cv2.circle(
                    frame,
                    (cx, cy),
                    int(14 + 8 * pulse),
                    pigment(hue + 7, 195, 170 + 55 * brightness),
                    -1,
                    cv2.LINE_AA,
                )
                rotation = t * (0.38 + 0.22 * energy_bias) + angular_offset
                for j in range(spokes):
                    angle = rotation + j * 2 * math.pi / spokes
                    radius = 101 + 10 * math.sin(t * 0.6 + j)
                    sx = int(cx + radius * math.cos(angle))
                    sy = int(cy + radius * math.sin(angle))
                    dot = int(2 + 4 * chroma[i, j % 12] + 2 * pulse)
                    cv2.circle(
                        frame,
                        (sx, sy),
                        dot,
                        pigment(hue + j * 9, 130, 145 + 70 * brightness),
                        -1,
                        cv2.LINE_AA,
                    )
            else:
                bar_levels = 0.8 * bar_levels + 0.2 * np.clip(chroma[i], 0, 1)
                baseline = 205 + int(8 * math.sin(t * 0.6 + angular_offset))
                for j in range(12):
                    x = 22 + j * 19
                    height = int(
                        18 + 110 * bar_levels[j] + 28 * pulse * float(j % 3 == 0)
                    )
                    cv2.rectangle(
                        frame,
                        (x, baseline - height),
                        (x + 12, baseline),
                        pigment(hue + j * 7, 175, 115 + 75 * brightness + 30 * pulse),
                        -1,
                    )
                    cv2.line(
                        frame,
                        (x, baseline - height),
                        (x + 12, baseline - height),
                        pigment(hue + j * 7, 100, 235),
                        2,
                        cv2.LINE_AA,
                    )
                wave_y = int(55 + 20 * math.sin(t * 0.8 + angular_offset))
                points = []
                for x in range(12, 245, 4):
                    y = wave_y + int(
                        (8 + 22 * band_levels[1])
                        * math.sin(x * 0.047 - t * (1.5 + 0.5 * energy_bias))
                    )
                    points.append((x, y))
                cv2.polylines(
                    frame,
                    [np.array(points, np.int32)],
                    False,
                    pigment(hue + 40, 150, 120 + 80 * brightness),
                    2,
                    cv2.LINE_AA,
                )
            painter.write(frame)
    finally:
        painter.release()
    return {
        "renderer": "persistent_music_reactive_v2",
        "style": style,
        "seed": seed,
        "mood_model": "laion/clap-htsat-unfused",
        "mood_proxy_top": top_mood,
        "mood_proxy_probs": mood_probs.tolist(),
        "palette_hue_bias": hue_bias,
        "fps": FPS,
        "resolution": [SIZE, SIZE],
        "frames": n,
        "mappings": {
            "chroma": "colour_and_bar_pattern",
            "beats_and_onsets": "pulse",
            "rms": "brightness",
            "frequency_bands": "ring_size_or_wave_amplitude",
            "frozen_audio_context": "palette_bias",
        },
        "source_clip": clip,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--styles", default="rings,bars")
    args = parser.parse_args()
    styles = args.styles.split(",")
    if any(style not in STYLES for style in styles):
        parser.error(f"Styles must be selected from {STYLES}")
    clips = [
        json.loads(line)
        for line in (ROOT / "data/manifests/clips.jsonl").read_text().splitlines()
    ]
    if args.limit:
        clips = clips[: args.limit]
    for style in styles:
        (OUT / style).mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips):
        with (
            np.load(ROOT / "data/features" / f"{clip['track_id']}.npz") as feature,
            np.load(
                ROOT / "data/global_audio_context" / f"{clip['track_id']}.npz"
            ) as mood,
        ):
            for style in styles:
                name = f"{clip['track_id']}_{clip['start_seconds']:.3f}"
                video = OUT / style / f"{name}.mp4"
                metadata = video.with_suffix(".json")
                if video.exists() and metadata.exists():
                    continue
                temporary = video.with_suffix(".partial.mp4")
                params = render(clip, feature, mood, style, temporary)
                temporary.replace(video)
                metadata.write_text(json.dumps(params, indent=2))
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            print(
                f"{i+1}/{len(clips)} clips rendered in {len(styles)} new styles",
                flush=True,
            )


if __name__ == "__main__":
    main()
