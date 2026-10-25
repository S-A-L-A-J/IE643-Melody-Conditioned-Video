import os

"""Train a small supervised audio-to-video-latent control prior on rings targets.

This is a learned control signal for later diffusion conditioning, not the
diffusion generator itself. Test tracks remain untouched until final scoring.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from train_contrastive import normalize_features

ROOT = Path(
    os.environ.get("IE643_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
)
OUT = ROOT / "outputs/audio_latent_prior"


# Earlier feed-forward latent predictor


class AudioLatentPrior(torch.nn.Module):
    def __init__(self, use_global=True):
        super().__init__()
        self.use_global = use_global
        self.local = torch.nn.Sequential(
            torch.nn.Conv1d(22, 128, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(128, 128, 3, padding=1),
            torch.nn.SiLU(),
        )
        self.global_projection = torch.nn.Linear(512, 128) if use_global else None
        self.output = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Conv1d(128, 4 * 16 * 16, 1)
        )

    def forward(self, music, context):
        frames = music.shape[1]
        hidden = self.local(music.transpose(1, 2))
        if self.global_projection is not None:
            hidden = hidden + self.global_projection(context)[:, :, None]
        output = self.output(hidden)
        return (
            output.reshape(len(music), 4, 16, 16, frames)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )


# Align rhythmic events to the video frame rate


def sample_rhythm(values, positions, first, stride, mode):
    if mode == "point":
        return values[positions]
    if mode != "causal_peak":
        raise ValueError(mode)
    # Retain events since the previous output frame, with renderer-like decay.
    # Do not read the preceding clip or future source frames.
    return np.asarray(
        [
            max(
                float(values[j]) * 0.78 ** (int(p) - j)
                for j in range(max(first, int(p) - stride + 1), int(p) + 1)
            )
            for p in positions
        ],
        dtype=np.float32,
    )


# Load aligned audio features and cached visual targets


def load_data(frames=8, stride=8, rhythm_mode="point", include_test=True):
    rows = [
        json.loads(line)
        for line in (ROOT / "data/manifests/synthetic_pairs_v2.jsonl")
        .read_text()
        .splitlines()
    ]
    rows = [
        row
        for row in rows
        if row["style"] == "rings" and (include_test or row["split"] != "test")
    ]
    groups = {
        name: []
        for name in (
            ("train", "validation", "test") if include_test else ("train", "validation")
        )
    }
    for row in rows:
        first = row["feature_start_frame"]
        positions = first + np.arange(frames) * stride
        with np.load(ROOT / row["features_path"]) as source:
            onset = sample_rhythm(
                source["onset_strength"], positions, first, stride, rhythm_mode
            )
            beat = sample_rhythm(
                source["beat_impulse"], positions, first, stride, rhythm_mode
            )
            chroma = source["chroma"][positions]
            pitch = source["dominant_pitch_hz"][positions]
            confidence = source["pitch_confidence"][positions]
            rms = source["rms"][positions]
            bands = source["band_energy"][positions]
        tempo = json.loads(
            (ROOT / row["features_path"]).with_suffix(".json").read_text()
        )["tempo_bpm"]
        item = {
            "rhythm": torch.from_numpy(
                np.stack([onset, beat, np.full(frames, tempo)], axis=-1).astype(
                    np.float32
                )
            ),
            "melody": torch.from_numpy(
                np.concatenate(
                    [chroma, pitch[:, None], confidence[:, None]], axis=-1
                ).astype(np.float32)
            ),
            "dynamics": torch.from_numpy(
                np.concatenate([rms[:, None], bands], axis=-1).astype(np.float32)
            ),
        }
        rhythm, melody, dynamics = normalize_features(item, torch.arange(frames))
        local = torch.cat([rhythm, melody, dynamics], dim=-1)
        with np.load(ROOT / row["global_audio_context_path"]) as source:
            context = torch.from_numpy(source["embedding"].copy())
        cache = list(
            (ROOT / "data/latent_cache" / f"rings_128_{frames}_{stride}").glob(
                f"{row['id']}_*.safetensors"
            )
        )
        if len(cache) != 1:
            raise RuntimeError(
                f'Expected one cached target for {row["id"]}; found {len(cache)}'
            )
        target = load_file(cache[0])["latents"][0].float()
        groups[row["split"]].append(
            {
                "id": row["id"],
                "track": row["track_id"],
                "music": local,
                "context": context,
                "target": target,
            }
        )
    packed = {}
    for split, records in groups.items():
        packed[split] = {
            "ids": [r["id"] for r in records],
            "tracks": [r["track"] for r in records],
            "music": torch.stack([r["music"] for r in records]),
            "context": torch.stack([r["context"] for r in records]),
            "target": torch.stack([r["target"] for r in records]),
        }
    return packed


# Select mismatched audio for the conditioning comparison


def wrong_track_indices(tracks):
    first = {}
    for i, track in enumerate(tracks):
        first.setdefault(track, i)
    order = list(first)
    next_track = {track: order[(i + 1) % len(order)] for i, track in enumerate(order)}
    return torch.tensor([first[next_track[track]] for track in tracks])


@torch.inference_mode()
# Evaluate reconstruction and conditioning controls


def evaluate(model, data, static_template, batch_size=64):
    music = data["music"].to("cuda")
    context = data["context"].to("cuda")
    target = data["target"].to("cuda")
    wrong = wrong_track_indices(data["tracks"]).to("cuda")
    right_losses = []
    wrong_losses = []
    shifted_losses = []
    for start in range(0, len(target), batch_size):
        sl = slice(start, start + batch_size)
        pred = model(music[sl], context[sl])
        swapped = model(music[wrong[sl]], context[wrong[sl]])
        shifted = model(music[sl].roll(music.shape[1] // 4, 1), context[sl])
        right_losses.append((pred - target[sl]).square().flatten(1).mean(1))
        wrong_losses.append((swapped - target[sl]).square().flatten(1).mean(1))
        shifted_losses.append((shifted - target[sl]).square().flatten(1).mean(1))
    right = torch.cat(right_losses)
    other = torch.cat(wrong_losses)
    shifted = torch.cat(shifted_losses)
    static = (target - static_template).square().flatten(1).mean(1)
    report = {
        "pairs": len(target),
        "correct_mse": float(right.mean()),
        "other_track_mse": float(other.mean()),
        "one_second_shift_mse": float(shifted.mean()),
        "static_template_mse": float(static.mean()),
        "mse_improvement_vs_static": float(1 - right.mean() / static.mean()),
        "correct_better_than_other_count": int((right < other).sum()),
        "correct_better_than_time_shift_count": int((right < shifted).sum()),
        "mean_other_track_advantage": float((other - right).mean()),
        "mean_time_shift_advantage": float((shifted - right).mean()),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--init", default="")
    parser.add_argument(
        "--rhythm-mode", choices=["point", "causal_peak"], default="point"
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    if args.rhythm_mode != "point" and not args.run_name:
        parser.error("A distinct --run-name is required for changed rhythm sampling")
    if (args.frames - 1) * args.stride >= 64:
        parser.error("Selected frames exceed the source clip")
    torch.manual_seed(643)
    torch.set_num_threads(4)
    output_dir = (
        OUT
        if args.frames == 8 and args.stride == 8
        else ROOT / "outputs" / f"audio_latent_prior_{args.frames}f"
    )
    if args.run_name:
        if not args.run_name.replace("_", "").isalnum():
            parser.error("Run name must contain letters, numbers or underscores")
        output_dir = ROOT / "outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(
        args.frames, args.stride, args.rhythm_mode, not args.validation_only
    )
    train = data["train"]
    static_template = train["target"].mean(0).to("cuda")
    model = AudioLatentPrior(use_global=not args.no_global).to("cuda")
    if args.init:
        model.load_state_dict(load_file(ROOT / args.init))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    music = train["music"].to("cuda")
    context = train["context"].to("cuda")
    target = train["target"].to("cuda")
    history = []
    best = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(target), device="cuda")
        losses = []
        for indices in order.split(args.batch_size):
            if len(indices) < 2:
                continue
            x = music[indices]
            c = context[indices]
            y = target[indices]
            pred = model(x, c)
            rolled = torch.roll(torch.arange(len(indices), device="cuda"), 1)
            wrong = model(x[rolled], c[rolled])
            per_correct = (pred - y).square().flatten(1).mean(1)
            per_wrong = (wrong - y).square().flatten(1).mean(1)
            reconstruction = per_correct.mean()
            temporal = F.mse_loss(
                pred[:, :, 1:] - pred[:, :, :-1], y[:, :, 1:] - y[:, :, :-1]
            )
            ranking = F.softplus(10 * (per_correct - per_wrong + 0.02)).mean() / 10
            loss = reconstruction + 0.5 * temporal + 0.5 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = evaluate(model, data["validation"], static_template)
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_correct_mse": validation["correct_mse"],
                "validation_swap_advantage": validation["mean_other_track_advantage"],
                "validation_shift_advantage": validation["mean_time_shift_advantage"],
            }
        )
        if validation["correct_mse"] < best - 1e-5:
            best = validation["correct_mse"]
            best_epoch = epoch
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                },
                output_dir
                / ("no_global.safetensors" if args.no_global else "model.safetensors"),
            )
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch}: train {history[-1]['training_loss']:.4f}, val MSE {validation['correct_mse']:.4f}, swap advantage {validation['mean_other_track_advantage']:.4f}, shift advantage {validation['mean_time_shift_advantage']:.4f}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            break
    filename = "no_global.safetensors" if args.no_global else "model.safetensors"
    model.load_state_dict(load_file(output_dir / filename))
    model.eval()
    report = {
        "status": "completed",
        "configuration": vars(args),
        "best_epoch": best_epoch,
        "split_status": "provisional_recording_grouped_song_identity_unverified",
        "model_type": "supervised_audio_to_video_latent_prior_not_diffusion",
        "counts": {key: len(value["ids"]) for key, value in data.items()},
        "history": history,
        "final": {
            split: evaluate(model, values, static_template)
            for split, values in data.items()
        },
        "note": "MSE and swap/shift tests use encoded rings targets. Human video ratings remain pending.",
    }
    stem = (
        f"audio_latent_prior_{args.frames}f"
        if args.frames != 8 or args.stride != 8
        else "audio_latent_prior"
    )
    if args.run_name:
        stem = args.run_name
    output = ROOT / "reports" / f'{stem}{"_no_global" if args.no_global else ""}.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"best_epoch": best_epoch, "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
