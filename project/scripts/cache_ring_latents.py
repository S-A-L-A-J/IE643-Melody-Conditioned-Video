import os

"""Cache 128px VAE targets for rings pairs using small GPU batches."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from pair_dataset import PairedClipDataset

ROOT = Path(
    os.environ.get("IE643_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if (args.frames - 1) * args.stride >= 64:
        parser.error("Selected frames exceed the four-second source clip")
    torch.set_num_threads(4)
    dataset = PairedClipDataset(manifest="data/manifests/synthetic_pairs_v2.jsonl")
    chosen = [i for i, row in enumerate(dataset.rows) if row["style"] == "rings"]
    if args.limit:
        chosen = chosen[: args.limit]
    output = ROOT / "data/latent_cache" / f"rings_128_{args.frames}_{args.stride}"
    output.mkdir(parents=True, exist_ok=True)
    vae = (
        AutoencoderKL.from_pretrained(
            ROOT / "models/diffusion/sd15",
            subfolder="vae",
            variant="fp16",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    vae.requires_grad_(False)
    frame_indices = torch.arange(args.frames) * args.stride
    existing = 0
    with torch.inference_mode():
        for n, index in enumerate(chosen, 1):
            row = dataset.rows[index]
            video = ROOT / row["video_path"]
            digest = hashlib.sha256(video.read_bytes()).hexdigest()[:12]
            path = output / f"{row['id']}_{digest}.safetensors"
            if path.exists():
                existing += 1
                continue
            item = dataset[index]
            frames = item["video"][:, frame_indices].permute(1, 0, 2, 3)
            frames = F.interpolate(
                frames, size=(128, 128), mode="bilinear", align_corners=False
            )
            encoded = []
            for chunk in frames.split(4):
                latent = (
                    vae.encode(chunk.to("cuda")).latent_dist.mode()
                    * vae.config.scaling_factor
                )
                encoded.append(latent.cpu())
            target = torch.cat(encoded).permute(1, 0, 2, 3)[None].contiguous()
            save_file({"latents": target}, path)
            if n % 50 == 0 or n == len(chosen):
                print(
                    f"{n}/{len(chosen)} ring targets cached ({existing} pre-existing)",
                    flush=True,
                )
    report = {
        "rings_pairs": len(chosen),
        "preexisting": existing,
        "newly_cached": len(chosen) - existing,
        "size": 128,
        "frames": args.frames,
        "frame_stride": args.stride,
    }
    (
        ROOT / "reports" / f"ring_latent_cache_{args.frames}_{args.stride}.json"
    ).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
