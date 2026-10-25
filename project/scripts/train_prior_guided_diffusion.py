import os

"""Train a small AnimateDiff LoRA to use a frozen learned audio latent prior.

The prior and original diffusion weights remain frozen. Validation/test tracks
stay out of optimizer steps. This is a focused control experiment at 128px/8f.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDPMScheduler, MotionAdapter
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from train_audio_latent_prior import AudioLatentPrior, load_data, wrong_track_indices

ROOT = Path(
    os.environ.get("IE643_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
)
OUT = ROOT / "outputs/prior_guided_diffusion"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    torch.manual_seed(643)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = load_data()
    prior = AudioLatentPrior().to("cuda").eval()
    prior.load_state_dict(
        load_file(ROOT / "outputs/audio_latent_prior/model.safetensors")
    )
    prior.requires_grad_(False)
    train = data["train"]
    validation = data["validation"]
    static = train["target"].mean(0).to("cuda", dtype=torch.float16)
    adapter = MotionAdapter.from_pretrained(
        ROOT / "models/diffusion/animatediff",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        ROOT / "models/diffusion/sd15",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    with torch.no_grad():
        text_model = pipe.text_encoder.float().cpu().eval()
        tokens = pipe.tokenizer(
            "abstract glowing concentric rings on a dark background",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text = (
            text_model(tokens.input_ids)[0]
            .mean(dim=1, keepdim=True)
            .to("cuda", dtype=torch.float16)
        )
        del text_model
    unet = pipe.unet
    del pipe, adapter
    unet.requires_grad_(False)
    targets = [
        name
        for name, module in unet.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(("to_q", "to_k", "to_v", "to_out.0"))
        and ("motion_modules" in name or "attn2" in name)
    ]
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=targets))
    unet.enable_gradient_checkpointing()
    unet.to("cuda", dtype=torch.float16).train()
    for name, module in unet.named_modules():
        if ".lora_A.default" in name or ".lora_B.default" in name:
            module.float()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    rng = torch.Generator(device="cuda").manual_seed(643)
    indices = torch.randperm(
        len(train["ids"]), generator=torch.Generator().manual_seed(643)
    ).tolist()
    train_wrong = wrong_track_indices(train["tracks"]).tolist()
    val_wrong = wrong_track_indices(validation["tracks"]).tolist()
    val_selected = []
    seen = set()
    for i, track in enumerate(validation["tracks"]):
        if track not in seen:
            val_selected.append(i)
            seen.add(track)
        if len(val_selected) == 8:
            break
    with torch.inference_mode():
        prior_train = (
            prior(train["music"].to("cuda"), train["context"].to("cuda")).half().cpu()
        )
        prior_val = (
            prior(validation["music"].to("cuda"), validation["context"].to("cuda"))
            .half()
            .cpu()
        )
    del prior
    torch.cuda.empty_cache()
    history = []
    beginning = 0
    if args.resume:
        if not (OUT / "training_state.pt").exists():
            raise FileNotFoundError("Missing checkpoint")
        _, unexpected = unet.load_state_dict(
            load_file(OUT / "lora.safetensors"), strict=False
        )
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")
        saved = torch.load(
            OUT / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        rng.set_state(saved["rng_state"])
        beginning = int(saved["step"])
        history = json.loads((OUT / "run.json").read_text())["history"]

    def noisy_input(noisy, template, timestep):
        alpha = float(scheduler.alphas_cumprod[int(timestep.item())].sqrt())
        return noisy + 0.3 * alpha * (template.to("cuda") - static)

    @torch.inference_mode()
    def validate():
        unet.eval()
        results = []
        timestep = torch.tensor([500], device="cuda")
        for n, i in enumerate(val_selected):
            target = validation["target"][i : i + 1].to("cuda", dtype=torch.float16)
            noise = torch.randn(
                target.shape,
                device="cuda",
                dtype=torch.float16,
                generator=torch.Generator(device="cuda").manual_seed(9000 + n),
            )
            noisy = scheduler.add_noise(target, noise, timestep)
            right = unet(
                noisy_input(noisy, prior_val[i : i + 1], timestep),
                timestep,
                encoder_hidden_states=text.expand(8, -1, -1),
            ).sample
            j = val_wrong[i]
            wrong = unet(
                noisy_input(noisy, prior_val[j : j + 1], timestep),
                timestep,
                encoder_hidden_states=text.expand(8, -1, -1),
            ).sample
            correct = float(F.mse_loss(right.float(), noise.float()))
            other = float(F.mse_loss(wrong.float(), noise.float()))
            results.append(
                {
                    "pair": validation["ids"][i],
                    "correct_loss": correct,
                    "other_song_loss": other,
                    "advantage": other - correct,
                }
            )
        unet.train()
        return {
            "pairs": len(results),
            "correct_better_count": sum(x["advantage"] > 0 for x in results),
            "mean_advantage": sum(x["advantage"] for x in results) / len(results),
            "mean_correct_loss": sum(x["correct_loss"] for x in results) / len(results),
        }

    report = {
        "status": "running",
        "configuration": vars(args),
        "train_pairs": len(train["ids"]),
        "validation_pairs": len(validation["ids"]),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "adapter_targets": len(targets),
        "guidance_scale": 0.3,
        "frames": 8,
        "size": 128,
        "training_timestep_range": [250, 750],
        "prior": "frozen_audio_latent_prior_model.safetensors",
        "note": "Provisional recording-grouped split; test data excluded from training and scale selection.",
    }
    if args.resume:
        report["before"] = json.loads((OUT / "run.json").read_text())["before"]
    else:
        report["before"] = validate()
    print("before", report["before"], "resume step", beginning, flush=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(beginning + 1, args.steps + 1):
        i = indices[(step - 1) % len(indices)]
        j = train_wrong[i]
        target = train["target"][i : i + 1].to("cuda", dtype=torch.float16)
        noise = torch.randn(
            target.shape, device="cuda", dtype=torch.float16, generator=rng
        )
        timestep = torch.randint(250, 751, (1,), device="cuda", generator=rng)
        noisy = scheduler.add_noise(target, noise, timestep)
        correct_input = noisy_input(noisy, prior_train[i : i + 1], timestep)
        wrong_input = noisy_input(noisy, prior_train[j : j + 1], timestep)
        condition = text.expand(8, -1, -1)
        with torch.no_grad():
            right_est = unet(
                correct_input, timestep, encoder_hidden_states=condition
            ).sample
            wrong_est = unet(
                wrong_input, timestep, encoder_hidden_states=condition
            ).sample
            right_est_loss = F.mse_loss(right_est.float(), noise.float())
            wrong_est_loss = F.mse_loss(wrong_est.float(), noise.float())
            weight = 2 * torch.sigmoid(50 * (right_est_loss - wrong_est_loss + 0.01))
            reported = (
                right_est_loss
                + 2 * F.softplus(50 * (right_est_loss - wrong_est_loss + 0.01)) / 50
            )
            del right_est, wrong_est
        optimizer.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            right = unet(
                correct_input, timestep, encoder_hidden_states=condition
            ).sample
        right_loss = F.mse_loss(right.float(), noise.float())
        scaler.scale((1 + weight) * right_loss).backward()
        del right, right_loss
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            wrong = unet(wrong_input, timestep, encoder_hidden_states=condition).sample
        wrong_loss = F.mse_loss(wrong.float(), noise.float())
        scaler.scale(-weight * wrong_loss).backward()
        if not torch.isfinite(reported):
            raise ValueError(f"Non-finite objective at step {step}")
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "step": step,
                "pair": train["ids"][i],
                "correct_loss": float(right_est_loss),
                "other_song_loss": float(wrong_est_loss),
                "objective": float(reported),
                "timestep": int(timestep.item()),
            }
        )
        del (
            target,
            noise,
            timestep,
            noisy,
            correct_input,
            wrong_input,
            wrong,
            wrong_loss,
        )
        if step % args.checkpoint_every == 0 or step == args.steps:
            report.update(
                status="checkpointed",
                step=step,
                history=history,
                validation=validate(),
                peak_vram_reserved_mb=round(
                    torch.cuda.max_memory_reserved() / 2**20, 1
                ),
                elapsed_seconds=round(time.monotonic() - start, 1),
            )
            save_file(
                {
                    name: p.detach().cpu().contiguous()
                    for name, p in unet.named_parameters()
                    if p.requires_grad and "lora_" in name
                },
                OUT / "lora.safetensors",
            )
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": rng.get_state(),
                },
                OUT / "training_state.pt",
            )
            (OUT / "run.json").write_text(json.dumps(report, indent=2))
            print(
                f"step {step}: val {report['validation']}; VRAM {report['peak_vram_reserved_mb']} MB",
                flush=True,
            )
    report["status"] = "completed"
    report["total_seconds"] = round(time.monotonic() - start, 1)
    (OUT / "run.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
