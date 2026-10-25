"""At most 64 update attempts for a low-noise spatial-domain LoRA diagnostic.

32 existing training recordings; eight fixed validation recordings; no test or
fresh-evaluation tensors. Audio conditioning remains upstream in the frozen GRU.
"""

import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / "geometry_lora_continuation"
sys.path[:0] = [
    str(HERE),
    str(
        Path(__file__).resolve().parent.joinpath("../../../../project").resolve()
        / "scripts"
    ),
]
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import LoraConfig
from safetensors.torch import save_file, load_file
from train_audio_latent_prior import load_data
from gpu_backend import load_pipeline, PROMPT


def dump(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def first_recordings(part, count):
    chosen = []
    seen = set()
    for i, track in enumerate(part["tracks"]):
        if track not in seen:
            chosen.append(i)
            seen.add(track)
        if len(chosen) == count:
            break
    assert len(chosen) == count
    return chosen


def main():
    if (HERE / "STOPPED.json").exists():
        raise RuntimeError("Experiment stopped by user; no further training permitted.")
    assert torch.cuda.is_available()
    torch.manual_seed(643)
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "run.json").exists(), "Preserve existing training run"
    started = time.monotonic()
    data = load_data(32, 2, "causal_peak", False)
    train = first_recordings(data["train"], 32)
    val = first_recordings(data["validation"], 8)
    targets = [data["train"]["target"][i : i + 1].contiguous() for i in train]
    val_targets = [
        data["validation"]["target"][i : i + 1, :, :8].contiguous() for i in val
    ]
    report = {
        "kind": "bounded low-noise spatial geometry LoRA, not direct audio training",
        "seed": 643,
        "max_update_attempts": 64,
        "continuation_from_step": 64,
        "maximum_total_attempts": 128,
        "wall_ceiling_seconds": 1200,
        "train_recordings": 32,
        "validation_recordings": 8,
        "train_ids": [data["train"]["ids"][i] for i in train],
        "validation_ids": [data["validation"]["ids"][i] for i in val],
        "frames_per_update": 8,
        "fps": 8,
        "size": 128,
        "rank": 2,
        "lr": 0.0001,
        "training_timesteps": [10, 120],
        "validation_timesteps": [50, 100],
        "objective": "standard epsilon MSE against noised encoded target clips",
        "base_and_motion_weights_frozen": True,
        "audio_features_used_in_optimizer": False,
        "fresh_or_test_used": False,
        "selection": "minimum fixed validation epsilon MSE among continuation steps 64,80,96,112,128",
        "limitations": [
            "One-second training windows; longer temporal behavior is untrained.",
            "Synthetic rings domain only; upstream GRU carries music input.",
            "Lower denoising loss alone is not an output-quality result.",
        ],
    }
    del data
    gc.collect()
    pipe = load_pipeline(offload=False)
    pipe.text_encoder = pipe.text_encoder.cuda()
    with torch.no_grad():
        embedding = pipe.encode_prompt(PROMPT, "cuda", 1, False)[0].detach()
    unet = pipe.unet
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    unet = unet.cuda().train()
    names = [
        name
        for name, module in unet.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(("to_q", "to_v", "to_out.0"))
        and "motion_modules" not in name
    ]
    unet.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=names))
    unet.enable_gradient_checkpointing()
    for name, module in unet.named_modules():
        if ".lora_A.default" in name or ".lora_B.default" in name:
            module.float()
    parameters = [p for p in unet.parameters() if p.requires_grad]
    assert parameters and all(p.dtype == torch.float32 for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    rng = torch.Generator(device="cuda").manual_seed(643)
    condition = embedding.repeat_interleave(8, dim=0)
    report.update(
        trainable_parameters=sum(p.numel() for p in parameters),
        adapter_target_modules=names,
        gpu=torch.cuda.get_device_name(0),
        base_dtype=str(next(unet.parameters()).dtype),
    )
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def validate():
        unet.eval()
        losses = []
        for i, z in enumerate(val_targets):
            z = z.cuda().half()
            noise = (
                torch.randn(z.shape, generator=torch.Generator().manual_seed(9000 + i))
                .cuda()
                .half()
            )
            for t in [50, 100]:
                timestep = torch.tensor([t], device="cuda")
                estimate = unet(
                    scheduler.add_noise(z, noise, timestep),
                    timestep,
                    encoder_hidden_states=condition,
                ).sample
                losses.append(float(F.mse_loss(estimate.float(), noise.float())))
        unet.train()
        return sum(losses) / len(losses)

    def checkpoint(step, loss):
        path = OUT / f"lora_step{step:03d}.safetensors"
        assert not path.exists()
        state = {
            n: p.detach().cpu().contiguous()
            for n, p in unet.named_parameters()
            if "lora_" in n
        }
        assert all(torch.isfinite(p).all() for p in state.values())
        save_file(state, path)
        if step:
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": rng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                },
                OUT / f"training_state_{step:03d}.pt",
            )
        report["validation"].append(
            {"step": step, "epsilon_mse": loss, "checkpoint": path.name}
        )
        dump(
            OUT / f"progress_{step:03d}.json",
            {**report, "elapsed_seconds": time.monotonic() - started},
        )
        print(
            "VALIDATION",
            step,
            loss,
            "VRAM MiB",
            torch.cuda.max_memory_allocated() / 2**20,
            flush=True,
        )

    previous = HERE / "geometry_lora"
    _, unexpected = unet.load_state_dict(
        load_file(previous / "lora_step064.safetensors"), strict=False
    )
    assert not unexpected
    saved = torch.load(
        previous / "training_state_064.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    rng.set_state(saved["rng_state"])
    del saved
    gc.collect()
    report["validation"] = []
    report["history"] = []
    baseline = validate()
    checkpoint(64, baseline)
    updates = 0
    for step in range(65, 129):
        tick = time.monotonic()
        z = targets[(step - 1) % len(targets)]
        first = int(torch.randint(0, 25, (1,), generator=rng, device="cuda").item())
        z = z[:, :, first : first + 8].cuda().half()
        noise = torch.randn(z.shape, generator=rng, device="cuda", dtype=torch.float16)
        t = torch.randint(10, 121, (1,), generator=rng, device="cuda")
        noisy = scheduler.add_noise(z, noise, t)
        optimizer.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            estimate = unet(noisy, t, encoder_hidden_states=condition).sample
        loss = F.mse_loss(estimate.float(), noise.float())
        assert torch.isfinite(loss)
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= scale_before
        updates += int(updated)
        report["history"].append(
            {
                "attempt": step,
                "optimizer_updated": updated,
                "epsilon_mse": float(loss.detach()),
                "timestep": int(t.item()),
                "gradient_norm": float(norm),
                "seconds": time.monotonic() - tick,
            }
        )
        print(
            "step",
            step,
            "loss",
            float(loss.detach()),
            "updated",
            updated,
            "seconds",
            round(time.monotonic() - tick, 2),
            flush=True,
        )
        del estimate, loss, z, noise, t, noisy
        if step % 16 == 0:
            val_loss = validate()
            checkpoint(step, val_loss)
            if step >= 96 and val_loss >= baseline:
                report["early_stop"] = (
                    "No fixed validation improvement at this checkpoint"
                )
                break
        if time.monotonic() - started > 1200:
            report["early_stop"] = "20 minute ceiling"
            break
    report.update(
        status="completed",
        optimizer_updates=updates,
        total_optimizer_updates=64 + updates,
        elapsed_seconds=time.monotonic() - started,
        peak_vram_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
        peak_vram_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
    )
    best = min(report["validation"], key=lambda r: r["epsilon_mse"])
    report["selected"] = best
    report["best_checkpoint_beats_frozen_loss"] = best["epsilon_mse"] < baseline
    dump(OUT / "run.json", report)
    print(
        "FINISHED",
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["history", "adapter_target_modules"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        failure = {
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        path = OUT / "failure.json"
        if not path.exists():
            dump(path, failure)
        raise
