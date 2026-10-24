# Reproduce the bounded follow-up

Workspace: `/home/salaj/.codex/.chatgpt-projects/g-p-6aae9e9b275881919a205c4774702722`.
Python: `/home/salaj/IE643_project/.venv/bin/python`.
Run with host GPU access: restricted sandbox execution hides CUDA. The available device is an RTX 3050 Laptop GPU, 4096 MiB. No downloads are required or allowed; loaders set offline mode and local-files-only.

Environment: torch 2.5.1+cu118, diffusers 0.35.1, transformers 4.49.0, accelerate 1.10.1, safetensors 0.8.0, numpy 1.26.4, scipy 1.15.3, OpenCV 4.11.0, ffmpeg 4.4.2. Native CUDA float16 denoiser, sequential CPU offload; float32 GRU and VAE decoding. CPU float32 seeded Gaussian noise is converted to CUDA float16. The existing custom short sampler matched the native installed pipeline exactly in the saved compatibility control (`../raw128_native_control.json`).

From the workspace:

```bash
PY=/home/salaj/IE643_project/.venv/bin/python
EXP=assessment/experiments/diffusion_refinement/gpu_extension
$PY "$EXP/resumed_validation.py"
$PY "$EXP/resumed_20261024/paired_validation.py"
$PY "$EXP/resumed_20261024/confirm_remaining.py"
$PY "$EXP/resumed_20261024/long_candidate.py"
$PY "$EXP/resumed_20261024/summarize.py"
$PY "$EXP/resumed_20261024/package_media.py"
```

These preserve and reuse existing latents/media. JSON result files deliberately use exclusive creation in sampling scripts; a repeat can stop at that guard after checking cached samples. To regenerate without overwriting evidence, copy the experiment tree to a new experiment directory, retain configs/checkpoints/scripts and saved predictor initialization files, and omit the new run's `validation8`, `paired_validation`, `confirmation3`, `long_validation`, `review`, and `review_v2` output directories. Run the same commands with EXP pointing there. There is no need to retrain. The original stopped entry points remain guarded and historical; do not remove their stop records.

Selected settings are frozen in `CHOSEN_CANDIDATE.json`: raw 128px predictor latents, strength .06, base 100 DDIM steps (actual timesteps 50,40,30,20,10,0), linear betas .00085–.012, epsilon prediction, no clipping, linspace, offset 1, eta 0, guidance 1 with the full 77-token fixed rings prompt. For each seed, independently refine with +epsilon and -epsilon; average the two final float32 latent tensors, then decode. No target latent is used in sampling. Saved unsigned noise hashes verify matched opposite noise. Validation seed is 643 + recording index; long controls share seed 643.

The adapter is `../geometry_lora_continuation/lora_step128.safetensors` (299,136 spatial parameters, rank 2). It had already completed 128 updates on 32 training recordings before this resumption; no new training occurred here. Its base/motion/text weights and the GRU remain preserved. See `../METHOD_NOTES.md`, training run JSON and optimizer/RNG checkpoints for training provenance.

Long GRU: rolling past-32-frame context, first window emitted whole, then each window's last frame. Long diffusion: 32-frame windows, hop 16, Hann blending of predicted epsilon at every global DDIM step, common global noise tensor. Selected follow-up uses raw128 latents, unlike the earlier 256px render/re-encode experiment. Processing 8/16 seconds increases duration, not native spatial resolution. Arbitrary-audio `../long_inference.py` remains untested and guarded; these commands use existing validation features.

API/scheduler audit: `../../API_AUDIT.md`, `../METHOD_NOTES.md`, and `../../CORRECTION.md`. The old API audit's CPU-only statement is historical and corrected. Official versioned references: https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/animatediff and https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py . The earlier scheduler and pooled-text concerns were hypotheses, not established causes; no causal attribution to them is warranted.
