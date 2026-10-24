# Melody-conditioned abstract video generation — IE643, Team Nullspace

Salaj Sonar, 24B2133. Repository: https://github.com/S-A-L-A-J/IE643-Melody-Conditioned-Video

Music-conditioned latent video prediction with optional adapted AnimateDiff appearance refinement. This repository also preserves unsuccessful direct audio-conditioned diffusion experiments. The demonstrated domain is synthetic abstract ring animation, not arbitrary cinematic video.

## Final architecture and findings

Accompaniment (pretrained Demucs) → 22 temporal music features + frozen CLAP context → trained phase-aware GRU → visual latents → optional adapted SD1.5/AnimateDiff refinement → frozen VAE decoder. The GRU supplies musical structure; diffusion does not directly receive audio. Selected refinement averages denoising results from opposite noise samples at strength .06, using a spatial rank-2 appearance LoRA trained for 128 total updates. The backbone and motion weights remain frozen.

Across 11 validation recordings, selected refinement reduced image reconstruction error by approximately 3.78% while preserving measured beat-event F1. This is validation-selected evidence, not an independent test or human preference result. The bounded strength pilot used one validation excerpt and one seed: greater denoising freedom eventually destroyed ring structure. Detailed results and limitations are under `assessment/experiments/diffusion_refinement/gpu_extension/resumed_20261024/RESULTS.md` and `assessment/experiments/prior_dependence_pilot/`.

## Setup and tested interface

Use Python 3.10 and FFmpeg. Install `project/requirements.txt` in a virtual environment; `project/requirements.lock.txt` records the working environment. Use mutually compatible torch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1 builds (original CUDA 11.8). Diffusion uses diffusers 0.35.1, accelerate 1.10.1 and PEFT (see lockfile).

```
python fetch_pretrained.py
python assessment/demo/app.py
```

Open http://127.0.0.1:7863. The tested interface generates a four-second **GRU-only** video, with optional vocal suppression. It does not expose final diffusion refinement. The original baseline remains selectable. Set an explicit visual phase to control initial arrangement.

```
python assessment/demo/inference.py assessment/evidence/unseen_synthetic_audio.wav --device cpu --visual-phase 90
```

The included audio is synthesized, not a Bollywood benchmark. A relocated code-package smoke test passed using the existing environment and local pretrained dependencies; a clean-machine dependency install was not validated.

## Diffusion refinement

Download full SD1.5 and AnimateDiff dependencies using `project/scripts/download_diffusion_models.py` (see its CLI/help). `fetch_pretrained.py` downloads only the dependencies required by the GRU interface. Publishers' licenses/access conditions apply. Backends expect the fp16 files in `project/models/diffusion/{sd15,animatediff}`. `IE643_PRETRAINED_ROOT` may point to a project directory containing `models/` instead.

A portable wrapper accepts a safetensors file with a `latents` tensor `[1,4,T,16,16]`:

```
python assessment/experiments/diffusion_refinement/refine_latents.py input.safetensors refined.safetensors
```

Requires CUDA; original device was RTX 3050 Laptop 4GB. This newly packaged wrapper was syntax-checked, not GPU-executed; its backend and selected weights were used in saved validation runs. Decode refined latents with the SD1.5 VAE, accounting for its scaling factor. `gpu_extension/extension.py` contains decode and rolling prediction helpers. Arbitrary-audio long inference is historical, guarded, and untested; do not claim it is the tested demo.

## Training, data, and reproduction limits

- `project/scripts/`: original preprocessing and direct diffusion/prior experiments.
- `project/outputs/`: retained own trained weights. Unnecessary optimizer states are omitted.
- `assessment/experiments/train_temporal_prior.py`: final GRU training and matched phase ablation.
- `assessment/experiments/diffusion_refinement/gpu_extension/`: appearance-adapter training, native backend, validation and duration-extension code.
- `assessment/experiments/diffusion_refinement/gpu_extension/resumed_20261024/`: selected paired-noise backend and numeric evidence.
- `assessment/evidence/`: earlier metric audits and synthesized smoke input.

Training requires reconstructing features and synthetic target videos from the source dataset; source audio/video and caches are excluded. Historical validation scripts also require those caches and sometimes saved intermediate tensors absent here. JSON files retain original absolute paths as provenance. Commands in historical notes refer to the original machine; they are not turnkey package commands. Do not remove preservation/stop guards to restart old experiments.

Original split: 100/11/14 source recordings for train/validation/provisional test (700/77/98 ring clips). Recording-disjoint is not necessarily song-disjoint. Original tests were repeatedly inspected. The later external excerpt evaluation concerns the frozen GRU, not an independent test of selected diffusion refinement. Metrics use estimated beats, not human annotations. No human ratings or semantic/mood accuracy are claimed.

## Sources and third-party components

- Audio source: [Navrasa-5000](https://huggingface.co/datasets/beastLucifer/navrasa-5000), revision `8056599a540110a31adab3cd297134e099aefc65`.
- External audio evidence: [Movie-MUSNOMIX](https://github.com/mrinmoy-iitg/Movie-MUSNOMIX), revision `715e701bbc39607bd870c02e0fd1e00a41d0eab6`.
- [Demucs](https://github.com/facebookresearch/demucs), [CLAP](https://huggingface.co/laion/clap-htsat-unfused), [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [AnimateDiff](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2).
- [Diffusers 0.35.1 AnimateDiff documentation](https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/animatediff).

No commercial recordings, third-party pretrained weights, or assessment recordings are redistributed. No blanket license is asserted over upstream material; retain original notices and follow each source's terms.
