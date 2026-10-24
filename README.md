# Melody-conditioned video generation

IE643 course project · Team Nullspace · Salaj Sonar (24B2133)

A temporal GRU predicts abstract video latents from music features and CLAP context. An adapted Stable Diffusion / AnimateDiff stage optionally refines their appearance. Earlier direct audio-conditioned diffusion experiments are retained for comparison.

## Run the demo

Use Python 3.10 and FFmpeg, then install `project/requirements.txt` with compatible PyTorch builds.

```sh
python fetch_pretrained.py
python assessment/demo/app.py
```

Open http://127.0.0.1:7863. The tested upload interface runs the GRU and pretrained VAE; diffusion refinement is a separate stage. Third-party pretrained weights and source recordings are not included.

## Read the code

Start with **[TemporalPrior](assessment/experiments/train_temporal_prior.py)**: the model class and `forward()` method show the complete music-to-latent network. Then follow the [short reading guide](docs/CODE_GUIDE.md).

| Location | Purpose |
|---|---|
| `assessment/demo/` | Upload interface and inference pipeline |
| `assessment/experiments/train_temporal_prior.py` | Final GRU architecture, losses and training |
| `assessment/experiments/diffusion_refinement/` | Appearance adaptation, refinement and evaluation |
| `project/scripts/` | Audio/data preparation, shared utilities and earlier model experiments |
| `project/outputs/` | Our trained checkpoints |
| `project/archive/` | Historical plotting and review helpers |

## Results and scope

The demonstrated domain is synthetic abstract animation. Selected paired-noise refinement reduced image MSE by 3.78% across 11 validation recordings; the combined figure includes model-selection data. Musical conditioning comes through the GRU, not a direct audio input to the final denoiser. Estimated-beat metrics do not establish melody transcription, mood accuracy or human preference.

See [setup, evidence, sources and reproduction limits](docs/REPRODUCIBILITY.md) for full details. The newly packaged latent-refinement wrapper has static checks only; clean-machine installation has not been validated.
