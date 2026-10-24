# Frozen AnimateDiff refinement pilot — results

**Decision: reject this configuration for the project demo; retain the improved supervised GRU predictor.** The latent-to-diffusion connection works, but this pilot shows no useful refinement benefit. Preserve the experiment as a negative result. This does not establish that other diffusion approaches cannot work.

Team Nullspace — Salaj Sonar. Executed 23 October 2026 IST. One four-second validation clip, `2c8e26308b2879a4_0.000__rings`, selected by manifest order before generation. Other-recording control: `399a3d13a9b756d9_0.000__rings`. No fresh-evaluation or provisional-test tensors were used. Recording/song independence remains unverified.

## Fixed method

The preserved temporal-phase GRU checkpoint predicts 32 scaled SD-VAE latent frames at 8 fps, 128×128. A frozen SD1.5 UNet with the existing AnimateDiff motion adapter starts from these predictions plus fixed Gaussian noise. Native AnimateDiff video-to-video denoising uses DDIM, 20 base steps, linear betas 0.00085–0.012, epsilon prediction, linspace spacing, offset 1, no clipping, eta 0, seed 643, and the full 77-token rings prompt. Guidance scale is 1.0. Strengths 0.10/0.20/0.30 execute 2/4/6 denoising steps respectively. There is no VAE encode/decode round trip before refinement.

The validation renderer-seeded visual phase is held fixed across conditions. Wrong-audio changes both local music features and the global audio context entering the GRU; shifted-audio circularly rolls only local features by eight frames (one second). All videos are played against the same correct reference audio. **Diffusion receives no audio and learns no parameters.** Control differences propagate through the predictor's initial latents; they are not evidence of learned diffusion audio conditioning.

## Measured results

| Variant | Beat F1 | RGB MAE | Change correlation | Warp residual | Denoising s |
|---|---:|---:|---:|---:|---:|
| Correct predictor | 0.800 | 0.00000 | 1.0000 | 0.02038 | — |
| Precision only | 0.800 | 0.00023 | 1.0000 | 0.02031 | — |
| Correct + refinement 0.10 | 0.800 | 0.02333 | 0.9926 | 0.02482 | 32.8 |
| Correct + refinement 0.20 | 0.800 | 0.03952 | 0.9793 | 0.03547 | 66.5 |
| Correct + refinement 0.30 | 0.800 | 0.05607 | 0.8990 | 0.04738 | 116.0 |
| Wrong-audio predictor | 0.308 | 0.00000 | 1.0000 | 0.01758 | — |
| Wrong + refinement 0.10 | 0.308 | 0.02241 | 0.9907 | 0.02303 | 49.6 |
| Shifted predictor | 0.750 | 0.00000 | 1.0000 | 0.01943 | — |
| Shifted + refinement 0.10 | 0.750 | 0.02274 | 0.9920 | 0.02271 | 33.7 |

RGB MAE is computed on retained decoded RGB arrays on a 0–1 scale against each condition's own predictor. Change correlation compares consecutive-frame change signals with that same predictor. Warp residual is estimated optical-flow reconstruction error on encoded video, also on a 0–1 scale. Lower warp residual may reward static/blurred video; higher sharpness may reward noise. These are descriptive diagnostics, not human quality ratings. All timing scores match estimated beats and visual events one-to-one within 125 ms; they do not establish semantic quality or mood accuracy.

The correct predictor and all three refinements match six of eight estimated beats against seven detected events (F1 0.800). Thus preserved F1 alone would conceal increasing distortion. The precision-only RGB MAE is 0.00023, whereas refinement changes pixels by 0.02333–0.05607. The outputs are visibly different, rather than an unchanged input being presented as success. At 0.30, warp residual is about 2.33 times the predictor value and frame-change correlation falls to 0.899. Sharpness falls from 1336 to 1301/1191/1002 across the three strengths, without justifying a standalone quality ranking.

Fixed snapshots at 0.5, 1.5, 2.5, and 3.5 seconds show growing ring deformation, uneven edges, broken highlights and patchy colors at 0.20/0.30; 0.10 has smaller but visible distortions and no demonstrated benefit. This is an assistant visual inspection of saved frames, not fabricated human evaluation. Temporal diagnostics worsen with strength, although optical flow on abstract rings is imperfect. Playback assets are provided for author review.

At 0.10, wrong-audio and shifted-audio F1 remain exactly at their predictor baselines, 0.308 and 0.750. This is consistent with timing inherited from the GRU. The one-second shift is a weak negative control for this nearly half-second beat period: it can alias approximately two beats. The separate +200 ms reference control, retained in JSON, is more disruptive. No population inference, significance test, mood/melody accuracy claim, or human preference claim is warranted from one validation recording.

## Runtime, compatibility and stopping

CUDA was unavailable despite the described RTX hardware. These are CPU observations, not RTX 3050 benchmarks or 4 GB VRAM feasibility results. Successful denoising used bfloat16 storage/activations with float32 Linear/Conv2d arithmetic; GRU and VAE decoding used float32. Per-case denoising took 32.8–116.0 seconds; peak process RSS was approximately 4,835 MiB (4.72 GiB). Runtime varies with CPU load/thermal state. Denoising times exclude model loading, decoding and rendering. The final evaluation pass decoded five new outputs, reused four cached outputs, and rendered comparisons in 3:10 wall time with about 1.13 GiB peak RSS (`evaluate_final.log`); this is not a full end-to-end benchmark.

The standard loader was killed before denoising, and a native bfloat16 CPU run was interrupted for slowness. Their logs are retained. A streaming loader verified complete coverage of 1,253 weight/buffer tensors, and the CPU arithmetic adjustment made the pilot practical. See `API_AUDIT.md` for installed-source evidence and official documentation. Old DDPM training/DDIM inference use identical forward-noise arrays; that suspected mismatch is not supported. Old text mean-pooling differs from native token conditioning, but neither it nor the scheduler recommendation is proven to explain earlier audio-control failures.

Stopped after five refinement runs: three correct-audio strengths and the two 0.10 controls, plus the precision-only diagnostic. Stronger audio-control cells and additional recordings were not run because visible degradation and unchanged timing did not support expansion. No training, model downloads, publication, upload, or changes to existing results/report/slides occurred. `verification.json` checks preserved input hashes, matched noise, 32-frame media, and identical decoded audio across all comparisons.

## Review artifacts

- `videos/strength_sweep_contact.png`: correct predictor followed by strengths 0.10/0.20/0.30, fixed frame times.
- `videos/pair_s0.10.mp4`, `pair_s0.20.mp4`, `pair_s0.30.mp4`: predictor versus refinement, same audio.
- `videos/controls_predictor_vs_refined_s0.10.mp4`: predictor controls on the top row, matched refinements below; columns correct/wrong/shifted, same reference audio.
- `results_09.json`: final complete diagnostics. Earlier `results_03.json` and `results_04.json` remain intact.
- `REPRODUCE.md`, `run_pilot.py`, `package_versions.json`, per-run latent JSON files and logs: reproduction and environment evidence.
