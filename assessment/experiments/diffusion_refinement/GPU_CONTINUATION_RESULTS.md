# GPU continuation — results at user stop

**Recommendation: retain the longer rolling-context predictor; stop further diffusion experiments.**

GPU access outside the sandbox worked. The original CPU artifacts remain unchanged. Only saved-latent rendering and verification occurred after the stop request.

| Completed output | Duration | Estimated-beat F1 | Warp residual |
|---|---:|---:|---:|
| Direct GRU | 16 s | 0.918 | 0.02331 |
| Rolling GRU | 8 s | 0.903 | 0.02207 |
| Rolling GRU | 16 s | 0.968 | 0.02280 |
| Frozen diffusion, rolling initialization | 8 s | 0.903 | 0.01793 |
| Frozen diffusion, rolling initialization | 16 s | 0.968 | 0.01853 |

These are one-recording validation diagnostics. Duration changes the beat population; compare direct versus rolling on the same duration. Warp residual may reward blur. Timing uses estimated beats and does not establish semantic quality or mood accuracy. No human ratings exist.

| Small geometry adapter check | Measured result |
|---|---:|
| Fixed eight-recording validation denoising MSE, frozen | 0.294196 |
| Same diagnostic after 64 updates | 0.179517 |
| Same diagnostic after 128 updates | 0.145059 |
| First-clip target latent MSE, predictor | 0.101597 |
| First-clip target latent MSE, frozen refinement | 0.113182 |
| First-clip target latent MSE, 64-update refinement | 0.106759 |

The adapter improves the denoiser, but no generated-video gain over the predictor is established. The step-128 checkpoint was not output-quality tested before stopping. Eight-recording generated-output evaluation, adapted long sampling and the generic audio-file CLI smoke test were not executed. Music control remains upstream in the GRU; the adapter learns synthetic-ring appearance.

Frozen 16-second sampling completed in 100.2 seconds at about 538 MiB allocated VRAM; spatial-adapter training peaked around 2,633 MiB. These are observations on this host with sequential offloading, not universal benchmarks.

Latest handoff: `gpu_extension/HANDOFF.md`. Method caveats: `gpu_extension/METHOD_NOTES.md`. Stop boundary: `gpu_extension/STOPPED.json`. Completed media passed 8-fps, 64/128-frame and matched-audio checks in `gpu_extension/final_media_verification.json`.
