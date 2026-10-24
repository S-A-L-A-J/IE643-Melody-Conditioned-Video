# Latest state — 24 October 2026 IST

**Stopped at the user's request. Do not start more training or diffusion sampling.**

Read [the GPU continuation handoff](gpu_extension/HANDOFF.md), [GPU-access correction](CORRECTION.md), and [stop record](gpu_extension/STOPPED.json).

Useful result: rolling-context GRU generation extends the preserved predictor to 8/16 seconds with clearer long output on the fixed validation recording. Diffusion runs on the host GPU and supports 8/16 seconds; a small appearance LoRA improves validation denoising loss, but no visual gain over the predictor is established. Do not turn a denoising-loss improvement into a generated-video quality claim.

All original CPU experiment artifacts, including the original `HANDOFF.md` and `RESULTS.md`, are preserved as historical evidence. Their host-GPU availability conclusion was corrected. The latest handoff is `gpu_extension/HANDOFF.md`.
