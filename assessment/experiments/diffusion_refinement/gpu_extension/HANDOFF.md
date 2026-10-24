# Handoff — stopped at user request, 24 October 2026 IST

**Stop further experiments.** The user asked to stop if diffusion was not producing a useful result. No further training, diffusion sampling, new audio-file smoke tests or generated-output validation should run without a new explicit request. `STOPPED.json` records the boundary; entry points have stop guards. Rendering saved latents and writing this handoff are finalization only.

Team Nullspace; sole member Salaj Sonar. Existing checkpoints, evaluation files, report/slides and `sources/` are preserved. All continuation code, weights, logs and media are inside `diffusion_refinement/`. Nothing was uploaded or submitted. Fresh evaluation was not loaded or reclassified.

## Useful result: longer predictor

The preserved GRU can be used for 8- and 16-second output with a rolling 32-frame (four-second) context. Emit the first window, then the last frame from each past-32-frame window. This avoids both recurrent extrapolation drift and blending incompatible geometric states. The same checkpoint is used; no GRU retraining occurred.

On the fixed first validation recording, eight-second estimated-beat F1 is 0.903 and sixteen-second F1 is 0.968. Direct extrapolation on the same sixteen seconds scores 0.918 and visibly distorts near the end. Concatenating four-second clips retains sharpness but resets appearance; overlap blending can ghost. Rolling-context frame grids retain clearer rings. This is single-recording validation evidence, not a general long-video quality guarantee. After startup, the normalized time feature stays at the trained window endpoint, so global geometric drift can flatten. Timing uses estimated beats and does not establish semantic quality or mood accuracy.

Relevant saved media:

- `videos/d08_correct_rolling.mp4` — 8-second predictor output.
- `videos/d16_correct_rolling.mp4` — 16-second predictor output.
- `videos/comparison_predictor16_4panels.mp4` — direct, concatenated, overlap-blended and rolling methods, same audio.
- `videos/rolling_preview.png` — fixed-frame assistant inspection, not human ratings.

## Diffusion: operational progress, no established visual win

The original sandbox hid CUDA. Host execution confirms the RTX 3050 4 GB GPU and native float16 execution. See `../CORRECTION.md` and `../gpu_confirmation/`. The first native two-step run took 5.2 seconds; full native confirmation retained the earlier distortion finding. Do not repeat the earlier host-level inference that the GPU was unavailable.

This continuation tested render/re-encode initialization, 256px processing, 100-step base schedules at low strengths and a CFG control. These reduce some obvious artifacts compared with raw-latent 128px refinement, but do not demonstrate a useful gain over predictor-only output. The custom short sampler exactly reproduces the native saved result (maximum latent difference 0.0).

Frozen diffusion also generated eight- and sixteen-second latent sequences using overlapping 32-frame UNet windows at each global scheduler step, with shared seeded noise. Eight-second correct/wrong/shifted predictor controls are saved. The sixteen-second sequence has 128 frames; sampling completed in 100.2 seconds with about 538 MiB peak allocated VRAM. It had completed before the stop request. Final rendering of these already-generated latents is recorded in `render_completed_outputs.log`; no additional denoising is performed by that stage. See `videos/comparison_long8_5panels.mp4` and `videos/comparison_long16_3panels.mp4` when finalization is complete.

A small spatial geometry LoRA (299,136 parameters) was trained for 64 updates, then one bounded 64-update continuation. The full pretrained model, motion modules, text encoder and GRU were frozen. Training used 32 existing training recordings and standard low-noise epsilon MSE. Fixed eight-recording validation denoising loss fell from 0.294196 to 0.145059 (50.7%). All checkpoints and optimizer/scaler/RNG states are preserved; `selected_adapter.json` points to step 128.

This is **not proof of improved generated video**. At step 64, target-latent MSE on the first validation clip improved relative to frozen refinement (0.11318 → 0.10676), but remained worse than predictor-only (0.10160). The step-128 checkpoint was checked only through denoising loss, not generated-output quality. The same validation groups were used for checkpoint selection; they are not an independent test. Diffusion receives no direct audio feature. The LoRA learns synthetic-ring appearance, while music control remains upstream in the GRU.

## What was deliberately not run after stopping

- Step-128 generated-output comparisons and adapted long-video sampling.
- `evaluate_adapter_validation.py` eight-recording generated-output evaluation.
- An arbitrary audio-file smoke test of `long_inference.py`.

Those scripts exist but are unexecuted. Do not call the generic CLI tested or deployable. The tested 8/16-second experiment path is `extension.py` with saved validation features. There are no further training jobs planned. Do not fill missing result cells with denoising-loss numbers or imply that an unexecuted comparison passed.

## Recommendation

Retain the rolling-context GRU extension for longer project demos, subject to author playback review. Keep diffusion and its adapter as research evidence; do not replace the selected generator or claim a diffusion quality improvement. Stop further diffusion work under the user's latest instruction. A higher denoising score is not needed to justify this decision, and no human ratings were fabricated.

`METHOD_NOTES.md` records implementation and metric caveats. `prepared.json`, `rolling_prepared.json`, per-run JSON, metrics snapshots and training reports retain configurations. `preservation_at_stop.json` checks all 19 tracked input files and all 76 original experiment artifacts unchanged. Original CPU conclusions remain historical; this handoff and the GPU correction supersede their hardware statements and stopping chronology.
