# GPU continuation methods and evidence boundaries

This continuation follows the GPU-access correction in `../CORRECTION.md`. Original CPU outputs are historical and preserved. The host GPU is the RTX 3050 4 GB Laptop GPU; restricted sandbox execution hides CUDA. Execute the GPU commands through the host execution boundary. All pretrained loads are local-only and Hugging Face offline mode is enabled.

## Native confirmation and stronger frozen baselines

`../gpu_confirmation/run_gpu_pilot.py` repeats the original fixed strengths with native CUDA float16, full CLIP token conditioning and sequential CPU offload. It reuses the original float32 predictor latents and CPU-generated float32 noise before dtype conversion. It runs all nine condition/strength cells, rather than stopping at five. The result is `../gpu_confirmation/results_12.json` (three predictor controls plus nine refinements).

`gpu_backend.py` builds the same installed UNetMotionModel configuration using meta initialization and streams every source tensor, asserting shape and complete key coverage. The spatial SD and AnimateDiff pretrained weights are not rewritten. Sequential offload moves submodules through GPU memory; it does not replace CUDA arithmetic with CPU kernels. The short custom-denoising control exactly matches the native pipeline's saved latent output (`raw128_native_control.json`, maximum absolute difference 0.0).

The frozen short-clip sweep compares raw latents, a rendered-video VAE round trip at 128px, and bicubic-upscaled 256px video followed by VAE posterior-mode encoding. At 256px it compares strengths .03/.06/.10 with 100 base DDIM steps, plus CFG 3 at .06. Fine timesteps reduce large scheduler jumps. VAE round-trip and bicubic-only controls are saved, so blur/smoothing from preprocessing cannot be attributed to diffusion. No added detail is claimed for upscaling.

## Longer GRU generation

At 8 fps, eight seconds contains 64 frames and sixteen seconds contains 128. Four alternatives are saved:

- Direct GRU extrapolation: one long forward pass. It can drift visually beyond the trained context even when timing F1 is high.
- Concatenation: independent 32-frame predictions. This retains sharp shapes but resets appearance at joins.
- Hann overlap: 32-frame predictions every 16 frames, blended as latents. Different predicted geometry can ghost; a low warp residual may simply measure blur.
- Rolling context: emit the first trained-length window, then emit the last frame of each past-32-frame window. This keeps recurrent context and normalized time within training support, avoiding direct long-state extrapolation and double-image blending. After startup its normalized-time feature stays at the trained endpoint; global geometric drift can flatten. It does not create a new long-trained model.

The rolling method was selected using only the existing first validation recording, including fixed end-frame grids and estimated-beat metrics. Its eight- and sixteen-second outputs and matched eight-second wrong-recording/shifted-feature controls are retained. No fresh evaluation was loaded or reclassified. Broader song generalization is unverified.

## Longer diffusion generation

The saved motion adapter's positional sequence limit is 32 frames. The continuation does not feed it an unsupported 64/128-frame attention sequence. Each global DDIM step evaluates overlapping 32-frame windows at a 16-frame hop, blends their epsilon estimates with Hann weights, then updates the entire latent sequence once. Initial noise is one seeded global tensor, not independently sampled per output chunk. This is an explicit overlap algorithm; it is not claimed as an implementation or evaluation of FreeNoise.

An 8/16-second predictor video is rendered and re-encoded to 256px VAE latents before these steps. All audio controls change only the upstream predictor. There is no direct audio encoder in diffusion. Each comparison uses one original correct reference audio track across all panels. Same-duration matched runs share settings and float32 base-noise hashes.

## Bounded geometry LoRA

After frozen refinement remained unconvincing, a small spatial adapter was trained: 299,136 parameters, rank 2 on spatial attention q/v/output projections; the base UNet, motion modules, text encoder and GRU stay frozen. It uses the first clip of each of 32 existing training recordings. Each update samples eight contiguous frames from a cached 32-frame, 128px target, noised at a timestep from 10–120 under the same linear forward-noise schedule as inference. The objective is ordinary epsilon MSE. No contrastive music loss or direct audio feature enters this training.

The first block is 64 updates; a single additional 64-update block was allowed after monotonically improving validation denoising loss and improved generated target-latent error relative to frozen diffusion. No further training is authorized by this experiment plan. Both blocks preserve weights, optimizer/scaler/RNG state at each 16-step checkpoint. Selection uses the same fixed first eight validation recordings and noise levels 50/100, with fixed noise. The chosen checkpoint is recorded in `selected_adapter.json`.

This is a **trained synthetic-ring appearance adapter in a music-conditioned pipeline**, not a demonstration of diffusion learning direct music, melody, semantics or mood conditioning. Denoising loss is not output quality. `validation8/` compares predictor, frozen refinement, adapted refinement and the VAE-decoded held-validation target on the same eight groups. These groups were also used for checkpoint selection, so the evidence remains validation evidence, not an independent final test.

Earlier per-refinement JSON files contain `trained_parameters: 0`; that field counted training inside the inference call. Adapted runs also identify their geometry-LoRA checkpoint, whose actual training is documented separately. Later records use explicit `trained_parameters_in_this_refinement_call` and `adapter_parameters_trained_before_inference` fields to remove that ambiguity. All LoRA inference is frozen during sampling.

## Metric interpretation

Extension diagnostics use retained raw RGB arrays, normalized to 128px for cross-resolution comparisons; the earlier CPU/native confirmation also used encoded-video event measurements. Do not silently equate these implementations. Matching is one-to-one within 125 ms of estimated beats. Timing cannot establish semantic quality or mood accuracy. Optical-flow residual and sharpness are descriptive and have blur/static/noise confounds. Frame changes at four-second boundaries may be legitimate musical responses. Encoded-target MSE is useful for this supervised synthetic domain but is not a human rating. Nearly unchanged output is not evidence of useful refinement, including secondary residual-anchor MSE diagnostics.

No human quality ratings were created. Frame-grid observations are assistant inspections; paired videos are provided for author playback review. Existing report/slides and original model/evaluation files are not modified.
