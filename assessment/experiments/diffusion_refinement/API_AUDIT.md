# Implementation and compatibility audit — 23 October 2026

Inspected the eight requested saved sources before execution. Their hashes and timestamps are in `input_inventory_before.json`. The temporal checkpoint is a supervised GRU plus convolutional latent decoder; its selected epoch is 80, validation latent MSE 0.103392. It generates 32 scaled SD-VAE latent frames at 8 fps and 128×128. Earlier diffusion-adapter evidence does not establish reliable audio conditioning. No old adapters are loaded in this experiment.

## Installed API and official references

Runtime: original project Python, torch 2.5.1+cu118, diffusers 0.35.1. Installed source excerpts, including `__call__`, `prepare_latents`, `get_timesteps`, and `encode_prompt`, are preserved in `installed_api_audit.txt`. The source file and its hash are recorded in `environment.json` after a successful run.

Official documentation consulted on 23 October 2026:

- [Diffusers 0.35.1 AnimateDiff, including video-to-video example](https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/animatediff)
- [Versioned pipeline implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py)
- [Versioned UNetMotionModel implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/unets/unet_motion_model.py)

The official example uses DDIM with a linear beta schedule, no sample clipping, linspace timesteps and offset 1. The documentation notes sensitivity to the beta schedule. This is a configuration recommendation, not proof of why earlier experiments failed.

## Scheduler hypotheses checked

The old prior-guided training script uses DDPM epsilon prediction and scaled-linear betas 0.00085–0.012; its inference script uses DDIM with the same training beta parameters and epsilon prediction. Their alpha-cumprod arrays match exactly (`scheduler_audit.json`). Thus a blanket claim of mismatched forward-noise schedules in those two scripts is unsupported. DDPM training with DDIM inference is itself a supported arrangement. A linear-beta pilot changes the effective noise at a given integer timestep; it does not isolate a causal scheduler improvement over old runs.

The pilot takes the final `int(20*strength)` timesteps: 2, 4, or 6 denoising evaluations for strengths 0.10, 0.20, or 0.30. This small number at low strength is a compute/quality limitation. Actual timesteps and full scheduler configurations are saved per output. No zero-step strength is allowed.

In the installed `prepare_latents`, supplied latents receive added noise only when `add_noise=True`. `__call__` passes `add_noise=enforce_inference_steps`. With the default `False` and supplied latents, blindly passing clean predictions would omit the intended forward noising. We explicitly add fixed noise to the predictor latents at the selected first timestep, then call the native pipeline with supplied noisy latents, `enforce_inference_steps=False`, and the same strength schedule. The native pipeline does not add a second noise sample in this path. `enforce_inference_steps=True` would instead change the effective base schedule; it is not used.

## Text conditioning

Old training and inference both mean-pool the CLIP token sequence to one token. This is shape-compatible cross-attention and internally consistent between those two scripts, but differs from the token sequence used by pretrained SD and the native pipeline. It may weaken text information; it is not established as the cause of the audio-control failure. This pilot calls native `encode_prompt`, asserts shape `[1,77,768]`, and lets the pipeline repeat that sequence per frame. A fixed rings prompt and guidance scale 1.0 avoid doubling the denoiser batch. Guidance scale 1 still uses the conditional text sequence but disables classifier-free guidance. No audio, CLAP vector, or music features enter diffusion directly.

## Memory and numerical limits

`nvidia-smi` fails; torch reports CUDA unavailable and zero devices. The named RTX 3050 is not accessible in this session. CPU bfloat16 is used for frozen diffusion, float32 for the GRU and VAE decoding. GPU runtime/VRAM claims cannot be made from this run. Peak process RSS is a cumulative high-water mark, not incremental model memory.

The initial standard `from_pretrained` loader was killed with exit 137 during combined model construction, before denoising (`refine_first.log`). The native 2D-to-motion conversion transiently constructs another model, which is expensive in RAM. The revised loader creates the exact installed native motion-model configuration on the meta device, then streams each immutable source tensor into one bfloat16 model. Every source key, shape, uniqueness and complete state-dict coverage is asserted. The run reports 1,253 loaded tensors. No model weights are randomly substituted or trained. Ignored legacy MotionAdapter config fields generate a warning retained in logs; native defaults and the installed conversion define the architecture.

This is direct latent video-to-video refinement: no decode/re-encode round trip or VAE posterior sampling is inserted. Input/output predictions are decoded with the same pretrained VAE. It tests refinement of the predictor's own latent representation, rather than refinement of an arbitrary uploaded encoded video.

### CPU arithmetic adjustment

The native bfloat16 attempt was interrupted with exit 130 after more than seven minutes without completing its first denoising step; no refined latent was produced. A cold-call matrix microbenchmark (`cpu_linear_microbenchmark.json`) measured 0.020 s float32 versus 0.684 s bfloat16 for its stated shapes. This is only a hardware diagnostic, not a whole-model speed claim.

The measured pilot therefore keeps bfloat16 weights and activations but computes each `Linear` and `Conv2d` operation in float32, casting its output back to the input dtype. The unchanged installed attention/norm and scheduler implementations are retained. This trades per-layer temporary memory for optimized CPU kernels without a full float32 model copy. Native bfloat16 versus this arithmetic path produced identical outputs on the two small sanity examples in `cpu_arithmetic_sanity.json`; full-model bitwise equality is untested. Actual diffusion output uses this explicit mixed arithmetic configuration throughout and cannot be claimed to reproduce native GPU float16 results. The first adjustment attempt exposed a keyword-argument bug before completing denoising (`refine_cpu_arithmetic_first.log`); the corrected wrapper accepts the native `input` argument. Both failed logs are preserved.
