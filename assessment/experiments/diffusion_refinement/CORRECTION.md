# GPU-access correction — 24 October 2026 IST

The initial pilot checked CUDA only inside the restricted execution sandbox. Inside that sandbox, `nvidia-smi` fails and torch reports zero CUDA devices. After the user's correction, the same read-only checks outside the sandbox succeeded: NVIDIA RTX 3050 4 GB Laptop GPU, driver 580.178.04, torch 2.5.1+cu118, CUDA available with one device.

The assistant should have checked this boundary before choosing CPU execution. Earlier statements that the hardware was unavailable apply only to the restricted sandbox and must not be read as a host-level finding. The CPU results are retained as historical measurements of that numerical configuration; they are not a sufficient basis to dismiss GPU execution.

`gpu_confirmation/` preserves the predictor latents and repeats the fixed pilot with native CUDA float16 and sequential CPU offloading. No CPU arithmetic wrappers are active. Every run records GPU name, allocated/reserved peak memory and runtime. The first two-step refinement completed in about 5.2 seconds, versus about 32.8 seconds in the CPU arithmetic configuration. Later durations are saved per case. Identical float32 base noise is generated on CPU before conversion to float16, so changes in RNG algorithms across devices are avoided.

The user subsequently authorized continued bounded work on useful diffusion refinement and longer-than-four-second generation for both methods. That work is stored separately under `gpu_extension/`. Prior CPU reports and `HANDOFF.md` are historical; consult this correction and the later continuation handoff before using their stopping recommendation. Existing checkpoints, validation evidence, fresh-evaluation status, report and slides remain preserved.
