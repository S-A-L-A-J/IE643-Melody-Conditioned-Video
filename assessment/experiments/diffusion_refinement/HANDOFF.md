# Handoff — bounded diffusion refinement experiment

**Status: complete; negative pilot; reject the current refinement configuration.**

Team Nullspace, sole member Salaj Sonar. Executed 23 October 2026 IST, before the 25 October 2026 11:59 PM IST project deadline. All new experiment work is contained in this directory. No model training, external publication, submission or upload occurred. No experiment processes remain running.

## Outcome

The improved temporal-phase GRU predictor was connected to frozen SD1.5 + AnimateDiff through its predicted scaled latents, explicit forward noising, and native video-to-video denoising. This is a real diffusion refinement experiment, not a new diffusion training result. Diffusion receives a fixed text prompt and noisy initial video latents; audio enters only the GRU.

One first-by-manifest-order validation clip was used at 32 frames, 8 fps and 128×128. Three correct-audio strengths (0.10/0.20/0.30), wrong-recording and one-second local-feature-shift controls at 0.10, and a precision-only diagnostic were executed. All five refinement runs used seed 643, identical noise, prompt and scheduler settings. Stronger control cells and additional target recordings were deliberately not run after the pilot showed degradation.

Correct-audio estimated-beat F1 stayed at 0.800, while RGB MAE versus predictor increased to 0.0233/0.0395/0.0561 and flow-warp residual increased from 0.0204 to 0.0248/0.0355/0.0474. Fixed snapshots show progressive ring deformation and broken/patchy highlights, without demonstrated useful refinement. Wrong/shifted F1 stayed at their upstream predictor values (0.308/0.750). These controls demonstrate propagation of GRU inputs, not conditioning learned by frozen diffusion. The one-second shift can alias roughly two beats on this clip.

**Retain the existing GRU + pretrained VAE as the selected demo.** Keep this folder as defensible negative evidence. Do not replace the checkpoint, rewrite existing evaluation results, or start another training campaign based on this pilot. A claim that diffusion now learns music conditioning would be false.

## Start here

1. [RESULTS.md](RESULTS.md): settings, compact measured table, visual observations, stopping decision and limitations.
2. [REPORT_PARAGRAPH.md](REPORT_PARAGRAPH.md): short report-ready wording; not inserted into any report or slides.
3. [Controls video](videos/controls_predictor_vs_refined_s0.10.mp4): top row predictor, bottom row refinement; correct/wrong/shifted columns; identical correct reference audio in all panels.
4. [Strength sweep snapshots](videos/strength_sweep_contact.png) and [0.30 comparison](videos/pair_s0.30.mp4): visible degradation at fixed frame times and a same-audio paired video. Pair videos for 0.10 and 0.20 are also present.
5. [REPRODUCE.md](REPRODUCE.md), [API_AUDIT.md](API_AUDIT.md), [results_09.json](results_09.json), [verification.json](verification.json): reproduction and evidence.

## Environment and limitations

The original Python environment was used: torch 2.5.1+cu118, diffusers 0.35.1; exact dependency versions and local pretrained hashes are saved. CUDA was unavailable and `nvidia-smi` failed, so no RTX 3050 performance or VRAM feasibility claim is supported. The standard loader was killed; a streaming load then verified all 1,253 tensors. Native CPU bfloat16 inference was too slow, so the successful CPU pilot used bfloat16 storage with float32 Linear/Conv2d arithmetic. Logs preserve the failed attempts and one corrected keyword-argument bug.

Denoising took 33–116 seconds per case, excluding loading/decoding/rendering, with about 4.72 GiB peak process RSS. The final evaluation/rendering pass took 3:10, with four earlier videos cached. These are session observations, not rigorous hardware benchmarks. Timing metrics use estimated beats, coarse 8-fps visual events and a 125-ms tolerance; they do not establish semantic quality or mood accuracy. Flow and sharpness measures have blur/static/noise confounds. No human ratings exist. Assistant visual checks used fixed saved frame grids; author playback review remains available through the videos and is not represented as completed human evaluation.

The requested saved evidence/checkpoint/scripts and pretrained config hashes remain unchanged (19 tracked files). Fourteen generated media files passed frame-rate, duration and identical decoded-audio checks. Existing weights, results, report, slides and `sources/` were not edited. `assessment/fresh_evaluation` was not loaded, tuned on or reclassified; its already-evaluated status remains intact. Original provisional-test tensors were not loaded. The wrong-recording control uses validation features from another recording, not a new independent target evaluation.

## Reproduction / continuation

The executed commands and numerical caveats are in `REPRODUCE.md`; code is `run_pilot.py`, `verify_artifacts.py`, and `make_summary.py`. Reproduce only in a new sibling experiment folder to preserve this evidence. Do not automatically fill the unexecuted stronger-control cells: the stopping decision is intentional.

No user input is required to close this experiment. The next useful project action is author review of the paired videos and optional use of the supplied negative-result paragraph in the report. There is no pending publication or submission action in this handoff.
