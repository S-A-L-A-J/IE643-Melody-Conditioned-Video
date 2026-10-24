# IE643: music-conditioned video generation

Goal: adapt AnimateDiff + Stable Diffusion 1.5 using synthetic music–video pairs.
The procedural renderer creates supervision; the eventual generator is diffusion.

## Environment

```bash
cd /home/salaj/IE643_project
source activate.sh
python scripts/check_environment.py
```

Python 3.10 venv, isolated from existing Conda and ROS environments. Its interpreter
is Ubuntu's `/usr/bin/python3.10`. Use `activate.sh` to exclude shell-exported ROS paths.
PyTorch/torchaudio 2.5.1 and torchvision 0.20.1 use CUDA 11.8 wheels to keep
Demucs audio I/O compatible. The host NVIDIA driver is not changed.
`requirements.lock.txt` records the actual installed packages after verification.
FFmpeg is supplied by the existing system installation.
Run `./setup.sh` only if recreating the environment from scratch.

## Audio dataset

Source: https://huggingface.co/datasets/beastLucifer/navrasa-5000-dataset
Revision: `8056599a540110a31adab3cd297134e099aefc65`.
Only `India/Bollywood.zip` is imported (320,774,369 bytes).
Archive SHA256: `7a7dca3a92d6cdef56054baf50a1a6778eae9d609dee8d184c41bd5a626353b3`.
Original card and catalog are retained in `data/manifests`.
The uploader labels the collection CC0; this is an uploader claim, not independent
verification of rights to underlying commercial recordings. Do not describe the
audio as independently rights-cleared or redistribute it with project code.

The archive contains 132 files: 125 eligible excerpts, one exact duplicate and six
excerpts shorter than four seconds (including one empty file). All originals are
retained. The catalog includes compilations, covers, remixes and instrumentals. File names
are source titles, not verified song identities. All data is DEVELOPMENT ONLY.
Before evaluation: identify songs, check acoustic duplicates and overlap across
compilations, and assign song-grouped splits. File hashes catch exact duplicates
only. No benchmark/test split is claimed by this starter.

```bash
python scripts/prepare_dataset.py
python scripts/separate_vocals.py --limit 3
python scripts/separate_vocals.py
python scripts/build_clips.py
python scripts/verify_dataset.py
python scripts/make_listening_report.py
```

Separation uses pretrained `htdemucs`, four-second inference segments, zero random
shifts and 25% overlap. This is a speed-oriented development configuration; compare
quality against more shifts on a small listening sample before final preprocessing.
It retains float32 vocal and accompaniment WAVs and records per-file completion,
runtime and GPU memory. Reruns skip completed outputs. Vocal suppression is imperfect;
the singer's melodic line may disappear and some instruments may be affected.
Keep originals and inspect both stems. No claim of perfect vocal removal is made.

`clips.jsonl` indexes four-second windows by parent path and start time, avoiding
duplicate audio. These are audio-only examples, not completed audio–video pairs.
Use `python scripts/build_clips.py --duration 8` to rebuild this index with eight-second
windows; original recordings and separated stems remain complete.

Open `reports/listening_review.html` for side-by-side original/instrumental/vocal playback.
The initial listening review found useful vocal/instrumental separation. Inspect
more songs before assigning final quality labels.

## Aligned features and preliminary synthetic pairs

```bash
python scripts/extract_features.py
python scripts/verify_features.py
python scripts/plot_features.py
python scripts/render_pairs.py
python scripts/build_pair_manifest.py
python scripts/pair_dataset.py
```

All 125 instrumentals have rhythm, harmonic pitch-class, dominant spectral pitch,
RMS and four frequency-band energy streams sampled at 16 Hz. The dominant pitch
is a polyphonic heuristic, not validated lead melody transcription. A frozen global
mood embedding is still pending. `reports/feature_overview.png` shows examples.

The first E0 dataset contains 875 deterministic four-second videos at 256×256 and
16 fps. `data/manifests/synthetic_pairs.jsonl` records exact audio offsets, feature
frame offsets, video paths and renderer parameters. `scripts/pair_dataset.py` loads
one sample with video shaped `[3, 64, 256, 256]` and separate rhythm, melody and
dynamics tensors. See `outputs/e0_audio_video_preview.mp4` for a synchronized
example. The video files themselves are silent; the manifest links them to audio.

The first renderer has one abstract visual family and tests the data pipeline
and timing. Musical correspondence and visual diversity still need human review
before final adaptation supervision is claimed.

Two additional music-reactive visual families, rings and bars, are now available
alongside the original geometric renderer. All 125 instrumentals also have a
frozen 512-dimensional CLAP whole-excerpt audio-context embedding. The six
audio-text mood similarities are exploratory; they are not validated emotion
labels and are heavily skewed toward “energetic” on this collection. The
expanded manifest has 2,625 verified aligned pairs. See
`reports/synthetic_styles_review.html` for synchronized examples.
The frozen model is `laion/clap-htsat-unfused`, pinned to revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.

```bash
python scripts/find_audio_overlap.py
python scripts/make_provisional_split.py
python scripts/extract_global_audio_context.py
python scripts/render_pairs_v2.py
python scripts/build_pair_manifest_v2.py
```

The new train/validation/test split groups two strong detected audio-overlap
candidates and keeps prior debug-training tracks out of test. Its 100/11/14
tracks are **provisional**: many source files are jukeboxes, mashups or covers,
and neither their song identities nor all cross-file song overlap is verified.
Do not present these partitions as a song-disjoint benchmark.

The programmed targets were assessed on all 14 provisional test tracks. Beat
event F1 is 0.812 geometric, 0.866 rings and 0.801 bars; shifting audio events
by 200 ms lowers them to 0.275, 0.151 and 0.242. This verifies timing of the
synthetic renderer, not learned diffusion control. See
`reports/provisional_renderer_benchmark.json` for precision, coverage and
frame-change diagnostics. A blank human-rating form is in
`reports/provisional_human_review_template.csv`.

## Diffusion setup

`scripts/download_diffusion_models.py` downloads pinned SD 1.5 and AnimateDiff
checkpoints and checks available storage first. Model revisions and selected files
are recorded in `reports/diffusion_download_plan.json`.
`scripts/smoke_diffusion.py` checks local loading and tiny 256x256 inference using
sequential CPU offloading. This debug test uses four frames and two denoising steps,
not a four-second final video. It has no musical conditioning and does not establish
training feasibility. Full output duration remains 4–8 seconds.

## First adapter-training diagnostic

The local RTX 3050 can train a small frozen-backbone AnimateDiff adapter at
256×256 using 16 frames sampled over four seconds and CPU activation offloading.
The trainable parts are rhythm, melody-proxy and dynamics projectors plus rank-2
LoRA in temporal and cross-attention layers. The 128-step, eight-song debug run
is reproducibly checkpointed under `outputs/tiny_e2/` and peaked at roughly
3.4 GB reserved GPU memory. Run or resume it with:

```bash
python scripts/train_tiny.py --steps 128 --pairs 8 --frames 16 --stride 4 --size 256 --checkpoint-every 16 --resume
python scripts/evaluate_tiny.py
python scripts/infer_tiny.py --steps 12
```

The first command requires an existing checkpoint when `--resume` is used; omit
that flag to start afresh. The swap evaluation and generation settings are saved
in `reports/tiny_e2_swap_evaluation.json` and
`reports/tiny_e2_generation.json`. Two generated videos with audio are
`outputs/tiny_e2_correct_with_audio.mp4` and
`outputs/tiny_e2_mismatched_with_audio.mp4`.

This first run did **not** establish music control: at a fixed noise level,
correct music gave a lower prediction loss on only 3 of 8 training clips, and
the two generated previews under different music were visually near-identical.
Use these diagnostics before spending compute on long training. There is no
held-out diffusion result yet.

A second, isolated run (`scripts/train_contrastive.py`) tried normalized inputs,
new rings targets and a paired wrong-song ranking loss for 32 steps. Correct
music won 4/8 training and 4/8 provisional test comparisons; a one-second
within-song shift was worse on only 3/8 in both groups. Its generated training
clip scored 0.20 beat-event F1, the same as a 200 ms shifted-audio control.
These checks also fail to establish reliable music control. Settings, weights,
swap results and video diagnostics are under `outputs/contrastive_rings/` and
`reports/contrastive_rings_*`.

A third experiment (`scripts/train_spatial.py`) tests direct spatial injection
of local music features into latent video frames. It uses 8 frames across four
seconds at 128×128 to probe conditioning cheaply before any full-size run.
It also failed the audio-control check (3/8 provisional test swaps).

The strongest learned-control result is a separate supervised latent prior
(`scripts/train_audio_latent_prior.py`) trained against 700 synthetic rings
targets, selected on 77 validation clips and evaluated on 98 provisional test
clips. Correct music gave lower encoded-target error than another track on
98/98 clips and than a one-second time shift on 97/98. It improved MSE by
about 7.4% over a static template. Decoded output is blurry at 2 fps and is
not the diffusion generator; visible beat synchronization is unproven. A
no-global-context ablation performed similarly, so the CLAP stream has not
shown a clear benefit. See `reports/audio_prior_review.html` and
`reports/audio_latent_prior.json`.

A 128-step diffusion LoRA trained to interpret this frozen prior generated
sharper ring-like scenes, but correct music won only 4/14 provisional test
swaps and 1/4 clearer single-title test swaps. Denoising from the prior with
the frozen diffusion model also worsened validation error. Neither route has
yet passed the music-control checkpoint. See
`reports/prior_guided_diffusion_test.json` and `PROGRESS.md`.

## Remaining work

1. Resolve song identities, duplicate performances and grouped splits before
   held-out evaluation.
2. Add a frozen global mood embedding and more visual styles to synthetic targets.
3. Make conditioning influence generation on a tiny set, then compare E1/E2/E3
   and ablations under matched budgets.
4. Held-out evaluation and the 4–8-second demo.

Memory offloading can reduce inference VRAM but does not establish training fit.
Frozen original diffusion weights, trainable projectors/fusion/adapters, standard
denoising loss. Mood remains part of the complete planned system.

### October 11: 8 fps learned prior

The 32-frame prior is trained and saved under `outputs/audio_latent_prior_32f/`. The model remains a direct supervised latent predictor, not diffusion. Its provisional test MSE is 0.34860 (8.55% below the static template); correct music beats wrong-track conditioning on 97/98 clips and a one-second shift on 98/98. Decoded images remain blurry.

Review audio/video comparisons in `reports/model_comparison_review.html`. Training metrics are in `reports/audio_latent_prior_32f.json`; full decoded timing results are written by `scripts/evaluate_prior_decoded.py` to `reports/prior_32f_decoded_test.json`. Track-level control summaries are generated by `scripts/summarize_decoded_controls.py`. Human ratings and song-identity verification remain pending.

### Completed decoded-video evaluation

All 98 provisional test clips were decoded under four conditions (392 videos total). At 8 fps, frame-change event F1 is 0.6878 with correct music, 0.5079 with wrong-track input, 0.5295 with a one-second input shift, and 0.9007 for the matched encoded renderer target. Shifting the reference beat times by 200 ms lowers correct-model F1 to 0.2831. This supports visible event timing under the exploratory detector; it does not establish human preference, mood accuracy, or diffusion control.

Per-track mean advantage is +0.1799 over wrong tracks (14/14 tracks; bootstrap 95% interval [0.1217, 0.2433]) and +0.1583 over shifted inputs (13/14; interval [0.0914, 0.2309]). Clips within each track are grouped for these intervals. Source-song overlap remains unverified, and repeated test inspection makes a fresh final holdout necessary.

- Playable model comparison: `reports/model_comparison_review.html`.
- Randomized human review: `reports/learned_human_review.html` (56 clips; no fabricated ratings). All file references and frame counts verified, generated JavaScript syntax checked; browser interaction remains untested because local-file navigation was blocked.
- Full numerical evidence: `reports/prior_32f_decoded_test.json` and `reports/prior_32f_track_controls.json`.
- Reproduction details: `reports/prior_32f_reproducibility.json`.

Next experiment: preserve between-frame rhythm events in the input, select changes on validation data, then address sharpness and the diffusion conditioning path. The learned prior's images remain blurry; the sharper diffusion baseline still fails music-control checks.

## Beat-preserving sampling experiment (completed)

The preferred timing candidate is now `outputs/prior_32f_causal_peak/model.safetensors`. It retains onset/beat peaks between output frames, decayed by 0.78 per source frame to resemble the renderer pulse. Sampling is causal and does not cross clip boundaries. Default sampling and all prior checkpoints remain unchanged.

Both candidates started from the same existing 32-frame checkpoint, used the same learning rate, maximum epochs, seed and early-stopping rule, and selected their best checkpoint at epoch 30. All 77 validation clips were used for encoded-target scoring; test inputs were excluded. Validation MSE improved from 0.34769 for the matched point-sampling continuation to 0.34454.

The first clip from each of 11 validation tracks was selected before scoring for decoded evaluation. Beat-event F1 improved from 0.68459 to 0.83695; a one-second input shift reduces the new model to 0.47961. Matched encoded renderer targets score 0.91244. Nine tracks improve, one ties, one worsens; mean paired gain is 0.15236, with an exploratory track-bootstrap 95% interval [0.08308, 0.22086]. These are validation diagnostics, not new held-out test results.

Sharpness remains unresolved: mean Laplacian variance changes only from 18.67 to 19.21, versus 1665.18 for targets, and visual inspection still shows blurred predictions. This metric also responds to noise and is not a human quality score. Diffusion conditioning remains unproven.

Review: `reports/rhythm_sampling_review.html`; full comparison: `reports/rhythm_validation_comparison.json`; retention decision: `reports/rhythm_sampling_decision.json`.

Next checkpoint: improve spatial detail while retaining timing, using validation controls before integrating the new prior into diffusion. A longer temporal state and explicit appearance conditioning are candidate changes; neither has been tested yet.
