# IE643 implementation progress

Updated 2026-10-11. All numbers below are from local runs on the RTX 3050 4 GB Laptop GPU unless stated otherwise.

## Completed and verified

- 132 Bollywood-labelled excerpts imported; 125 eligible after one exact duplicate and six very short/empty excerpts were excluded. Originals retained.
- All 125 eligible excerpts separated into instrumental and vocal stems with Demucs; structural alignment and file checks passed. User approved representative separation quality.
- Rhythm, harmonic pitch-class, dominant spectral pitch proxy and dynamics extracted for all 125 at 16 feature samples/second. Frozen CLAP context is now available; mood labels remain unvalidated. The dominant pitch is not validated lead-melody transcription.
- 875 four-second E0 synthetic audio-video pairs rendered at 256x256, 16 fps. Pair manifest links source audio offsets, feature frame offsets, renderer parameters and targets; every pair passed shape/alignment checks. This original renderer has one style; the expanded dataset below contains three styles.
- AnimateDiff + SD 1.5 checkpoints downloaded and tiny inference completed using CPU offloading. This was not music-conditioned generation.
- Real denoising adapter optimizer steps succeeded at 128x128x4 frames, 192x192x4 frames and 256x256x4/8/16 frames. The 256x256x16 test used CPU activation offloading and reserved ~3.4 GB GPU memory. 256x256x4 without offloading ran out of GPU memory.
- Trainable components: local rhythm/melody/dynamics projectors and rank-2 LoRA in temporal and cross-attention layers. Original video model weights remain frozen. A probe verified finite gradients and an actual weight update.

## Latest tiny adaptation checkpoint

- E2-style tiny training on eight different source tracks completed 128 optimizer steps. The model, projectors and optimizer/scaler state are saved in `outputs/tiny_e2/`, and video latents are cached for reproducible continuation.
- On one tracked clip with fixed image noise and timestep 500, denoising loss fell from 0.09546 to 0.07142. Swapping in a different song produced a slightly *lower* loss (0.07137).
- The paired swap test over all eight training clips favored the correct music for only 3/8, with mean advantage (wrong minus correct loss) -0.0000569. Under identical generation noise the two four-second previews differed by only 2.75/255 pixel values on average. Both looked nearly identical. The current adapter therefore has **not** learned reliable music control; lower denoising loss alone was not sufficient.
- Preliminary generated clips with their corresponding instrumentals are `outputs/tiny_e2_correct_with_audio.mp4` and `outputs/tiny_e2_mismatched_with_audio.mp4`. The underlying silent outputs, generation settings, and swap test are in `outputs/` and `reports/`.
- The training uses 16 frames sampled across four seconds at 4 fps. It fits in about 3.4 GB of reserved GPU memory with CPU activation offloading. This is a tiny training-set diagnostic, not a held-out result or a final-quality video generator.

## Expanded data and provisional evaluation

- A frozen CLAP whole-excerpt audio-context embedding (512 values) has been extracted for all 125 eligible instrumentals. Six zero-shot mood similarities are stored as exploratory proxies; the top label is heavily skewed toward “energetic” and is **not** a validated emotion label.
- Two additional persistent music-reactive target styles, rings and bars, were rendered and visually checked. The expanded manifest contains 2,625 verified pairs across geometric, rings and bars styles. See `reports/synthetic_styles_review.html`.
- Feature-fingerprint review found two strong audio-overlap candidates (scores above 0.99), which are grouped together. A provisional recording-grouped split contains 100 train, 11 validation and 14 test tracks; all eight tracks exposed in the first diffusion debug run remain in train. Most files are compilations or mashups, so this is **not** a verified song-disjoint benchmark.
- On the 14 provisional test tracks, programmed renderer beat-event F1 is 0.812 geometric, 0.866 rings and 0.801 bars; 200 ms audio shifts fall to 0.275, 0.151 and 0.242 respectively. These are renderer timing checks, not diffusion-model scores. Precision, event coverage, frame-change diagnostics and a human-rating template are in `reports/provisional_renderer_benchmark.json` and `reports/provisional_human_review_template.csv`.
- The second 32-step normalized/ranking adapter still failed music control (correct music won 4/8 train and 4/8 provisional test swaps). A third 64-step spatial-injection adapter also failed (3/8 provisional test swaps).
- A supervised audio-to-video-latent prior trained on 700 ring clips, selected on 77 validation clips, and scored on 98 provisional test clips. It beat another track on 98/98 test targets and a one-second shift on 97/98; encoded-target MSE was 0.350 versus 0.378 for a static template. Its decoded images are blurry and only 2 fps, so this is a **working learned control signal**, not a final video generator or verified visible beat synchronization. Removing the global CLAP context gave similar results, so its added value is unproven.
- A 128-step diffusion adapter guided by the frozen learned prior made much sharper ring-like images at 128×128/2 fps, but failed the held-out audio-control check: correct music won 4/14 provisional test track swaps and 1/4 clearer single-title swaps. On one held-out preview, visible beat-event coverage was zero. The diffusion output is visually improved, but music alignment is not established.
- The 8 fps learned prior has completed training (best validation checkpoint at epoch 43). On 98 provisional test clips it beats wrong-track conditioning on 97/98 and one-second shifts on 98/98; encoded-target MSE is 0.34860 versus 0.38120 for a static template (8.55% lower). Decoded-video evaluation is complete across all 98 clips: beat-event F1 0.6878 for correct music, 0.5079 for another track, 0.5295 for one-second-shifted input, and 0.9007 for the encoded synthetic target at matched 8 fps. Correct music wins per-track mean F1 on 14/14 tracks versus wrong music and 13/14 versus shifted input. Images remain blurry. Human preference and mood judgments still need participants.

Key review files: `reports/provisional_results.png`, `reports/audio_prior_review.html`,
`reports/audio_latent_prior.json`, `reports/prior_guided_diffusion_test.json`, and
`reports/provisional_human_review_template.csv`.

## Remaining checkpoints

1. Establish **visible** audio alignment in learned video: more frames and a trained diffusion-conditioning path must pass song-swap, within-song shift and beat-event controls. The supervised prior passes encoded-target controls, while all tested diffusion adapters fail.
2. Verify song identities, medley overlap and rights before claiming a song-disjoint benchmark; validate a global mood representation rather than treating CLAP zero-shot labels as ground truth.
3. Compare E0 renderer, E1 pooled embedding, E2 local streams, E3 with validated mood and incremental ablations under matched budgets.
4. Complete held-out continuity, mood and human ratings. Programmed renderer beat coverage and provisional model controls are measured; subjective ratings remain pending.
5. Build the upload, feature inspection, diffusion generation, preview, metrics and download interface for 4-8-second clips.

The current 16-frame local training window samples the four-second target at 4 fps. Full 16-fps training across all 64 frames has not been shown to fit locally; limited L4/A100 time may be needed.

## October 11 continuation

- Completed all 875 cached ring targets at 32 frames across four seconds (8 fps), trained the prior, and saved `outputs/audio_latent_prior_32f/model.safetensors`.
- Packaged seven audio/video comparisons in `reports/model_comparison_review.html`, including the 8 fps prior and existing 2 fps diffusion. All seven files have verified video/audio streams and four-second duration. Browser security policy blocked interactive opening of the local page; playback interaction is not browser-tested.
- Fixed human-review CSV export escaping. Generated JavaScript passes the syntax check and its newline expression produces actual CSV rows.
- Sampling audit: 8 fps exact-frame sampling retains 2,826/5,632 train beat impulses, 284/595 validation impulses, and 384/815 test impulses. Test an event-preserving input representation on validation data in a future experiment.

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
