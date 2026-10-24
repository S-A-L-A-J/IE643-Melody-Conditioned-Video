# Bounded diffusion follow-up — 24 October 2026 IST

**Retain adapted paired refinement as an optional experimental visual polish step; reject frozen pretrained refinement as the default. Keep the GRU as the music-conditioned generator.** This is a modest synthetic-domain improvement, not a solved general music-to-video diffusion model.

## Controlled four-second validation

All outputs use 128px, 32 frames at 8fps. Diffusion uses strength .06, six actual DDIM steps, guidance 1 and fixed seed 643 + recording index. Paired variants average final outputs from +noise and -noise; the same settings/noise are matched across frozen/adapted models. Target videos are evaluation references, not sampling inputs.

| Validation population | Variant | Latent MSE | Image MSE | Image error reduction vs GRU | Estimated-beat F1 | Flow-warp residual |
|---|---|---:|---:|---:|---:|---:|
| selection8 | prior | 0.104950 | 0.012006 | +0.00% | 0.923 | 0.02069 |
| selection8 | frozen | 0.115626 | 0.013115 | -9.23% | 0.913 | 0.02638 |
| selection8 | adapted | 0.106189 | 0.012165 | -1.32% | 0.923 | 0.02192 |
| selection8 | frozen_pair | 0.101568 | 0.011996 | +0.09% | 0.923 | 0.01758 |
| selection8 | adapted_pair | 0.099122 | 0.011577 | +3.57% | 0.923 | 0.01711 |
| confirmation3 | prior | 0.093325 | 0.011199 | +0.00% | 1.000 | 0.01827 |
| confirmation3 | frozen_pair | 0.090107 | 0.011141 | +0.52% | 1.000 | 0.01506 |
| confirmation3 | candidate | 0.087940 | 0.010709 | +4.37% | 1.000 | 0.01498 |

The first eight recordings were used for checkpoint/candidate selection. The paired adapted candidate passed the declared ≥2% latent/image improvement and timing/sharpness gates. It improved both errors on 8/8 recordings. Settings were then frozen before evaluating the remaining three recordings; both errors improved on 3/3 there and beat F1 remained 1.000. All eleven remain validation, not an independent test. Frozen paired refinement improved decoded-image error by only 0.09% on the first eight and reduced sharpness, failing the declared gate. Single-sample adapted refinement also failed to beat the predictor on average.

Across all eleven recordings, adapted paired refinement reduced latent error by 5.61% and image error by 3.78%; estimated-beat F1 stayed 0.944. The combined figure includes selection data. No significance test or human preference study was performed.

Fixed frame inspections on recordings 0,3,7 and all three confirmation recordings showed modestly more defined ring edges without obvious new geometry defects. The composition remains similar and the target's small orbiting details are still missing. Mean pixel change from the predictor was 0.0166 on a [0,1] RGB scale (about 4.2/255); this is visible modest refinement, not a dramatic content improvement. Per-frame edge/flow proxies have confounds and do not replace author playback review.

## Runtime and memory

Median single diffusion pass: 7.76s; paired work approximately 15.52s per four-second clip. Peak sampling allocated/reserved memory: 165.8/216.0 MiB. Sequential CPU offload keeps GPU memory low. These measurements exclude model loading, GRU, VAE decode and video mux; PyTorch allocation is not total-device memory. A spot device query during sampling reported 197 MiB used, not a measured total-device peak.

## Longer output and controls

The preserved GRU checkpoint generates 8/16 seconds using rolling four-second context. The selected adapted paired method uses overlapping 32-frame denoiser windows per global DDIM step, avoiding unsupported long motion-attention inputs. All controls play the same correct reference audio.

| Duration / upstream audio | GRU estimated-beat F1 | Refined F1 | GRU warp | Refined warp | Mean RGB change | Paired sampling seconds |
|---|---:|---:|---:|---:|---:|---:|
| d08_correct_rolling | 0.903 | 0.903 | 0.02207 | 0.01734 | 0.0176 | 45.4 |
| d08_wrong_rolling | 0.370 | 0.370 | 0.01887 | 0.01518 | 0.0171 | 46.1 |
| d08_shift_rolling | 0.875 | 0.875 | 0.02102 | 0.01646 | 0.0175 | 48.1 |
| d16_correct_rolling | 0.968 | 0.968 | 0.02280 | 0.01732 | 0.0178 | 107.7 |

Long results cover one validation recording; no long target-error comparison was performed. Wrong audio replaces upstream music/context; shifted audio moves local features by one second with context fixed. Diffusion has no direct audio input. These controls show inherited conditioning, not conditioning learned by diffusion. A shift can coincide with another beat; timing scores alone cannot rank semantic correctness. The tested output resolution is 128px; 256px presentation panels are display enlargement.

## Limits and recommendation

Retain the paired adapted method for an explicitly labeled optional project demo, subject to Salaj's playback judgment. Keep predictor-only available because the visual gain is small and sampling doubles inference work. Reject the broad claim that frozen diffusion reliably improves these outputs. No further training or tuning is justified for this deadline. An arbitrary-audio CLI has not been tested here.

The spatial LoRA was trained on synthetic visual targets, with base diffusion, motion and text models frozen. It is not audio-conditioned diffusion training. Melody sensitivity, semantics and mood accuracy remain unestablished. Timing metrics use estimated beats; synthetic decoded-target MSE is not human quality. No human ratings were fabricated. No independent fresh evaluation was used, re-tuned or reclassified. Original weights, reports/slides, sources and all inventoried earlier artifacts are preserved; see PRESERVATION.json.

See REPRODUCE.md for exact commands/environment; REPORT_PARAGRAPH.md for reusable prose; SUMMARY.json and raw results for measurements. Review videos are in review_v2/videos. Earlier review/ and long_validation/videos/comparison_*.mp4 composites are interim incorrect-size builds, retained for provenance and excluded from deliverables; square individual outputs are unaffected.
