# Handoff — bounded resumed experiment, Team Nullspace

Owner: Salaj Sonar. Deadline: 25 October 2026, 11:59 PM IST.

The user explicitly resumed work after the historical stop. This follow-up completes generated-output testing of the existing 128-update appearance adapter, adds a predeclared paired-noise inference test, confirms settings on the remaining validation groups, and tests 8/16-second refinement. No new training occurred during this resumption. Historical STOPPED.json and earlier handoffs remain unchanged as provenance; this directory is the current evidence.

**Recommendation: retain paired adapted refinement as an optional, modest visual polish step; keep the music-conditioned GRU as the primary generator.** Frozen diffusion and single-sample adapted refinement failed the useful-improvement checks. The selected candidate averages final latent predictions from +noise/-noise at strength .06, six DDIM steps, fixed text/guidance 1. It uses the existing spatial LoRA step128; no target is used to generate the output.

The first eight validation recordings show 5.6% lower target-latent MSE and 3.6% lower image MSE, with 8/8 improving. After freezing the candidate, all three remaining validation recordings improve (4.4% image-error reduction). Estimated-beat timing is unchanged. Combined eleven-recording image-error reduction is 3.78%, but includes selection data. No independent final-test claim is warranted. Details, failed controls and compact tables are in RESULTS.md and SUMMARY.json.

The visual difference is modestly cleaner ring edges. Missing target details remain. No semantic/mood accuracy or human preference is established. Estimated-beat F1 does not measure these. Music controls enter through the GRU; diffusion has no direct audio encoder. Appearance-adapter training is not audio-conditioned diffusion training.

Deliverables:

- `review_v2/videos/comparison_validation_0.mp4` (also 3 and 7): target, GRU, frozen paired refinement and adapted paired refinement, same audio.
- `review_v2/videos/comparison_08s_v2.mp4`: GRU, selected refinement, wrong- and shifted-audio controls through GRU, same correct reference audio.
- `review_v2/videos/comparison_16s_v2.mp4`: 16-second GRU/refinement comparison.
- `long_validation/videos/`: individual square 8/16-second clips and raw RGB arrays.
- `REPRODUCE.md`: commands, exact settings, environment, scheduler/text compatibility references and reuse rules.
- `REPORT_PARAGRAPH.md`: short report-ready method/results paragraph.
- `MEDIA_VERIFICATION.json`, `PRESERVATION.json`: media checks and original-file hash checks.
- `CHOSEN_CANDIDATE.json`, `VALIDATION_PLAN.json`, `PAIRED_PLAN.json`: choice chronology and predeclared checks.
- `ARTIFACT_MANIFEST.json`: hashes of this follow-up's final artifacts.

Interim `review/` and `long_validation/videos/comparison_*.mp4` contain an incorrect-size rectangular composite build and are retained only for provenance; use review_v2 deliverables. Individual square media are unaffected. See MEDIA_BUILD_NOTE.md. No fabricated ratings or discarded failed experiments.

Long generation is an inference extension, not a newly trained long model: rolling four-second GRU context and overlap-add diffusion predictions at each global step. Only one validation recording has long-duration diagnostics. Both methods were actually run at 8/16 seconds; output remains native128px. The generic `../long_inference.py` arbitrary-audio CLI remains untested and guarded. Do not present it as deployed or tested.

All 19 inventoried inputs, 76 original artifacts and 376 earlier GPU artifacts match their saved hashes. Existing checkpoints/evaluations/reports/slides and sources remain unchanged. Fresh evaluation was not used or reclassified. Everything stayed local. No remaining training or GPU jobs should be running after finalization. Further work should prioritize author playback and reporting, rather than extending the training campaign before the deadline.
