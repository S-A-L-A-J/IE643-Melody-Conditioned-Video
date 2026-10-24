# Bounded prior-control pilot — completed

Reviewed 25 October 2026 from saved evidence only; no new inference or training.

## Status and protocol

All planned cells completed: two GRU priors (correct and wrong audio) × strengths 0.06, 0.30, 0.70 × opposite noise signs. All 12 signed latent results, six paired results, two baseline videos, six refined videos, and two comparison videos are saved. `run.log` ends COMPLETE; saved runtime is 592.45 seconds. No missing cells.

The same seed 643 and same unsigned Gaussian noise were used throughout (one unique saved unsigned-noise hash). Each final output averages two denoising passes initialized with opposite noise. Saved geometry appearance LoRA, prompt, guidance=1, base DDIM steps=100, and 32 frames were held fixed. The first four seconds of a previously inspected validation example were used; this was not a fresh test.

## Existing numerical findings

| Strength | Actual denoising steps per signed pass | Correct departure, latent MSE | Wrong departure, latent MSE | Between-output latent separation / input-prior separation | Correct beat F1 | Wrong beat F1 |
|---|---:|---:|---:|---:|---:|---:|
| Prior only | 0 | 0 | 0 | 1.000 | 0.800 | 0.308 |
| 0.06 | 6 | 0.00619 | 0.00628 | 0.936 | 0.800 | 0.308 |
| 0.30 | 30 | 0.01890 | 0.01893 | 0.794 | 0.800 | 0.308 |
| 0.70 | 70 | 0.28377 | 0.27434 | 0.098 | 0.625 | 0.462 |

Input correct-versus-wrong latent MSE was 0.26695. The ratio is a squared-distance ratio, not a probability or percentage of audio information retained. RGB separation MAE likewise fell from baseline 0.07393 to 0.07443, 0.06944, and 0.01922 at increasing strengths.

## Visual inspection

`comparison.png` shows both priors as distinct colored concentric rings. At 0.06 their appearance remains close to the respective priors. At 0.30 rings remain recognizable and the two inputs remain visually distinct, with more correction. At 0.70 both paired outputs collapse to blurred colored patterns and largely lose ring structure. Thus larger change was not useful improvement in this pilot.

## Interpretation and limits

This supports a practical preservation-versus-denoising-freedom tradeoff: low/medium strengths preserve prior differences, while high strength makes the two outputs less separated and damages structure. It does not establish that diffusion learned audio dependence: audio reaches the frozen sampler through GRU initialization, not direct audio conditioning. It does not validate a new trained conditioning branch.

Strength also changes denoising depth (6/30/70 steps); it is not an isolated noise-only causal comparison. There is only one inspected excerpt, two prior inputs, one seed, and fixed paired-noise averaging. A high-strength collapse may reflect this particular model, domain, sampler, and averaging, not all diffusion methods. Wrong-audio videos and beat scores use the correct reference soundtrack. Estimated-beat F1 is a limited timing proxy; appearance/sharpness diagnostics are not human quality or semantic measures. No target-quality gain, human preference, generalization, or final-test claim is warranted. Do not select a new production setting from this pilot.

## Presentation-ready wording

“We swept refinement strength on two distinct audio-predicted priors using identical paired noise. Low and medium strengths preserved their structural differences and measured timing. High strength substantially increased departure but collapsed the rings and reduced separation between outputs. This illustrates why our final system restricts diffusion freedom; stronger learned conditioning would be needed before granting it more generative responsibility.”

## Artifacts

- `results.json`: exact diagnostics and limitations.
- `comparison.png`: eight-row contact sheet (priors, then correct/wrong at each strength).
- `videos/correct_strength_comparison.mp4`: correct prior versus three strengths.
- `videos/wrong_strength_comparison.mp4`: wrong prior versus three strengths.
- `PLAN.json`, `README.md`, `run.py`, `run.log`, `latents/`: protocol and complete raw evidence.
