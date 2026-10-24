# Prior dependence versus refinement strength pilot

24 October 2026. Explicitly authorized inference-only extension; no training or change to the selected main method.

Two distinct GRU prior sequences (correct and wrong audio, first four seconds of the existing long-validation example) are refined with the saved appearance LoRA. The only swept control is noising strength: .06, .30, .70 using 100 base DDIM steps, giving 6, 30, and 70 actual denoising steps. All runs use the same seed 643, fixed prompt, guidance 1, and the same paired opposite Gaussian noise. Pairing is held fixed to match the selected final approach.

## Reproduction

From the project workspace, with local GPU access:

```
/home/salaj/IE643_project/.venv/bin/python assessment/experiments/prior_dependence_pilot/run.py
```

The sampler imports the preserved backend from the earlier experiment. It loads only local weights. Checkpoint and source evidence are not modified. The saved plan precedes all new results. Do not rerun into a completed directory: JSON creation is exclusive to preserve evidence.

## Interpretation

- Departure from each input prior is measured by latent MSE and normalized RGB MAE.
- Separation between outputs initialized from the two priors is compared with their initial separation.
- Estimated-beat F1 and optical-flow/edge diagnostics provide limited timing/appearance proxies.
- Every video uses the same correct reference soundtrack; the wrong-audio prior is intentionally evaluated against that reference.
- Stronger noise also entails more denoising steps under this standard schedule: the pilot tests the practical noise-strength setting, not noise amount isolated from sampling depth.
- Increasing output separation is not automatically better adherence or quality. It can reflect distorted trajectories.
- This is one previously inspected validation example and two prior inputs, not independent test evidence.
- No direct audio input reaches diffusion. Dependence is inherited through initialization; this pilot cannot prove that diffusion learned musical conditioning.
- No new human ratings, target-quality improvement claim, or setting selection is warranted.
