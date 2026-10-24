# Nullspace music-to-video demo

The current interface offers the new temporal model and the preserved beat-aware baseline, both decoded by the pretrained SD 1.5 VAE. Neither is diffusion inference. The temporal model produces clearer rings on inspected validation examples. The baseline remains available for comparison.

## Start locally

```bash
source /home/salaj/IE643_project/activate.sh
python /home/salaj/.codex/.chatgpt-projects/g-p-6aae9e9b275881919a205c4774702722/assessment/demo/app.py
```

Open http://127.0.0.1:7863. Upload an audio file, select a start time with at least four seconds remaining, and choose whether to suppress singing. Click Generate video. Download the video and result details. Processing uses at most a 30-second context window around the chosen segment.

The system analyses onset, beats, tempo, chroma, spectral pitch proxy, RMS and four frequency bands at 16 Hz. It preserves intervening beat/onset peaks when selecting 32 output frames at 8 fps. The selected audio segment and feature grid share the same offset. A frozen CLAP encoder averages normalized 10-second chunk embeddings. Both models use 22 local features plus 512 context values. The temporal model additionally receives normalized clip time and sine/cosine of the visual arrangement angle. It uses a 128-unit GRU followed by a convolutional spatial decoder to predict latents. The frozen VAE decodes them. Visual arrangement is an explicit style control, not a ground-truth feature inferred from audio.

Outputs are isolated under `../demo_outputs/<run-id>/`, including extracted features, input context, video, plot and JSON. Existing training data/checkpoints are not modified. The JSON records the checkpoint hash, effective start time, runtime and limitations.

## Metric

Beat-event F1 combines one-to-one matched precision and coverage within 125 ms. Visual events are peaks in consecutive grayscale frame differences. This is the existing project's exploratory metric, not mood accuracy, perceptual quality, or an independent benchmark. No detected audio beats produces an unavailable metric. Shifted-reference score is included in the JSON as a diagnostic.

## Dependencies and current limits

Uses the original project's environment and locally downloaded Demucs, CLAP, SD 1.5 VAE and causal-peak checkpoint. FFmpeg must be installed. CPU fallback exists but has not been timed. GPU runs passed on RTX 3050 4 GB. Output is 128x128, 8 fps, four seconds. Source project paths can be overridden with `IE643_PROJECT_ROOT`; full portable packaging is pending.
