# Reproduce the bounded pilot

Run from `/home/salaj/.codex/.chatgpt-projects/g-p-6aae9e9b275881919a205c4774702722` using the existing environment. No installation or download is required. Every pretrained load is local-only; Hugging Face offline mode is set inside the script. Existing pretrained weights and the GRU checkpoint are read-only inputs.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/salaj/IE643_project/.venv/bin/python assessment/experiments/diffusion_refinement/run_pilot.py prepare
PYTHONDONTWRITEBYTECODE=1 /home/salaj/IE643_project/.venv/bin/python assessment/experiments/diffusion_refinement/run_pilot.py refine --max-cases 1
PYTHONDONTWRITEBYTECODE=1 /home/salaj/IE643_project/.venv/bin/python assessment/experiments/diffusion_refinement/run_pilot.py evaluate
```

`prepare` is intentionally single-use and fails if its outputs already exist. `refine` skips existing latent files and `--max-cases` bounds the number of additional cases. Without `--max-cases` it attempts the fixed nine-case matrix (three strengths × three predictor inputs), with a three-hour per-invocation ceiling. It does not select settings automatically. `evaluate` decodes all available latents, retains RGB frame arrays, evaluates one-to-one beat timing and descriptive motion measures, and creates four-panel videos when all conditions at a strength exist. Existing media are skipped; results use a new filename based on available case count.

The executed bounded sequence is: prepare; refine `--max-cases 1`; evaluate; refine `--only-correct --max-cases 2`; refine `--max-cases 2` to add only the two 0.10 audio controls; evaluate. Add the precision-only control before the last evaluation. The unexecuted stronger audio-control cells are not results. `--only-correct` lets the scheduled strength sweep run without expanding the control matrix. No second validation recording is evaluated as a target clip.

To reproduce from scratch without overwriting this experiment, copy `run_pilot.py` into a new sibling directory under `assessment/experiments/` and run its stages there. The script resolves the preserved GRU checkpoint and helper modules relative to its parent and the original project at its fixed location. Keep all old outputs for audit. Original failed loader evidence is in `refine_first.log`; the current script uses the verified streaming loader described in `API_AUDIT.md`.

CPU execution uses bfloat16 diffusion and float32 predictor/VAE decoding. This host lacks working CUDA. A future CUDA rerun would use float16 and is a distinct numerical environment; it must not be merged into the CPU result table as an exact replication. Identical seed does not promise identical outputs across devices or library versions.

For CPU speed, the current script stores denoiser weights/activations in bfloat16 but uses float32 arithmetic inside its Linear/Conv2d layers, casting outputs back. See `API_AUDIT.md`. `package_versions.json` records dependencies, `hardware.txt` records this host, and `environment.json` describes the first successful refined run, rather than the total experiment duration. Per-case JSON files contain denoising runtimes and cumulative process peak memory; VAE decoding, loading and rendering are excluded from these case runtimes.

The additional precision-only control is reproducible with:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/salaj/IE643_project/.venv/bin/python - <<'PY'
from pathlib import Path
from safetensors.torch import load_file, save_file
p = Path('assessment/experiments/diffusion_refinement')
z = load_file(p/'predictor_latents.safetensors')['correct']
out = p/'refined_latents/correct_s0.00_precision.safetensors'
assert not out.exists(), 'Preserve the existing control; use a new reproduction folder.'
save_file({'latents': z.bfloat16().float().contiguous()}, out)
PY
```

Run that snippet only in a new reproduction folder before evaluation. The saved control metadata describes zero denoising steps; it is not a zero-strength diffusion run. It measures predictor-latent rounding alone, not denoiser-weight rounding.

The original audio for the selected validation clip is muxed into every video, including wrong/shifted predictor controls. Controls deliberately sound identical to the reference in comparisons; their changed conditioning is upstream, in GRU feature/context inputs. The shifted control rolls only the local feature sequence by one second; its global context and visual phase stay fixed. All comparisons use the same prompt, seed, frame rate, resolution, scheduler, noise tensor, and VAE.

Interpret metrics with the caveats in `API_AUDIT.md` and the results JSON. Frame-change event detection is inherited from the original evaluation code, while beat-event matching uses the newer one-to-one implementation. Timing uses estimated beats and cannot establish semantics, musical mood, or human preference. No human ratings are generated.
