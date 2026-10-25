# Code reading guide

Read one component at a time. Earlier experiment scripts need not be read before the selected model.

1. `assessment/experiments/train_temporal_prior.py`: read only `TemporalPrior.__init__` and `forward()` first. Inputs are music `[B,T,22]`, CLAP context `[B,512]`, and phase `[B,2]`. Output is `[B,4,T,16,16]`. Then read `main()` for the three losses and optimizer, and `evaluate()` for swaps and shifts.
2. `assessment/demo/inference.py`: follow `generate()`, then its helpers `conditioning()`, `context_embedding()` and `separate()`. This orchestrates the upload path, including the final hybrid and GRU-only comparisons. Read `assessment/demo/diffusion_refinement.py` for the connection to the selected trained diffusion backend.
3. `project/scripts/extract_features.py`: `extract()` builds the temporal musical features.
4. `assessment/experiments/diffusion_refinement/refine_latents.py`: selected paired-noise inference wrapper. Follow its import to `gpu_extension/resumed_20261024/paired_backend.py`, especially `refine()`. Wrapper portability is static-checked only.
5. `assessment/experiments/diffusion_refinement/gpu_extension/continue_geometry_lora.py`: appearance-adapter training. Read alongside `train_geometry_lora.py` for setup.
6. `assessment/demo/metrics.py`: estimated-beat event matching. Detailed validation evidence remains in the experiment directories.
7. `assessment/demo/app.py`: interface widgets and callback wiring; no model training.

For earlier architectures, see `project/scripts/train_tiny.py` plus `probe_adapter_step.py`, then `train_prior_guided_diffusion.py` and its inference/evaluation counterparts. These are historical experiments, not the selected upload path.

The detailed reproducibility notes are retained separately so the main README stays concise.
