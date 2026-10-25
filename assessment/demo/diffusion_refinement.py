"""Selected appearance-adapted AnimateDiff refinement for uploaded audio latents."""
from pathlib import Path
import gc
import hashlib
import importlib.util
import time
import torch


def refine_latents(latents, pretrained_root, progress=None, seed=643):
    if not torch.cuda.is_available():
        raise RuntimeError('The final diffusion hybrid requires CUDA. Choose GRU only for CPU generation.')
    folder=Path(__file__).resolve().parents[1]/'experiments/diffusion_refinement/gpu_extension'
    weights=folder/'geometry_lora_continuation/lora_step128.safetensors'
    backend_file=folder/'resumed_20261024/paired_backend.py'
    if not weights.is_file():
        raise RuntimeError('The selected trained diffusion adapter is missing.')
    spec=importlib.util.spec_from_file_location('ie643_paired_diffusion_backend',backend_file)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    # All entry points use the same models directory, including relocated checkouts.
    backend.ROOT=Path(pretrained_root).resolve().parent
    started=time.monotonic();pipe=None
    try:
        if progress:progress('Loading adapted Stable Diffusion and AnimateDiff')
        pipe=backend.load_pipeline(offload=False,lora_path=weights)
        # Move complete components rather than each individual layer; this
        # avoids repeated CPU/GPU transfers on the live demo's small latents.
        pipe.enable_model_cpu_offload(gpu_id=0)
        if progress:progress('Diffusion refinement: positive-noise pass (1/2)')
        positive,positive_info=backend.refine(pipe,latents,strength=.06,base_steps=100,guidance=1.,seed=seed,noise_sign=1.)
        if progress:progress('Diffusion refinement: opposite-noise pass (2/2)')
        negative,negative_info=backend.refine(pipe,latents,strength=.06,base_steps=100,guidance=1.,seed=seed,noise_sign=-1.)
        refined=(positive+negative)/2
        assert refined.shape==latents.shape and torch.isfinite(refined).all()
        return refined.contiguous(),{
            'method':'appearance-adapted AnimateDiff; mean of opposite-noise latent predictions',
            'adapter_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'strength':.06,'actual_steps_per_pass':positive_info['steps'],'passes':2,
            'seed':seed,'audio_enters_diffusion_directly':False,
            'duration_seconds_including_model_load':round(time.monotonic()-started,2),
            'latent_mse_change_from_gru':float((refined-latents).square().mean()),
            'positive_pass':positive_info,'negative_pass':negative_info,
        }
    finally:
        if pipe is not None:
            del pipe
        gc.collect();torch.cuda.empty_cache()
