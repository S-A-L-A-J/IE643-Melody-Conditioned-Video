"""Download pinned pretrained dependencies for the demo."""
from pathlib import Path
import os
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parent/'project'
snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5',revision='451f4fe16113bff5a5d2269ed5ad43b0592e9a14',allow_patterns=['vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors'],local_dir=root/'models/diffusion/sd15')
snapshot_download('laion/clap-htsat-unfused',revision='8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a',allow_patterns=['*.json','*.txt','*.bin','*.safetensors'],local_dir=root/'models/clap')
os.environ['TORCH_HOME']=str(root/'models/torch')
from demucs.pretrained import get_model
get_model('htdemucs')
print('Demo dependencies downloaded. Full AnimateDiff experiment weights require project/scripts/download_diffusion_models.py.')
