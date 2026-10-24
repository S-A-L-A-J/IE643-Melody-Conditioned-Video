"""Selected paired-noise refinement of supplied GRU latents; requires CUDA.

Input safetensors must contain key 'latents', shape [1,4,T,16,16].
No audio or target video is consumed by this stage. This portable wrapper has
static checks only; selected backend was exercised in saved validation runs.
"""
import argparse
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file, save_file
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'gpu_extension/resumed_20261024'))
from paired_backend import load_pipeline,refine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=643)
    p.add_argument('--strength',type=float,default=.06)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    clean=load_file(a.input)['latents']
    assert clean.ndim==5 and clean.shape[0:2]==(1,4) and clean.shape[-2:]==(16,16)
    assert torch.isfinite(clean).all()
    model=load_pipeline(lora_path=HERE/'gpu_extension/geometry_lora_continuation/lora_step128.safetensors')
    positive,_=refine(model,clean,a.strength,100,1.,a.seed,noise_sign=1.)
    negative,_=refine(model,clean,a.strength,100,1.,a.seed,noise_sign=-1.)
    save_file({'latents':((positive+negative)/2).contiguous()},a.output)
if __name__=='__main__':main()
