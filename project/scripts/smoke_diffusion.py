import os
"""Small unconditioned diffusion inference check; NOT a trained music-to-video result."""
import argparse
import json
import time
from pathlib import Path
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--steps',type=int,default=2)
    args=p.parse_args()
    torch.set_num_threads(4)
    start=time.monotonic()
    adapter=MotionAdapter.from_pretrained(ROOT/'models/diffusion/animatediff',
                                         torch_dtype=torch.float16,variant='fp16',local_files_only=True)
    pipe=AnimateDiffPipeline.from_pretrained(ROOT/'models/diffusion/sd15',motion_adapter=adapter,
                                            torch_dtype=torch.float16,variant='fp16',local_files_only=True)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False,
                                             timestep_spacing='linspace',steps_offset=1)
    pipe.enable_vae_slicing()
    pipe.unet.enable_forward_chunking(chunk_size=1,dim=1)
    pipe.enable_sequential_cpu_offload()
    torch.cuda.reset_peak_memory_stats()
    frames=pipe(prompt='abstract colorful geometric shapes floating on a dark background',
                width=256,height=256,num_frames=args.frames,num_inference_steps=args.steps,
                guidance_scale=1.0,generator=torch.Generator('cpu').manual_seed(643)).frames[0]
    torch.cuda.synchronize()
    export_to_gif(frames,ROOT/'outputs/diffusion_smoke.gif')
    report=dict(status='passed',frames=args.frames,steps=args.steps,resolution=[256,256],
                peak_vram_mb=round(torch.cuda.max_memory_allocated()/2**20,1),
                seconds=round(time.monotonic()-start,2),
                offload='sequential_cpu',music_conditioned=False,training_tested=False,
                note='Plumbing check only. Not evidence of final video quality or adapter-training feasibility.')
    (ROOT/'reports/diffusion_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
