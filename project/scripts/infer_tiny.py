import os
"""Generate preliminary four-second videos with the trained music adapters.

Uses 16 frames across four seconds (4 fps) to fit the local GPU. It is a
research diagnostic, not a quality or generalization claim.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from peft import LoraConfig
from safetensors.torch import load_file

from pair_dataset import PairedClipDataset
from probe_adapter_step import LocalAudioProjector

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--seed',type=int,default=643)
    parser.add_argument('--target-index',type=int,default=0)
    parser.add_argument('--other-index',type=int,default=1)
    args=parser.parse_args()
    start=time.monotonic()
    torch.set_num_threads(4)
    checkpoint=ROOT/'outputs/tiny_e2'
    run=json.loads((checkpoint/'run.json').read_text())
    config=run['configuration']
    dataset=PairedClipDataset()
    # Match the two first distinct tracks used by training.
    chosen=[];seen=set()
    for i,row in enumerate(dataset.rows):
        if row['track_id'] not in seen:
            chosen.append(i);seen.add(row['track_id'])
        if len(chosen)>=max(args.target_index,args.other_index)+1:
            break
    target=dataset[chosen[args.target_index]]
    other=dataset[chosen[args.other_index]]
    indices=torch.arange(config['frames'])*config['stride']

    adapter=MotionAdapter.from_pretrained(ROOT/'models/diffusion/animatediff',variant='fp16',
                                          torch_dtype=torch.float16,local_files_only=True)
    pipe=AnimateDiffPipeline.from_pretrained(ROOT/'models/diffusion/sd15',motion_adapter=adapter,
                                             torch_dtype=torch.float16,variant='fp16',local_files_only=True)
    with torch.no_grad():
        text_model=pipe.text_encoder.float().cpu().eval()
        inputs=pipe.tokenizer('abstract colorful geometric shapes on a dark background',
                              padding='max_length',max_length=pipe.tokenizer.model_max_length,
                              truncation=True,return_tensors='pt')
        text=text_model(inputs.input_ids)[0].mean(dim=1,keepdim=True).to('cuda',dtype=torch.float16)
        del text_model
    unet=pipe.unet
    vae=pipe.vae.float().cpu().eval()
    del pipe,adapter
    targets=[name for name,module in unet.named_modules() if isinstance(module,torch.nn.Linear)
             and name.endswith(('to_q','to_v')) and ('motion_modules' in name or 'attn2' in name)]
    unet.add_adapter(LoraConfig(r=config['rank'],lora_alpha=config['rank'],target_modules=targets))
    unet.load_state_dict(load_file(checkpoint/'lora.safetensors'),strict=False)
    unet.to('cuda',dtype=torch.float16).eval()
    for name,module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    projector=LocalAudioProjector().to('cuda').eval()
    projector.load_state_dict(load_file(checkpoint/'projector.safetensors'))

    def condition(item):
        with torch.no_grad():
            audio=projector(item['rhythm'][indices].to('cuda'),item['melody'][indices].to('cuda'),
                            item['dynamics'][indices].to('cuda')).to(dtype=torch.float16)
            return torch.cat([text.expand(config['frames'],-1,-1),audio],dim=1)

    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon',clip_sample=False,
                            timestep_spacing='linspace',steps_offset=1)
    scheduler.set_timesteps(args.steps,device='cuda')
    latents0=torch.randn((1,4,config['frames'],config['size']//8,config['size']//8),
                         generator=torch.Generator('cpu').manual_seed(args.seed),dtype=torch.float16).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    results=[]
    for label,item in [('correct',target),('mismatched',other)]:
        current=latents0.clone()
        conditioning=condition(item)
        with torch.inference_mode():
            for timestep in scheduler.timesteps:
                predicted=unet(current,timestep,encoder_hidden_states=conditioning).sample
                current=scheduler.step(predicted,timestep,current).prev_sample
        frames=[]
        with torch.inference_mode():
            for chunk in current[0].permute(1,0,2,3).cpu().float().split(4):
                decoded=vae.decode(chunk/vae.config.scaling_factor).sample
                rgb=((decoded/2+0.5).clamp(0,1)*255).byte().permute(0,2,3,1).numpy()
                frames.extend(rgb)
        output=ROOT/'outputs'/f'tiny_e2_{label}.mp4'
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),4,(config['size'],config['size']))
        if not writer.isOpened():
            raise RuntimeError('Failed to open video writer')
        for frame in frames:
            writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        writer.release()
        results.append(dict(condition=label,video=str(output.relative_to(ROOT)),
                            latent_mean=float(current.float().mean()),
                            latent_std=float(current.float().std()),frames=len(frames)))
        print(label,output,flush=True)
    report=dict(status='completed',training_step=run['step'],target_pair=target['id'],
                other_pair=other['id'],denoising_steps=args.steps,frames=config['frames'],
                fps=4,resolution=[config['size'],config['size']],seed=args.seed,
                peak_vram_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20,1),
                seconds=round(time.monotonic()-start,1),results=results,
                note='Tiny synthetic-data experiment; compare videos under identical seed to inspect audio control.')
    (ROOT/'reports/tiny_e2_generation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
