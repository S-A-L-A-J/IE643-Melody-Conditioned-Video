import os
"""Decode held-out latent-prior predictions for visual and audio-swap review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file

from train_audio_latent_prior import AudioLatentPrior,load_data

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--stride',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(4)
    data=load_data(args.frames,args.stride)['test']
    preferred='46319c3a949c0b23'  # Clear single-title source: Koi Aur Nahi.
    index=next(i for i,(track,pair) in enumerate(zip(data['tracks'],data['ids']))
               if track==preferred and pair.endswith('_0.000__rings'))
    other=next(i for i,track in enumerate(data['tracks']) if track!=preferred)
    model=AudioLatentPrior().to('cuda').eval()
    model_dir=ROOT/'outputs'/('audio_latent_prior' if args.frames==8 and args.stride==8
                              else f'audio_latent_prior_{args.frames}f')
    model.load_state_dict(load_file(model_dir/'model.safetensors'))
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15',subfolder='vae',
                                     variant='fp16',torch_dtype=torch.float32,
                                     local_files_only=True).to('cuda').eval()
    vae.requires_grad_(False)
    music=data['music'].to('cuda')
    context=data['context'].to('cuda')
    target=data['target'][index:index+1].to('cuda')
    with torch.inference_mode():
        latents={
            'correct':model(music[index:index+1],context[index:index+1]),
            'other_song':model(music[other:other+1],context[other:other+1]),
            'shifted_one_second':model(music[index:index+1].roll(args.frames//4,1),context[index:index+1]),
            'target':target,
        }
        results={}
        for label,latent in latents.items():
            frames=[]
            for chunk in latent[0].permute(1,0,2,3).split(4):
                decoded=vae.decode(chunk/vae.config.scaling_factor).sample
                rgb=((decoded/2+0.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
                frames.extend(rgb)
            prefix='audio_prior' if args.frames==8 else f'audio_prior_{args.frames}f'
            output=ROOT/'outputs'/f'{prefix}_{label}.mp4'
            writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),args.frames/4,(128,128))
            if not writer.isOpened():
                raise RuntimeError(f'Cannot write {output}')
            for frame in frames:
                writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            writer.release()
            results[label]={'video':str(output.relative_to(ROOT)),
                            'latent_mse_to_target':float((latent-target).square().mean())}
        report={'status':'completed','pair_id':data['ids'][index],
                'other_pair_id':data['ids'][other],
                'split':'provisional_test','fps':args.frames/4,'frames':args.frames,'resolution':[128,128],
                'outputs':results,
                'note':'These are direct decoded predictions of the learned prior, not diffusion generations.'}
    report_name='audio_prior_preview.json' if args.frames==8 else f'audio_prior_{args.frames}f_preview.json'
    (ROOT/'reports'/report_name).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
