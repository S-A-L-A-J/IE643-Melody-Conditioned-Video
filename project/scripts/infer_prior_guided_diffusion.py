import os
"""Generate same-seed held-out videos from the prior-guided AnimateDiff adapter."""
import argparse
import json
from pathlib import Path

import cv2
import torch
from diffusers import AnimateDiffPipeline,DDIMScheduler,MotionAdapter
from peft import LoraConfig
from safetensors.torch import load_file

from train_audio_latent_prior import AudioLatentPrior,load_data,wrong_track_indices

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--lora',default='lora.safetensors')
    parser.add_argument('--seed',type=int,default=643)
    args=parser.parse_args()
    torch.set_num_threads(4)
    data=load_data()
    test=data['test']
    preferred='46319c3a949c0b23'
    index=next(i for i,(track,pair) in enumerate(zip(test['tracks'],test['ids']))
               if track==preferred and pair.endswith('_0.000__rings'))
    other=int(wrong_track_indices(test['tracks'])[index])
    prior=AudioLatentPrior().to('cuda').eval()
    prior.load_state_dict(load_file(ROOT/'outputs/audio_latent_prior/model.safetensors'))
    with torch.inference_mode():
        music=test['music'][index:index+1].to('cuda')
        context=test['context'][index:index+1].to('cuda')
        templates={
            'correct':prior(music,context).half(),
            'other_song':prior(test['music'][other:other+1].to('cuda'),
                               test['context'][other:other+1].to('cuda')).half(),
            'shifted_one_second':prior(music.roll(2,1),context).half(),
        }
    static=data['train']['target'].mean(0).to('cuda',dtype=torch.float16)
    adapter=MotionAdapter.from_pretrained(ROOT/'models/diffusion/animatediff',variant='fp16',
                                          torch_dtype=torch.float16,local_files_only=True)
    pipe=AnimateDiffPipeline.from_pretrained(ROOT/'models/diffusion/sd15',motion_adapter=adapter,
                                             torch_dtype=torch.float16,variant='fp16',local_files_only=True)
    with torch.no_grad():
        text_model=pipe.text_encoder.float().cpu().eval()
        tokens=pipe.tokenizer('abstract glowing concentric rings on a dark background',
                              padding='max_length',max_length=pipe.tokenizer.model_max_length,
                              truncation=True,return_tensors='pt')
        text=text_model(tokens.input_ids)[0].mean(dim=1,keepdim=True).to('cuda',dtype=torch.float16)
        del text_model
    unet=pipe.unet
    vae=pipe.vae.float().cpu().eval()
    del pipe,adapter
    targets=[name for name,module in unet.named_modules() if isinstance(module,torch.nn.Linear)
             and name.endswith(('to_q','to_k','to_v','to_out.0'))
             and ('motion_modules' in name or 'attn2' in name)]
    unet.add_adapter(LoraConfig(r=4,lora_alpha=4,target_modules=targets))
    _,unexpected=unet.load_state_dict(load_file(ROOT/'outputs/prior_guided_diffusion'/args.lora),strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected LoRA keys: {unexpected[:5]}')
    unet.to('cuda',dtype=torch.float16).eval()
    for name,module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon',
                            clip_sample=False,timestep_spacing='linspace',steps_offset=1)
    scheduler.set_timesteps(args.steps,device='cuda')
    initial=torch.randn((1,4,8,16,16),device='cuda',dtype=torch.float16,
                        generator=torch.Generator(device='cuda').manual_seed(args.seed))
    report={'checkpoint_file':args.lora,'test_pair':test['ids'][index],
            'other_pair':test['ids'][other],'seed':args.seed,'steps':args.steps,
            'frames':8,'fps':2,'size':128,'videos':{}}
    with torch.inference_mode():
        for label,template in templates.items():
            current=initial.clone()
            for timestep in scheduler.timesteps:
                alpha=float(scheduler.alphas_cumprod[int(timestep.item())].sqrt())
                conditioned=current+0.3*alpha*(template-static)
                eps=unet(conditioned,timestep,encoder_hidden_states=text.expand(8,-1,-1)).sample
                current=scheduler.step(eps,timestep,current).prev_sample
            frames=[]
            for chunk in current[0].permute(1,0,2,3).cpu().float().split(4):
                decoded=vae.decode(chunk/vae.config.scaling_factor).sample
                rgb=((decoded/2+0.5).clamp(0,1)*255).byte().permute(0,2,3,1).numpy()
                frames.extend(rgb)
            output=ROOT/'outputs'/f'prior_guided_{label}.mp4'
            writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),2,(128,128))
            if not writer.isOpened():
                raise RuntimeError(f'Cannot write {output}')
            for frame in frames:
                writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            writer.release()
            report['videos'][label]=str(output.relative_to(ROOT))
            print(label,output,flush=True)
    report['note']='Adapted diffusion outputs on one provisional held-out track; visual quality and audio alignment require direct inspection.'
    (ROOT/'reports/prior_guided_generation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
