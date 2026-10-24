import os
"""Validation-only img2img-style diffusion refinement of learned audio priors."""
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline,DDIMScheduler,MotionAdapter
from safetensors.torch import load_file

from train_audio_latent_prior import AudioLatentPrior,load_data,wrong_track_indices

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    torch.set_num_threads(4)
    data=load_data()['validation']
    index=0
    other=int(wrong_track_indices(data['tracks'])[index])
    prior=AudioLatentPrior().to('cuda').eval()
    prior.load_state_dict(load_file(ROOT/'outputs/audio_latent_prior/model.safetensors'))
    with torch.inference_mode():
        source=prior(data['music'][index:index+1].to('cuda'),
                     data['context'][index:index+1].to('cuda')).half()
        swapped=prior(data['music'][other:other+1].to('cuda'),
                      data['context'][other:other+1].to('cuda')).half()
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
    unet=pipe.unet.to('cuda',dtype=torch.float16).eval()
    vae=pipe.vae.float().cpu().eval()
    del pipe,adapter
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon',
                            clip_sample=False,timestep_spacing='linspace',steps_offset=1)
    scheduler.set_timesteps(20,device='cuda')
    noise=torch.randn(source.shape,device='cuda',dtype=torch.float16,
                      generator=torch.Generator(device='cuda').manual_seed(643))
    target=data['target'][index:index+1].to('cuda',dtype=torch.float16)
    report={'split':'provisional_validation','pair_id':data['ids'][index],
            'other_pair_id':data['ids'][other],'prior_mse':float(F.mse_loss(source.float(),target.float())),
            'conditions':{}}
    with torch.inference_mode():
        for strength in (0.35,0.65):
            start_index=round((1-strength)*len(scheduler.timesteps))
            first_timestep=scheduler.timesteps[start_index]
            for label,original in [('correct',source),('other_song',swapped)]:
                current=scheduler.add_noise(original,noise,first_timestep)
                for timestep in scheduler.timesteps[start_index:]:
                    eps=unet(current,timestep,encoder_hidden_states=text.expand(8,-1,-1)).sample
                    current=scheduler.step(eps,timestep,current).prev_sample
                frames=[]
                for chunk in current[0].permute(1,0,2,3).cpu().float().split(4):
                    decoded=vae.decode(chunk/vae.config.scaling_factor).sample
                    rgb=((decoded/2+0.5).clamp(0,1)*255).byte().permute(0,2,3,1).numpy()
                    frames.extend(rgb)
                name=f'prior_refine_{strength}_{label}'
                output=ROOT/'outputs'/f'{name}.mp4'
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),2,(128,128))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot write {output}')
                for frame in frames:
                    writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                writer.release()
                report['conditions'][name]={'mse_to_target':float(F.mse_loss(current.float(),target.float())),
                                            'video':str(output.relative_to(ROOT)),
                                            'denoising_steps':len(scheduler.timesteps[start_index:])}
                print(name,report['conditions'][name],flush=True)
    (ROOT/'reports/prior_refinement_probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
