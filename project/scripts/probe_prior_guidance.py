import os
"""Check whether a learned audio-latent prior helps frozen diffusion denoising.

Selects guidance strength on provisional validation tracks only. No weights
are updated and the test partition is not examined by this probe.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDPMScheduler, MotionAdapter
from safetensors.torch import load_file

from train_audio_latent_prior import AudioLatentPrior,load_data,wrong_track_indices

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    torch.set_num_threads(4)
    data=load_data()
    validation=data['validation']
    chosen=[];seen=set()
    for i,track in enumerate(validation['tracks']):
        if track not in seen:
            chosen.append(i);seen.add(track)
    wrong=wrong_track_indices(validation['tracks'])
    prior=AudioLatentPrior().to('cuda').eval()
    prior.load_state_dict(load_file(ROOT/'outputs/audio_latent_prior/model.safetensors'))
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
    unet=pipe.unet.to('cuda',dtype=torch.float16).eval()
    del pipe,adapter
    scheduler=DDPMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon')
    timestep=torch.tensor([500],device='cuda')
    alpha=float(scheduler.alphas_cumprod[500].sqrt())
    scales=(0.0,0.2,0.5,1.0)
    results={str(scale):[] for scale in scales}
    with torch.inference_mode():
        for n,i in enumerate(chosen):
            target=validation['target'][i:i+1].to('cuda',dtype=torch.float16)
            noise=torch.randn(target.shape,device='cuda',dtype=torch.float16,
                              generator=torch.Generator(device='cuda').manual_seed(643+n))
            noisy=scheduler.add_noise(target,noise,timestep)
            right=prior(validation['music'][i:i+1].to('cuda'),
                        validation['context'][i:i+1].to('cuda')).half()
            j=int(wrong[i])
            other=prior(validation['music'][j:j+1].to('cuda'),
                        validation['context'][j:j+1].to('cuda')).half()
            condition=text.expand(8,-1,-1)
            for scale in scales:
                right_input=noisy+scale*alpha*(right-static)
                wrong_input=noisy+scale*alpha*(other-static)
                right_eps=unet(right_input,timestep,encoder_hidden_states=condition).sample
                wrong_eps=unet(wrong_input,timestep,encoder_hidden_states=condition).sample
                correct_loss=float(F.mse_loss(right_eps.float(),noise.float()))
                wrong_loss=float(F.mse_loss(wrong_eps.float(),noise.float()))
                results[str(scale)].append({'pair':validation['ids'][i],
                                            'correct_loss':correct_loss,'other_song_loss':wrong_loss,
                                            'advantage':wrong_loss-correct_loss})
            print(f'{n+1}/{len(chosen)} validation tracks',flush=True)
    summary={key:{'pairs':len(values),
                  'correct_mse':sum(x['correct_loss'] for x in values)/len(values),
                  'correct_better_count':sum(x['advantage']>0 for x in values),
                  'mean_other_song_advantage':sum(x['advantage'] for x in values)/len(values)}
             for key,values in results.items()}
    report={'status':'completed','split':'provisional_validation','timestep':500,
            'scales':summary,'details':results,
            'note':'Frozen base UNet; no adapter fine-tuning. Positive wrong-minus-correct advantage favors music-conditioned prior. This probe selects scale on validation, not test.'}
    (ROOT/'reports/prior_guidance_probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
