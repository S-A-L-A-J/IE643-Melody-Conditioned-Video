import os
"""Final music-swap and one-second-shift checks for prior-guided AnimateDiff."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline,DDPMScheduler,MotionAdapter
from peft import LoraConfig
from safetensors.torch import load_file

from train_audio_latent_prior import AudioLatentPrior,load_data,wrong_track_indices

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    torch.set_num_threads(4)
    data=load_data()
    run=json.loads((ROOT/'outputs/prior_guided_diffusion/run.json').read_text())
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
    unet=pipe.unet
    del pipe,adapter
    targets=[name for name,module in unet.named_modules() if isinstance(module,torch.nn.Linear)
             and name.endswith(('to_q','to_k','to_v','to_out.0'))
             and ('motion_modules' in name or 'attn2' in name)]
    unet.add_adapter(LoraConfig(r=4,lora_alpha=4,target_modules=targets))
    _,unexpected=unet.load_state_dict(load_file(ROOT/'outputs/prior_guided_diffusion/lora.safetensors'),strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected LoRA keys: {unexpected[:5]}')
    unet.to('cuda',dtype=torch.float16).eval()
    for name,module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    scheduler=DDPMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon')
    timestep=torch.tensor([500],device='cuda')
    alpha=float(scheduler.alphas_cumprod[500].sqrt())
    clear_title_ids={'46319c3a949c0b23','4b6c42f5dc2b5059',
                     '7637ab3eb8248572','62449208ca20fded'}
    all_records=[]
    with torch.inference_mode():
        test=data['test']
        selected=[];seen=set()
        for i,track in enumerate(test['tracks']):
            if track not in seen:
                selected.append(i);seen.add(track)
        wrong=wrong_track_indices(test['tracks'])
        for n,i in enumerate(selected):
            target=test['target'][i:i+1].to('cuda',dtype=torch.float16)
            noise=torch.randn(target.shape,device='cuda',dtype=torch.float16,
                              generator=torch.Generator(device='cuda').manual_seed(9000+n))
            noisy=scheduler.add_noise(target,noise,timestep)
            music=test['music'][i:i+1].to('cuda')
            context=test['context'][i:i+1].to('cuda')
            j=int(wrong[i])
            templates={
                'correct':prior(music,context).half(),
                'other_song':prior(test['music'][j:j+1].to('cuda'),
                                   test['context'][j:j+1].to('cuda')).half(),
                'shifted_one_second':prior(music.roll(2,1),context).half(),
            }
            losses={}
            for name,template in templates.items():
                model_input=noisy+run['guidance_scale']*alpha*(template-static)
                predicted=unet(model_input,timestep,encoder_hidden_states=text.expand(8,-1,-1)).sample
                losses[name]=float(F.mse_loss(predicted.float(),noise.float()))
            all_records.append({'pair_id':test['ids'][i],'track_id':test['tracks'][i],
                                'clear_single_title_source':test['tracks'][i] in clear_title_ids,
                                'correct_loss':losses['correct'],
                                'other_song_loss':losses['other_song'],
                                'shifted_one_second_loss':losses['shifted_one_second'],
                                'other_song_advantage':losses['other_song']-losses['correct'],
                                'time_shift_advantage':losses['shifted_one_second']-losses['correct']})
            print(f'{n+1}/{len(selected)} held-out tracks',flush=True)
    def summary(records):
        return {'tracks':len(records),'correct_better_than_other_song':sum(x['other_song_advantage']>0 for x in records),
                'correct_better_than_one_second_shift':sum(x['time_shift_advantage']>0 for x in records),
                'mean_correct_loss':sum(x['correct_loss'] for x in records)/len(records),
                'mean_other_song_advantage':sum(x['other_song_advantage'] for x in records)/len(records),
                'mean_time_shift_advantage':sum(x['time_shift_advantage'] for x in records)/len(records)}
    report={'status':'completed','checkpoint_step':run['step'],'timestep':500,
            'split_status':'provisional_recording_grouped_song_identity_unverified',
            'all_test_tracks':summary(all_records),
            'clear_single_title_subset':summary([x for x in all_records if x['clear_single_title_source']]),
            'tracks':all_records,'note':'Positive advantage means correct music predicts diffusion noise better. This is a latent diagnostic, not generated-video quality.'}
    (ROOT/'reports/prior_guided_diffusion_test.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='tracks'},indent=2))


if __name__=='__main__':
    main()
