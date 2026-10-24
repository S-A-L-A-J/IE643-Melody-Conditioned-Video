import os
"""Paired music-swap and one-second-shift diagnostics for spatial adapter."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDPMScheduler, MotionAdapter
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from pair_dataset import PairedClipDataset
from probe_adapter_step import LocalAudioProjector
from train_contrastive import normalize_features
from spatial_condition import AudioSpatialInjector

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-tracks',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(4)
    checkpoint=ROOT/'outputs/spatial_rings'
    run=json.loads((checkpoint/'run.json').read_text())
    config=run['configuration']
    dataset=PairedClipDataset(manifest='data/manifests/synthetic_pairs_v2.jsonl')
    rows_by_id={row['id']:i for i,row in enumerate(dataset.rows)}
    test_indices=[]; seen=set()
    for i,row in enumerate(dataset.rows):
        if row['style']=='rings' and row['split']=='test' and row['track_id'] not in seen:
            test_indices.append(i);seen.add(row['track_id'])
        if len(test_indices)>=args.test_tracks:
            break
    clear_title_ids={
        '46319c3a949c0b23',  # Koi Aur Nahi
        '4b6c42f5dc2b5059',  # Lag Ja Gale Se Phir Lofi
        '7637ab3eb8248572',  # Malhari
        '62449208ca20fded',  # Laal Pari
    }
    clear_title_indices=[]
    for i,row in enumerate(dataset.rows):
        if row['style']=='rings' and row['split']=='test' and row['track_id'] in clear_title_ids \
                and row['audio_start_seconds']==0:
            clear_title_indices.append(i)
    if len(clear_title_indices)!=len(clear_title_ids):
        raise RuntimeError('Clear single-title test subset changed')
    groups={'train':[rows_by_id[x] for x in run['pair_ids']],
            'provisional_test':test_indices,
            'clear_single_title_test':clear_title_indices}
    frame_indices=torch.arange(config['frames'])*config['stride']
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
        vae=pipe.vae.float().to('cuda').eval()
        cache_dir=ROOT/'data/latent_cache'/f"rings_{config['size']}_{config['frames']}_{config['stride']}"
        cache_dir.mkdir(parents=True,exist_ok=True)
        examples={}
        for split,indices in groups.items():
            examples[split]=[]
            for index in indices:
                row=dataset.rows[index]
                item=dataset[index]
                video=ROOT/row['video_path']
                digest=hashlib.sha256(video.read_bytes()).hexdigest()[:12]
                cache=cache_dir/f"{row['id']}_{digest}.safetensors"
                if cache.exists():
                    latent=load_file(cache)['latents']
                else:
                    frames=item['video'][:,frame_indices].permute(1,0,2,3)
                    if config['size']!=256:
                        frames=F.interpolate(frames,size=(config['size'],config['size']),
                                             mode='bilinear',align_corners=False)
                    chunks=[]
                    for chunk in frames.split(4):
                        encoded=vae.encode(chunk.to('cuda')).latent_dist.mode()*vae.config.scaling_factor
                        chunks.append(encoded.cpu())
                    latent=torch.cat(chunks).permute(1,0,2,3)[None].contiguous()
                    save_file({'latents':latent},cache)
                rhythm,melody,dynamics=normalize_features(item,frame_indices)
                examples[split].append({'id':row['id'],'track_id':row['track_id'],'latents':latent,
                                        'rhythm':rhythm.to('cuda'),'melody':melody.to('cuda'),
                                        'dynamics':dynamics.to('cuda')})
        del vae
        torch.cuda.empty_cache()
    unet=pipe.unet
    del pipe,adapter
    targets=[name for name,module in unet.named_modules() if isinstance(module,torch.nn.Linear)
             and name.endswith(('to_q','to_v')) and ('motion_modules' in name or 'attn2' in name)]
    unet.add_adapter(LoraConfig(r=config['rank'],lora_alpha=config['rank'],target_modules=targets))
    _,unexpected=unet.load_state_dict(load_file(checkpoint/'lora.safetensors'),strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected LoRA keys: {unexpected[:5]}')
    unet.to('cuda',dtype=torch.float16).eval()
    for name,module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    projector=LocalAudioProjector().to('cuda').eval()
    projector.load_state_dict(load_file(checkpoint/'projector.safetensors'))
    injector=AudioSpatialInjector(config['size']//8).to('cuda').eval()
    injector.load_state_dict(load_file(checkpoint/'injector.safetensors'))
    scheduler=DDPMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon')
    timestep=torch.tensor([500],device='cuda')

    def condition(example,shift=0):
        rhythm=example['rhythm'].roll(shift,0) if shift else example['rhythm']
        melody=example['melody'].roll(shift,0) if shift else example['melody']
        dynamics=example['dynamics'].roll(shift,0) if shift else example['dynamics']
        audio=projector(rhythm,melody,dynamics).half()
        return torch.cat([text.expand(config['frames'],-1,-1),audio],dim=1)

    def injected(noisy,example,shift=0):
        rhythm=example['rhythm'].roll(shift,0) if shift else example['rhythm']
        melody=example['melody'].roll(shift,0) if shift else example['melody']
        dynamics=example['dynamics'].roll(shift,0) if shift else example['dynamics']
        delta=injector(rhythm,melody,dynamics).half()
        signal_scale=float(scheduler.alphas_cumprod[500].sqrt())
        return noisy+delta*signal_scale

    results={}
    with torch.inference_mode():
        for split,group in examples.items():
            records=[]
            for i,example in enumerate(group):
                other=group[(i+1)%len(group)]
                noise=torch.randn(example['latents'].shape,device='cuda',dtype=torch.float16,
                                  generator=torch.Generator(device='cuda').manual_seed(643+i))
                noisy=scheduler.add_noise(example['latents'].to('cuda',dtype=torch.float16),noise,timestep)
                right=unet(injected(noisy,example),timestep,
                           encoder_hidden_states=condition(example)).sample.float()
                swapped=unet(injected(noisy,other),timestep,
                             encoder_hidden_states=condition(other)).sample.float()
                shifted=unet(injected(noisy,example,shift=2),timestep,
                             encoder_hidden_states=condition(example,shift=2)).sample.float()
                correct=float(F.mse_loss(right,noise.float()))
                wrong=float(F.mse_loss(swapped,noise.float()))
                late=float(F.mse_loss(shifted,noise.float()))
                records.append({'pair':example['id'],'other_pair':other['id'],
                                'correct_loss':correct,'other_song_loss':wrong,'shifted_one_second_loss':late,
                                'other_song_advantage':wrong-correct,'time_shift_advantage':late-correct})
            results[split]={'pairs':records,
                            'correct_better_than_other_song':sum(x['other_song_advantage']>0 for x in records),
                            'correct_better_than_time_shift':sum(x['time_shift_advantage']>0 for x in records),
                            'mean_other_song_advantage':sum(x['other_song_advantage'] for x in records)/len(records),
                            'mean_time_shift_advantage':sum(x['time_shift_advantage'] for x in records)/len(records)}
            print(split,{key:value for key,value in results[split].items() if key!='pairs'},flush=True)
    report={'checkpoint_step':run['step'],'timestep':500,'frames':config['frames'],
            'test_status':'provisional_recording_grouped_song_identity_unverified',
            'clear_title_note':'Four files have clear single-song source titles, but acoustic identity is not independently verified.',
            'advantage_definition':'wrong minus correct loss; positive favors correct music',
            'results':results}
    output=ROOT/'reports/spatial_rings_swap_evaluation.json'
    output.write_text(json.dumps(report,indent=2))
    print(output)


if __name__=='__main__':
    main()
