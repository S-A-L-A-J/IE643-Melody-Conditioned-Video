import os
"""Third tiny experiment: direct spatial music injection plus paired ranking.

This is an isolated debugging run on the new rings targets, not a final model.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, MotionAdapter, DDPMScheduler
from peft import LoraConfig
from safetensors.torch import save_file, load_file

from pair_dataset import PairedClipDataset
from probe_adapter_step import LocalAudioProjector
from spatial_condition import AudioSpatialInjector

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))

def normalize_features(item,indices):
    rhythm=item['rhythm'][indices].clone()
    melody=item['melody'][indices].clone()
    dynamics=item['dynamics'][indices].clone()
    rhythm[:,0]=torch.log1p(rhythm[:,0].clamp_min(0))/math.log1p(20)
    rhythm[:,2]=((rhythm[:,2]-120)/80).clamp(-2,2)
    melody[:,12]=(torch.log2(melody[:,12].clamp_min(40)/220)/3).clamp(-2,2)
    melody[:,13]=(melody[:,13]*4).clamp(0,2)
    dynamics[:,0]=(dynamics[:,0]*4).clamp(0,2)
    dynamics[:,1:]=(torch.log1p(dynamics[:,1:].clamp_min(0))/math.log1p(30)).clamp(0,2)
    return rhythm,melody,dynamics

def choose_pairs(dataset,count):
    preferred=['d568b4accdc027f2','9caf45d667658a8f','bdef0eefdd8e5d25',
               '4f5993b4ce0bb534','07a450f4730f2f7c','ac5e0ce9ffa800d8',
               '1882fda314c37da1','4c1c40fe06d60c8f']
    chosen=[]
    for track_id in preferred[:count]:
        matches=[i for i,row in enumerate(dataset.rows) if row.get('style')=='rings'
                 and row.get('split')=='train' and row['track_id']==track_id
                 and row['audio_start_seconds']==0]
        if len(matches)!=1:
            raise ValueError(f'Expected one rings training clip for {track_id}')
        chosen.extend(matches)
    if len(chosen)<count:
        raise ValueError('Not enough distinct tracks')
    return chosen

def save_checkpoint(path,unet,projector,injector,optimizer,scaler,rng,step,report):
    path.mkdir(parents=True,exist_ok=True)
    lora={name:param.detach().cpu().contiguous()
          for name,param in unet.named_parameters() if param.requires_grad and 'lora_' in name}
    projection={name:t.detach().cpu().contiguous() for name,t in projector.state_dict().items()}
    save_file(lora,path/'lora.safetensors')
    save_file(projection,path/'projector.safetensors')
    save_file({name:t.detach().cpu().contiguous() for name,t in injector.state_dict().items()},
              path/'injector.safetensors')
    torch.save(dict(step=step,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    rng_state=rng.get_state()),path/'training_state.pt')
    (path/'run.json').write_text(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--pairs',type=int,default=8)
    parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--stride',type=int,default=8)
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--rank',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--checkpoint-every',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    frame_indices=torch.arange(args.frames)*args.stride
    if frame_indices[-1]>=64:
        parser.error('Frames must fit in the four-second source pair')
    torch.manual_seed(643)
    torch.set_num_threads(4)
    run_dir=ROOT/'outputs/spatial_rings'
    run_dir.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    dataset=PairedClipDataset(manifest='data/manifests/synthetic_pairs_v2.jsonl')
    selected=choose_pairs(dataset,args.pairs)
    adapter=MotionAdapter.from_pretrained(ROOT/'models/diffusion/animatediff',variant='fp16',
                                          torch_dtype=torch.float16,local_files_only=True)
    pipe=AnimateDiffPipeline.from_pretrained(ROOT/'models/diffusion/sd15',motion_adapter=adapter,
                                             torch_dtype=torch.float16,variant='fp16',local_files_only=True)
    cache_dir=ROOT/'data/latent_cache'/f'rings_{args.size}_{args.frames}_{args.stride}'
    cache_dir.mkdir(parents=True,exist_ok=True)
    cache_paths=[]
    for index in selected:
        row=dataset.rows[index]
        video_path=ROOT/row['video_path']
        digest=hashlib.sha256()
        with video_path.open('rb') as video_file:
            for block in iter(lambda:video_file.read(1024*1024),b''):
                digest.update(block)
        short_hash=digest.hexdigest()[:12]
        cache_paths.append(cache_dir/f"{row['id']}_{short_hash}.safetensors")
    with torch.no_grad():
        text_model=pipe.text_encoder.float().cpu().eval()
        inputs=pipe.tokenizer('abstract glowing concentric rings on a dark background',
                              padding='max_length',max_length=pipe.tokenizer.model_max_length,
                              truncation=True,return_tensors='pt')
        text_tokens=text_model(inputs.input_ids)[0].detach().mean(dim=1,keepdim=True).to('cuda',dtype=torch.float16)
        del text_model
        vae=pipe.vae.float().cpu().eval() if not all(p.exists() for p in cache_paths) else None
        examples=[]
        for index,cache_path in zip(selected,cache_paths):
            item=dataset[index]
            if cache_path.exists():
                stored=load_file(cache_path)['latents']
            else:
                frames=item['video'][:,frame_indices].permute(1,0,2,3)
                if args.size!=256:
                    frames=F.interpolate(frames,size=(args.size,args.size),mode='bilinear',align_corners=False)
                latent=(vae.encode(frames).latent_dist.mode()*vae.config.scaling_factor)
                stored=latent.permute(1,0,2,3)[None].contiguous()
                save_file({'latents':stored},cache_path)
            rhythm,melody,dynamics=normalize_features(item,frame_indices)
            examples.append(dict(id=item['id'],latents=stored,
                                 rhythm=rhythm,melody=melody,dynamics=dynamics))
        del vae
    unet=pipe.unet
    del pipe,adapter
    unet.requires_grad_(False)
    targets=[name for name,module in unet.named_modules() if isinstance(module,torch.nn.Linear)
             and name.endswith(('to_q','to_v')) and ('motion_modules' in name or 'attn2' in name)]
    unet.add_adapter(LoraConfig(r=args.rank,lora_alpha=args.rank,target_modules=targets))
    unet.enable_gradient_checkpointing()
    unet.to('cuda',dtype=torch.float16).train()
    for name,module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    projector=LocalAudioProjector().to('cuda').train()
    injector=AudioSpatialInjector(args.size//8).to('cuda').train()
    trainable=[p for p in list(unet.parameters())+list(projector.parameters())+list(injector.parameters())
               if p.requires_grad]
    assert all(p.dtype==torch.float32 for p in trainable)
    optimizer=torch.optim.AdamW(trainable,lr=args.lr)
    scaler=torch.amp.GradScaler('cuda')
    scheduler=DDPMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,
                            beta_schedule='scaled_linear',prediction_type='epsilon')
    rng=torch.Generator(device='cuda').manual_seed(643)
    fixed_noise=torch.randn(examples[0]['latents'].shape,device='cuda',dtype=torch.float16,generator=rng)
    history=[]
    report=dict(status='running',configuration=vars(args),pair_ids=[x['id'] for x in examples],
                trainable_parameters=sum(p.numel() for p in trainable),adapter_targets=len(targets),
                video_fps=16,training_frame_indices=frame_indices.tolist(),
                objective='denoising_mse_plus_2x_softplus_music_swap_margin_0.01_scale_50',
                normalization='log_onset_centered_tempo_log_pitch_scaled_loudness_log_band_energy',
                spatial_condition='22-channel local audio to four-channel latent radial basis delta',
                note='Eight clearer-title rings training clips; source song identities unverified.')
    start_step=0
    if args.resume:
        if not (run_dir/'training_state.pt').exists():
            raise FileNotFoundError('No training checkpoint to resume')
        lora_state=load_file(run_dir/'lora.safetensors')
        missing,unexpected=unet.load_state_dict(lora_state,strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected LoRA weights: {unexpected[:5]}')
        projector.load_state_dict(load_file(run_dir/'projector.safetensors'))
        injector.load_state_dict(load_file(run_dir/'injector.safetensors'))
        saved=torch.load(run_dir/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        if 'rng_state' in saved:
            rng.set_state(saved['rng_state'])
        start_step=int(saved['step'])
        previous=json.loads((run_dir/'run.json').read_text())
        if previous['pair_ids']!=report['pair_ids']:
            raise RuntimeError('Checkpoint pair selection changed')
        report['before']=previous['before']
        history=list(previous['history'])
        report['resumed_from_step']=start_step

    def conditioning(example):
        rhythm=example['rhythm'].to('cuda')
        melody=example['melody'].to('cuda')
        dynamics=example['dynamics'].to('cuda')
        audio=projector(rhythm,melody,dynamics).to(dtype=torch.float16)
        return torch.cat([text_tokens.expand(args.frames,-1,-1),audio],dim=1)

    def injected(noisy,example,timestep):
        delta=injector(example['rhythm'].to('cuda'),example['melody'].to('cuda'),
                       example['dynamics'].to('cuda'))
        signal_scale=float(scheduler.alphas_cumprod[int(timestep.item())].sqrt())
        return noisy+delta.to(dtype=torch.float16)*signal_scale

    @torch.no_grad()
    def measure(index=0,alternate=1):
        unet.eval(); projector.eval()
        target=examples[index]['latents'].to('cuda',dtype=torch.float16)
        timestep=torch.tensor([500],device='cuda')
        noisy=scheduler.add_noise(target,fixed_noise,timestep)
        right=conditioning(examples[index])
        wrong=conditioning(examples[alternate])
        prediction=unet(injected(noisy,examples[index],timestep),timestep,encoder_hidden_states=right).sample
        wrong_prediction=unet(injected(noisy,examples[alternate],timestep),timestep,
                              encoder_hidden_states=wrong).sample
        result=dict(correct_loss=float(F.mse_loss(prediction.float(),fixed_noise.float())),
                    mismatched_loss=float(F.mse_loss(wrong_prediction.float(),fixed_noise.float())),
                    prediction_difference=float((prediction.float()-wrong_prediction.float()).abs().mean()))
        del target,noisy,right,wrong,prediction,wrong_prediction
        unet.train(); projector.train(); injector.train()
        return result

    if not args.resume:
        report['before']=measure()
    print('before',report['before'],'resume step',start_step,flush=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,args.steps+1):
        example=examples[(step-1)%len(examples)]
        latents=example['latents'].to('cuda',dtype=torch.float16)
        noise=torch.randn(latents.shape,device='cuda',dtype=torch.float16,generator=rng)
        timestep=torch.randint(0,1000,(1,),device='cuda',generator=rng)
        noisy=scheduler.add_noise(latents,noise,timestep)
        other=examples[step%len(examples)]
        # The paired objective is differentiated in two serial passes. Holding
        # both UNet graphs at once exceeded the host's 15 GiB RAM during CPU
        # activation offloading. A no-grad estimate provides the exact local
        # derivative weights for the softplus ranking term.
        with torch.no_grad():
            estimate_correct=unet(injected(noisy,example,timestep),timestep,
                                  encoder_hidden_states=conditioning(example)).sample
            estimate_wrong=unet(injected(noisy,other,timestep),timestep,
                                encoder_hidden_states=conditioning(other)).sample
            estimate_correct_loss=F.mse_loss(estimate_correct.float(),noise.float())
            estimate_wrong_loss=F.mse_loss(estimate_wrong.float(),noise.float())
            ranking_weight=2*torch.sigmoid(50*(estimate_correct_loss-estimate_wrong_loss+0.01))
            reported_loss=estimate_correct_loss+2*F.softplus(
                50*(estimate_correct_loss-estimate_wrong_loss+0.01))/50
            del estimate_correct,estimate_wrong
        optimizer.zero_grad(set_to_none=True)
        condition=conditioning(example)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            prediction=unet(injected(noisy,example,timestep),timestep,
                            encoder_hidden_states=condition).sample
        correct_loss=F.mse_loss(prediction.float(),noise.float())
        scaler.scale((1+ranking_weight)*correct_loss).backward()
        del condition,prediction,correct_loss
        wrong_condition=conditioning(other)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            wrong_prediction=unet(injected(noisy,other,timestep),timestep,
                                  encoder_hidden_states=wrong_condition).sample
        wrong_loss=F.mse_loss(wrong_prediction.float(),noise.float())
        scaler.scale(-ranking_weight*wrong_loss).backward()
        if not torch.isfinite(reported_loss):
            raise ValueError(f'Non-finite loss at step {step}')
        scaler.step(optimizer)
        scaler.update()
        history.append(dict(step=step,pair=example['id'],loss=float(reported_loss),
                            correct_loss=float(estimate_correct_loss),
                            mismatched_loss=float(estimate_wrong_loss),
                            timestep=int(timestep.item()),grad_scale=float(scaler.get_scale())))
        del latents,noise,timestep,noisy,wrong_condition,wrong_prediction,wrong_loss
        if step%args.checkpoint_every==0 or step==args.steps:
            report.update(status='checkpointed',step=step,history=history,
                          latest=measure(),peak_vram_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20,1),
                          elapsed_seconds=round(time.monotonic()-start,1))
            save_checkpoint(run_dir,unet,projector,injector,optimizer,scaler,rng,step,report)
            print(f"step {step}: {history[-1]['loss']:.4f}; fixed {report['latest']}; "
                  f"VRAM {report['peak_vram_reserved_mb']} MB",flush=True)
    report['status']='completed'
    report['total_seconds']=round(time.monotonic()-start,1)
    save_checkpoint(run_dir,unet,projector,injector,optimizer,scaler,rng,args.steps,report)

if __name__=='__main__':
    main()
