import os
"""One real denoising update on an E0 pair; measures 4 GB GPU feasibility."""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, MotionAdapter, DDPMScheduler
from peft import LoraConfig

from pair_dataset import PairedClipDataset

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))

class LocalAudioProjector(torch.nn.Module):
    def __init__(self,dim=768):
        super().__init__()
        self.rhythm=torch.nn.Sequential(torch.nn.Linear(3,dim),torch.nn.LayerNorm(dim))
        self.melody=torch.nn.Sequential(torch.nn.Linear(14,dim),torch.nn.LayerNorm(dim))
        self.dynamics=torch.nn.Sequential(torch.nn.Linear(5,dim),torch.nn.LayerNorm(dim))
        self.kind=torch.nn.Parameter(torch.zeros(3,dim))

    def forward(self,rhythm,melody,dynamics):
        streams=[self.rhythm(rhythm),self.melody(melody),self.dynamics(dynamics)]
        tokens=torch.stack(streams,dim=1)+self.kind[None]
        frame_index=torch.arange(len(rhythm),device=rhythm.device,dtype=torch.float32)
        div=torch.exp(torch.arange(0,tokens.shape[-1],2,device=rhythm.device)*(-math.log(10000.0)/tokens.shape[-1]))
        timing=torch.zeros(len(rhythm),tokens.shape[-1],device=rhythm.device)
        timing[:,0::2]=torch.sin(frame_index[:,None]*div)
        timing[:,1::2]=torch.cos(frame_index[:,None]*div)
        return tokens+timing[:,None]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--rank',type=int,default=2)
    parser.add_argument('--frame-stride',type=int,default=4)
    parser.add_argument('--activation-offload',action='store_true')
    args=parser.parse_args()
    start=time.monotonic()
    torch.manual_seed(643)
    torch.set_num_threads(4)
    dataset=PairedClipDataset()
    item=dataset[0]
    frame_indices=torch.arange(args.frames)*args.frame_stride
    if frame_indices[-1]>=64:
        parser.error('Selected frames exceed the four-second source clip')
    video=item['video'][:,frame_indices].permute(1,0,2,3)
    video=F.interpolate(video,size=(args.size,args.size),mode='bilinear',align_corners=False)

    model_root=ROOT/'models/diffusion'
    adapter=MotionAdapter.from_pretrained(model_root/'animatediff',variant='fp16',
                                          torch_dtype=torch.float16,local_files_only=True)
    pipe=AnimateDiffPipeline.from_pretrained(model_root/'sd15',motion_adapter=adapter,
                                             torch_dtype=torch.float16,variant='fp16',
                                             local_files_only=True)
    tokenizer=pipe.tokenizer
    with torch.no_grad():
        vae=pipe.vae.float().cpu().eval()
        latent_frames=vae.encode(video).latent_dist.mode()*vae.config.scaling_factor
        text_model=pipe.text_encoder.float().cpu().eval()
        text_inputs=tokenizer('abstract colorful geometric shapes on a dark background',
                              padding='max_length',max_length=tokenizer.model_max_length,
                              truncation=True,return_tensors='pt')
        text_tokens=text_model(text_inputs.input_ids)[0].detach().mean(dim=1,keepdim=True).to('cuda',dtype=torch.float16)
    unet=pipe.unet
    del vae,text_model
    latents=latent_frames.permute(1,0,2,3)[None].to('cuda',dtype=torch.float16)
    del latent_frames,pipe,adapter
    unet.requires_grad_(False)
    targets=[name for name,module in unet.named_modules()
             if isinstance(module,torch.nn.Linear)
             and name.endswith(('to_q','to_v'))
             and ('motion_modules' in name or 'attn2' in name)]
    if not targets or not any('motion_modules' in name for name in targets) or not any('attn2' in name for name in targets):
        raise RuntimeError('Did not find both temporal and cross-attention adapter targets')
    unet.add_adapter(LoraConfig(r=args.rank,lora_alpha=args.rank,target_modules=targets))
    unet.enable_gradient_checkpointing()
    unet.to('cuda',dtype=torch.float16).train()
    # GradScaler cannot unscale FP16 trainable parameters. PEFT casts its inputs
    # to adapter weight dtype internally, so keep only LoRA matrices in FP32.
    for name,module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    if not all(p.dtype==torch.float32 for name,p in unet.named_parameters() if 'lora_' in name):
        raise RuntimeError('LoRA adapter parameters must remain FP32')
    projector=LocalAudioProjector().to('cuda').train()
    trainable=[p for p in list(unet.parameters())+list(projector.parameters()) if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-4)

    rhythm=item['rhythm'][frame_indices].to('cuda')
    melody=item['melody'][frame_indices].to('cuda')
    dynamics=item['dynamics'][frame_indices].to('cuda')
    audio_tokens=projector(rhythm,melody,dynamics).to(dtype=torch.float16)
    conditioning=torch.cat([text_tokens.expand(args.frames,-1,-1),audio_tokens],dim=1)
    noise=torch.randn_like(latents)
    timestep=torch.tensor([500],device='cuda',dtype=torch.long)
    scheduler=DDPMScheduler(num_train_timesteps=1000,beta_start=0.00085,
                            beta_end=0.012,beta_schedule='scaled_linear',prediction_type='epsilon')
    noisy=scheduler.add_noise(latents,noise,timestep)
    torch.cuda.reset_peak_memory_stats()
    begin_step=time.monotonic()
    scaler=torch.amp.GradScaler('cuda')
    try:
        probe_param=next(p for name,p in unet.named_parameters() if 'lora_B' in name and p.requires_grad)
        before_projector=next(projector.parameters()).detach().clone()
        before_adapter=probe_param.detach().clone()
        offload_context=(torch.autograd.graph.save_on_cpu(pin_memory=False)
                         if args.activation_offload else contextlib.nullcontext())
        with offload_context:
            predicted=unet(noisy,timestep,encoder_hidden_states=conditioning).sample
        loss=F.mse_loss(predicted.float(),noise.float())
        if not torch.isfinite(loss):
            raise ValueError('Non-finite denoising loss')
        scaler.scale(loss).backward()
        projector_grad=sum(p.grad is not None and bool(torch.isfinite(p.grad).all())
                           for p in projector.parameters())
        lora_grad=sum(p.grad is not None and bool(torch.isfinite(p.grad).all())
                      for name,p in unet.named_parameters() if 'lora_' in name)
        if not projector_grad or not lora_grad:
            raise RuntimeError('No finite gradients reached projectors or LoRA')
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        parameter_change=max(float((before_projector-next(projector.parameters())).abs().max()),
                             float((before_adapter-probe_param).abs().max()))
        if parameter_change<=0:
            raise RuntimeError('Optimizer did not change trainable weights')
        result=dict(status='passed',input_pair=item['id'],size=args.size,frames=args.frames,
                    frame_stride=args.frame_stride,activation_offload=args.activation_offload,
                    lora_rank=args.rank,adapter_targets=len(targets),
                    trainable_parameters=sum(p.numel() for p in trainable),
                    loss=float(loss.detach()),projector_grad_tensors=projector_grad,
                    lora_grad_tensors=lora_grad,peak_allocated_mb=round(torch.cuda.max_memory_allocated()/2**20,1),
                    largest_checked_weight_change=parameter_change,
                    peak_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20,1),
                    step_seconds=round(time.monotonic()-begin_step,2),
                    total_seconds=round(time.monotonic()-start,2),
                    note='One E2-style denoising update; no checkpoint trained or video generated from music.')
        output=ROOT/'reports'/f'adapter_probe_{args.size}_{args.frames}.json'
        output.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    except Exception as exc:
        result=dict(status='failed',error=str(exc),size=args.size,frames=args.frames,
                    frame_stride=args.frame_stride,activation_offload=args.activation_offload,
                    peak_allocated_mb=round(torch.cuda.max_memory_allocated()/2**20,1),
                    peak_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20,1))
        (ROOT/'reports'/f'adapter_probe_{args.size}_{args.frames}.json').write_text(json.dumps(result,indent=2))
        raise

if __name__=='__main__':
    main()
