import os
"""Local frozen AnimateDiff backend with native CUDA arithmetic and bounded windows."""
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler, UNet2DConditionModel, UNetMotionModel, AutoencoderKL
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTokenizer

ROOT=Path(os.environ.get('IE643_PRETRAINED_ROOT', str(Path(__file__).resolve().parent.joinpath('../../../../../project').resolve())))
PROMPT='abstract glowing concentric rings on a dark background'

def load_pipeline(offload=True,lora_path=None):
    assert torch.cuda.is_available(), 'Use host GPU execution; sandbox hides CUDA'
    dtype=torch.float16
    with init_empty_weights():
        base=UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(ROOT/'models/diffusion/sd15/unet'))
        adapter=MotionAdapter.from_config(MotionAdapter.load_config(ROOT/'models/diffusion/animatediff'))
        unet=UNetMotionModel.from_unet2d(base,adapter,load_weights=False)
    del base,adapter
    expected=unet.state_dict();loaded=set()
    for folder in ['sd15/unet','animatediff']:
        with safe_open(ROOT/'models/diffusion'/folder/'diffusion_pytorch_model.fp16.safetensors',framework='pt',device='cpu') as f:
            for key in f.keys():
                assert key in expected and key not in loaded,key
                tensor=f.get_tensor(key);assert tensor.shape==expected[key].shape,key
                set_module_tensor_to_device(unet,key,'cpu',value=tensor,dtype=dtype);loaded.add(key)
        del tensor;gc.collect()
    assert loaded==set(expected)
    del expected
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=dtype,local_files_only=True)
    text=CLIPTextModel.from_pretrained(ROOT/'models/diffusion/sd15/text_encoder',variant='fp16',torch_dtype=dtype,local_files_only=True)
    tokenizer=CLIPTokenizer.from_pretrained(ROOT/'models/diffusion/sd15/tokenizer',local_files_only=True)
    scheduler=DDIMScheduler.from_pretrained(ROOT/'models/diffusion/sd15/scheduler',local_files_only=True,
                  beta_schedule='linear',clip_sample=False,timestep_spacing='linspace',steps_offset=1)
    pipe=AnimateDiffVideoToVideoPipeline(vae,text,tokenizer,unet,None,scheduler)
    if lora_path is not None:
        from peft import LoraConfig
        from safetensors.torch import load_file
        targets=[name for name,module in pipe.unet.named_modules() if isinstance(module,torch.nn.Linear)
                 and name.endswith(('to_q','to_v','to_out.0')) and 'motion_modules' not in name]
        pipe.unet.add_adapter(LoraConfig(r=2,lora_alpha=2,target_modules=targets))
        state=load_file(lora_path)
        expected_lora={name for name,p in pipe.unet.named_parameters() if 'lora_' in name}
        assert set(state)==expected_lora,'LoRA keys differ'
        _,unexpected=pipe.unet.load_state_dict(state,strict=False);assert not unexpected
        for name,module in pipe.unet.named_modules():
            if '.lora_A.default' in name or '.lora_B.default' in name:module.float()
    pipe.unet.requires_grad_(False).eval()
    pipe._geometry_lora_path=str(lora_path) if lora_path else None
    if offload:pipe.enable_sequential_cpu_offload(gpu_id=0)
    print('Loaded native CUDA backend; tensors',len(loaded),flush=True)
    return pipe

def windows(frames,length=32,hop=16):
    if frames<=length:return [(0,frames)]
    starts=list(range(0,frames-length+1,hop))
    if starts[-1]!=frames-length:starts.append(frames-length)
    return [(s,s+length) for s in starts]

@torch.inference_mode()
def refine(pipe,clean,strength=.06,base_steps=100,guidance=1.,seed=643,noise_sign=1.):
    """Blend overlapping noise estimates at EACH global DDIM step, not final clips.

    Every denoiser call stays within the native motion positional length. One
    global latent/noise field and scheduler step avoid independently re-noising
    adjacent clips. This explicit overlap algorithm is not claimed as FreeNoise.
    """
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    frames=clean.shape[2];parts=windows(frames)
    prompt,negative=pipe.encode_prompt(PROMPT,'cuda',1,guidance>1,negative_prompt='')
    pipe.scheduler.set_timesteps(base_steps,device='cuda')
    times,n=pipe.get_timesteps(base_steps,pipe.scheduler.timesteps,strength,'cuda')
    assert n>0
    clean=clean.to('cuda',dtype=torch.float16)
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
    unsigned_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest()
    noise=noise*noise_sign
    noise_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest()
    current=pipe.scheduler.add_noise(clean,noise.to('cuda',dtype=torch.float16),times[:1])
    for i,t in enumerate(times):
        prediction=torch.zeros_like(current,dtype=torch.float32)
        count=torch.zeros((1,1,frames,1,1),device='cuda')
        for start,end in parts:
            x=pipe.scheduler.scale_model_input(current[:,:,start:end],t)
            embedding=prompt.repeat_interleave(end-start,dim=0)
            if guidance>1:
                x=torch.cat([x,x]);embedding=torch.cat([negative,prompt]).repeat_interleave(end-start,dim=0)
            eps=pipe.unet(x,t,encoder_hidden_states=embedding).sample
            if guidance>1:
                uncond,cond=eps.chunk(2);eps=uncond+guidance*(cond-uncond)
            weight=(torch.ones(end-start,device='cuda') if len(parts)==1 else torch.hann_window(end-start,periodic=False,device='cuda').clamp_min(.001)).view(1,1,-1,1,1)
            prediction[:,:,start:end]+=eps.float()*weight;count[:,:,start:end]+=weight
        current=pipe.scheduler.step((prediction/count).to(current.dtype),t,current,eta=0.).prev_sample
        print('denoise',i+1,'/',n,'t',int(t),'windows',len(parts),'elapsed',round(time.monotonic()-started,2),flush=True)
    assert torch.isfinite(current).all()
    return current.float().cpu(),{'strength':strength,'base_steps':base_steps,'steps':n,'timesteps':times.cpu().tolist(),
        'guidance':guidance,'seed':seed,'noise_sign':noise_sign,'unsigned_seed_noise_sha256':unsigned_hash,'frames':frames,'windows':parts,'seconds':time.monotonic()-started,
        'gpu':torch.cuda.get_device_name(0),'native_dtype':'float16','peak_vram_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
        'peak_vram_reserved_mb':torch.cuda.max_memory_reserved()/2**20,'base_float32_noise_sha256':noise_hash,
        'scheduler':dict(pipe.scheduler.config),'audio_enters_diffusion_directly':False,'trained_parameters_in_this_refinement_call':0,
        'geometry_lora':getattr(pipe,'_geometry_lora_path',None),
        'adapter_parameters_trained_before_inference':299136 if getattr(pipe,'_geometry_lora_path',None) else 0,
        'long_method':'overlap-add denoiser estimates per global scheduler step'}
