"""Validation-only frozen latent video-to-video pilot. Existing inputs are read-only.

Stages keep the large denoiser and float32 VAE out of memory simultaneously.
All outputs are exclusive-create or skipped on resume; no checkpoint is trained.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import MethodType

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('IE643_PRETRAINED_ROOT', str(Path(__file__).resolve().parent.joinpath('../../../../project').resolve())))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(HERE.parent.parent))
import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from train_temporal_prior import TemporalPrior
from train_audio_latent_prior import sample_rhythm
from train_contrastive import normalize_features

CHECKPOINT = HERE.parent.parent/'temporal_phase/model.safetensors'
PROMPT = 'abstract glowing concentric rings on a dark background'
STRENGTHS = [.10, .20, .30]

def cpu_float32_arithmetic(model):
    """Keep compact weights; use optimized float32 CPU kernels, cast outputs back.

    No parameter changes, training, or architectural changes. This avoids very
    slow emulated bf16 matrix/conv kernels on AVX2-only hardware. It is an explicit
    numerical configuration, not a promise of bitwise equality to native bf16.
    """
    def linear(self, input):
        x=input
        return torch.nn.functional.linear(x.float(),self.weight.float(),
                    self.bias.float() if self.bias is not None else None).to(x.dtype)
    def conv(self, input):
        x=input
        return self._conv_forward(x.float(),self.weight.float(),
                    self.bias.float() if self.bias is not None else None).to(x.dtype)
    counts={'linear':0,'conv2d':0}
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):
            module.forward=MethodType(linear,module);counts['linear']+=1
        elif isinstance(module,torch.nn.Conv2d):
            module.forward=MethodType(conv,module);counts['conv2d']+=1
    return counts

def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def features(row):
    first = row['feature_start_frame']; positions = first+np.arange(32)*2
    with np.load(ROOT/row['features_path']) as f:
        onset = sample_rhythm(f['onset_strength'], positions, first, 2, 'causal_peak')
        beat = sample_rhythm(f['beat_impulse'], positions, first, 2, 'causal_peak')
        tempo = json.loads((ROOT/row['features_path']).with_suffix('.json').read_text())['tempo_bpm']
        item = {'rhythm': torch.tensor(np.stack([onset, beat, np.full(32, tempo)], -1), dtype=torch.float32),
                'melody': torch.tensor(np.concatenate([f['chroma'][positions], f['dominant_pitch_hz'][positions,None], f['pitch_confidence'][positions,None]], -1)),
                'dynamics': torch.tensor(np.concatenate([f['rms'][positions,None], f['band_energy'][positions]], -1))}
    x = torch.cat(normalize_features(item, torch.arange(32)), -1).float()[None]
    with np.load(ROOT/row['global_audio_context_path']) as f:
        context = torch.tensor(f['embedding'].copy())[None]
    return x, context

@torch.inference_mode()
def prepare():
    if (HERE/'config.json').exists() or (HERE/'predictor_latents.safetensors').exists():
        raise FileExistsError('Preparation already exists; use a new experiment directory for a fresh reproduction.')
    # Read manifest metadata, then load features for exactly two validation rows.
    rows = [r for r in map(json.loads, (ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())
            if r['split']=='validation' and r['style']=='rings']
    row = rows[0]; wrong = next(r for r in rows if r['track_id']!=row['track_id'])
    x,c = features(row); wx,wc = features(wrong)
    meta = json.loads((ROOT/row['renderer_parameters_path']).read_text())
    angle = float(np.random.default_rng(meta['seed']).uniform(0, 2*math.pi))
    phase = torch.tensor([[math.sin(angle),math.cos(angle)]])
    model = TemporalPrior().eval(); model.load_state_dict(load_file(CHECKPOINT))
    started = time.monotonic()
    z = {'correct':model(x,c,phase), 'wrong_audio':model(wx,wc,phase),
         'shifted_audio':model(x.roll(8,1),c,phase)}
    save_file(z, HERE/'predictor_latents.safetensors')
    config = {'team':'Nullspace','member':'Salaj Sonar','selection':'first validation rings row in saved manifest; wrong is first other validation recording',
              'row':row,'wrong_row':wrong,'frames':32,'fps':8,'size':128,'seed':643,'strengths':STRENGTHS,
              'num_inference_steps':20,'guidance_scale':1.,'eta':0.,'prompt':PROMPT,
              'phase':'same renderer-seeded phase for every condition','shift':'circular roll of 8 music-feature frames (1 second); context unchanged',
              'checkpoint_sha256':digest(CHECKPOINT),'predictor_seconds_all_three':time.monotonic()-started,
              'scope':'one validation recording, exploratory; no fresh_evaluation or test tensors loaded',
              'stop_rule':'no expansion unless visible benefit with timing and stability preserved; stop at 3 hours for this pilot, below half-day ceiling'}
    write_json(HERE/'config.json',config)
    print('Prepared',row['id'],'wrong',wrong['id'],flush=True)

@torch.inference_mode()
def refine(max_cases, only_correct=False):
    import diffusers
    from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler, UNet2DConditionModel, UNetMotionModel, AutoencoderKL
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import CLIPTextModel, CLIPTokenizer
    started=time.monotonic()
    assert torch.cuda.is_available(), 'GPU confirmation requires CUDA; do not fall back to CPU'
    device='cuda'
    dtype=torch.float16 if device=='cuda' else torch.bfloat16
    print('Loading local frozen models',device,dtype,flush=True)
    # Standard from_unet2d transiently allocates a float32 duplicate and was killed
    # on this host. Build the exact native configuration on meta, then stream each
    # tensor from the two immutable safetensors checkpoints into one model.
    with init_empty_weights():
        base=UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(ROOT/'models/diffusion/sd15/unet'))
        adapter=MotionAdapter.from_config(MotionAdapter.load_config(ROOT/'models/diffusion/animatediff'))
        unet=UNetMotionModel.from_unet2d(base,adapter,load_weights=False)
    del base,adapter
    expected=unet.state_dict();loaded=set();weight_files=[]
    for directory in [ROOT/'models/diffusion/sd15/unet',ROOT/'models/diffusion/animatediff']:
        weight=directory/'diffusion_pytorch_model.fp16.safetensors'
        weight_files.append(str(weight))
        with safe_open(weight,framework='pt',device='cpu') as f:
            for key in f.keys():
                assert key in expected and key not in loaded,key
                tensor=f.get_tensor(key)
                assert tensor.shape==expected[key].shape,(key,tensor.shape,expected[key].shape)
                set_module_tensor_to_device(unet,key,'cpu',value=tensor,dtype=dtype)
                loaded.add(key)
        del tensor;gc.collect()
    assert loaded==set(expected),(set(expected)-loaded)
    assert not any(p.is_meta for p in unet.parameters())
    del expected
    print('Verified and loaded',len(loaded),'tensors',flush=True)
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15',subfolder='vae',variant='fp16',torch_dtype=dtype,local_files_only=True)
    text=CLIPTextModel.from_pretrained(ROOT/'models/diffusion/sd15/text_encoder',variant='fp16',torch_dtype=dtype,local_files_only=True)
    tokenizer=CLIPTokenizer.from_pretrained(ROOT/'models/diffusion/sd15/tokenizer',local_files_only=True)
    scheduler=DDIMScheduler.from_pretrained(ROOT/'models/diffusion/sd15/scheduler',local_files_only=True)
    pipe=AnimateDiffVideoToVideoPipeline(vae=vae,text_encoder=text,tokenizer=tokenizer,unet=unet,motion_adapter=None,scheduler=scheduler)
    del vae,text,tokenizer,unet;gc.collect()
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,beta_schedule='linear',clip_sample=False,
                                           timestep_spacing='linspace',steps_offset=1)
    pipe.unet.requires_grad_(False).eval()
    cpu_arithmetic=cpu_float32_arithmetic(pipe.unet) if device=='cpu' else None
    if device=='cpu':cpu_float32_arithmetic(pipe.text_encoder)
    print('CPU float32 arithmetic modules',cpu_arithmetic,flush=True)
    pipe.enable_vae_slicing()
    pipe.enable_sequential_cpu_offload(gpu_id=0)
    embeds,_=pipe.encode_prompt(PROMPT,device,1,False)
    assert embeds.shape==(1,77,768),embeds.shape
    # encode_prompt is the native token-sequence path; no pooling, no audio.
    z=load_file(HERE/'predictor_latents.safetensors')
    (HERE/'refined_latents').mkdir(exist_ok=True)
    jobs=[(s,k) for s in STRENGTHS for k in (['correct'] if only_correct else ['correct','wrong_audio','shifted_audio'])]
    completed=0
    for strength,key in jobs:
        stem=f'{key}_s{strength:.2f}'
        path=HERE/'refined_latents'/f'{stem}.safetensors'
        if path.exists():continue
        if max_cases and completed>=max_cases:break
        if time.monotonic()-started>3*3600:break
        tick=time.monotonic()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        pipe.scheduler.set_timesteps(20,device=device)
        timesteps,n=pipe.get_timesteps(20,pipe.scheduler.timesteps,strength,device)
        assert n==int(20*strength) and n>0
        generator=torch.Generator(device=device).manual_seed(643)
        clean=z[key].to(device,dtype=dtype)
        # Fixed float32 noise is identical across conditions/strengths, then cast.
        base_noise=torch.randn(clean.shape,generator=torch.Generator(device='cpu').manual_seed(643),dtype=torch.float32)
        noise=base_noise.to(device=device,dtype=dtype)
        noisy=pipe.scheduler.add_noise(clean,noise,timesteps[:1])
        def callback(p,i,t,kwargs):
            print(stem,'step',i+1,'t',int(t),'elapsed',round(time.monotonic()-tick,1),flush=True)
            if time.monotonic()-started>3*3600:raise TimeoutError('Pilot ceiling reached')
            return kwargs
        result=pipe(latents=noisy,prompt_embeds=embeds,height=128,width=128,strength=strength,
                    num_inference_steps=20,enforce_inference_steps=False,guidance_scale=1.,eta=0.,
                    generator=generator,output_type='latent',callback_on_step_end=callback).frames
        assert torch.isfinite(result).all()
        save_file({'latents':result.float().cpu().contiguous()},path)
        report={'condition':key,'strength':strength,'denoising_steps':n,'timesteps':timesteps.cpu().tolist(),
                'seconds':time.monotonic()-tick,'device':device,'dtype':str(dtype),
                'peak_process_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'peak_vram_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
                'diffusers':diffusers.__version__,'torch':torch.__version__,'text_shape':list(embeds.shape),
                'scheduler':dict(pipe.scheduler.config),'latent_mae':float((result.float()-clean.float()).abs().mean()),
                'noise_sha256':hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(),
                'base_float32_noise_sha256':hashlib.sha256(base_noise.numpy().tobytes()).hexdigest(),
                'offload':'sequential CPU offload; native CUDA float16 operations',
                'peak_vram_reserved_mb':torch.cuda.max_memory_reserved()/2**20,
                'gpu_name':torch.cuda.get_device_name(0),
                'audio_input_to_diffusion':False,'trained_parameters':0}
        report['cpu_float32_arithmetic_modules']=cpu_arithmetic
        write_json(path.with_suffix('.json'),report)
        completed+=1;print('Saved',stem,json.dumps(report),flush=True)
    if not (HERE/'environment.json').exists():
        write_json(HERE/'environment.json',{'python':sys.executable,'torch':torch.__version__,'diffusers':diffusers.__version__,
                  'cuda_available':torch.cuda.is_available(),'device':device,'dtype':str(dtype),'load_and_run_seconds':time.monotonic()-started,
                  'cpu_float32_arithmetic_modules':cpu_arithmetic,
                  'pipeline_source':str(Path(__import__('inspect').getfile(AnimateDiffVideoToVideoPipeline))),
                  'pipeline_source_sha256':digest(Path(__import__('inspect').getfile(AnimateDiffVideoToVideoPipeline)))})

def video_frames(path):
    cap=cv2.VideoCapture(str(path));out=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        out.append(frame)
    cap.release();return np.stack(out)

@torch.inference_mode()
def evaluate():
    from diffusers import AutoencoderKL
    from audit_video_metrics import matched_score,continuity
    from evaluate_e0 import video_events
    config=json.loads((HERE/'config.json').read_text());row=config['row']
    with np.load(ROOT/row['features_path']) as f:
        beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
    (HERE/'videos').mkdir(exist_ok=True)
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    latents={f'{k}_predictor':v for k,v in load_file(HERE/'predictor_latents.safetensors').items()}
    for path in sorted((HERE/'refined_latents').glob('*.safetensors')):
        latents[path.stem]=load_file(path)['latents']
    results=[]
    for name,z in latents.items():
        path=HERE/'videos'/f'{name}.mp4'
        if not path.exists():
            frames=[]
            for chunk in z[0].permute(1,0,2,3).split(2):
                frames.extend(((vae.decode(chunk.to('cuda')/vae.config.scaling_factor).sample/2+.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy())
            # Lossless RGB frames retained for an artifact-independent pixel comparison.
            np.savez_compressed(HERE/'videos'/f'{name}.npz',rgb=np.stack(frames))
            subprocess.run(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s','128x128','-r','8','-i','pipe:0',
                            '-ss',str(row['audio_start_seconds']),'-i',str(ROOT/row['audio_path']),'-map','0:v:0','-map','1:a:0','-t','4',
                            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)],
                           input=np.stack(frames).tobytes(),check=True)
        events,signal=video_events(path,fps=8)
        rgb=np.load(path.with_suffix('.npz'))['rgb'].astype(float)/255
        condition=name.split('_s')[0] if '_s' in name else name.removesuffix('_predictor')
        base=np.load(HERE/'videos'/f'{condition}_predictor.npz')['rgb'].astype(float)/255
        base_events,base_signal=video_events(HERE/'videos'/f'{condition}_predictor.mp4',fps=8)
        item={'name':name,'timing':matched_score(beats,events),'shifted_reference_200ms':matched_score(beats+.2,events),
              'events_seconds':events.tolist(),'continuity':continuity(path),'rgb_mae_vs_own_predictor':float(np.abs(rgb-base).mean()),
              'frame_change_correlation_vs_own_predictor':float(np.corrcoef(signal[1:],base_signal[1:])[0,1]),
              'sharpness_laplacian':float(np.mean([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for f in (rgb*255).astype(np.uint8)]))}
        results.append(item)
    result_path=HERE/f'results_{len(results):02d}.json'
    if not result_path.exists():write_json(result_path,{'config':config,'estimated_beats_seconds':beats.tolist(),'results':results,
            'limitations':['One validation recording; no population inference.','Timing uses estimated beats, not semantic quality or mood accuracy.',
                           'Warp residual rewards blur/static output; sharpness rewards noise.','No human ratings. Audio enters GRU only; diffusion is frozen.']})
    # Each comparison has the exact same original audio track across all panels.
    for strength in STRENGTHS:
        pair=[HERE/'videos'/'correct_predictor.mp4',HERE/'videos'/f'correct_s{strength:.2f}.mp4']
        pair_out=HERE/'videos'/f'pair_s{strength:.2f}.mp4'
        if all(p.exists() for p in pair) and not pair_out.exists():
            subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(pair[0]),'-i',str(pair[1]),
                '-filter_complex',f'[0:v]scale=384:384,pad=384:414:0:30:black,drawtext=text=Predictor correct:x=10:y=8:fontsize=18:fontcolor=white[a];[1:v]scale=384:384,pad=384:414:0:30:black,drawtext=text=Refined strength {strength:.2f}:x=10:y=8:fontsize=18:fontcolor=white[b];[a][b]hstack=inputs=2[v]',
                '-map','[v]','-map','0:a:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(pair_out)],check=True)
        paths=[HERE/'videos'/f'correct_predictor.mp4',HERE/'videos'/f'correct_s{strength:.2f}.mp4',
               HERE/'videos'/f'wrong_audio_s{strength:.2f}.mp4',HERE/'videos'/f'shifted_audio_s{strength:.2f}.mp4']
        out=HERE/'videos'/f'comparison_s{strength:.2f}.mp4'
        if all(p.exists() for p in paths) and not out.exists():
            args=['ffmpeg','-v','error','-nostdin']
            for p in paths:args+=['-i',str(p)]
            labels=['Predictor / correct','Refined / correct','Refined / wrong','Refined / shift 1s']
            filters=[]
            for i,label in enumerate(labels):
                filters.append(f'[{i}:v]scale=256:256,pad=256:284:0:28:black,drawtext=text={label}:x=8:y=8:fontsize=15:fontcolor=white[v{i}]')
            filters.append('[v0][v1][v2][v3]hstack=inputs=4[v]')
            args+=['-filter_complex',';'.join(filters),'-map','[v]','-map','0:a:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(out)]
            subprocess.run(args,check=True)
    control_names=['correct_predictor','wrong_audio_predictor','shifted_audio_predictor',
                   'correct_s0.10','wrong_audio_s0.10','shifted_audio_s0.10']
    control_out=HERE/'videos'/'controls_predictor_vs_refined_s0.10.mp4'
    if all((HERE/'videos'/f'{n}.mp4').exists() for n in control_names) and not control_out.exists():
        args=['ffmpeg','-v','error','-nostdin'];filters=[]
        labels=['Predictor correct','Predictor wrong','Predictor shifted 1s',
                'Refined 0.10 correct','Refined 0.10 wrong','Refined 0.10 shifted 1s']
        for i,(name,label) in enumerate(zip(control_names,labels)):
            args+=['-i',str(HERE/'videos'/f'{name}.mp4')]
            filters.append(f'[{i}:v]scale=256:256,pad=256:284:0:28:black,drawtext=text={label}:x=8:y=8:fontsize=15:fontcolor=white[v{i}]')
        filters.append('[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout=0_0|256_0|512_0|0_284|256_284|512_284[v]')
        args+=['-filter_complex',';'.join(filters),'-map','[v]','-map','0:a:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(control_out)]
        subprocess.run(args,check=True)
    # Frame grids for direct visual audit at fixed times, without selecting flattering frames.
    for strength in STRENGTHS:
        names=['correct_predictor',f'correct_s{strength:.2f}',f'wrong_audio_s{strength:.2f}',f'shifted_audio_s{strength:.2f}']
        if not all((HERE/'videos'/f'{n}.npz').exists() for n in names):
            names=names[:2]
            if not all((HERE/'videos'/f'{n}.npz').exists() for n in names):continue
        strips=[]
        for name in names:
            rgb=np.load(HERE/'videos'/f'{name}.npz')['rgb']
            strip=np.concatenate([cv2.resize(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR),(192,192)) for i in [4,12,20,28]],1)
            strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(strip,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            strips.append(strip)
        out=HERE/'videos'/f'contact_s{strength:.2f}_{len(names)}rows.png'
        if not out.exists():cv2.imwrite(str(out),np.concatenate(strips,0))
    sweep=HERE/'videos'/'strength_sweep_contact.png'
    names=['correct_predictor']+[f'correct_s{s:.2f}' for s in STRENGTHS]
    if not sweep.exists() and all((HERE/'videos'/f'{n}.npz').exists() for n in names):
        strips=[]
        for name in names:
            rgb=np.load(HERE/'videos'/f'{name}.npz')['rgb']
            strip=np.concatenate([cv2.resize(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR),(192,192)) for i in [4,12,20,28]],1)
            strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(strip,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            strips.append(strip)
        cv2.imwrite(str(sweep),np.concatenate(strips,0))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','refine','evaluate']);parser.add_argument('--max-cases',type=int,default=0);parser.add_argument('--only-correct',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(643);cv2.setNumThreads(1)
    if args.stage=='prepare':prepare()
    elif args.stage=='refine':refine(args.max_cases,args.only_correct)
    else:evaluate()
