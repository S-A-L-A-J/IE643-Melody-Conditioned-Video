"""Validation-only longer predictor and rendered-video diffusion experiment."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
EXPERIMENTS=BASE.parent
ROOT=Path(os.environ.get('IE643_PRETRAINED_ROOT', str(Path(__file__).resolve().parent.joinpath('../../../../project').resolve())))
sys.path[:0]=[str(EXPERIMENTS),str(ROOT/'scripts')]
import cv2
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from diffusers import AutoencoderKL
from train_temporal_prior import TemporalPrior
from train_audio_latent_prior import sample_rhythm
from train_contrastive import normalize_features
from audit_video_metrics import matched_score
from gpu_backend import load_pipeline,refine,windows
from scipy.signal import find_peaks

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

def features(row,frames):
    first=row['feature_start_frame'];positions=first+np.arange(frames)*2
    with np.load(ROOT/row['features_path']) as f:
        assert positions[-1]<len(f['times'])
        onset=sample_rhythm(f['onset_strength'],positions,first,2,'causal_peak')
        beat=sample_rhythm(f['beat_impulse'],positions,first,2,'causal_peak')
        tempo=json.loads((ROOT/row['features_path']).with_suffix('.json').read_text())['tempo_bpm']
        item={'rhythm':torch.tensor(np.stack([onset,beat,np.full(frames,tempo)],-1),dtype=torch.float32),
              'melody':torch.tensor(np.concatenate([f['chroma'][positions],f['dominant_pitch_hz'][positions,None],f['pitch_confidence'][positions,None]],-1)),
              'dynamics':torch.tensor(np.concatenate([f['rms'][positions,None],f['band_energy'][positions]],-1))}
    x=torch.cat(normalize_features(item,torch.arange(frames)),-1).float()[None].cuda()
    with np.load(ROOT/row['global_audio_context_path']) as f:c=torch.tensor(f['embedding'].copy())[None].cuda()
    return x,c

@torch.inference_mode()
def predict(model,x,c,phase,method):
    if method=='direct':return model(x,c,phase).cpu()
    if method=='concat':return torch.cat([model(part,c,phase).cpu() for part in x.split(32,dim=1)],dim=2)
    if method=='rolling':
        initial=model(x[:,:32],c,phase).cpu()
        remaining=[]
        for indices in torch.arange(32,x.shape[1]).split(16):
            batch=torch.cat([x[:,int(i)-31:int(i)+1] for i in indices],dim=0)
            output=model(batch,c.expand(len(indices),-1),phase.expand(len(indices),-1))
            remaining.append(output[:,:,-1].permute(1,0,2,3)[None].cpu())
        return torch.cat([initial]+remaining,dim=2)
    frames=x.shape[1];total=torch.zeros((1,4,frames,16,16));counts=torch.zeros((1,1,frames,1,1))
    for start,end in windows(frames):
        z=model(x[:,start:end],c,phase).cpu()
        w=torch.hann_window(end-start,periodic=False).clamp_min(.001).view(1,1,-1,1,1)
        total[:,:,start:end]+=z*w;counts[:,:,start:end]+=w
    return total/counts

def mux(name,rgb,row):
    out=HERE/'videos'/f'{name}.mp4'
    if out.exists():return
    size=rgb.shape[1];duration=len(rgb)/8
    np.savez_compressed(out.with_suffix('.npz'),rgb=rgb)
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size}x{size}','-r','8','-i','pipe:0',
          '-ss',str(row['audio_start_seconds']),'-i',str(ROOT/row['audio_path']),'-map','0:v:0','-map','1:a:0','-t',str(duration),
          '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(out)],input=rgb.tobytes(),check=True)

@torch.inference_mode()
def decode(vae,z):
    chunks=[]
    for part in z[0].permute(1,0,2,3).split(2):
        rgb=((vae.decode(part.cuda()/vae.config.scaling_factor).sample/2+.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
        chunks.append(rgb)
    return np.concatenate(chunks)

@torch.inference_mode()
def encode(vae,rgb):
    chunks=[]
    tensor=torch.tensor(rgb.astype(np.float32)/127.5-1).permute(0,3,1,2)
    for part in tensor.split(2):chunks.append((vae.encode(part.cuda()).latent_dist.mode()*vae.config.scaling_factor).cpu())
    return torch.cat(chunks).permute(1,0,2,3)[None].contiguous()

@torch.inference_mode()
def prepare():
    assert not (HERE/'prepared.json').exists(),'Already prepared; preserve outputs'
    (HERE/'inputs').mkdir(exist_ok=True);(HERE/'videos').mkdir(exist_ok=True);(HERE/'refined').mkdir(exist_ok=True)
    old=json.loads((BASE/'config.json').read_text());row=old['row'];wrong=old['wrong_row']
    meta=json.loads((ROOT/row['renderer_parameters_path']).read_text())
    angle=float(np.random.default_rng(meta['seed']).uniform(0,2*np.pi))
    phase=torch.tensor([[np.sin(angle),np.cos(angle)]],dtype=torch.float32,device='cuda')
    model=TemporalPrior().cuda().eval();model.load_state_dict(load_file(EXPERIMENTS/'temporal_phase/model.safetensors'))
    z={};timings={}
    for seconds in [8,16]:
        x,c=features(row,seconds*8);wx,wc=features(wrong,seconds*8)
        tick=time.monotonic()
        for method in ['direct','concat','overlap']:
            z[f'd{seconds:02d}_correct_{method}']=predict(model,x,c,phase,method)
        if seconds==8:
            z['d08_wrong_overlap']=predict(model,wx,wc,phase,'overlap')
            z['d08_shift_overlap']=predict(model,x.roll(8,1),c,phase,'overlap')
        timings[str(seconds)]=time.monotonic()-tick
    z['d04_correct_raw128']=load_file(BASE/'predictor_latents.safetensors')['correct']
    save_file(z,HERE/'predictor_latents.safetensors')
    del model,x,c,wx,wc;gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    for name,latent in z.items():
        rgb=decode(vae,latent)
        if name=='d04_correct_raw128':rgb=np.load(BASE/'videos/correct_predictor.npz')['rgb']
        mux(name,rgb,row)
        if 'overlap' in name or name=='d04_correct_raw128':
            if name=='d04_correct_raw128':
                raw=HERE/'inputs/d04_raw128.safetensors'
                if not raw.exists():save_file({'latents':latent},raw)
                r128=encode(vae,rgb);save_file({'latents':r128},HERE/'inputs/d04_reencoded128.safetensors')
                mux('d04_roundtrip128',decode(vae,r128),row)
            large=np.stack([cv2.resize(frame,(256,256),interpolation=cv2.INTER_CUBIC) for frame in rgb])
            prefix=name.replace('_raw128','').replace('_overlap','')
            mux(prefix+'_upscaled256',large,row)
            rec=encode(vae,large);save_file({'latents':rec},HERE/'inputs'/f'{prefix}_reencoded256.safetensors')
            mux(prefix+'_roundtrip256',decode(vae,rec),row)
        print('Prepared',name,len(rgb),'frames',flush=True)
    dump(HERE/'prepared.json',{'row':row,'wrong_row':wrong,'durations':[8,16],'fps':8,'grus':'same checkpoint, direct extrapolation / concat four-second windows / Hann overlap two-second hop',
        'predictor_seconds':timings,'validation_only':True,'fresh_evaluation_used':False,'new_training':False,
        'reference_audio':'correct recording for every media condition','vae_encode':'posterior mode, float32 CUDA; no posterior sampling',
        'upsampling':'bicubic to 256, a baseline not claimed as added detail','decode':'float32 CUDA, same VAE for all extension outputs'})

@torch.inference_mode()
def prepare_rolling():
    assert not (HERE/'rolling_prepared.json').exists(),'Rolling preparation already exists'
    old=json.loads((BASE/'config.json').read_text());row=old['row'];wrong=old['wrong_row']
    meta=json.loads((ROOT/row['renderer_parameters_path']).read_text())
    angle=float(np.random.default_rng(meta['seed']).uniform(0,2*np.pi))
    phase=torch.tensor([[np.sin(angle),np.cos(angle)]],dtype=torch.float32,device='cuda')
    model=TemporalPrior().cuda().eval();model.load_state_dict(load_file(EXPERIMENTS/'temporal_phase/model.safetensors'))
    outputs={};timings={}
    for seconds in [8,16]:
        x,c=features(row,seconds*8);tick=time.monotonic()
        outputs[f'd{seconds:02d}_correct_rolling']=predict(model,x,c,phase,'rolling')
        if seconds==8:
            wx,wc=features(wrong,seconds*8)
            outputs['d08_wrong_rolling']=predict(model,wx,wc,phase,'rolling')
            outputs['d08_shift_rolling']=predict(model,x.roll(8,1),c,phase,'rolling')
        timings[str(seconds)]=time.monotonic()-tick
    save_file(outputs,HERE/'rolling_latents.safetensors');del model;gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    for name,z in outputs.items():
        rgb=decode(vae,z);mux(name,rgb,row)
        large=np.stack([cv2.resize(f,(256,256),interpolation=cv2.INTER_CUBIC) for f in rgb])
        mux(name+'_upscaled256',large,row)
        rec=encode(vae,large);save_file({'latents':rec},HERE/'inputs'/f'{name}_reencoded256.safetensors')
        mux(name+'_roundtrip256',decode(vae,rec),row)
        print('Prepared rolling context',name,flush=True)
    dump(HERE/'rolling_prepared.json',{'row':row,'wrong_row':wrong,'durations':[8,16],'predictor_seconds':timings,
        'method':'Use trained 32-frame GRU context. Emit first window, then emit only last prediction for each rolling past-32-frame window.',
        'reason':'Direct extrapolation visibly drifts; overlap of different geometric states ghosts; concatenation resets appearance.',
        'limitation':'After startup, normalized time feature is the trained end-of-window value. Geometric time drift can flatten; rhythm comes from ongoing local audio.',
        'training':False,'selection_population':'same validation recording, no fresh data used'})

JOBS=[
    ('raw128_native_control','d04_raw128',.1,20,1.),
    ('raw128_fine','d04_raw128',.06,100,1.),
    ('reenc128_fine','d04_reencoded128',.06,100,1.),
    ('reenc256_003','d04_correct_reencoded256',.03,100,1.),
    ('reenc256_006','d04_correct_reencoded256',.06,100,1.),
    ('reenc256_010','d04_correct_reencoded256',.10,100,1.),
    ('reenc256_006_cfg3','d04_correct_reencoded256',.06,100,3.),
]

@torch.inference_mode()
def run(group):
    lora=None
    if group.startswith('adapt'):
        if (HERE/'selected_adapter.json').exists():lora=Path(json.loads((HERE/'selected_adapter.json').read_text())['checkpoint'])
        else:
            chosen=json.loads((HERE/'geometry_lora/run.json').read_text())['selected']['checkpoint']
            lora=HERE/'geometry_lora'/chosen
    pipe=load_pipeline(lora_path=lora)
    long_jobs=[(f'd08_{c}_refined_rolling',f'd08_{c}_rolling_reencoded256',.06,100,1.) for c in ['correct','wrong','shift']]+[('d16_correct_refined_rolling','d16_correct_rolling_reencoded256',.06,100,1.)]
    final_short=[('adapt128_raw128_006','d04_raw128',.06,100,1.)]+[(f'adapt128_reenc256_{s}','d04_correct_reencoded256',strength,100,1.) for s,strength in [('003',.03),('006',.06),('010',.10)]]
    if group=='short':jobs=JOBS
    elif group=='adapt_final':jobs=final_short+[('adapt_'+name,source,s,n,g) for name,source,s,n,g in long_jobs]
    elif group in ['adapt_short','adapt_short128']:
        prefix='adapt128' if group=='adapt_short128' else 'adapt'
        jobs=[(f'{prefix}_raw128_006','d04_raw128',.06,100,1.)]+[(f'{prefix}_reenc256_{s}', 'd04_correct_reencoded256',strength,100,1.) for s,strength in [('003',.03),('006',.06),('010',.10)]]
    elif group=='adapt_long':jobs=[('adapt_'+name,source,s,n,g) for name,source,s,n,g in long_jobs]
    else:jobs=long_jobs
    for name,source,strength,steps,cfg in jobs:
        out=HERE/'refined'/f'{name}.safetensors'
        if out.exists():continue
        z=load_file(HERE/'inputs'/f'{source}.safetensors')['latents']
        print('Start',name,flush=True);result,report=refine(pipe,z,strength,steps,cfg)
        report['source']=source
        report['geometry_lora']=str(lora) if lora else None
        if name=='raw128_native_control':
            previous=load_file(BASE/'gpu_confirmation/refined_latents/correct_s0.10.safetensors')['latents']
            report['max_abs_difference_native_pipeline']=float((result-previous).abs().max())
        save_file({'latents':result},out);dump(out.with_suffix('.json'),report)
        print('Saved',name,json.dumps(report),flush=True)

def diagnostics(rgb,beats):
    # Normalize all quality/timing diagnostics to 128px to compare resolutions.
    small=np.stack([cv2.resize(f,(128,128),interpolation=cv2.INTER_AREA) for f in rgb])
    gray=np.stack([cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in small])
    changes=np.r_[0,np.abs(np.diff(gray.astype(float),axis=0)).mean((1,2))]
    prominence=max(float(np.percentile(changes[1:],75))*.18,.15)
    peaks,_=find_peaks(changes,prominence=prominence,distance=2)
    residuals=[];yy,xx=np.mgrid[:128,:128].astype(np.float32)
    for prev,current in zip(gray,gray[1:]):
        flow=cv2.calcOpticalFlowFarneback(current,prev,None,.5,3,15,3,5,1.2,0)
        mx=xx+flow[:,:,0];my=yy+flow[:,:,1];valid=(mx>=0)&(mx<127)&(my>=0)&(my<127)
        warped=cv2.remap(prev,mx,my,cv2.INTER_LINEAR)
        residuals.append(float(np.abs(warped.astype(float)-current)[valid].mean()/255))
    seam_indices=np.arange(32,len(rgb),32)
    return {'beat_f1':matched_score(beats,peaks/8),'reference_shift_200ms':matched_score(beats+.2,peaks/8),
        'events_seconds':(peaks/8).tolist(),'frame_changes':changes.tolist(),'mean_change':float(changes[1:].mean()/255),
        'warp_residual':float(np.mean(residuals)), 'sharpness':float(np.mean([cv2.Laplacian(f,cv2.CV_64F).var() for f in gray])),
        'four_second_boundary_changes':[float(changes[i]/255) for i in seam_indices],
        'limitation':'All beats estimated; optical flow/edge proxies do not establish semantic or mood accuracy; boundary changes may be valid audio responses.'}

@torch.inference_mode()
def render():
    config=json.loads((HERE/'prepared.json').read_text());row=config['row']
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    for path in sorted((HERE/'refined').glob('*.safetensors')):
        if not (HERE/'videos'/f'{path.stem}.mp4').exists():mux(path.stem,decode(vae,load_file(path)['latents']),row)
    with np.load(ROOT/row['features_path']) as f:beats=f['beat_times']-row['audio_start_seconds']
    records={}
    for path in sorted((HERE/'videos').glob('*.npz')):
        rgb=np.load(path)['rgb'];b=beats[(beats>=0)&(beats<len(rgb)/8)]
        records[path.stem]={'frames':len(rgb),'seconds':len(rgb)/8,'size':rgb.shape[1],**diagnostics(rgb,b)}
        if path.stem.startswith('reenc256'):
            base=np.load(HERE/'videos/d04_correct_roundtrip256.npz')['rgb']
            original=np.load(HERE/'videos/d04_correct_upscaled256.npz')['rgb']
            records[path.stem]['rgb_mae_vs_roundtrip']=float(np.abs(rgb.astype(float)-base).mean()/255)
            records[path.stem]['rgb_mae_vs_upscaled_predictor']=float(np.abs(rgb.astype(float)-original).mean()/255)
    out=HERE/f'metrics_{len(records):02d}.json'
    if not out.exists():dump(out,{'results':records,'population':'first fixed validation recording only; settings selected here, not on fresh evaluation','human_ratings':None})
    groups={
      'short256':['d04_correct_upscaled256','d04_correct_roundtrip256','reenc256_003','reenc256_006','reenc256_010','reenc256_006_cfg3'],
      'predictor8':['d08_correct_direct','d08_correct_concat','d08_correct_overlap','d08_correct_rolling'],
      'predictor16':['d16_correct_direct','d16_correct_concat','d16_correct_overlap','d16_correct_rolling'],
      'long8':['d08_correct_rolling_upscaled256','d08_correct_rolling_roundtrip256','d08_correct_refined_rolling','d08_wrong_refined_rolling','d08_shift_refined_rolling'],
      'long16':['d16_correct_rolling_upscaled256','d16_correct_rolling_roundtrip256','d16_correct_refined_rolling']}
    groups['adapter4']=['d04_correct_upscaled256','d04_correct_roundtrip256','reenc256_006','adapt_reenc256_003','adapt_reenc256_006','adapt_reenc256_010']
    groups['adapter128_4']=['d04_correct_upscaled256','d04_correct_roundtrip256','reenc256_006','adapt_reenc256_006','adapt128_reenc256_003','adapt128_reenc256_006','adapt128_reenc256_010']
    groups['adapter8']=['d08_correct_rolling_upscaled256','d08_correct_refined_rolling','adapt_d08_correct_refined_rolling','adapt_d08_wrong_refined_rolling','adapt_d08_shift_refined_rolling']
    groups['adapter16']=['d16_correct_rolling_upscaled256','d16_correct_refined_rolling','adapt_d16_correct_refined_rolling']
    for group,names in groups.items():
        names=[n for n in names if (HERE/'videos'/f'{n}.mp4').exists()]
        if len(names)<2:continue
        output=HERE/'videos'/f'comparison_{group}_{len(names)}panels.mp4'
        if not output.exists():
            args=['ffmpeg','-v','error','-nostdin'];filters=[]
            for i,name in enumerate(names):
                args+=['-i',str(HERE/'videos'/f'{name}.mp4')]
                label=name.replace('_',' ')
                filters.append(f'[{i}:v]scale=256:256,pad=256:288:0:32:black,drawtext=text={label}:x=6:y=10:fontsize=12:fontcolor=white[v{i}]')
            filters.append(''.join(f'[v{i}]' for i in range(len(names)))+f'hstack=inputs={len(names)}[v]')
            args+=['-filter_complex',';'.join(filters),'-map','[v]','-map','0:a:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(output)]
            subprocess.run(args,check=True)
        contact=output.with_suffix('.png')
        if not contact.exists():
            strips=[]
            for name in names:
                rgb=np.load(HERE/'videos'/f'{name}.npz')['rgb']
                indices=np.linspace(4,len(rgb)-4,4).astype(int)
                strip=np.concatenate([cv2.resize(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR),(192,192)) for i in indices],1)
                strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(strip,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);strips.append(strip)
            cv2.imwrite(str(contact),np.concatenate(strips,0))
    print(json.dumps({n:{'F1':r['beat_f1']['f1'],'warp':r['warp_residual'],'sharpness':r['sharpness'],'boundary':r['four_second_boundary_changes']} for n,r in records.items()},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','rolling','short','long','adapt_short','adapt_short128','adapt_long','adapt_final','render']);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(643)
    if (HERE/'STOPPED.json').exists() and args.stage!='render':
        raise RuntimeError('Experiment stopped by user. Do not run further preparation, training or sampling without a new explicit request.')
    assert torch.cuda.is_available(),'GPU required; use execution outside sandbox'
    if args.stage=='prepare':prepare()
    elif args.stage=='rolling':prepare_rolling()
    elif args.stage=='render':render()
    else:run(args.stage)
