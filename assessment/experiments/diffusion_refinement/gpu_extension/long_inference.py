"""Experimental 4–24 second audio inference; existing demo/checkpoints untouched.

Use host GPU execution. Rolling GRU context avoids extrapolating recurrent state;
optional diffusion uses render/re-encode and overlapping global DDIM steps.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
HERE=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'cache/numba')
os.environ['MPLCONFIGDIR']=str(HERE/'cache/matplotlib')
sys.dont_write_bytecode=True
import cv2
import numpy as np
import soundfile as sf
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file,load_file
import extension as ext
from gpu_backend import load_pipeline,refine

def video(path,rgb,audio,offset):
    assert not path.exists()
    size=rgb.shape[1]
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size}x{size}','-r','8','-i','pipe:0',
        '-ss',str(offset),'-i',str(audio),'-map','0:v:0','-map','1:a:0','-t',str(len(rgb)/8),'-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)],input=rgb.tobytes(),check=True)

@torch.inference_mode()
def main(args):
    if (HERE/'STOPPED.json').exists():raise RuntimeError('Experiment stopped by user; this untested CLI was not executed.')
    assert torch.cuda.is_available(),'CUDA is required; run outside the restricted sandbox'
    assert Path(args.audio).is_file()
    assert 4<=args.duration<=24,'Supported experimental duration is 4–24 seconds; 8 and 16 seconds have validation diagnostics'
    assert args.start>=0 and np.isfinite(args.start)
    frames=round(args.duration*8);duration=frames/8
    root=HERE/'inference_runs';root.mkdir(exist_ok=True)
    output=root/uuid.uuid4().hex[:12];output.mkdir()
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(643);cv2.setNumThreads(1)
    print('Output',output,flush=True)
    # Preserve a 30-second context and enough audio after the selected start.
    window_start=max(0.,args.start-6.)
    offset=round((args.start-window_start)*16)/16
    decoded=output/'analysed_audio.wav'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(window_start),'-i',str(Path(args.audio).resolve()),
        '-t','30','-vn','-ac','1','-ar','44100','-c:a','pcm_f32le',str(decoded)],check=True)
    audio,sr=sf.read(decoded,dtype='float32')
    if len(audio)/sr<offset+duration-1e-4:raise ValueError('Not enough audio remains for the requested duration')
    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2))<1e-7:raise ValueError('Silent or invalid audio')
    demo_path=ext.EXPERIMENTS.parent/'demo/inference.py'
    sys.path.insert(0,str(demo_path.parent))
    spec=importlib.util.spec_from_file_location('preserved_demo_helpers',demo_path)
    demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    print('Extract features',flush=True)
    arrays,metadata=demo.extract(decoded)
    feature_path=output/'features.npz';np.savez_compressed(feature_path,**arrays)
    feature_path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    if args.context:
        with np.load(args.context) as f:context=f['embedding'].copy()
    else:
        print('Encode global audio context',flush=True)
        context=demo.context_embedding(decoded,'cuda').numpy()
    context_path=output/'context.npz';np.savez_compressed(context_path,embedding=context)
    row={'features_path':str(feature_path),'global_audio_context_path':str(context_path),
         'feature_start_frame':int(round(offset*16)),'audio_path':str(decoded),'audio_start_seconds':offset}
    music,context=ext.features(row,frames)
    model=ext.TemporalPrior().cuda().eval()
    checkpoint=ext.EXPERIMENTS/'temporal_phase/model.safetensors';model.load_state_dict(load_file(checkpoint))
    angle=np.deg2rad(args.visual_phase)
    phase=torch.tensor([[np.sin(angle),np.cos(angle)]],dtype=torch.float32,device='cuda')
    latent=ext.predict(model,music,context,phase,'rolling')
    save_file({'latents':latent},output/'predictor_latents.safetensors')
    del model,music,context;gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    rgb=ext.decode(vae,latent);video(output/'predictor.mp4',rgb,decoded,offset)
    beats=arrays['beat_times']-offset;beats=beats[(beats>=0)&(beats<duration)]
    report={'duration':duration,'frames':frames,'fps':8,'input':str(Path(args.audio).resolve()),
        'actual_start':window_start+offset,'visual_phase':args.visual_phase,'predictor':'preserved temporal-phase GRU, rolling 32-frame context',
        'predictor_metrics':ext.diagnostics(rgb,beats),'cached_context':args.context,
        'diffusion':None,'limitations':['Long-context method is an inference extension, not newly trained long-video generation.',
             'Timing uses estimated beats and cannot establish semantic quality or mood accuracy.',
             '8 and 16 seconds have single-recording validation diagnostics; broader generalization is unverified.',
             'Audio enters diffusion through the predictor video only; no direct diffusion audio encoder.']}
    if args.diffusion:
        large=np.stack([cv2.resize(f,(256,256),interpolation=cv2.INTER_CUBIC) for f in rgb])
        video(output/'predictor_256.mp4',large,decoded,offset)
        init=ext.encode(vae,large);save_file({'latents':init},output/'reencoded_latents.safetensors')
        roundtrip=ext.decode(vae,init);video(output/'roundtrip_256.mp4',roundtrip,decoded,offset)
        del vae;gc.collect();torch.cuda.empty_cache()
        pipe=load_pipeline(lora_path=Path(args.geometry_lora) if args.geometry_lora else None)
        result,diff_report=refine(pipe,init,args.strength,100,args.guidance)
        save_file({'latents':result},output/'refined_latents.safetensors')
        del pipe;gc.collect();torch.cuda.empty_cache()
        vae=AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
        refined=ext.decode(vae,result);video(output/'diffusion_256.mp4',refined,decoded,offset)
        report['diffusion']={**diff_report,'geometry_lora':args.geometry_lora,'metrics':ext.diagnostics(refined,beats),
            'roundtrip_metrics':ext.diagnostics(roundtrip,beats),'rgb_mae_vs_roundtrip':float(np.abs(refined.astype(float)-roundtrip).mean()/255)}
        inputs=['predictor_256','roundtrip_256','diffusion_256'];cmd=['ffmpeg','-v','error','-nostdin'];filters=[]
        for i,name in enumerate(inputs):
            cmd+=['-i',str(output/f'{name}.mp4')]
            filters.append(f'[{i}:v]pad=256:284:0:28:black,drawtext=text={name}:x=8:y=8:fontsize=14:fontcolor=white[v{i}]')
        filters.append('[v0][v1][v2]hstack=inputs=3[v]')
        cmd+=['-filter_complex',';'.join(filters),'-map','[v]','-map','0:a:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(output/'comparison.mp4')]
        subprocess.run(cmd,check=True)
    report['seconds_total']=time.monotonic()-started
    (output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(output),**report},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('--duration',type=float,default=8.);p.add_argument('--start',type=float,default=0.)
    p.add_argument('--diffusion',action='store_true');p.add_argument('--strength',type=float,default=.06);p.add_argument('--guidance',type=float,default=1.)
    p.add_argument('--geometry-lora');p.add_argument('--visual-phase',type=float,default=0.);p.add_argument('--context',help='Optional precomputed context for a known validation recording')
    main(p.parse_args())
