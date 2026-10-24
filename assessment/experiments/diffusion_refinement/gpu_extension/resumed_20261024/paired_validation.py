"""Matched +/- noise sampling; reuse positive outputs, no additional training."""
import gc
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;BASE=HERE/'validation8'
sys.path[:0]=[str(HERE),str(WORK),str(WORK.parent.parent)]
import cv2
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from paired_backend import load_pipeline,refine
import extension as ext

OUT=HERE/'paired_validation';OUT.mkdir(exist_ok=True)
(OUT/'latents').mkdir(exist_ok=True);(OUT/'videos').mkdir(exist_ok=True)

def dump(path,x):
    with path.open('x') as f:json.dump(x,f,indent=2)

@torch.inference_mode()
def main():
    assert (HERE/'RESUME_AUTHORIZATION.json').exists()
    assert (BASE/'results.json').exists(),'Finish the base comparison first'
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(643);cv2.setNumThreads(1)
    prepared=load_file(BASE/'prepared_latents.safetensors')
    records=json.loads((BASE/'prepared.json').read_text())['records']
    selection=json.loads((WORK/'selected_adapter.json').read_text())
    for variant,lora in [('frozen',None),('adapted',(WORK/'geometry_lora_continuation/lora_step128.safetensors'))]:
        pipe=load_pipeline(lora_path=lora)
        for record in records:
            i=record['index'];out=OUT/'latents'/f'{variant}_negative_{i}.safetensors'
            if not out.exists():
                value,meta=refine(pipe,prepared[f'prior_{i}'],.06,100,1.,record['seed'],noise_sign=-1.)
                meta['source_positive']=str(BASE/'latents'/f'{variant}_{i}.safetensors')
                positive_meta=json.loads((BASE/'latents'/f'{variant}_{i}.json').read_text())
                assert positive_meta['base_float32_noise_sha256']==meta['unsigned_seed_noise_sha256']
                save_file({'latents':value},out);dump(out.with_suffix('.json'),meta)
            positive=load_file(BASE/'latents'/f'{variant}_{i}.safetensors')['latents']
            negative=load_file(out)['latents']
            mean=(positive+negative)/2
            mean_path=OUT/'latents'/f'{variant}_pair_{i}.safetensors'
            if not mean_path.exists():save_file({'latents':mean.contiguous()},mean_path)
        del pipe;gc.collect();torch.cuda.empty_cache()
    vae=ext.AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    results=[]
    for record in records:
        i=record['index'];row=record['row'];prior=prepared[f'prior_{i}'];target=prepared[f'target_{i}']
        variants={v:load_file(OUT/'latents'/f'{v}_{i}.safetensors')['latents'] for v in ['frozen_pair','adapted_pair']}
        variants['adapted_pair_anchor050']=prior+.5*(variants['adapted_pair']-prior)
        target_rgb=np.load(BASE/'videos'/f'target_{i}.npz')['rgb']
        prior_rgb=np.load(BASE/'videos'/f'prior_{i}.npz')['rgb']
        with np.load(ext.ROOT/row['features_path']) as f:
            beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
        metrics={};frames={}
        for variant,z in variants.items():
            path=OUT/'videos'/f'{variant}_{i}.npz'
            if not path.exists():
                rgb=ext.decode(vae,z);old=ext.HERE;ext.HERE=OUT
                try:ext.mux(f'{variant}_{i}',rgb,row)
                finally:ext.HERE=old
            else:rgb=np.load(path)['rgb']
            frames[variant]=rgb
            metrics[variant]={'latent_mse_to_target':float((z-target).square().mean()),
                'rgb_mse_to_decoded_target':float(np.square((rgb.astype(float)-target_rgb)/255).mean()),
                'rgb_mae_vs_prior':float(np.abs(rgb.astype(float)-prior_rgb).mean()/255),**ext.diagnostics(rgb,beats)}
        results.append({'id':record['id'],'variants':metrics})
        if i in [0,3,7]:
            strips=[]
            images={'target':target_rgb,'prior':prior_rgb,**frames}
            for label,rgb in images.items():
                strip=np.concatenate([cv2.resize(cv2.cvtColor(rgb[j],cv2.COLOR_RGB2BGR),(192,192)) for j in [4,12,20,28]],1)
                strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(strip,label,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);strips.append(strip)
            path=OUT/'videos'/f'comparison_{i}.png'
            if not path.exists():cv2.imwrite(str(path),np.concatenate(strips,0))
    aggregate={}
    for variant in results[0]['variants']:
        aggregate[variant]={k:float(np.mean([r['variants'][variant][k] for r in results])) for k in ['latent_mse_to_target','rgb_mse_to_decoded_target','rgb_mae_vs_prior','warp_residual','sharpness']}
        aggregate[variant]['estimated_beat_f1']=float(np.mean([r['variants'][variant]['beat_f1']['f1'] for r in results]))
    base=json.loads((BASE/'results.json').read_text())
    comparisons={}
    for variant in aggregate:
        comparisons[variant]={other:{'latent_wins':sum(a['variants'][variant]['latent_mse_to_target']<b['variants'][other]['latent_mse_to_target'] for a,b in zip(results,base['results'])),
            'rgb_wins':sum(a['variants'][variant]['rgb_mse_to_decoded_target']<b['variants'][other]['rgb_mse_to_decoded_target'] for a,b in zip(results,base['results']))} for other in ['prior','frozen','adapted']}
    report={'population':'same eight validation groups, not independent test','aggregate':aggregate,'comparisons':comparisons,'results':results,
        'base_result':str(BASE/'results.json'),'no_new_training':True,'human_ratings':None,
        'limitations':'Antithetic averaging is an inference heuristic. Estimated beats and synthetic-target errors do not establish semantics/mood or human preference.'}
    dump(OUT/'results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)

if __name__=='__main__':main()
