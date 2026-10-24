"""Frozen candidate confirmation on validation recordings 9–11, not fresh test data."""
import gc
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path[:0]=[str(HERE),str(WORK),str(WORK.parent.parent),str(Path(__file__).resolve().parent.joinpath('../../../../../project').resolve()/'scripts')]
import cv2
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from train_temporal_prior import TemporalPrior,data
from paired_backend import load_pipeline,refine
import extension as ext
OUT=HERE/'confirmation3';OUT.mkdir(exist_ok=True)
(OUT/'latents').mkdir(exist_ok=True);(OUT/'videos').mkdir(exist_ok=True)

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

@torch.inference_mode()
def main():
    assert torch.cuda.is_available();torch.manual_seed(643);torch.set_num_threads(4);cv2.setNumThreads(1)
    chosen=json.loads((HERE/'CHOSEN_CANDIDATE.json').read_text())
    assert chosen['candidate']=='adapted_pair' and chosen['anchor_alpha']==1.
    if not (OUT/'prepared.json').exists():
        part=data()['validation'];indices=[];seen=set()
        for i,track in enumerate(part['tracks']):
            if track not in seen:indices.append(i);seen.add(track)
        assert len(indices)==11
        rows={r['id']:r for r in map(json.loads,(ext.ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
        model=TemporalPrior().cuda().eval();model.load_state_dict(load_file(ext.EXPERIMENTS/'temporal_phase/model.safetensors'))
        z={};records=[]
        for number,i in enumerate(indices[8:],start=8):
            z[f'prior_{number}']=model(part['music'][i:i+1].cuda(),part['context'][i:i+1].cuda(),part['phase'][i:i+1].cuda()).cpu()
            z[f'target_{number}']=part['target'][i:i+1]
            records.append({'index':number,'id':part['ids'][i],'row':rows[part['ids'][i]],'seed':643+number})
        save_file(z,OUT/'prepared_latents.safetensors');dump(OUT/'prepared.json',{'records':records,'split':'remaining validation recordings, not used for LoRA checkpoint or resumed candidate selection; not an independent final test'})
        del model,part,z;gc.collect();torch.cuda.empty_cache()
    prepared=load_file(OUT/'prepared_latents.safetensors');records=json.loads((OUT/'prepared.json').read_text())['records']
    for variant,lora in [('frozen_pair',None),('candidate',(WORK/'geometry_lora_continuation/lora_step128.safetensors'))]:
        pipe=load_pipeline(lora_path=lora)
        for record in records:
            i=record['index'];signed=[]
            for sign in [1,-1]:
                path=OUT/'latents'/f'{variant}_{i}_sign{sign}.safetensors'
                if not path.exists():
                    value,meta=refine(pipe,prepared[f'prior_{i}'],chosen['strength'],chosen['base_steps'],chosen['guidance'],record['seed'],noise_sign=sign)
                    save_file({'latents':value},path);dump(path.with_suffix('.json'),meta)
                signed.append(load_file(path)['latents'])
            mean=(signed[0]+signed[1])/2
            path=OUT/'latents'/f'{variant}_{i}.safetensors'
            if not path.exists():save_file({'latents':mean.contiguous()},path)
        del pipe;gc.collect();torch.cuda.empty_cache()
    vae=ext.AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    results=[]
    for record in records:
        i=record['index'];row=record['row'];prior=prepared[f'prior_{i}'];target=prepared[f'target_{i}']
        values={'target':target,'prior':prior,'frozen_pair':load_file(OUT/'latents'/f'frozen_pair_{i}.safetensors')['latents'],
                'candidate':load_file(OUT/'latents'/f'candidate_{i}.safetensors')['latents']}
        with np.load(ext.ROOT/row['features_path']) as f:
            beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
        metrics={};frames={}
        for variant,z in values.items():
            path=OUT/'videos'/f'{variant}_{i}.npz'
            if not path.exists():
                rgb=ext.decode(vae,z);old=ext.HERE;ext.HERE=OUT
                try:ext.mux(f'{variant}_{i}',rgb,row)
                finally:ext.HERE=old
            else:rgb=np.load(path)['rgb']
            frames[variant]=rgb
            metrics[variant]={'latent_mse_to_target':float((z-target).square().mean()),**ext.diagnostics(rgb,beats)}
        for variant,rgb in frames.items():
            metrics[variant]['rgb_mse_to_decoded_target']=float(np.square((rgb.astype(float)-frames['target'])/255).mean())
            metrics[variant]['rgb_mae_vs_prior']=float(np.abs(rgb.astype(float)-frames['prior']).mean()/255)
        results.append({'id':record['id'],'variants':metrics})
        strips=[]
        for variant,rgb in frames.items():
            strip=np.concatenate([cv2.resize(cv2.cvtColor(rgb[j],cv2.COLOR_RGB2BGR),(192,192)) for j in [4,12,20,28]],1)
            strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(strip,variant,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);strips.append(strip)
        path=OUT/'videos'/f'comparison_{i}.png'
        if not path.exists():cv2.imwrite(str(path),np.concatenate(strips,0))
    aggregate={}
    for variant in results[0]['variants']:
        aggregate[variant]={k:float(np.mean([r['variants'][variant][k] for r in results])) for k in ['latent_mse_to_target','rgb_mse_to_decoded_target','rgb_mae_vs_prior','warp_residual','sharpness']}
        aggregate[variant]['estimated_beat_f1']=float(np.mean([r['variants'][variant]['beat_f1']['f1'] for r in results]))
    comparisons={other:{'latent_wins':sum(r['variants']['candidate']['latent_mse_to_target']<r['variants'][other]['latent_mse_to_target'] for r in results),
        'rgb_wins':sum(r['variants']['candidate']['rgb_mse_to_decoded_target']<r['variants'][other]['rgb_mse_to_decoded_target'] for r in results)} for other in ['prior','frozen_pair']}
    report={'population':'three remaining validation recordings; candidate frozen before loading these confirmation outputs; not final test',
        'candidate_config':chosen,'aggregate':aggregate,'comparisons':comparisons,'results':results,'human_ratings':None,
        'limitations':'Synthetic target errors and estimated beats do not establish semantic quality, mood accuracy or human preference.'}
    dump(OUT/'results.json',report);print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)

if __name__=='__main__':main()
