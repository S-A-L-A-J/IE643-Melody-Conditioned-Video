"""Matched generation comparison on the eight validation recordings used for selection.

This is validation evidence, not an independent final test. Includes predictor
and encoded-target reference, so lower denoising loss cannot conceal bad output.
"""
import gc
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent.parent),str(Path(__file__).resolve().parent.joinpath('../../../../project').resolve()/'scripts')]
import cv2
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from train_temporal_prior import TemporalPrior,data
from gpu_backend import load_pipeline,refine
import extension as ext

OUT=HERE/'resumed_20261024/validation8';OUT.mkdir(exist_ok=True)
(OUT/'latents').mkdir(exist_ok=True);(OUT/'videos').mkdir(exist_ok=True)

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

@torch.inference_mode()
def main():
    assert (HERE/'resumed_20261024/RESUME_AUTHORIZATION.json').exists(), 'Explicit resumption record required'
    torch.manual_seed(643);torch.set_num_threads(4);cv2.setNumThreads(1);assert torch.cuda.is_available()
    if not (OUT/'prepared.json').exists():
        part=data()['validation'];selected=[];seen=set()
        for i,track in enumerate(part['tracks']):
            if track not in seen:selected.append(i);seen.add(track)
            if len(selected)==8:break
        model=TemporalPrior().cuda().eval();model.load_state_dict(load_file(ext.EXPERIMENTS/'temporal_phase/model.safetensors'))
        z={};records=[]
        rows={r['id']:r for r in map(json.loads,(ext.ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
        for j,i in enumerate(selected):
            z[f'prior_{j}']=model(part['music'][i:i+1].cuda(),part['context'][i:i+1].cuda(),part['phase'][i:i+1].cuda()).cpu()
            z[f'target_{j}']=part['target'][i:i+1]
            records.append({'index':j,'id':part['ids'][i],'row':rows[part['ids'][i]],'seed':643+j})
        save_file(z,OUT/'prepared_latents.safetensors');dump(OUT/'prepared.json',{'records':records,'split':'validation; same eight groups used for adapter selection','test_or_fresh_used':False})
        del model,part,z;gc.collect();torch.cuda.empty_cache()
    records=json.loads((OUT/'prepared.json').read_text())['records']
    prepared=load_file(OUT/'prepared_latents.safetensors')
    selection=json.loads((HERE/'selected_adapter.json').read_text())
    for variant,lora in [('frozen',None),('adapted',(WORK/'geometry_lora_continuation/lora_step128.safetensors'))]:
        pipe=load_pipeline(lora_path=lora)
        for r in records:
            j=r['index'];out=OUT/'latents'/f'{variant}_{j}.safetensors'
            if out.exists():continue
            value,meta=refine(pipe,prepared[f'prior_{j}'],.06,100,1.,r['seed'])
            meta['geometry_lora']=str(lora) if lora else None;meta['updates_in_this_refinement_call']=0
            meta['adapter_training_parameter_count']=299136 if lora else 0
            save_file({'latents':value},out);dump(out.with_suffix('.json'),meta)
        del pipe;gc.collect();torch.cuda.empty_cache()
    vae=ext.AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    results=[]
    for record in records:
        j=record['index'];row=record['row']
        target=prepared[f'target_{j}'];prior=prepared[f'prior_{j}']
        values={'prior':prior,'target':target}
        values.update({variant:load_file(OUT/'latents'/f'{variant}_{j}.safetensors')['latents'] for variant in ['frozen','adapted']})
        delta=values['adapted']-prior
        values['anchor025']=prior+.25*delta
        values['anchor050']=prior+.50*delta
        frames={}
        with np.load(ext.ROOT/row['features_path']) as f:
            beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
        metrics={}
        for variant,z in values.items():
            rgb_path=OUT/'videos'/f'{variant}_{j}.npz'
            if not rgb_path.exists():
                rgb=ext.decode(vae,z);np.savez_compressed(rgb_path,rgb=rgb)
                ext_here=ext.HERE;ext.HERE=OUT
                try:ext.mux(f'{variant}_{j}',rgb,row)
                finally:ext.HERE=ext_here
            else:rgb=np.load(rgb_path)['rgb']
            frames[variant]=rgb
            metrics[variant]={'latent_mse_to_target':float((z-target).square().mean()),**ext.diagnostics(rgb,beats)}
        for variant in values:
            metrics[variant]['rgb_mse_to_decoded_target']=float(np.square((frames[variant].astype(float)-frames['target'])/255).mean())
            metrics[variant]['rgb_mae_vs_prior']=float(np.abs(frames[variant].astype(float)-frames['prior']).mean()/255)
        delta=values['adapted']-prior
        anchor={str(a):float((prior+a*delta-target).square().mean()) for a in [0,.25,.5,.75,1.]}
        results.append({'id':record['id'],'variants':metrics,'anchor_mse_secondary_diagnostic':anchor})
        strips=[]
        for variant in ['target','prior','frozen','adapted','anchor025','anchor050']:
            rgb=frames[variant];strip=np.concatenate([cv2.resize(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR),(160,160)) for i in [4,12,20,28]],1)
            strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(strip,variant,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);strips.append(strip)
        image=OUT/'videos'/f'comparison_{j}.png'
        if not image.exists():cv2.imwrite(str(image),np.concatenate(strips,0))
    aggregate={}
    for variant in ['prior','frozen','adapted','anchor025','anchor050','target']:
        aggregate[variant]={metric:float(np.mean([r['variants'][variant][metric] for r in results])) for metric in ['latent_mse_to_target','rgb_mse_to_decoded_target','rgb_mae_vs_prior','warp_residual','sharpness']}
        aggregate[variant]['estimated_beat_f1']=float(np.mean([r['variants'][variant]['beat_f1']['f1'] for r in results]))
    comparisons={other:{'recordings_improved_latent_mse':sum(r['variants']['adapted']['latent_mse_to_target']<r['variants'][other]['latent_mse_to_target'] for r in results),
                       'mean_latent_mse_gain':float(np.mean([r['variants'][other]['latent_mse_to_target']-r['variants']['adapted']['latent_mse_to_target'] for r in results]))} for other in ['prior','frozen']}
    report={'population':'eight fixed validation recordings also used for adapter selection; exploratory, not independent test',
        'aggregate':aggregate,'paired_comparisons':comparisons,'results':results,'adapter_selection':selection,
        'anchoring_note':'Secondary fixed-alpha diagnostics only. Small latent-MSE gains or nearly unchanged frames are not evidence of useful visual improvement.',
        'human_ratings':None,'limitations':'Estimated beats do not establish semantics/mood; synthetic encoded targets favor the trained ring domain.'}
    dump(OUT/'results.json',report);print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)

if __name__=='__main__':main()
