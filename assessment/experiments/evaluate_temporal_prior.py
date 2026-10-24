import os
"""Decode fixed validation clips and compare prior variants under music controls."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from train_temporal_prior import ROOT,HERE,TemporalPrior,data
from train_audio_latent_prior import AudioLatentPrior,wrong_track_indices
from evaluate_e0 import video_events,score


def decode(vae,latent,path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(128,128))
    if not writer.isOpened():raise RuntimeError(str(path))
    edges=[]
    try:
        for chunk in latent[0].permute(1,0,2,3).split(4):
            frames=((vae.decode(chunk/vae.config.scaling_factor).sample/2+.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
            for frame in frames:
                bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR);writer.write(bgr)
                edges.append(cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
    finally:writer.release()
    return float(np.mean(edges))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='temporal_phase');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(643)
    d=data()['validation'];wrong=wrong_track_indices(d['tracks']);chosen=[];seen=set()
    for i,track in enumerate(d['tracks']):
        if track not in seen:chosen.append(i);seen.add(track)
    directory=HERE/args.run;out=directory/'decoded';out.mkdir(exist_ok=True)
    model=TemporalPrior(phase=args.run!='temporal_no_phase').cuda().eval()
    model.load_state_dict(load_file(directory/'model.safetensors'))
    baseline=AudioLatentPrior().cuda().eval();baseline.load_state_dict(load_file(ROOT/'outputs/prior_32f_causal_peak/model.safetensors'))
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    rows={r['id']:r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
    results=[]
    for i in chosen:
        x=d['music'][i:i+1].cuda();c=d['context'][i:i+1].cuda();phase=d['phase'][i:i+1].cuda();j=int(wrong[i])
        outputs={'temporal':model(x,c,phase),'wrong_song':model(d['music'][j:j+1].cuda(),d['context'][j:j+1].cuda(),phase),
                 'shifted':model(x.roll(8,1),c,phase),'baseline':baseline(x,c),'target':d['target'][i:i+1].cuda()}
        row=rows[d['ids'][i]]
        with np.load(ROOT/row['features_path']) as f:
            beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
        result={'id':d['ids'][i],'track':d['tracks'][i],'variants':{}}
        for label,latent in outputs.items():
            path=out/f'{i}_{label}.mp4';sharpness=decode(vae,latent,path)
            events,changes=video_events(path,fps=8)
            result['variants'][label]={'f1':score(beats,events)['f1'],'coverage':score(beats,events)['coverage'],
                                       'sharpness_proxy':sharpness,'mean_frame_change':float(changes[1:].mean()),'path':str(path)}
        results.append(result);print(f'{len(results)}/{len(chosen)} validation tracks',flush=True)
    aggregate={label:{key:float(np.mean([r['variants'][label][key] for r in results]))
                       for key in ['f1','coverage','sharpness_proxy','mean_frame_change']} for label in outputs}
    report={'selection':'first clip per validation track, same 11-track diagnostic as previous rhythm comparison',
            'aggregate':aggregate,'results':results,'limits':'Validation only, no test data. Sharpness proxy can increase from noise. Human review pending.'}
    (directory/'decoded_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
