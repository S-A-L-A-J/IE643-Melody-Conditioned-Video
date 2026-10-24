import os
"""Compare fixed first clips from each validation track; never reads test inputs."""
import json
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from train_audio_latent_prior import ROOT,AudioLatentPrior,load_data
from evaluate_e0 import video_events,score

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    modes={'point_control':'point','causal_peak':'causal_peak'}
    datasets={name:load_data(32,2,mode,False)['validation'] for name,mode in modes.items()}
    models={}
    for name in modes:
        model=AudioLatentPrior().cuda().eval();model.load_state_dict(load_file(ROOT/'outputs'/f'prior_32f_{name}'/'model.safetensors'));models[name]=model
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    rows={r['id']:r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
    out=ROOT/'outputs/rhythm_validation';out.mkdir(exist_ok=True)
    selected=[i for i,p in enumerate(datasets['point_control']['ids']) if p.endswith('_0.000__rings')]
    assert len(selected)==11
    results=[]
    for i in selected:
        pair=datasets['point_control']['ids'][i];row=rows[pair];assert row['split']=='validation'
        with np.load(ROOT/row['features_path']) as f:
            beats=f['beat_times'];beats=beats[(beats>=0)&(beats<4)]
        variants={}
        for name,data in datasets.items():
            x=data['music'][i:i+1].cuda();c=data['context'][i:i+1].cuda()
            variants[name]=models[name](x,c)
            variants[name+'_shifted']=models[name](x.roll(8,1),c)
        variants['target']=datasets['point_control']['target'][i:i+1].cuda()
        result={'id':pair,'scores':{}}
        for label,latent in variants.items():
            path=out/f'{pair}_{label}.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(128,128))
            if not writer.isOpened():raise RuntimeError(str(path))
            edges=[]
            for chunk in latent[0].permute(1,0,2,3).split(4):
                frames=((vae.decode(chunk/vae.config.scaling_factor).sample/2+.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
                for frame in frames:
                    bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR);writer.write(bgr)
                    edges.append(float(cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var()))
            writer.release();events,changes=video_events(path,8)
            result['scores'][label]={'aligned':score(beats,events),'audio_shift_200ms':score(beats+.2,events),'laplacian_variance':float(np.mean(edges))}
        results.append(result);print(f'{len(results)}/11 validation tracks',flush=True)
    aggregate={label:{'f1':float(np.mean([r['scores'][label]['aligned']['f1'] for r in results])),'laplacian_variance':float(np.mean([r['scores'][label]['laplacian_variance'] for r in results]))} for label in variants}
    report={'selection':'first four-second clip from all 11 validation tracks; fixed before scoring','aggregate':aggregate,'results':results,'limits':'Frame-change event F1 is exploratory; Laplacian variance measures edges/noise, not perceptual quality. No test clips evaluated.'}
    (ROOT/'reports/rhythm_validation_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
