import os
"""Decode all provisional test clips; compare timing with matched controls."""
import json
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from train_audio_latent_prior import ROOT, AudioLatentPrior, load_data, wrong_track_indices
from evaluate_e0 import video_events, score

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    data=load_data(32,2)['test']
    model=AudioLatentPrior().cuda().eval()
    model.load_state_dict(load_file(ROOT/'outputs/audio_latent_prior_32f/model.safetensors'))
    vae=AutoencoderKL.from_pretrained(ROOT/'models/diffusion/sd15',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    rows={r['id']:r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
    directory=ROOT/'outputs/prior_32f_test';directory.mkdir(exist_ok=True)
    wrong=wrong_track_indices(data['tracks'])
    results=[]
    for i,pair in enumerate(data['ids']):
        x=data['music'][i:i+1].cuda();c=data['context'][i:i+1].cuda();j=int(wrong[i])
        variants={'correct':model(x,c),'other_track':model(data['music'][j:j+1].cuda(),data['context'][j:j+1].cuda()),'shifted_1s':model(x.roll(8,1),c),'target':data['target'][i:i+1].cuda()}
        row=rows[pair]
        with np.load(ROOT/row['features_path']) as f:
            beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
        result={'id':pair,'track':data['tracks'][i],'variants':{}}
        for label,latent in variants.items():
            path=directory/f'{pair}_{label}.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(128,128))
            if not writer.isOpened():raise RuntimeError(str(path))
            for chunk in latent[0].permute(1,0,2,3).split(4):
                frames=((vae.decode(chunk/vae.config.scaling_factor).sample/2+.5).clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
                for frame in frames:writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            writer.release()
            events,changes=video_events(path,fps=8)
            result['variants'][label]={'aligned':score(beats,events),'shifted_audio_200ms':score(beats+.2,events),'events':len(events),'mean_frame_change':float(changes[1:].mean())}
        results.append(result)
        if (i+1)%14==0:print(f'{i+1}/{len(data["ids"])} clips decoded and scored',flush=True)
    aggregate={label:{'mean_f1':float(np.mean([r['variants'][label]['aligned']['f1'] for r in results])),'mean_shifted_audio_f1':float(np.mean([r['variants'][label]['shifted_audio_200ms']['f1'] for r in results])),'mean_coverage':float(np.mean([r['variants'][label]['aligned']['coverage'] for r in results]))} for label in variants}
    report={'pairs':len(results),'tracks':len(set(data['tracks'])),'fps':8,'split':'provisional_recording_grouped','aggregate':aggregate,'results':results,'limitations':'Exploratory frame-difference peak metric; clips within tracks are correlated. No human ratings or verified song-disjoint split. Direct prior decoding, not diffusion.'}
    (ROOT/'reports/prior_32f_decoded_test.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
