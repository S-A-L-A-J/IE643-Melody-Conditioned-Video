import os
"""Bounded validation-only test of temporal memory and explicit visual phase.

No provisional test examples are loaded. Appearance phase comes from renderer
metadata during training and is an explicit user control at inference.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[2] / 'project')))
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from train_audio_latent_prior import load_data, wrong_track_indices


class TemporalPrior(nn.Module):
    def __init__(self, phase=True):
        super().__init__()
        self.phase=phase
        self.context=nn.Linear(512,128)
        self.input=nn.Linear(25,128)
        self.temporal=nn.GRU(128,128,batch_first=True)
        self.spatial=nn.Linear(128,64*4*4)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),
                                   nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU(),nn.Conv2d(16,4,3,padding=1))

    def forward(self,music,context,phase):
        b,t,_=music.shape
        if not self.phase:phase=phase*0
        times=torch.linspace(0,1,t,device=music.device).view(1,t,1).expand(b,t,1)
        x=torch.cat([music,phase[:,None,:].expand(-1,t,-1),times],dim=-1)
        x=F.silu(self.input(x))
        hidden,_=self.temporal(x,self.context(context)[None])
        z=self.decoder(self.spatial(hidden).reshape(b*t,64,4,4))
        return z.reshape(b,t,4,16,16).permute(0,2,1,3,4).contiguous()


def data():
    packed=load_data(32,2,'causal_peak',False)
    rows={r['id']:r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
    for split,part in packed.items():
        phases=[]
        for ident in part['ids']:
            meta=json.loads((ROOT/rows[ident]['renderer_parameters_path']).read_text())
            angle=float(np.random.default_rng(meta['seed']).uniform(0,2*math.pi))
            phases.append([math.sin(angle),math.cos(angle)])
        part['phase']=torch.tensor(phases,dtype=torch.float32)
    return packed


@torch.inference_mode()
def evaluate(model,part):
    values=[]
    wrong=wrong_track_indices(part['tracks']).cuda()
    for ids in torch.arange(len(part['ids']),device='cuda').split(16):
        x=part['music'][ids];c=part['context'][ids];phase=part['phase'][ids];y=part['target'][ids]
        pred=model(x,c,phase)
        other=model(part['music'][wrong[ids]],part['context'][wrong[ids]],phase)
        shift=model(x.roll(8,1),c,phase)
        values.append(torch.stack([(v-y).square().flatten(1).mean(1) for v in [pred,other,shift]],1).cpu())
    values=torch.cat(values)
    return {'correct_mse':float(values[:,0].mean()),'other_mse':float(values[:,1].mean()),
            'shift_mse':float(values[:,2].mean()),'swap_wins':int((values[:,0]<values[:,1]).sum()),
            'shift_wins':int((values[:,0]<values[:,2]).sum()),'clips':len(values)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--no-phase',action='store_true');args=p.parse_args()
    torch.manual_seed(643);torch.set_num_threads(4)
    directory=HERE/('temporal_no_phase' if args.no_phase else 'temporal_phase');directory.mkdir(exist_ok=True)
    packed=data()
    for part in packed.values():
        for key in ['music','context','target','phase']:part[key]=part[key].cuda()
    model=TemporalPrior(not args.no_phase).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    part=packed['train'];best=float('inf');history=[];started=time.monotonic();best_epoch=0
    wrong=wrong_track_indices(part['tracks']).cuda()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for ids in torch.randperm(len(part['ids']),device='cuda').split(16):
            pred=model(part['music'][ids],part['context'][ids],part['phase'][ids]);y=part['target'][ids]
            others=wrong[ids]
            swapped=model(part['music'][others],part['context'][others],part['phase'][ids])
            correct=(pred-y).square().flatten(1).mean(1)
            incorrect=(swapped-y).square().flatten(1).mean(1)
            temporal=F.mse_loss(pred[:,:,1:]-pred[:,:,:-1],y[:,:,1:]-y[:,:,:-1])
            ranking=F.softplus(10*(correct-incorrect+.02)).mean()/10
            loss=correct.mean()+.5*temporal+.5*ranking
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append(float(loss.detach()))
        model.eval();val=evaluate(model,packed['validation']);history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),**val})
        if val['correct_mse']<best-1e-5:
            best=val['correct_mse'];best_epoch=epoch
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},directory/'model.safetensors')
        report={'configuration':vars(args),'model':'temporal GRU + convolutional latent decoder','phase_control':not args.no_phase,
                'seed':643,'counts':{k:len(v['ids']) for k,v in packed.items()},'selection':'validation MSE; no test data loaded',
                'best_epoch':best_epoch,'best_mse':best,'elapsed_seconds':time.monotonic()-started,'history':history}
        (directory/'training.json').write_text(json.dumps(report,indent=2))
        if epoch==1 or epoch%5==0:print(json.dumps(history[-1]),flush=True)
        if epoch-best_epoch>=15:break
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)


if __name__=='__main__':main()
