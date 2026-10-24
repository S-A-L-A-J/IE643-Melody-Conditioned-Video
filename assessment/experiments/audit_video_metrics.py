import os
"""Independent validation diagnostics from saved videos; no model/test loading.

One-to-one event matching avoids giving repeated events credit for one beat.
Flow-warp residual is descriptive: static/blurred video can score deceptively well.
"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[2] / 'project')))
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_e0 import video_events


def matched_score(beats,events,tolerance=.125):
    beats=np.asarray(beats);events=np.asarray(events)
    if not len(beats) or not len(events):
        return {'precision':0.,'recall':0.,'f1':0.,'matches':0,'audio_beats':len(beats),'visual_events':len(events)}
    distances=np.abs(beats[:,None]-events[None,:])
    # Infeasible matches cost more than all feasible distances combined.
    cost=np.where(distances<=tolerance,distances,1000.)
    rows,columns=linear_sum_assignment(cost)
    count=int(np.sum(distances[rows,columns]<=tolerance))
    precision=count/len(events);recall=count/len(beats)
    return {'precision':precision,'recall':recall,'f1':2*count/(len(beats)+len(events)),
            'matches':count,'audio_beats':len(beats),'visual_events':len(events)}


def continuity(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    cap.release()
    changes=[];residuals=[];flows=[]
    for prev,current in zip(frames,frames[1:]):
        flow=cv2.calcOpticalFlowFarneback(current,prev,None,.5,3,15,3,5,1.2,0)
        yy,xx=np.mgrid[:current.shape[0],:current.shape[1]].astype(np.float32)
        mx=xx+flow[:,:,0];my=yy+flow[:,:,1]
        valid=(mx>=0)&(mx<current.shape[1]-1)&(my>=0)&(my<current.shape[0]-1)
        warped=cv2.remap(prev,mx,my,cv2.INTER_LINEAR)
        residuals.append(float(np.abs(warped.astype(float)-current)[valid].mean()/255))
        changes.append(float(np.abs(prev.astype(float)-current).mean()/255))
        flows.append(float(np.linalg.norm(flow,axis=-1).mean()))
    return {'mean_frame_change_0_1':float(np.mean(changes)),
            'flow_warp_residual_0_1':float(np.mean(residuals)),
            'mean_flow_pixels':float(np.mean(flows)),
            'static_frame_control_warp_residual':0.}


def main():
    cv2.setNumThreads(1)
    assert matched_score([1.],[.95,1.,1.05])['matches']==1
    assert matched_score([0.,1.],[0.,1.])['f1']==1
    assert matched_score([0.],[1.])['f1']==0
    records=json.loads((HERE/'temporal_phase/decoded_validation.json').read_text())['results']
    rows={r['id']:r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
    output=[]
    for record in records:
        row=rows[record['id']]
        with np.load(ROOT/row['features_path']) as f:
            beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
        values={}
        for label,item in record['variants'].items():
            events,_=video_events(item['path'],fps=8)
            values[label]={'one_to_one':matched_score(beats,events),'shifted_reference_200ms':matched_score(beats+.2,events),
                           'continuity':continuity(item['path'])}
        values['dense_event_control']={'one_to_one':matched_score(beats,np.arange(0,4,.125))}
        values['static_control']={'one_to_one':matched_score(beats,[])}
        output.append({'track':record['track'],'id':record['id'],'variants':values})
    aggregate={label:{'one_to_one_f1':float(np.mean([r['variants'][label]['one_to_one']['f1'] for r in output]))}
               for label in output[0]['variants']}
    for label in record['variants']:
        aggregate[label].update({key:float(np.mean([r['variants'][label]['continuity'][key] for r in output]))
                                 for key in ['mean_frame_change_0_1','flow_warp_residual_0_1','mean_flow_pixels']})
    rng=np.random.default_rng(643);comparisons={}
    for control in ['baseline','wrong_song','shifted']:
        gains=np.array([r['variants']['temporal']['one_to_one']['f1']-r['variants'][control]['one_to_one']['f1'] for r in output])
        boot=rng.choice(gains,(10000,len(gains)),replace=True).mean(1)
        comparisons[control]={'mean_gain':float(gains.mean()),'tracks_improved':int((gains>0).sum()),
                              'bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist()}
    report={'population':'11 fixed validation clips, one per recording; not independent final test',
            'event_match':'maximum feasible one-to-one matching within 125 ms',
            'aggregate':aggregate,'paired_comparisons':comparisons,'results':output,
            'limitations':['Optical flow is estimated and unreliable on textureless/blurred frames.',
                          'Low warp residual can reward blur or static video; no coherence superiority claim.',
                          'Bootstrap intervals are exploratory after validation-driven model selection.',
                          'Synthetic target supervision favours event metrics; human evaluation still required.']}
    (HERE.parent/'evidence/validation_metric_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':main()
