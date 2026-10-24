import os
"""Measure event timing in E0 videos with shifted-audio and static controls."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
cv2.setNumThreads(1)

def video_events(path,fps=16):
    capture=cv2.VideoCapture(str(path))
    signal=[]
    previous=None
    while True:
        ok,frame=capture.read()
        if not ok:
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        signal.append(0.0 if previous is None else float(np.abs(gray-previous).mean()))
        previous=gray
    capture.release()
    signal=np.asarray(signal,dtype=np.float32)
    if len(signal)<3 or signal.max()<1e-5:
        return np.array([]),signal
    prominence=max(float(np.percentile(signal[1:],75))*0.18,0.15)
    peaks,_=find_peaks(signal,prominence=prominence,distance=2)
    return peaks/fps,signal

def score(audio_events,visual_events,tolerance=.125):
    if not len(audio_events) or not len(visual_events):
        return dict(precision=0.0,coverage=0.0,f1=0.0)
    precision=float(np.mean([np.min(np.abs(audio_events-event))<=tolerance for event in visual_events]))
    coverage=float(np.mean([np.min(np.abs(visual_events-event))<=tolerance for event in audio_events]))
    f1=2*precision*coverage/(precision+coverage+1e-12)
    return dict(precision=precision,coverage=coverage,f1=f1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tracks',type=int,default=25)
    parser.add_argument('--manifest',default='data/manifests/synthetic_pairs.jsonl')
    parser.add_argument('--split',default='')
    parser.add_argument('--style',default='')
    parser.add_argument('--report',default='reports/e0_alignment.json')
    args=parser.parse_args()
    pairs=[json.loads(x) for x in (ROOT/args.manifest).read_text().splitlines()]
    chosen=[];seen=set()
    for row in pairs:
        if args.split and row.get('split')!=args.split:
            continue
        if args.style and row.get('style')!=args.style:
            continue
        if row['track_id'] not in seen:
            chosen.append(row);seen.add(row['track_id'])
        if args.tracks and len(chosen)==args.tracks:
            break
    results=[]
    for row in chosen:
        with np.load(ROOT/row['features_path']) as source:
            start=row['audio_start_seconds']
            beats=source['beat_times']
            beats=beats[(beats>=start)&(beats<start+row['audio_duration_seconds'])]-start
        visual,signal=video_events(ROOT/row['video_path'])
        aligned=score(beats,visual)
        shifted=score(beats+.2,visual)
        results.append(dict(id=row['id'],beats=len(beats),visual_events=len(visual),
                            aligned=aligned,shifted_audio_200ms=shifted,static_video=score(beats,np.array([]))))
    aggregate={key:round(float(np.mean([r[key]['f1'] for r in results])),4)
               for key in ['aligned','shifted_audio_200ms','static_video']}
    report=dict(tracks=len(results),split=args.split or 'unspecified',style=args.style or 'unspecified',
                manifest=args.manifest,tolerance_seconds=.125,
                event_method='peaks of consecutive-frame grayscale difference',
                aggregate_f1=aggregate,results=results,
                note='E0 is programmed from music features. This checks timing/metric plumbing, not learned musical control.')
    (ROOT/args.report).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':
    main()
