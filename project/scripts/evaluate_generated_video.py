import os
"""Diagnostic beat-event and frame-change metrics for a generated video."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_e0 import score,video_events

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--pair-id',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for line in (ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines()]
    row=next((r for r in rows if r['id']==args.pair_id),None)
    if row is None:
        raise ValueError(f'Unknown pair: {args.pair_id}')
    video=ROOT/args.video
    capture=cv2.VideoCapture(str(video))
    fps=float(capture.get(cv2.CAP_PROP_FPS))
    frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    visual,changes=video_events(video,fps=fps)
    with np.load(ROOT/row['features_path']) as source:
        start=row['audio_start_seconds']
        beats=source['beat_times']
        beats=beats[(beats>=start)&(beats<start+row['audio_duration_seconds'])]-start
    tolerance=max(0.125,0.5/fps)
    report={'video':args.video,'pair_id':args.pair_id,'source_split':row['split'],
            'frames':frames,'fps':fps,'duration_seconds':frames/fps,
            'beat_events':len(beats),'visual_events':len(visual),'tolerance_seconds':tolerance,
            'aligned':score(beats,visual,tolerance),
            'audio_shifted_200ms':score(beats+0.2,visual,tolerance),
            'mean_frame_change':float(changes[1:].mean()),
            'p95_frame_change':float(np.percentile(changes[1:],95)),
            'note':'Exploratory single-clip metric. Visual event extraction is sensitive to frame rate and texture; human ratings remain pending.'}
    output=ROOT/args.report
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
