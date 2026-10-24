import os
"""Combine provisional held-out renderer timing results and prepare review."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FILES={'geometric':'e0_alignment_provisional_test.json',
       'rings':'e0_rings_alignment_provisional_test.json',
       'bars':'e0_bars_alignment_provisional_test.json'}


def frame_changes(video):
    capture=cv2.VideoCapture(str(video))
    previous=None
    changes=[]
    while True:
        ok,frame=capture.read()
        if not ok:
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            changes.append(float(np.abs(gray-previous).mean()))
        previous=gray
    capture.release()
    if not changes:
        raise ValueError(f'No frame changes in {video}')
    return np.asarray(changes)


def main():
    manifest=[json.loads(line) for line in (ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines()]
    by_id={row['id']:row for row in manifest}
    summary={}
    review=[]
    for style,filename in FILES.items():
        report=json.loads((ROOT/'reports'/filename).read_text())
        measurements=[]
        for result in report['results']:
            row=by_id[result['id'] if '__' in result['id'] else result['id']+'__geometric']
            changes=frame_changes(ROOT/row['video_path'])
            measurements.append({'mean_frame_change':float(changes.mean()),
                                 'p95_frame_change':float(np.percentile(changes,95))})
            review_id=hashlib.sha256(('IE643-review-v1:'+row['id']).encode()).hexdigest()[:10]
            review.append({'review_id':review_id,'video_path':row['video_path'],
                           'instrumental_path':row['audio_path'],
                           'audio_start_seconds':row['audio_start_seconds'],
                           'beat_match_1_to_5':'','continuity_1_to_5':'',
                           'mood_match_1_to_5':'','overall_1_to_5':'','reviewer':'','notes':''})
        results=report['results']
        summary[style]={
            'tracks':len(results),
            'beat_precision':float(np.mean([r['aligned']['precision'] for r in results])),
            'beat_coverage':float(np.mean([r['aligned']['coverage'] for r in results])),
            'beat_f1':float(np.mean([r['aligned']['f1'] for r in results])),
            'shifted_200ms_f1':float(np.mean([r['shifted_audio_200ms']['f1'] for r in results])),
            'mean_adjacent_gray_change':float(np.mean([x['mean_frame_change'] for x in measurements])),
            'mean_p95_adjacent_gray_change':float(np.mean([x['p95_frame_change'] for x in measurements]))}
    review.sort(key=lambda row:row['review_id'])
    rating_path=ROOT/'reports/provisional_human_review_template.csv'
    with rating_path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    output={'status':'provisional','test_tracks':14,'scores':summary,
            'human_ratings':'pending; blank review template created',
            'method_note':'Beat events are estimated from frame differences. Renderer timing is programmed from music features. Frame-change values are descriptive, not a validated continuity metric.',
            'split_warning':'Detected recording overlaps are grouped, but song identities and medley overlap are unverified.'}
    (ROOT/'reports/provisional_renderer_benchmark.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
