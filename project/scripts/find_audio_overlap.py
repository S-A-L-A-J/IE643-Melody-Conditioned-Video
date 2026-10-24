import os
"""Flag likely shared recordings using aligned 4-second feature fingerprints.

High scores are candidates for grouping, not verified song identities. The
fingerprint uses chroma, onset and loudness, so covers or remixes may be missed.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracks = [json.loads(line) for line in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines()]
    tracks = [row for row in tracks if row['eligible']]
    windows = []
    owners = []
    offsets = []
    for i, track in enumerate(tracks):
        with np.load(ROOT/'data/features'/f"{track['id']}.npz") as source:
            chroma = source['chroma']
            onset = np.log1p(source['onset_strength'])[:,None]
            loudness = np.log1p(source['rms'] * 20)[:,None]
            x = np.concatenate([chroma,onset,loudness],axis=1).astype(np.float32)
        for start in range(0,len(x)-64+1,16):
            window = x[start:start+64].copy()
            window -= window.mean(axis=0,keepdims=True)
            window /= np.maximum(window.std(axis=0,keepdims=True),0.05)
            flattened = window.reshape(-1)
            flattened /= max(float(np.linalg.norm(flattened)),1e-8)
            windows.append(flattened)
            owners.append(i)
            offsets.append(start/16)
    windows = np.stack(windows)
    owners = np.asarray(owners)
    offsets = np.asarray(offsets)
    # Restrict to different source files; each 4-second window is shifted by
    # one second so clips with distinct excerpt offsets can still align.
    candidates = []
    for i in range(len(tracks)):
        left = windows[owners==i]
        right = windows[owners>i]
        scores = left @ right.T
        if not scores.size:
            continue
        right_owners = owners[owners>i]
        right_offsets = offsets[owners>i]
        for j in range(i+1,len(tracks)):
            positions = np.where(right_owners==j)[0]
            if not len(positions):
                continue
            local = scores[:,positions]
            a,b = np.unravel_index(local.argmax(),local.shape)
            value = float(local[a,b])
            if value>=0.72:
                candidates.append({'track_a':tracks[i]['id'],'track_b':tracks[j]['id'],
                                   'score':round(value,5),'offset_a_s':float(offsets[owners==i][a]),
                                   'offset_b_s':float(right_offsets[positions[b]]),
                                   'title_a':tracks[i]['source_title'],'title_b':tracks[j]['source_title']})
    candidates.sort(key=lambda row:row['score'],reverse=True)
    output = ROOT/'reports/audio_overlap_candidates.csv'
    with output.open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=['track_a','track_b','score','offset_a_s','offset_b_s','title_a','title_b'])
        writer.writeheader()
        writer.writerows(candidates)
    report = {'eligible_tracks':len(tracks),'candidate_pairs_above_0_72':len(candidates),
              'candidate_pairs_above_0_90':sum(row['score']>=0.9 for row in candidates),
              'highest_scores':candidates[:12],
              'note':'Candidates are not verified identity labels; absence of a match does not prove different songs.'}
    (ROOT/'reports/audio_overlap_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
