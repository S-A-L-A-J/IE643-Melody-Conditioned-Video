import os
"""Validate synthetic videos and link each target to its aligned audio/features."""
import json
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
clips=[json.loads(line) for line in (ROOT/'data/manifests/clips.jsonl').read_text().splitlines()]
rows=[]
for clip in clips:
    name=f"{clip['track_id']}_{clip['start_seconds']:.3f}"
    video=ROOT/'data/synthetic_videos'/f'{name}.mp4'
    params=video.with_suffix('.json')
    feature=ROOT/'data/features'/f"{clip['track_id']}.npz"
    if not video.exists() or not params.exists() or not feature.exists():
        raise FileNotFoundError(name)
    capture=cv2.VideoCapture(str(video))
    frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(capture.get(cv2.CAP_PROP_FPS))
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if (frames,width,height)!=(64,256,256) or abs(fps-16)>.01:
        raise ValueError(f'Video shape mismatch {name}: {frames} frames, {width}x{height}, {fps} fps')
    frame_offset=round(clip['start_seconds']*16)
    rows.append(dict(id=name,track_id=clip['track_id'],
                     audio_path=clip['path'],audio_start_seconds=clip['start_seconds'],
                     audio_duration_seconds=clip['duration'],
                     features_path=str(feature.relative_to(ROOT)),
                     feature_start_frame=frame_offset,feature_frames=64,feature_fps=16,
                     video_path=str(video.relative_to(ROOT)),video_frames=64,video_fps=16,
                     video_resolution=[256,256],
                     renderer_parameters_path=str(params.relative_to(ROOT)),
                     split='development',song_identity_review='pending',
                     mood_embedding='pending'))
manifest=ROOT/'data/manifests/synthetic_pairs.jsonl'
manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
report=dict(status='passed',pairs=len(rows),tracks=len({row['track_id'] for row in rows}),
            duration_seconds=4,frames=64,fps=16,resolution=[256,256],
            note='Preliminary E0 videos are deterministic renderer outputs, not diffusion generations.')
(ROOT/'reports/pair_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
