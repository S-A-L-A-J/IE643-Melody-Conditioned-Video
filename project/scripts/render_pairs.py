import os
"""Render deterministic, aligned E0 videos from instrumental features.

These videos are synthetic training targets and an evaluation baseline. The final
music-to-video generator remains the adapted diffusion model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FPS = 16
SIZE = 256
OUT = ROOT/'data/synthetic_videos'

def color(hue, saturation=180, brightness=220):
    hsv=np.array([[[int(hue)%180, int(np.clip(saturation,0,255)),
                    int(np.clip(brightness,0,255))]]],dtype=np.uint8)
    return tuple(int(x) for x in cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0])

def render(clip, features, metadata, output):
    seed=int(hashlib.sha256(f"{clip['track_id']}:{clip['start_seconds']:.3f}".encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    offsets=rng.uniform(0,2*math.pi,3)
    radii=rng.uniform(28,48,3)
    positions=rng.uniform(60,196,(3,2))
    shape_types=rng.choice(['circle','triangle','diamond'],size=3,replace=False).tolist()
    start=round(clip['start_seconds']*FPS)
    count=round(clip['duration']*FPS)
    chroma=features['chroma'][start:start+count]
    onset=features['onset_strength'][start:start+count]
    beats=features['beat_impulse'][start:start+count]
    rms=features['rms'][start:start+count]
    if len(chroma)!=count:
        raise ValueError('Features do not cover this clip')
    onset_scale=max(float(np.percentile(features['onset_strength'],95)),1e-4)
    rms_scale=max(float(np.percentile(features['rms'],90)),1e-4)
    tempo=max(float(metadata['tempo_bpm']),40)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(SIZE,SIZE))
    if not writer.isOpened():
        raise RuntimeError('Could not open MP4 video writer')
    yy,xx=np.mgrid[0:SIZE,0:SIZE]
    radial=np.sqrt((xx-SIZE/2)**2+(yy-SIZE/2)**2)/180
    phase=np.exp(2j*np.pi*np.arange(12)/12)
    hue_vector=1+0j
    afterglow=0.0
    try:
        for i in range(count):
            seconds=i/FPS
            # Circular smoothing prevents hue jumps at the pitch-class wraparound.
            target=np.sum(chroma[i]*phase)
            if abs(target)>1e-5:
                hue_vector=.88*hue_vector+.12*target/abs(target)
            hue=(np.angle(hue_vector)%(2*math.pi))*180/(2*math.pi)
            energy=float(np.clip(rms[i]/rms_scale,0,1.5))
            attack=float(np.clip(onset[i]/onset_scale,0,2))
            afterglow=max(afterglow*.72, float(beats[i]))
            pulse=np.clip(.4*attack+.8*afterglow,0,1.5)
            base=np.empty((SIZE,SIZE,3),dtype=np.float32)
            base[:,:,0]=12+14*(1-radial)+10*pulse
            base[:,:,1]=8+7*(1-radial)+7*energy
            base[:,:,2]=20+17*(1-radial)+14*pulse
            frame=np.clip(base,0,255).astype(np.uint8)
            movement=10+30*energy
            for j in range(3):
                speed=(tempo/120)*(0.55+0.18*j)
                cx=int(positions[j,0]+movement*math.sin(2*math.pi*speed*seconds/3+offsets[j]))
                cy=int(positions[j,1]+movement*math.cos(2*math.pi*speed*seconds/4+offsets[j]))
                radius=int(radii[j]*(1+.15*energy+.2*pulse))
                pigment=color(hue+j*52,175+30*energy,170+35*energy+22*pulse)
                overlay=frame.copy()
                if shape_types[j]=='circle':
                    cv2.circle(overlay,(cx,cy),radius,pigment,-1,cv2.LINE_AA)
                else:
                    sides=3 if shape_types[j]=='triangle' else 4
                    angle=seconds*(tempo/120)*.35+offsets[j]
                    pts=np.array([[
                        cx+radius*math.cos(angle+2*math.pi*k/sides),
                        cy+radius*math.sin(angle+2*math.pi*k/sides)] for k in range(sides)],dtype=np.int32)
                    cv2.fillConvexPoly(overlay,pts,pigment,cv2.LINE_AA)
                frame=cv2.addWeighted(overlay,.35+.1*pulse,frame,.65-.1*pulse,0)
                cv2.circle(frame,(cx,cy),radius+5,color(hue+j*52,130,130+50*pulse),1,cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    params=dict(renderer='persistent_geometric_v1',seed=seed,shapes=shape_types,
                initial_positions=positions.round(2).tolist(),base_radii=radii.round(2).tolist(),
                phases=offsets.round(3).tolist(),fps=FPS,resolution=[SIZE,SIZE],frames=count,
                mappings={'chroma':'hue','beats_and_onsets':'pulse','tempo':'motion_speed','rms':'brightness_and_motion'},
                source_clip=clip)
    return params

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    clips=[json.loads(line) for line in (ROOT/'data/manifests/clips.jsonl').read_text().splitlines()]
    if args.limit:
        clips=clips[:args.limit]
    for i,clip in enumerate(clips):
        name=f"{clip['track_id']}_{clip['start_seconds']:.3f}"
        video=OUT/f'{name}.mp4'
        parameters=OUT/f'{name}.json'
        if video.exists() and parameters.exists():
            continue
        feature_path=ROOT/'data/features'/f"{clip['track_id']}.npz"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        with np.load(feature_path) as features:
            metadata=json.loads(feature_path.with_suffix('.json').read_text())
            temporary=video.with_suffix('.partial.mp4')
            params=render(clip,features,metadata,temporary)
        temporary.replace(video)
        parameters.write_text(json.dumps(params,indent=2))
        print(f'{i+1}/{len(clips)} {video.name}',flush=True)

if __name__=='__main__':
    main()
