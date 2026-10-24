import os
"""PyTorch loader for aligned preliminary E0 music/video pairs."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[1]

class PairedClipDataset(Dataset):
    def __init__(self, root=ROOT, manifest='data/manifests/synthetic_pairs.jsonl'):
        self.root=Path(root)
        self.rows=[json.loads(line) for line in (self.root/manifest).read_text().splitlines()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        capture=cv2.VideoCapture(str(self.root/row['video_path']))
        frames=[]
        try:
            while len(frames)<row['video_frames']:
                ok,frame=capture.read()
                if not ok:
                    raise ValueError(f"Short video: {row['video_path']}")
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        video=np.stack(frames).astype(np.float32)/127.5-1
        video=torch.from_numpy(video).permute(3,0,1,2).contiguous()
        first=row['feature_start_frame']
        last=first+row['feature_frames']
        with np.load(self.root/row['features_path']) as source:
            onset=source['onset_strength'][first:last]
            beat=source['beat_impulse'][first:last]
            chroma=source['chroma'][first:last]
            pitch=source['dominant_pitch_hz'][first:last]
            confidence=source['pitch_confidence'][first:last]
            rms=source['rms'][first:last]
            bands=source['band_energy'][first:last]
        feature_metadata=json.loads((self.root/row['features_path']).with_suffix('.json').read_text())
        tempo=np.full(len(onset),feature_metadata['tempo_bpm'],dtype=np.float32)
        rhythm=np.stack([onset,beat,tempo],axis=-1).astype(np.float32)
        melody=np.concatenate([chroma,pitch[:,None],confidence[:,None]],axis=-1).astype(np.float32)
        dynamics=np.concatenate([rms[:,None],bands],axis=-1).astype(np.float32)
        if not (len(onset)==len(chroma)==len(rms)==video.shape[1]==row['video_frames']):
            raise ValueError(f"Misaligned pair: {row['id']}")
        item=dict(id=row['id'],video=video,
                  rhythm=torch.from_numpy(rhythm),melody=torch.from_numpy(melody),
                  dynamics=torch.from_numpy(dynamics))
        if 'global_audio_context_path' in row:
            with np.load(self.root/row['global_audio_context_path']) as context:
                item['global_context']=torch.from_numpy(context['embedding'].copy())
                item['mood_softmax']=torch.from_numpy(context['mood_softmax'].copy())
            item['style']=row['style']
            item['split']=row['split']
        return item

if __name__=='__main__':
    dataset=PairedClipDataset()
    item=dataset[0]
    print(dict(pairs=len(dataset),id=item['id'],
               shapes={key:list(value.shape) for key,value in item.items() if isinstance(value,torch.Tensor)},
               video_range=[float(item['video'].min()),float(item['video'].max())]))
