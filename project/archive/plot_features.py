import os
"""Plot representative feature streams for visual inspection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tracks = [json.loads(x) for x in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines() if x]
chosen = [r for r in tracks if r['eligible']][:3]
fig, axes = plt.subplots(len(chosen), 3, figsize=(16, 9), constrained_layout=True)
for row, axis in zip(chosen,axes):
    path = ROOT/'data/features'/f"{row['id']}.npz"
    with np.load(path) as features:
        t = features['times']
        axis[0].plot(t,features['onset_strength'],color='#3b82f6',lw=1)
        for event in features['beat_times']:
            axis[0].axvline(event,color='#f97316',alpha=.3,lw=.7)
        axis[0].set_title('Onset strength and detected beats')
        axis[1].imshow(features['chroma'].T,aspect='auto',origin='lower',
                       extent=(0,t[-1]+1/16,0,12),cmap='magma')
        axis[1].set_yticks(np.arange(12)+.5, ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])
        axis[1].set_title('Harmonic pitch classes')
        axis[2].plot(t,features['rms'],color='#10b981',lw=1)
        axis[2].set_title('Energy (RMS)')
    for ax in axis:
        ax.set_xlim(0,float(t[-1])+1/16)
        ax.set_xlabel('Seconds')
    axis[0].set_ylabel(row['source_title'][:34].replace('_',' '),fontsize=9)
fig.suptitle('IE643 instrumental feature checks; pitch classes are not lead-melody transcription')
out=ROOT/'reports/feature_overview.png'
fig.savefig(out,dpi=140)
print(out)
