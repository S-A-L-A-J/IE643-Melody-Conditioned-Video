import os
"""Verify feature coverage and exact clip-to-feature indexing."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tracks = [json.loads(line) for line in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines()]
eligible = {row['id'] for row in tracks if row['eligible']}
clips = [json.loads(line) for line in (ROOT/'data/manifests/clips.jsonl').read_text().splitlines()]
checked = {}
for track_id in eligible:
    data_path = ROOT/'data/features'/f'{track_id}.npz'
    meta_path = data_path.with_suffix('.json')
    if not data_path.exists() or not meta_path.exists():
        raise AssertionError(f'Missing features: {track_id}')
    meta = json.loads(meta_path.read_text())
    with np.load(data_path) as data:
        times = data['times']
        length = len(times)
        assert meta['frames'] == length
        assert np.allclose(times, np.arange(length)/16, atol=1e-6)
        assert data['chroma'].shape == (length,12)
        assert data['band_energy'].shape == (length,4)
        for name in ['onset_strength','beat_impulse','rms','dominant_pitch_hz','pitch_confidence']:
            assert data[name].shape == (length,), (track_id,name)
        assert np.isfinite(data['chroma']).all()
        assert np.isfinite(data['band_energy']).all()
        assert np.isfinite(data['rms']).all()
        assert np.isfinite(data['onset_strength']).all()
        assert np.all(data['rms'] >= 0)
        assert np.all(data['chroma'] >= 0)
        assert np.all((data['beat_times'] >= 0) & (data['beat_times'] < meta['duration_seconds']))
    checked[track_id] = length
for clip in clips:
    fps = 16
    start = round(clip['start_seconds']*fps)
    count = round(clip['duration']*fps)
    assert abs(start/fps-clip['start_seconds']) < 1e-6
    assert count == 64
    assert start+count <= checked[clip['track_id']], clip
report = dict(status='passed',tracks=len(checked),feature_fps=16,
              clips=len(clips),clip_frames=64,
              total_feature_frames=sum(checked.values()),
              mood_status='pending_frozen_encoder',
              pitch_note='Dominant spectral pitch is a heuristic, not verified melodic transcription.')
(ROOT/'reports/feature_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
