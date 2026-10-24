import os
"""Check structural integrity, not perceptual quality of vocal separation."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
tracks=[json.loads(x) for x in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines()]
stats=[]
missing=[]
for row in tracks:
    if not row['eligible']:
        continue
    dest=ROOT/'data/stems'/row['id']
    if not (dest/'complete.json').exists():
        missing.append(row['id'])
        continue
    mix,sr=sf.read(ROOT/row['path'],always_2d=True)
    vocal,v_sr=sf.read(dest/'vocals.wav',always_2d=True)
    inst,i_sr=sf.read(dest/'instrumental.wav',always_2d=True)
    assert sr==v_sr==i_sr
    assert mix.shape==vocal.shape==inst.shape
    assert np.isfinite(vocal).all() and np.isfinite(inst).all()
    stats.append(dict(id=row['id'], instrumental_peak=float(np.abs(inst).max()),
                      reconstruction_relative_rms=float(np.sqrt(np.mean((mix-vocal-inst)**2))/(np.sqrt(np.mean(mix**2))+1e-12))))
report=dict(indexed_files=len(tracks), unique_files=sum(not r['duplicate'] for r in tracks),
            eligible_files=sum(r['eligible'] for r in tracks),
            separated_files=len(stats),missing=missing,integrity='passed' if not missing else 'incomplete',
            note='Integrity checks do not establish vocal removal quality; listening review is pending.',tracks=stats)
(ROOT/'reports/dataset_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='tracks'},indent=2))
if missing:
    raise SystemExit(1)
