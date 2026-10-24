import os
"""Index original audio, retaining provenance and withholding unverified song splits."""
import hashlib
import json
from pathlib import Path
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    records = []
    seen = set()
    for path in sorted((ROOT / 'data/raw').glob('*')):
        if path.suffix.lower() not in {'.wav', '.mp3', '.flac'}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        info = sf.info(path)
        samples, _ = sf.read(path, dtype='float32')
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        reasons = []
        if digest in seen:
            reasons.append('exact_duplicate')
        if info.duration < 4:
            reasons.append('shorter_than_4_seconds')
        if not np.isfinite(samples).all() or rms < 1e-5:
            reasons.append('silent_empty_or_nonfinite')
        records.append(dict(id=digest[:16], path=str(path.relative_to(ROOT)),
                            sha256=digest, duplicate=digest in seen,
                            eligible=not reasons, exclusion_reasons=reasons, rms=rms,
                            duration=info.duration, sample_rate=info.samplerate,
                            channels=info.channels, source_title=path.stem,
                            song_id=None, split='development',
                            review_status='needs_song_identity_and_audio_quality_review',
                            source_dataset='beastLucifer/navrasa-5000-dataset',
                            source_revision='8056599a540110a31adab3cd297134e099aefc65'))
        seen.add(digest)
    output = ROOT / 'data/manifests/tracks.jsonl'
    output.write_text(''.join(json.dumps(r)+'\n' for r in records))
    report = dict(files=len(records), unique_files=len(seen),
                  eligible_files=sum(r['eligible'] for r in records),
                  excluded_files=sum(not r['eligible'] for r in records),
                  total_minutes=sum(r['duration'] for r in records)/60,
                  sample_rates=sorted({r['sample_rate'] for r in records}),
                  split_status='Development only: song identity unresolved; no held-out claims.')
    (ROOT/'reports/dataset_inventory.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
