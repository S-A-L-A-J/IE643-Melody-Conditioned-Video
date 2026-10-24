import os
"""Create a local listening comparison for completed eligible excerpts."""
import html
import json
from pathlib import Path
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[1]
tracks=[json.loads(x) for x in (ROOT/'data/manifests/tracks.jsonl').read_text().splitlines()]
sections=[]
for row in tracks:
    if not row['eligible'] or not (ROOT/'data/stems'/row['id']/'complete.json').exists():
        continue
    name=row['source_title']
    # Favor single-song titles for convenient initial review.
    if any(w in name.lower() for w in ['jukebox','top_','collection','nonstop','non_stop']):
        continue
    paths=[('Original','../'+row['path']),('Instrumental',f"../data/stems/{row['id']}/instrumental.wav"),
           ('Vocals',f"../data/stems/{row['id']}/vocals.wav")]
    players=''.join(f'<div><p>{label}</p><audio controls preload="none" src="{quote(path)}"></audio></div>' for label,path in paths)
    sections.append(f'<section><h2>{html.escape(name.replace("_"," "))}</h2><div class="players">{players}</div></section>')
    if len(sections)==8:
        break
page='''<!doctype html><meta charset="utf-8"><title>IE643 vocal separation review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f6f7fa;color:#172033}section{background:white;padding:24px;margin:20px 0;border-radius:12px}h2{font-size:18px}.players{display:flex;gap:24px;flex-wrap:wrap}audio{width:300px}p{line-height:1.6}</style>
<h1>Vocal separation listening review</h1><p>Compare the original, instrumental and vocals. Listen for remaining words or humming, missing instruments, distorted transients and gaps. Separation is not guaranteed to preserve the singer’s melodic line. These development excerpts have not passed a perceptual quality review.</p>'''+''.join(sections)
(ROOT/'reports/listening_review.html').write_text(page)
print(f'Created listening review for {len(sections)} excerpts')
