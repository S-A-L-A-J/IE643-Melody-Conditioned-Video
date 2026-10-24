import os
"""Package existing model outputs with the same reference instrumental."""
import json
import subprocess
from pathlib import Path
ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))

def main():
    report=json.loads((ROOT/'reports/audio_prior_32f_preview.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['id']==report['pair_id'])
    cards=[]
    for prefix,labels in [('audio_prior_32f',['correct','other_song','shifted_one_second','target']),('prior_guided',['correct','other_song','shifted_one_second'])]:
        for label in labels:
            source=ROOT/'outputs'/f'{prefix}_{label}.mp4'
            dest=source.with_name(source.stem+'_with_audio.mp4')
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-ss',str(row['audio_start_seconds']),'-i',str(ROOT/row['audio_path']),'-t','4','-map','0:v:0','-map','1:a:0','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(dest)],check=True)
            title=('8 fps learned prior' if prefix=='audio_prior_32f' else '2 fps diffusion')+' · '+label.replace('_',' ')
            cards.append(f'<section><h2>{title}</h2><video controls playsinline preload="metadata" src="../outputs/{dest.name}"></video></section>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IE643 model comparison</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;background:#101217;color:#eff1f5;font:16px/1.5 system-ui} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}section{padding:1rem;background:#1b2029;border-radius:12px}video{width:100%;image-rendering:auto}h2{font-size:18px}</style><h1>Music control: current model comparison</h1><p>All clips play the same reference instrumental. “Other song” and “shifted one second” change the model input while keeping that reference audio, so you can compare timing. The synthetic target is programmed from music.</p><p>The 8 fps prior is a direct learned prediction, not diffusion; its images remain blurry. The diffusion model produces sharper rings at 2 fps but has failed the music-control tests. This is a provisional test excerpt: source-song identity and compilation overlap remain unverified.</p><div class="grid">'''+''.join(cards)+'</div><p>Four seconds per clip, 128 × 128 pixels. Human ratings are pending.</p></html>'
    (ROOT/'reports/model_comparison_review.html').write_text(html)
    print('Packaged seven playable comparisons.')
if __name__=='__main__':main()
