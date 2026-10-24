import os
"""Package the preselected first validation example with reference audio."""
import json,subprocess
from pathlib import Path
ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))

def main():
    report=json.loads((ROOT/'reports/rhythm_validation_comparison.json').read_text())
    pair=report['results'][0]['id']
    row=next(r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines()) if r['id']==pair)
    cards=[]
    for variant,title in [('point_control','Previous sampling, matched training'),('causal_peak','Beat-preserving sampling'),('target','Synthetic target')]:
        src=ROOT/'outputs/rhythm_validation'/f'{pair}_{variant}.mp4';dst=src.with_name(src.stem+'_with_audio.mp4')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),'-i',str(ROOT/row['audio_path']),'-t','4','-map','0:v:0','-map','1:a:0','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(dst)],check=True)
        cards.append(f'<section><h2>{title}</h2><video controls playsinline src="../{dst.relative_to(ROOT)}"></video></section>')
    metrics=''.join(f'<tr><td>{name}</td><td>{v["f1"]:.3f}</td><td>{v["laplacian_variance"]:.2f}</td></tr>' for name,v in report['aggregate'].items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IE643 rhythm sampling experiment</title><style>body{max-width:1100px;margin:2rem auto;padding:1rem;background:#101217;color:#eef1f5;font:16px/1.5 system-ui}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}section{background:#1b2029;padding:1rem;border-radius:10px}video{width:100%}h2{font-size:18px}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #46505d}</style><h1>Does preserving between-frame beats help?</h1><p>Both models started from the same checkpoint and used the same training budget and validation selection rule. This page shows the first validation example, selected before scoring. All clips are four seconds at 8 fps; these are direct learned predictions, not diffusion.</p><div class="grid">'''+''.join(cards)+'</div><h2>All 11 validation tracks</h2><table><tr><th>Condition</th><th>Beat-event F1</th><th>Edge variance</th></tr>'+metrics+'</table><p>Event timing and edge variance are exploratory measures. Edge variance can increase with noise and does not establish perceptual quality. Human ratings remain pending. No new test-set evaluation was run for this experiment.</p></html>'
    (ROOT/'reports/rhythm_sampling_review.html').write_text(page)
    print('Saved rhythm_sampling_review.html')
if __name__=='__main__':main()
