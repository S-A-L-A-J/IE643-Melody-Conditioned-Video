import os
"""Build randomized human review of learned predictions and matched controls."""
import hashlib
import json
import random
import re
from pathlib import Path
ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))

def main():
    rows=[r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines()) if r['split']=='test' and r['style']=='rings' and r['audio_start_seconds']==0]
    assert len(rows)==14
    entries=[]
    for row in rows:
        for variant in ['correct','other_track','shifted_1s','target']:
            path=ROOT/'outputs/prior_32f_test'/f'{row["id"]}_{variant}.mp4'
            if not path.is_file():raise FileNotFoundError(path)
            entries.append({'review_id':hashlib.sha256(('IE643-learned-32f:'+row['id']+variant).encode()).hexdigest()[:10],'video':'../'+str(path.relative_to(ROOT)),'audio':'../'+row['audio_path'],'style':variant,'pair_id':row['id']})
    random.Random(644).shuffle(entries)
    page=(ROOT/'reports/provisional_human_review.html').read_text()
    page=re.sub(r'const entries=.*?;\nconst key=',lambda _: 'const entries='+json.dumps(entries)+';\nconst key=',page,count=1)
    page=page.replace('IE643_review_v2','IE643_learned_review_32f').replace('IE643_human_ratings.csv','IE643_learned_ratings.csv').replace('Provisional held-out video review','Learned model and control review').replace('The visual style is hidden during review.','Model and control labels are hidden during review. The 56 clips include predictions, wrong-song and shifted-input controls, and synthetic targets, all shown at 8 fps. Rate the visible match to the reference audio.')
    (ROOT/'reports/learned_human_review.html').write_text(page)
    print('Created 56-clip randomized review; ratings remain empty.')
if __name__=='__main__':main()
