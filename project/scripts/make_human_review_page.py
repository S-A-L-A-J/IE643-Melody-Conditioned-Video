import os
"""Create an offline rating page for provisional held-out synthetic videos."""
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    rows=[json.loads(line) for line in (ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines()]
    entries=[row for row in rows if row['split']=='test' and row['audio_start_seconds']==0]
    assert len(entries)==42
    rng=random.Random(643)
    rng.shuffle(entries)
    visible=[{'review_id':hashlib.sha256(('IE643-review-v2:'+row['id']).encode()).hexdigest()[:10],
              'video':'../'+row['video_path'],'audio':'../'+row['audio_path'],
              'style':row['style'],'pair_id':row['id']} for row in entries]
    page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IE643 provisional video review</title>
<style>
body{max-width:780px;margin:2rem auto;padding:0 1rem;background:#101217;color:#eef1f5;font:16px/1.5 system-ui,sans-serif}
video{display:block;width:100%;max-width:420px;background:#000;border-radius:8px}
.panel{background:#1c222b;border:1px solid #353e4c;border-radius:12px;padding:1.2rem;margin-top:1rem}
.ratings{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem}
label{display:block}select,textarea,button{font:inherit}select,textarea{width:100%;box-sizing:border-box;background:#fff;color:#111;border-radius:6px;padding:.3rem}
textarea{min-height:70px}button{padding:.55rem .8rem;border:0;border-radius:7px;background:#82d9c8;color:#102622;cursor:pointer;margin:.5rem .4rem .5rem 0}
small{color:#b4bfca}
</style></head><body>
<h1>Provisional held-out video review</h1>
<p>Rate each four-second video after listening to its synchronized instrumental. The visual style is hidden during review. This local page saves progress in your browser and can download a CSV; it does not send ratings anywhere.</p>
<p id="progress"></p><div class="panel"><video id="video" controls playsinline preload="metadata"></video><audio id="audio" preload="metadata"></audio>
<small>Use the video play button. If audio does not start automatically, press play again.</small></div>
<div class="panel"><div class="ratings">
<label>Beat match<select id="beat"><option value="">Choose</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<label>Continuity<select id="continuity"><option value="">Choose</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<label>Mood match<select id="mood"><option value="">Choose</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<label>Overall<select id="overall"><option value="">Choose</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
</div><label>Notes<textarea id="notes"></textarea></label>
<button id="previous">Previous</button><button id="save">Save and next</button><button id="download">Download ratings CSV</button></div>
<p><small>Scale: 1 = poor, 5 = excellent. These files may include jukeboxes or mashups; the split is provisional.</small></p>
<script>
const entries=__ENTRIES__;
const key='IE643_review_v2';
let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
let index=0;
const video=document.getElementById('video'),audio=document.getElementById('audio');
function show(){let item=entries[index],score=ratings[item.review_id]||{};video.pause();audio.pause();video.src=item.video;audio.src=item.audio;
  for(let field of ['beat','continuity','mood','overall','notes'])document.getElementById(field).value=score[field]||'';
  document.getElementById('progress').textContent=`Clip ${index+1} of ${entries.length} · ${Object.keys(ratings).length} saved · review ID ${item.review_id}`;}
function save(){let item=entries[index],score={};for(let field of ['beat','continuity','mood','overall','notes'])score[field]=document.getElementById(field).value;
  ratings[item.review_id]=score;try{localStorage.setItem(key,JSON.stringify(ratings))}catch(e){};}
video.addEventListener('play',()=>{audio.currentTime=video.currentTime;audio.play().catch(()=>{});});
video.addEventListener('pause',()=>audio.pause());
video.addEventListener('seeking',()=>{audio.currentTime=video.currentTime;});
video.addEventListener('ended',()=>audio.pause());
document.getElementById('previous').onclick=()=>{save();index=Math.max(0,index-1);show();};
document.getElementById('save').onclick=()=>{save();index=Math.min(entries.length-1,index+1);show();};
document.getElementById('download').onclick=()=>{save();let header=['review_id','pair_id','style','beat','continuity','mood','overall','notes'];
  let quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';let lines=[header.join(',')];
  for(let item of entries){let score=ratings[item.review_id];if(score)lines.push(header.map(k=>quote(score[k]??item[k]??'')).join(','));}
  let blob=new Blob([lines.join('\\n')+'\\n'],{type:'text/csv'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='IE643_human_ratings.csv';a.click();URL.revokeObjectURL(url);};
show();
</script></body></html>'''
    page=page.replace('__ENTRIES__',json.dumps(visible))
    output=ROOT/'reports/provisional_human_review.html'
    output.write_text(page)
    print(output)


if __name__=='__main__':
    main()
