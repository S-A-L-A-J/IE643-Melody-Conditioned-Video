"""Read-only media/input verification; writes a new verification artifact."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records=json.loads((HERE/'input_inventory_before.json').read_text())
after={}
for name,old in records.items():
    path=Path(name)
    current={'bytes':path.stat().st_size,'sha256':sha(path),'mtime_ns':path.stat().st_mtime_ns}
    after[name]={'unchanged':old==current,**current}
assert all(v['unchanged'] for v in after.values())

videos={}
for path in sorted((HERE/'videos').glob('*.mp4')):
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],
                                   check=True,capture_output=True,text=True).stdout)
    v=next(s for s in probe['streams'] if s['codec_type']=='video')
    a=next(s for s in probe['streams'] if s['codec_type']=='audio')
    audio=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:a:0','-c:a','pcm_s16le','-f','md5','-'],
                         check=True,capture_output=True,text=True).stdout.strip()
    assert v['nb_frames']=='32' and v['avg_frame_rate']=='8/1',path
    assert abs(float(v['duration'])-4.)<.01,path
    videos[path.name]={'video_frames':int(v['nb_frames']),'fps':v['avg_frame_rate'],'video_duration':v['duration'],
                       'resolution':[v['width'],v['height']],'audio_duration':a.get('duration'),
                       'decoded_audio_pcm_md5':audio,'file_sha256':sha(path)}
assert videos
assert len({v['decoded_audio_pcm_md5'] for v in videos.values()})==1,'Audio differs across comparisons'

runs=[]
for path in (HERE/'refined_latents').glob('*.json'):
    record=json.loads(path.read_text())
    if record.get('denoising_steps',0)>0:runs.append(record)
assert len(runs)==5,len(runs)
assert len({r['noise_sha256'] for r in runs})==1
assert all(r['audio_input_to_diffusion'] is False and r['trained_parameters']==0 for r in runs)
assert all(r['text_shape']==[1,77,768] for r in runs)
assert len({json.dumps({k:r['scheduler'][k] for k in ['beta_schedule','beta_start','beta_end','prediction_type','timestep_spacing','set_alpha_to_one','steps_offset','clip_sample']},sort_keys=True) for r in runs})==1

result={'input_files':after,'videos':videos,'matched_noise_across_all_five_runs':True,
        'diffusion_receives_audio':False,'trained_parameters':0,'finite_refined_latents_asserted_at_save':True,
        'existing_checkpoint_and_requested_evidence_unchanged':True,
        'scope':'Checks requested source/evidence hashes and all generated media; not a filesystem-wide attestation.'}
with (HERE/'verification.json').open('x') as f:json.dump(result,f,indent=2)
print(f'PASS: {len(after)} preserved inputs; {len(videos)} four-second videos with identical decoded audio; five matched-noise refinement runs.')
