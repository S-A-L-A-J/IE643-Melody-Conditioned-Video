"""Build same-audio comparisons and verify saved media; no sampling."""
import os,sys,json,subprocess
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path[:0]=[str(WORK),str(WORK.parent.parent),str(Path(__file__).resolve().parent.joinpath('../../../../../project').resolve()/'scripts')]
import numpy as np,cv2
import extension as ext
def mux_comparison(name,rgb,row):
 path=OUT/'videos'/f'{name}.mp4'
 if path.exists():return
 height,width=rgb.shape[1:3]
 subprocess.run(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','8','-i','pipe:0','-ss',str(row['audio_start_seconds']),'-i',str(ext.ROOT/row['audio_path']),'-map','0:v:0','-map','1:a:0','-t',str(len(rgb)/8),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)],input=rgb.tobytes(),check=True)

OUT=HERE/'review_v2';OUT.mkdir(exist_ok=True);(OUT/'videos').mkdir(exist_ok=True)
for number in [0,3,7]:
 record=json.loads((HERE/'validation8/prepared.json').read_text())['records'][number]
 panels=[]
 for directory,name,label in [('validation8','target','synthetic target'),('validation8','prior','music GRU predictor'),('paired_validation','frozen_pair','frozen paired diffusion'),('paired_validation','adapted_pair','adapted paired diffusion')]:
  frames=np.load(HERE/directory/'videos'/f'{name}_{number}.npz')['rgb'];panel=np.stack([cv2.resize(f,(256,256)) for f in frames]);panel=np.pad(panel,((0,0),(28,0),(0,0),(0,0)))
  for frame in panel:cv2.putText(frame,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
  panels.append(panel)
 old=ext.HERE;ext.HERE=OUT
 try:mux_comparison(f'comparison_validation_{number}',np.concatenate(panels,2),record['row'])
 finally:ext.HERE=old
# Render long comparisons from saved square frames with explicit rectangular dimensions.
longdir=HERE/'long_validation/videos'
if (HERE/'long_validation/results.json').exists():
 row=json.loads((WORK.parent/'config.json').read_text())['row']
 for seconds in [8,16]:
  names=[f'd{seconds:02d}_correct_rolling_prior',f'd{seconds:02d}_correct_rolling_paired']
  labels=['music GRU predictor','adapted paired diffusion']
  if seconds==8:
   names+=['d08_wrong_rolling_paired','d08_shift_rolling_paired'];labels+=['wrong audio via GRU + diffusion','shifted audio via GRU + diffusion']
  panels=[]
  for name,label in zip(names,labels):
   frames=np.load(longdir/f'{name}.npz')['rgb'];panel=np.stack([cv2.resize(f,(256,256)) for f in frames]);panel=np.pad(panel,((0,0),(28,0),(0,0),(0,0)))
   for frame in panel:cv2.putText(frame,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
   panels.append(panel)
  mux_comparison(f'comparison_{seconds:02d}s_v2',np.concatenate(panels,2),row)
checks={}
paths=list((OUT/'videos').glob('*.mp4'))+[p for p in (HERE/'long_validation/videos').glob('*.mp4') if not p.name.startswith('comparison_')]
for path in paths:
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]));v=next(s for s in info['streams'] if s['codec_type']=='video');a=next(s for s in info['streams'] if s['codec_type']=='audio')
 result=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-f','null','-'],capture_output=True,text=True);assert result.returncode==0,result.stderr
 ahash=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-f','md5','-']).decode().strip()
 checks[str(path.relative_to(HERE))]={'frames':int(v['nb_frames']),'fps':v['r_frame_rate'],'duration':info['format']['duration'],'size':[v['width'],v['height']],'audio_decode_md5':ahash,'decode_ok':True}
# All long panels and controls use the same source audio and mux settings per duration.
for seconds in [8,16]:
 subset=[x for p,x in checks.items() if f'd{seconds:02d}_' in p or f'comparison_{seconds:02d}s' in p]
 if subset:assert len({x['audio_decode_md5'] for x in subset})==1
(HERE/'MEDIA_VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n');print('Verified',len(checks),'videos')
