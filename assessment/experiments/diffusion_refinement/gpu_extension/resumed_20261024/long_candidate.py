"""Fixed selected paired refinement on existing long validation predictor outputs."""
import os,sys,json,gc,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path[:0]=[str(HERE),str(WORK),str(WORK.parent.parent),str(Path(__file__).resolve().parent.joinpath('../../../../../project').resolve()/'scripts')]
import torch,numpy as np,cv2,subprocess
from safetensors.torch import load_file,save_file
from paired_backend import load_pipeline,refine
import extension as ext
OUT=HERE/'long_validation';OUT.mkdir(exist_ok=True)
for name in ['latents','videos']:(OUT/name).mkdir(exist_ok=True)
def mux_comparison(name,rgb,row):
 path=OUT/'videos'/f'{name}.mp4'
 if path.exists():return
 height,width=rgb.shape[1:3]
 subprocess.run(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','8','-i','pipe:0','-ss',str(row['audio_start_seconds']),'-i',str(ext.ROOT/row['audio_path']),'-map','0:v:0','-map','1:a:0','-t',str(len(rgb)/8),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)],input=rgb.tobytes(),check=True)

def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
@torch.inference_mode()
def main():
 torch.set_num_threads(4);cv2.setNumThreads(1)
 chosen=json.loads((HERE/'CHOSEN_CANDIDATE.json').read_text());source=load_file(WORK/'rolling_latents.safetensors')
 row=json.loads((WORK.parent/'config.json').read_text())['row']
 keys=['d08_correct_rolling','d08_wrong_rolling','d08_shift_rolling','d16_correct_rolling']
 pipe=load_pipeline(lora_path=(WORK/'geometry_lora_continuation/lora_step128.safetensors'))
 for key in keys:
  signed=[]
  for sign in [1,-1]:
   path=OUT/'latents'/f'{key}_sign{sign}.safetensors'
   if not path.exists():
    z,meta=refine(pipe,source[key],.06,100,1.,643,noise_sign=sign)
    save_file({'latents':z},path);dump(path.with_suffix('.json'),meta)
   signed.append(load_file(path)['latents'])
  path=OUT/'latents'/f'{key}_paired.safetensors'
  if not path.exists():save_file({'latents':((signed[0]+signed[1])/2).contiguous()},path)
  print('sampled',key,flush=True)
 del pipe;gc.collect();torch.cuda.empty_cache()
 vae=ext.AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
 results={};frames={}
 with np.load(ext.ROOT/row['features_path']) as f:allbeats=f['beat_times']-row['audio_start_seconds']
 for key in keys:
  duration=source[key].shape[2]/8;beats=allbeats[(allbeats>=0)&(allbeats<duration)]
  results[key]={}
  for kind,z in [('prior',source[key]),('paired',load_file(OUT/'latents'/f'{key}_paired.safetensors')['latents'])]:
   name=f'{key}_{kind}';path=OUT/'videos'/f'{name}.npz'
   if path.exists():rgb=np.load(path)['rgb']
   else:
    rgb=ext.decode(vae,z);old=ext.HERE;ext.HERE=OUT
    try:ext.mux(name,rgb,row)
    finally:ext.HERE=old
   frames[name]=rgb;results[key][kind]=ext.diagnostics(rgb,beats)
  results[key]['rgb_mae_vs_prior']=float(np.abs(frames[key+'_paired'].astype(float)-frames[key+'_prior']).mean()/255)
  rgb=frames[key+'_prior'];other=frames[key+'_paired'];indices=[4,len(rgb)//3,len(rgb)*2//3,len(rgb)-4]
  strips=[]
  for name,imgs in [('predictor',rgb),('paired adapted diffusion',other)]:
   strip=np.concatenate([cv2.resize(cv2.cvtColor(imgs[j],cv2.COLOR_RGB2BGR),(192,192)) for j in indices],1)
   strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(strip,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);strips.append(strip)
  cv2.imwrite(str(OUT/'videos'/f'{key}_comparison.png'),np.concatenate(strips,0))
 # Same audio for every panel, including controls propagated through the predictor.
 for seconds in [8,16]:
  names=[f'd{seconds:02d}_correct_rolling_prior',f'd{seconds:02d}_correct_rolling_paired']
  labels=['music GRU predictor','paired adapted diffusion']
  if seconds==8:
   names+=['d08_wrong_rolling_paired','d08_shift_rolling_paired'];labels+=['wrong audio via GRU + diffusion','shifted audio via GRU + diffusion']
  panels=[]
  for name,label in zip(names,labels):
   panel=np.stack([cv2.resize(x,(256,256)) for x in frames[name]])
   panel=np.pad(panel,((0,0),(28,0),(0,0),(0,0)))
   for frame in panel:cv2.putText(frame,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
   panels.append(panel)
  old=ext.HERE;ext.HERE=OUT
  try:mux_comparison(f'comparison_{seconds:02d}s',np.concatenate(panels,axis=2),row)
  finally:ext.HERE=old
 dump(OUT/'results.json',{'results':results,'seed':643,'candidate_config':chosen,'limitation':'One validation recording for length check; no long target evaluation. Controls enter through GRU; diffusion has no direct audio input. Timing uses estimated beats.'})
 print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
