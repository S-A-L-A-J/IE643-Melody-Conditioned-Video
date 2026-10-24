"""Inference-only prior preservation/noise-strength pilot; no training or selection."""
import os,sys,json,gc,time,subprocess
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=HERE.parent/'diffusion_refinement/gpu_extension';PREV=WORK/'resumed_20261024'
sys.path[:0]=[str(PREV),str(WORK),str(HERE.parent),str(Path(__file__).resolve().parent.joinpath('../../../project').resolve()/'scripts')]
import torch,numpy as np,cv2
from safetensors.torch import load_file,save_file
from paired_backend import load_pipeline,refine
import extension as ext

def dump(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
@torch.inference_mode()
def main():
 started=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(1)
 for folder in ['latents','videos']:(HERE/folder).mkdir(exist_ok=True)
 strengths=[.06,.30,.70]
 plan={'strengths':strengths,'base_steps':100,'noise_signs':[1,-1],'seed':643,'frames':32,'priors':['correct','wrong'],'no_training':True,'purpose':'Inference dependence on prior vs size of correction, not learned conditioning or new quality selection','data':'first four seconds of previously inspected long-validation recording, two GRU priors; no fresh/test use'}
 if not (HERE/'PLAN.json').exists():dump(HERE/'PLAN.json',plan)
 source=load_file(WORK/'rolling_latents.safetensors');priors={k:source[f'd08_{k}_rolling'][:,:,:32].contiguous() for k in ['correct','wrong']}
 save_file(priors,HERE/'priors.safetensors')
 chosen=json.loads((PREV/'CHOSEN_CANDIDATE.json').read_text());pipe=load_pipeline(lora_path=(WORK/'geometry_lora_continuation/lora_step128.safetensors'))
 outputs={};metas=[]
 for strength in strengths:
  for kind,prior in priors.items():
   values=[]
   for sign in [1,-1]:
    name=f'{kind}_s{strength:.2f}_sign{sign}';path=HERE/'latents'/f'{name}.safetensors'
    if path.exists():value=load_file(path)['latents'];meta=json.loads(path.with_suffix('.json').read_text())
    else:
     value,meta=refine(pipe,prior,strength,100,1.,643,noise_sign=sign)
     save_file({'latents':value},path);dump(path.with_suffix('.json'),meta)
    values.append(value);metas.append(meta)
   mean=(values[0]+values[1])/2;outputs[f'{kind}_s{strength:.2f}']=mean
   save_file({'latents':mean.contiguous()},HERE/'latents'/f'{kind}_s{strength:.2f}_paired.safetensors')
   print('Finished',kind,strength,'wall',time.monotonic()-started,flush=True)
 del pipe;gc.collect();torch.cuda.empty_cache()
 vae=ext.AutoencoderKL.from_pretrained(ext.ROOT/'models/diffusion/sd15/vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
 row=json.loads((WORK.parent/'config.json').read_text())['row'];ext.HERE=HERE
 with np.load(ext.ROOT/row['features_path']) as f:beats=f['beat_times']-row['audio_start_seconds'];beats=beats[(beats>=0)&(beats<4)]
 frames={}
 for name,z in {**{k+'_prior':v for k,v in priors.items()},**outputs}.items():
  rgb=ext.decode(vae,z);frames[name]=rgb;ext.mux(name,rgb,row)
 records=[];distance_before=float((priors['correct']-priors['wrong']).square().mean())
 rgb_before=float(np.abs(frames['correct_prior'].astype(float)-frames['wrong_prior']).mean()/255)
 for strength in strengths:
  a=outputs[f'correct_s{strength:.2f}'];b=outputs[f'wrong_s{strength:.2f}'];pair={'strength':strength,'latent_between_prior_outputs_mse':float((a-b).square().mean()),'latent_between_input_priors_mse':distance_before,'input_output_separation_ratio':float((a-b).square().mean())/distance_before,'rgb_between_outputs_mae':float(np.abs(frames[f'correct_s{strength:.2f}'].astype(float)-frames[f'wrong_s{strength:.2f}']).mean()/255),'rgb_between_input_priors_mae':rgb_before,'outputs':{}}
  for kind in priors:
   name=f'{kind}_s{strength:.2f}';rgb=frames[name]
   pair['outputs'][kind]={'latent_mse_from_own_prior':float((outputs[name]-priors[kind]).square().mean()),'rgb_mae_from_own_prior':float(np.abs(rgb.astype(float)-frames[kind+'_prior']).mean()/255),**ext.diagnostics(rgb,beats)}
  records.append(pair)
 strips=[]
 for name in ['correct_prior','wrong_prior']+[f'{k}_s{s:.2f}' for s in strengths for k in priors]:
  strip=np.concatenate([cv2.resize(cv2.cvtColor(frames[name][i],cv2.COLOR_RGB2BGR),(192,192)) for i in [4,12,20,28]],1)
  strip=cv2.copyMakeBorder(strip,26,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(strip,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);strips.append(strip)
 cv2.imwrite(str(HERE/'comparison.png'),np.concatenate(strips,0))
 for kind in priors:
  names=[kind+'_prior']+[f'{kind}_s{s:.2f}' for s in strengths];panels=[]
  for name in names:
   panel=np.stack([cv2.resize(x,(256,256)) for x in frames[name]]);panel=np.pad(panel,((0,0),(28,0),(0,0),(0,0)))
   for f in panel:cv2.putText(f,name,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1)
   panels.append(panel)
  composite=np.concatenate(panels,axis=2);h,w=composite.shape[1:3]
  subprocess.run(['/usr/bin/ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','8','-i','pipe:0','-ss',str(row['audio_start_seconds']),'-i',str(ext.ROOT/row['audio_path']),'-map','0:v:0','-map','1:a:0','-t','4','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(HERE/'videos'/f'{kind}_strength_comparison.mp4')],input=composite.tobytes(),check=True)
 dump(HERE/'results.json',{'plan':plan,'records':records,'baseline_diagnostics':{k:ext.diagnostics(frames[k+'_prior'],beats) for k in priors},'runtime_seconds':time.monotonic()-started,'unsigned_noise_hashes':sorted(set(m['unsigned_seed_noise_sha256'] for m in metas)),'limitations':'One previously inspected validation excerpt, two prior inputs. No direct diffusion audio conditioning; separation is inherited from initialized prior. No quality improvement/final-test claim. Wrong audio evaluated against same correct reference soundtrack and beats. Paired-noise averaging held fixed; only strength varied.'})
 print('COMPLETE',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
