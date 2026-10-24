"""Aggregate saved measured validation evidence; no model execution."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
 base=json.loads((HERE/'validation8/results.json').read_text());paired=json.loads((HERE/'paired_validation/results.json').read_text());confirm=json.loads((HERE/'confirmation3/results.json').read_text())
 rows=[]
 for split,prior,others in [('selection8',base['aggregate']['prior'],{**base['aggregate'],**paired['aggregate']}),('confirmation3',confirm['aggregate']['prior'],confirm['aggregate'])]:
  for name,v in others.items():
   if name=='target':continue
   rows.append({'split':split,'variant':name,'latent_mse':v['latent_mse_to_target'],'rgb_mse':v['rgb_mse_to_decoded_target'],'latent_error_reduction_percent':100*(1-v['latent_mse_to_target']/prior['latent_mse_to_target']),'rgb_error_reduction_percent':100*(1-v['rgb_mse_to_decoded_target']/prior['rgb_mse_to_decoded_target']),'estimated_beat_f1':v['estimated_beat_f1'],'flow_warp_residual':v['warp_residual'],'sharpness':v['sharpness'],'rgb_mae_vs_prior':v['rgb_mae_vs_prior']})
 aggregate={}
 for name,a,b in [('prior',base['aggregate']['prior'],confirm['aggregate']['prior']),('frozen_pair',paired['aggregate']['frozen_pair'],confirm['aggregate']['frozen_pair']),('adapted_pair',paired['aggregate']['adapted_pair'],confirm['aggregate']['candidate'])]:
  aggregate[name]={k:(8*a[k]+3*b[k])/11 for k in a}
 meta=[]
 for p in list((HERE/'validation8/latents').glob('adapted_*.json'))+list((HERE/'paired_validation/latents').glob('adapted_negative_*.json'))+list((HERE/'confirmation3/latents').glob('candidate_*_sign*.json')):
  meta.append(json.loads(p.read_text()))
 result={'table':rows,'combined11_validation_aggregate':aggregate,'runtime_4s':{'individual_diffusion_pass_median_seconds':float(np.median([x['seconds'] for x in meta])),'pair_diffusion_median_seconds_approximately':float(2*np.median([x['seconds'] for x in meta])),'maximum_sampling_allocated_MiB':max(x['peak_vram_allocated_mb'] for x in meta),'maximum_sampling_reserved_MiB':max(x['peak_vram_reserved_mb'] for x in meta),'excludes':'model load, GRU, VAE decode, media mux; PyTorch allocated memory is not total device use'},'confirmation_wins':confirm['comparisons'],'limitations':['All eleven recordings are validation; no independent final test.','First eight used for adapter/candidate selection; remaining three evaluated after freezing candidate.','Synthetic target MSE and estimated-beat F1 do not establish semantic quality, mood accuracy or human preference.','Audio enters GRU; diffusion uses fixed text and initialized video latents.','Adapter trained only on visual domain; no new training during resumption.']}
 (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='table'},indent=2))
if __name__=='__main__':main()
