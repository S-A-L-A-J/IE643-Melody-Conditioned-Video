"""Read-only numerical and preservation checks on saved experiment evidence."""
import json,hashlib,datetime
from pathlib import Path
import torch
from safetensors.torch import load_file
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
checks=[]
for model in ['frozen','adapted']:
 for i in range(8):
  plus=HERE/'validation8/latents'/f'{model}_{i}.safetensors';minus=HERE/'paired_validation/latents'/f'{model}_negative_{i}.safetensors';mean=HERE/'paired_validation/latents'/f'{model}_pair_{i}.safetensors'
  p=json.loads(plus.with_suffix('.json').read_text());m=json.loads(minus.with_suffix('.json').read_text());assert p['base_float32_noise_sha256']==m['unsigned_seed_noise_sha256']
  expected=(load_file(plus)['latents']+load_file(minus)['latents'])/2;actual=load_file(mean)['latents'];assert torch.equal(expected,actual) and torch.isfinite(actual).all();checks.append(str(mean.relative_to(HERE)))
for folder,variants,indices in [('confirmation3',['frozen_pair','candidate'],range(8,11))]:
 for model in variants:
  for i in indices:
   paths=[HERE/folder/'latents'/f'{model}_{i}_sign{sign}.safetensors' for sign in [1,-1]];meta=[json.loads(p.with_suffix('.json').read_text()) for p in paths];assert meta[0]['unsigned_seed_noise_sha256']==meta[1]['unsigned_seed_noise_sha256']
   expected=sum(load_file(p)['latents'] for p in paths)/2;mean=HERE/folder/'latents'/f'{model}_{i}.safetensors';assert torch.equal(expected,load_file(mean)['latents']);checks.append(str(mean.relative_to(HERE)))
long=json.loads((HERE/'long_validation/results.json').read_text())
noise={}
for key in long['results']:
 paths=[HERE/'long_validation/latents'/f'{key}_sign{sign}.safetensors' for sign in [1,-1]];meta=[json.loads(p.with_suffix('.json').read_text()) for p in paths];assert meta[0]['unsigned_seed_noise_sha256']==meta[1]['unsigned_seed_noise_sha256'];noise[key]=meta[0]['unsigned_seed_noise_sha256']
 expected=sum(load_file(p)['latents'] for p in paths)/2;mean=HERE/'long_validation/latents'/f'{key}_paired.safetensors';assert torch.equal(expected,load_file(mean)['latents']) and torch.isfinite(expected).all();checks.append(str(mean.relative_to(HERE)))
assert len({v for k,v in noise.items() if k.startswith('d08')})==1
inventory={}
for name in ['input_inventory_before.json','ARTIFACT_MANIFEST.json','ARTIFACT_MANIFEST_GPU.json']:
 data=json.loads((ROOT/name).read_text());mismatches=[]
 for name2,value in data.items():
  if not isinstance(value,dict) or 'sha256' not in value:continue
  path=Path(name2) if name2.startswith('/') else ROOT/name2
  if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=value['sha256']:mismatches.append(name2)
 assert not mismatches;inventory[name]={'tracked':len(data),'mismatches':mismatches}
report={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pair_means_exactly_verified':len(checks),'signed_noise_hashes_match':True,'long_controls_share_noise':True,'original_inventories':inventory,'deliverable_videos_verified':len(json.loads((HERE/'MEDIA_VERIFICATION.json').read_text())),'fresh_evaluation':'Not loaded or reclassified','human_ratings':None,'more_training_performed_during_resumption':False,'status':'completed; no further training or sampling planned'}
(HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
manifest={}
for path in sorted(HERE.rglob('*')):
 if path.is_file() and path.name!='ARTIFACT_MANIFEST.json' and '__pycache__' not in path.parts:
  manifest[str(path.relative_to(HERE))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(HERE/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
