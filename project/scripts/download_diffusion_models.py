import os
"""Download pinned diffusion checkpoints only when a safe disk reserve remains."""
import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT=Path(__file__).resolve().parents[1]
MODELS=[
    ('stable-diffusion-v1-5/stable-diffusion-v1-5','451f4fe16113bff5a5d2269ed5ad43b0592e9a14','sd15'),
    ('guoyww/animatediff-motion-adapter-v1-5-2','6167b88ffe39b4441fdf2113e77b99a6f56b7906','animatediff'),
]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=ROOT/'models/diffusion')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    plan=[]
    for repo,revision,folder in MODELS:
        info=HfApi().model_info(repo,revision=revision,files_metadata=True)
        selected=[f for f in info.siblings if (
            f.rfilename.endswith('.fp16.safetensors') or
            (f.rfilename.endswith(('.json','.txt')) and not f.rfilename.startswith(('onnx/','flax/')))
        )]
        missing=sum(f.size or 0 for f in selected if not (args.directory/folder/f.rfilename).exists())
        plan.append(dict(repo=repo,revision=revision,folder=folder,
                         files=[f.rfilename for f in selected],download_bytes=missing))
    free=shutil.disk_usage(args.directory).free
    needed=sum(p['download_bytes'] for p in plan)
    reserve=3*1024**3
    report=dict(models=plan,free_bytes=free,download_bytes=needed,reserve_bytes=reserve,
                fits=free>=needed+reserve)
    (ROOT/'reports/diffusion_download_plan.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='models'},indent=2))
    if args.check_only:
        return
    if not report['fits']:
        raise SystemExit('Insufficient disk headroom. Choose --directory on a larger volume.')
    for p in plan:
        snapshot_download(p['repo'],revision=p['revision'],allow_patterns=p['files'],
                          local_dir=args.directory/p['folder'],max_workers=2)
    print('Pinned checkpoints downloaded. Inference/training feasibility still requires testing.')

if __name__=='__main__':
    main()
