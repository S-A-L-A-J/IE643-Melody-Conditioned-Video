import os
"""Small correctness checks for training/inference parity and invalid inputs."""
import json
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import torch
from inference import ROOT,conditioning,generate
from train_audio_latent_prior import load_data
from metrics import score


def main():
    assert score([1.],[.95,1.,1.05])['matches']==1
    assert score([0.,1.],[0.,1.])['f1']==1
    assert score([0.],[1.])['f1']==0
    d=load_data(32,2,'causal_peak',False)['validation']
    rows={r['id']:r for r in map(json.loads,(ROOT/'data/manifests/synthetic_pairs_v2.jsonl').read_text().splitlines())}
    for i in [0,1,6]:
        row=rows[d['ids'][i]]
        with np.load(ROOT/row['features_path']) as arrays:
            meta=json.loads((ROOT/row['features_path']).with_suffix('.json').read_text())
            actual=conditioning(arrays,meta,row['feature_start_frame'])
        torch.testing.assert_close(actual,d['music'][i],rtol=0,atol=0)
    checks=['exact training/inference feature parity for three segment offsets']
    with tempfile.TemporaryDirectory() as directory:
        for name,samples,start,expected in [
            ('silent',np.zeros(44100*5),0,'silent'),
            ('short',np.sin(np.arange(44100)*.1)*.1,0,'four seconds'),
            ('negative',np.zeros(44100),-1,'nonnegative'),
        ]:
            path=Path(directory)/f'{name}.wav';sf.write(path,samples,44100)
            try:generate(str(path),start,False,'cpu')
            except ValueError as error:
                assert expected in str(error),(name,str(error));checks.append(f'{name} input rejected')
            else:raise AssertionError(f'{name} was accepted')
    print(json.dumps({'passed':checks},indent=2))


if __name__=='__main__':main()
