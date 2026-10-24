import os
"""Track-level summaries, avoiding treating seven clips per track as independent."""
import json
import numpy as np
from train_audio_latent_prior import ROOT

def main():
    source=json.loads((ROOT/'reports/prior_32f_decoded_test.json').read_text())
    tracks=sorted({r['track'] for r in source['results']})
    comparisons={}
    rng=np.random.default_rng(643)
    for other in ['other_track','shifted_1s','target']:
        differences=np.array([np.mean([r['variants']['correct']['aligned']['f1']-r['variants'][other]['aligned']['f1'] for r in source['results'] if r['track']==track]) for track in tracks])
        samples=rng.choice(differences,(10000,len(tracks)),replace=True).mean(1)
        comparisons[other]={'mean_track_f1_difference':float(differences.mean()),'tracks_correct_higher':int((differences>0).sum()),'tracks':len(tracks),'bootstrap_95_percent_interval':np.quantile(samples,[.025,.975]).tolist()}
    result={'comparison_metric':'correct F1 minus control F1; equal weight per source track','comparisons':comparisons,'limits':'Exploratory metric with provisional recording groups. Bootstrap intervals do not resolve shared-song leakage or human perceptual validity.'}
    (ROOT/'reports/prior_32f_track_controls.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
