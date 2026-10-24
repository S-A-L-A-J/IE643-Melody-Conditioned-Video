import os
"""Static research figure for renderer timing and learned-prior controls."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    renderer=json.loads((ROOT/'reports/provisional_renderer_benchmark.json').read_text())['scores']
    prior=json.loads((ROOT/'reports/audio_latent_prior.json').read_text())['final']['test']
    styles=['geometric','rings','bars']
    aligned=[renderer[style]['beat_f1'] for style in styles]
    shifted=[renderer[style]['shifted_200ms_f1'] for style in styles]
    labels=['Correct music','Other track','Shifted 1 s','Static template']
    losses=[prior['correct_mse'],prior['other_track_mse'],
            prior['one_second_shift_mse'],prior['static_template_mse']]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    x=np.arange(3)
    axes[0].bar(x-0.19,aligned,0.36,color='#3a8f82',label='Aligned audio')
    axes[0].bar(x+0.19,shifted,0.36,color='#cb765d',label='Audio shifted 200 ms')
    axes[0].set_xticks(x,styles)
    axes[0].set_ylim(0,1)
    axes[0].set_ylabel('Beat-event F1 (higher is better)')
    axes[0].set_title('Programmed renderer · 14 test tracks')
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].bar(np.arange(4),losses,color=['#3a8f82','#cb765d','#d8a753','#7b8394'])
    axes[1].set_xticks(np.arange(4),labels,rotation=20,ha='right')
    axes[1].set_ylabel('Encoded-video MSE (lower is better)')
    axes[1].set_title('Learned audio prior · 98 test clips')
    axes[1].set_ylim(0,0.5)
    for i,value in enumerate(losses):
        axes[1].text(i,value+0.009,f'{value:.3f}',ha='center',fontsize=9)
    fig.suptitle('IE643 provisional synthetic-data evaluation',fontsize=14)
    fig.text(0.5,-0.01,'Source songs and medley overlap are not fully verified; the learned prior is not the diffusion generator.',
             ha='center',fontsize=9,color='#555555')
    output=ROOT/'reports/provisional_results.png'
    fig.savefig(output,dpi=180,bbox_inches='tight')
    print(output)


if __name__=='__main__':
    main()
