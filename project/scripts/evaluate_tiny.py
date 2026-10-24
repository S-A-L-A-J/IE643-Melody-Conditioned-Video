import os
"""Paired correct/swapped-music denoising diagnostic on the tiny training set."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, DDPMScheduler, MotionAdapter
from peft import LoraConfig
from safetensors.torch import load_file

from pair_dataset import PairedClipDataset
from probe_adapter_step import LocalAudioProjector

ROOT = Path(os.environ.get('IE643_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


def main():
    torch.set_num_threads(4)
    checkpoint = ROOT / 'outputs/tiny_e2'
    run = json.loads((checkpoint / 'run.json').read_text())
    config = run['configuration']
    dataset = PairedClipDataset()
    rows_by_id = {row['id']: i for i, row in enumerate(dataset.rows)}
    frame_indices = torch.arange(config['frames']) * config['stride']
    examples = []
    cache_dir = ROOT / 'data/latent_cache' / f"{config['size']}_{config['frames']}_{config['stride']}"
    for pair_id in run['pair_ids']:
        item = dataset[rows_by_id[pair_id]]
        cache_paths = list(cache_dir.glob(f'{pair_id}_*.safetensors'))
        if len(cache_paths) != 1:
            raise RuntimeError(f'Expected one cached target for {pair_id}; found {len(cache_paths)}')
        examples.append({
            'id': pair_id,
            'latents': load_file(cache_paths[0])['latents'].to('cuda', dtype=torch.float16),
            'rhythm': item['rhythm'][frame_indices].to('cuda'),
            'melody': item['melody'][frame_indices].to('cuda'),
            'dynamics': item['dynamics'][frame_indices].to('cuda'),
        })
    adapter = MotionAdapter.from_pretrained(ROOT / 'models/diffusion/animatediff', variant='fp16',
                                            torch_dtype=torch.float16, local_files_only=True)
    pipe = AnimateDiffPipeline.from_pretrained(ROOT / 'models/diffusion/sd15', motion_adapter=adapter,
                                               torch_dtype=torch.float16, variant='fp16', local_files_only=True)
    with torch.no_grad():
        text_model = pipe.text_encoder.float().cpu().eval()
        tokens = pipe.tokenizer('abstract colorful geometric shapes on a dark background',
                                padding='max_length', max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
        text = text_model(tokens.input_ids)[0].mean(dim=1, keepdim=True).to('cuda', dtype=torch.float16)
        del text_model
    unet = pipe.unet
    del pipe, adapter
    targets = [name for name, module in unet.named_modules() if isinstance(module, torch.nn.Linear)
               and name.endswith(('to_q', 'to_v')) and ('motion_modules' in name or 'attn2' in name)]
    unet.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['rank'], target_modules=targets))
    _, unexpected = unet.load_state_dict(load_file(checkpoint / 'lora.safetensors'), strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected LoRA keys: {unexpected[:5]}')
    unet.to('cuda', dtype=torch.float16).eval()
    for name, module in unet.named_modules():
        if '.lora_A.default' in name or '.lora_B.default' in name:
            module.float()
    projector = LocalAudioProjector().to('cuda').eval()
    projector.load_state_dict(load_file(checkpoint / 'projector.safetensors'))
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule='scaled_linear', prediction_type='epsilon')
    generator = torch.Generator(device='cuda').manual_seed(643)
    noise = torch.randn(examples[0]['latents'].shape, device='cuda', dtype=torch.float16, generator=generator)
    timestep = torch.tensor([500], device='cuda')

    def conditioning(example):
        audio = projector(example['rhythm'], example['melody'], example['dynamics']).half()
        return torch.cat([text.expand(config['frames'], -1, -1), audio], dim=1)

    results = []
    with torch.inference_mode():
        for i, example in enumerate(examples):
            other = examples[(i + 1) % len(examples)]
            noisy = scheduler.add_noise(example['latents'], noise, timestep)
            right = unet(noisy, timestep, encoder_hidden_states=conditioning(example)).sample
            wrong = unet(noisy, timestep, encoder_hidden_states=conditioning(other)).sample
            correct_loss = float(F.mse_loss(right.float(), noise.float()))
            mismatched_loss = float(F.mse_loss(wrong.float(), noise.float()))
            results.append({'pair': example['id'], 'other_pair': other['id'],
                            'correct_loss': correct_loss, 'mismatched_loss': mismatched_loss,
                            'advantage': mismatched_loss - correct_loss,
                            'prediction_difference': float((right.float() - wrong.float()).abs().mean())})
    advantages = [r['advantage'] for r in results]
    report = {'training_step': run['step'], 'training_pairs': len(results),
              'timestep': int(timestep.item()), 'same_noise_per_pair': True,
              'mean_advantage': sum(advantages) / len(advantages),
              'correct_better_count': sum(a > 0 for a in advantages),
              'interpretation': 'Positive advantage means the correct music predicted noise better. This is training-set only.',
              'pairs': results}
    output = ROOT / 'reports/tiny_e2_swap_evaluation.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
