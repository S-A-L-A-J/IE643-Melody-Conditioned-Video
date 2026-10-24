import os
import importlib
import json
import sys
from pathlib import Path
import torch
root = Path(__file__).resolve().parents[1]
report = {'python':sys.version,'base_interpreter':sys.base_prefix}
for name in ['torch','torchaudio','torchvision','librosa','demucs','diffusers','transformers','accelerate','peft','soundfile','cv2','imageio','gradio']:
    module = importlib.import_module(name)
    report[name] = getattr(module,'__version__','installed')
from diffusers import AnimateDiffPipeline, MotionAdapter
from transformers import ClapModel, ClapProcessor
report['cuda_available'] = torch.cuda.is_available()
if report['cuda_available']:
    a = torch.ones((256,256),device='cuda', requires_grad=True)
    loss=(a@a).mean()
    loss.backward()
    torch.cuda.synchronize()
    assert loss.item()==256 and a.grad is not None
    report['gpu'] = torch.cuda.get_device_name(0)
    report['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
    report['forward_backward'] = 'passed'
(root/'reports/environment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
