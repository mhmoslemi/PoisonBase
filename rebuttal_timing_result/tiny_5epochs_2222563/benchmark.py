"""Five-epoch timing only, inside the user's existing Slurm allocation."""
import gc
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

ROOT = Path('/scratch/mmoslem3/PoisonBase')
OUTPUT = Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID') == '2222563'
assert socket.gethostname().split('.')[0] == 'rack09-15'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'sbatch/rebuttal_missing_cells_vulcan/code'))

import torch
import final_update as fu
from victim_aug import VictimAugmenter

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.cuda.is_available()
device = torch.device('cuda:0')
started = time.monotonic()
data = torch.load(ROOT / 'data/tinyimagenet.pt', map_location='cpu',
                  mmap=True, weights_only=False)
assert tuple(data['images_train'].shape) == (100000, 3, 64, 64)
assert data['images_train'].dtype == torch.uint8
images = data['images_train'].to(device=device, dtype=torch.float32).div_(255)
labels = data['labels_train'].to(device=device, dtype=torch.long)
assert labels.shape == (100000,) and labels.unique().numel() == 200
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
images.sub_(mean).div_(std)
del data
gc.collect()
torch.cuda.synchronize()
ctx = dict(norm=lambda x: (x - mean) / std, denorm=lambda x: x * std + mean,
           dsa_param=None, im_size=(64, 64))
augmenter = VictimAugmenter('standard', ctx)
report = dict(job=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
              gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
              samples=100000, image_size=[3, 64, 64], batch_size=128,
              optimizer='SGD', learning_rate=0.1, momentum=0.9, weight_decay=0,
              augmentation='per-image random crop padding 4 and horizontal flip',
              precision='float32, no AMP',
              cudnn_benchmark=torch.backends.cudnn.benchmark,
              cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
              setup_seconds=time.monotonic() - started, models={})
print('SETUP', json.dumps(report), flush=True)

for name in (sys.argv[1:] or ('ResNet18BN', 'VGG11BN')):
    net = fu.build_network(name, 3, 200, (64, 64), device, seed=42)
    if name in ('VGG11BN', 'VGG13BN'):
        # The repository VGG head expects 512 features; pool its 64x64 input to match.
        net.features[-1] = torch.nn.AdaptiveAvgPool2d(1)
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
    net.eval()
    with torch.no_grad():
        assert net(images[:128]).shape == (128, 200)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    epoch_seconds = []
    previous = time.monotonic()

    def progress(epoch, total):
        global previous
        torch.cuda.synchronize()
        now = time.monotonic()
        epoch_seconds.append(now - previous)
        previous = now
        assert 1 <= epoch <= 5
        print(f'{name}: epoch {epoch}/{total}: {epoch_seconds[-1]:.2f}s', flush=True)

    fu.train_from_scratch(net, images, labels, 5, 0.1, 128, [100, 150], device,
                          augmenter=augmenter, progress_callback=progress)
    assert len(epoch_seconds) == 5
    result = dict(epochs=5, epoch_seconds=epoch_seconds,
                  total_seconds=sum(epoch_seconds),
                  mean_seconds=sum(epoch_seconds) / 5,
                  mean_epochs_2_to_5=sum(epoch_seconds[1:]) / 4,
                  peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
    report['models'][name] = result
    filename = 'timings_' + '_'.join(sys.argv[1:]) + '.json' if sys.argv[1:] else 'timings.json'
    (OUTPUT / filename).write_text(json.dumps(report, indent=2) + '\n')
    print('TIMING', name, json.dumps(result), flush=True)
    del net
    gc.collect()
    torch.cuda.empty_cache()

print('DONE: five epochs per network; no experiment checkpoints written.', flush=True)
