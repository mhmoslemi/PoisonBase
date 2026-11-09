"""Read target metadata and time GM crafting; never produce experiment poisons."""
import json
import resource
from pathlib import Path
import sys
import time

script = Path(__file__).with_name('benchmark.py')
source = script.read_text().split("for name in (sys.argv[1:] or", 1)[0]
exec(compile(source, str(script), 'exec'))
data = torch.load(ROOT / 'data/tinyimagenet.pt', map_location='cpu', mmap=True,
                  weights_only=False)
classes = data['classes']
test_labels = data['labels_val'].clone()
test_images = data['images_val'].to(device=device, dtype=torch.float32).div_(255)
test_images.sub_(mean).div_(std)
del data
pairs = [('alligator_tarantula', 'n01698640', 'n01774750'),
         ('bullfrog_dining', 'n01641577', 'n03201208'),
         ('bullfrog_wagon', 'n01641577', 'n02814533')]
metadata = []
for name, poison, target in pairs:
    own, true = classes.index(poison), classes.index(target)
    candidates = (test_labels == true).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(42)
    chosen = candidates[torch.randperm(len(candidates), generator=generator)[:6]]
    assert len(chosen) == 6 and (test_labels[chosen] == true).all()
    metadata.append(dict(id=name, poison_wnid=poison, target_wnid=target,
                         poison_class=own, target_class=true, targets=chosen.tolist()))
    print('TARGETS', json.dumps(metadata[-1]), flush=True)
(OUTPUT / 'new_table_targets.json').write_text(json.dumps(metadata, indent=2) + '\n')
nets = []
for index in range(5):
    net = fu.build_network('VGG13BN', 3, 200, (64, 64), device, seed=42 + index)
    net.features[-1] = torch.nn.AdaptiveAvgPool2d(1)
    nets.append(net.eval())
own = metadata[0]['poison_class']
target = metadata[0]['targets'][0]
timings = []
for count in (100, 50):
    base = images[(labels == own).nonzero(as_tuple=True)[0][:count]]
    base01 = ctx['denorm'](base).clamp(0, 1)
    torch.cuda.synchronize()
    began = time.monotonic()
    adv, objective = fu.craft_gradmatch(nets, base01, test_images[target], own,
        ctx['norm'], 8 / 255, 1 / 255, 10, 1, device,
        dsa_strategy='color_crop_cutout_flip_scale_rotate', dsa_param=fu.ParamDiffAug(),
        fast=False, schedule=False, lowmem=False, chunk=256)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - began
    assert torch.isfinite(adv).all() and (adv - base01).abs().max() <= 8 / 255 + 1e-6
    item = dict(poisons=count, measured_steps=10, seconds=elapsed,
                estimated_250_steps_8_restarts_seconds=elapsed * 200,
                peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
                timing_only_randomly_initialized_surrogates=True)
    timings.append(item)
    print('CRAFT_TIMING', json.dumps(item), flush=True)
    del adv
(OUTPUT / 'new_table_craft_timings.json').write_text(json.dumps(timings, indent=2) + '\n')
