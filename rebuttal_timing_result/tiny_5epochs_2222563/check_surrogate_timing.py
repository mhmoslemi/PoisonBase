"""Five no-augmentation epochs, for sizing the 150-epoch surrogate jobs."""
from pathlib import Path
import sys
import time

script = Path(__file__).with_name('benchmark.py')
exec(compile(script.read_text().split("for name in (sys.argv[1:] or", 1)[0], str(script), 'exec'))
net = fu.build_network('VGG13BN', 3, 200, (64, 64), device, seed=42)
net.features[-1] = torch.nn.AdaptiveAvgPool2d(1)
torch.cuda.synchronize()
previous = time.monotonic()
times = []


def progress(epoch, total):
    global previous
    torch.cuda.synchronize()
    now = time.monotonic()
    times.append(now - previous)
    previous = now
    print(f'NO-AUG epoch {epoch}/{total}: {times[-1]:.2f}s', flush=True)


fu.train_from_scratch(net, images, labels, 5, 0.1, 128, [], device, progress_callback=progress)
assert len(times) == 5
result = dict(epoch_seconds=times, mean_seconds=sum(times) / len(times),
              estimated_two_150_epoch_seconds=300 * sum(times) / len(times))
(OUTPUT / 'surrogate_timing.json').write_text(json.dumps(result, indent=2) + '\n')
print('SURROGATE_TIMING', json.dumps(result), flush=True)
