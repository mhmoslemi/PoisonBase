#!/usr/bin/env bash
#
# Refuse to start a sweep on a machine that cannot actually train.
#
# defense.py's resolve_gpus() falls back to the cpu when it finds no gpu, and a
# cpu victim is ~100x slower, so a login-node launch sits there looking like it
# is working for days. Run this first and bail on a non-zero exit:
#
#     sh "$(dirname "$0")/preflight_cuda.sh" || exit 1
#
# Checks: torch imports, CUDA is available, at least one device, gpu 0 can
# actually allocate and compute, and enough memory is free to fit a victim.

source /home/mmoslem3/ENV/bin/activate

python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit('preflight FAILED: cannot import torch: %s' % e)

if not torch.cuda.is_available():
    sys.exit('preflight FAILED: CUDA is not available. This is almost certainly '
             'a login node -- get a gpu allocation (salloc / srun --jobid=... '
             '--overlap) and rerun. Refusing to fall back to the cpu.')

n = torch.cuda.device_count()
if n < 1:
    sys.exit('preflight FAILED: torch.cuda.is_available() is True but '
             'device_count() is 0')

try:
    x = torch.zeros(2048, 2048, device='cuda:0')
    float((x + 1).sum().item())
    del x
    free, total = torch.cuda.mem_get_info(0)
except Exception as e:
    sys.exit('preflight FAILED: gpu 0 is visible but unusable: %s' % e)

if free < 3 * 2 ** 30:
    sys.exit('preflight FAILED: only %.1f GiB free on gpu 0; a victim needs a '
             'few GiB. Something else is using this gpu.' % (free / 2 ** 30))

print('preflight ok: %d gpu(s), %s, %.1f/%.1f GiB free'
      % (n, torch.cuda.get_device_name(0), free / 2 ** 30, total / 2 ** 30))
PY
