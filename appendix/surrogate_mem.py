"""Peak GPU memory of training one surrogate -- the last cell of tab:computational-cost.

profile_selection.py reads the K=20 surrogate training TIME off the shared cache and
never trains, so it has nothing to measure peak memory from. This trains exactly one
surrogate into a throwaway cache directory with the CUDA peak counter reset around it.

    python appendix/surrogate_mem.py --model ConvNetBN
"""
import argparse, os, shutil, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import final_update as FU

p = argparse.ArgumentParser()
p.add_argument('--model', default='ConvNetBN')
p.add_argument('--dataset', default='CIFAR10')
p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
p.add_argument('--class_pair', default='dog-bird')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--surrogate_epochs', type=int, default=60)
p.add_argument('--surrogate_lr', type=float, default=0.1)
p.add_argument('--surrogate_bs', type=int, default=128)
p.add_argument('--surrogate_decay', nargs='*', type=int, default=[35, 45])
p.add_argument('--surrogate_wd', type=float, default=0.0)
p.add_argument('--surrogate_aug', action='store_true', default=False)
p.add_argument('--dsa_strategy', default=FU.DSA_DEFAULT)
a = p.parse_args()

if not torch.cuda.is_available():
    sys.exit('surrogate_mem: no CUDA device visible -- get a GPU allocation first')
dev = 'cuda'

ctx = FU.build_context(a, dev)
a.num_surrogates = 1
a.cache_dir = tempfile.mkdtemp(prefix='surromem_')
try:
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    FU.get_surrogates(a, ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'],
                      ctx['test_labs'], ctx['channel'], ctx['num_classes'],
                      ctx['im_size'], dev, ctx['dsa_param'], only_id=0)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    print('%s surrogate: %.0f s, peak GPU memory %.1f MB' % (a.model, time.time() - t0, peak))
finally:
    shutil.rmtree(a.cache_dir, ignore_errors=True)
