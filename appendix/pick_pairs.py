"""Sample the ordered target->adversarial class pairs for the cross-dataset table.

appendix.tex says the pairs are "sampled uniformly without replacement from the
classes using a fixed seed". This is that sampling, kept in one place so the run
scripts and the paper cannot drift apart. It writes a JSON and prints one
'<adversarial>-<target>' token per line, in the order the table lists them.

    python appendix/pick_pairs.py --dataset CIFAR100 --num_pairs 5

Sampling is without replacement over CLASSES, so the ten classes involved in the
five pairs are all distinct -- no class is ever both a target and an adversarial
class, and no class appears twice.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from torchvision import datasets

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='CIFAR100')
p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
p.add_argument('--num_pairs', type=int, default=5)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--out', default='target_sets/xdata_pairs_CIFAR100.json')
a = p.parse_args()

if a.dataset == 'CIFAR100':
    names = datasets.CIFAR100(a.data_path, train=True, download=True).classes
elif a.dataset == 'CIFAR10':
    names = datasets.CIFAR10(a.data_path, train=True, download=True).classes
elif a.dataset == 'SVHN':
    # torchvision's SVHN carries no .classes; the labels are the digits themselves.
    # With exactly ten classes, five pairs consume all of them -- still without
    # replacement, so no digit is both a target and an adversarial class.
    names = [str(c) for c in range(10)]
else:
    sys.exit('pick_pairs: unsupported dataset %r' % a.dataset)

rng = np.random.default_rng(a.seed)
picked = rng.choice(len(names), size=2 * a.num_pairs, replace=False)
pairs = []
for i in range(a.num_pairs):
    tgt, adv = int(picked[2 * i]), int(picked[2 * i + 1])
    pairs.append({'instance': i + 1, 'target_class': names[tgt],
                  'adversarial_class': names[adv],
                  'class_pair': '%s-%s' % (names[adv], names[tgt])})

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
tmp = '%s.%d.tmp' % (a.out, os.getpid())
with open(tmp, 'w') as f:
    json.dump({'_generated_by': 'appendix/pick_pairs.py',
               '_dataset': a.dataset, '_seed': a.seed,
               '_note': 'class_pair is <adversarial>-<target>, i.e. --pair_order '
                        'poison-target; the table reads target -> adversarial',
               'pairs': pairs}, f, indent=1)
os.replace(tmp, a.out)

for q in pairs:
    print(q['class_pair'])
