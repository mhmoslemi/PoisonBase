#!/usr/bin/env python
"""
make_aug_table.py -- emit aug_table.tex for tab:augmentation-robustness.

The No Aug. column is NOT rerun: it is recomputed from the undefended attack
runs in ours_result/, restricted to exactly the trials the augmented runs use --
the first --num_targets of the sorted Random-vs-DPP target intersection, and
victim ids < --num_victims. build_network() seeds from
seed*100000 + tidx*100 + victim_id, so victims 0..3 of an old 6-victim run are
bit-for-bit the same models the reduced protocol trains. The comparison is
therefore paired, not approximate.

The three augmented columns are read from defense_result/ as the aug shards
finish; a cell with no data yet is left as '--'.

    python make_aug_table.py                    # -> aug_table.tex
    python make_aug_table.py --num_targets 0    # use every paired target
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import _old_.defense as DEF
import _old_.final_update as FU

# (pair, attack, budget, epsilon label). GM runs at 2e-2 rather than the 1e-2 in
# the table draft: there are no DPP gradmatch poisons on disk at 1e-2 (dog-bird
# has 2 targets, frog-airplane 0), so that row cannot be filled.
ROWS = [
    ('dog-bird',      'gradmatch', 0.002, r'$2{\times}10^{-3}$'),
    ('dog-bird',      'gradmatch', 0.02,  r'$2{\times}10^{-2}$'),
    ('dog-bird',      'fc',        0.002, r'$2{\times}10^{-3}$'),
    ('dog-bird',      'fc',        0.01,  r'$1{\times}10^{-2}$'),
    ('frog-airplane', 'gradmatch', 0.002, r'$2{\times}10^{-3}$'),
    ('frog-airplane', 'gradmatch', 0.02,  r'$2{\times}10^{-2}$'),
    ('frog-airplane', 'fc',        0.002, r'$2{\times}10^{-3}$'),
    ('frog-airplane', 'fc',        0.01,  r'$1{\times}10^{-2}$'),
]
SELS = [('random', 'Random'), ('dpp', 'DPP')]
AUGS = [('none', 'No Aug.'), ('standard', 'Crop+Flip'),
        ('randaug', 'RandAugment'), ('cutout', 'Cutout')]
ATTACK_TEX = {'gradmatch': 'GM', 'fc': 'FC', 'sapa': 'SAPA'}
PAIR_TEX = {'dog-bird': 'dog--bird', 'frog-airplane': 'frog--airplane'}


def make_ns(args, pair, attack, base, budget, dpp, tgt):
    import argparse as _a
    return _a.Namespace(
        dataset='CIFAR10', model=args.model, attack=attack, base=base,
        class_pair=pair, budget=budget, epsilon=args.epsilon, seed=args.seed,
        lambda_margin=1.0, base_dist='cosine', sel_filter=False, sel_pca=False,
        sel_mmr=False, sel_dpp=dpp, sel_pool=3.0, sel_mu=1.0,
        sel_alpha=args.sel_alpha, fc_mode='sample', sharp_mode='worst',
        sharp_sigma=0.05, sharp_samples=20, craft_ensemble=5,
        target_select=int(tgt))


def score(path, keep_targets, num_victims):
    """(asr%, cta%, n_targets, n_trials) from a results.csv, or None."""
    if not os.path.exists(path):
        return None
    per, cta = defaultdict(list), []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if not (r.get('target_idx') and r.get('success')):
                continue
            t, v = int(r['target_idx']), int(r['victim_id'])
            if keep_targets is not None and t not in keep_targets:
                continue
            if num_victims and v >= num_victims:
                continue
            per[t].append(int(r['success']))
            cta.append(float(r['clean_test_acc']))
    if not per:
        return None
    asr = 100.0 * float(np.mean([np.mean(v) for v in per.values()]))
    return asr, 100.0 * float(np.mean(cta)), len(per), len(cta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='ResNet20BN')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epsilon', type=float, default=0.0313725)
    p.add_argument('--sel_alpha', type=float, default=2.0)
    p.add_argument('--num_targets', type=int, default=5,
                   help='first N of the sorted paired intersection; 0 = all')
    p.add_argument('--num_victims', type=int, default=4, help='0 = all')
    p.add_argument('--out_dir', default='ours_result')
    p.add_argument('--defense_out_dir', default='defense_result')
    p.add_argument('--out', default='latex/aug_table.tex')
    a = p.parse_args()

    import json
    cfg = json.load(open('sweep_config.json'))

    body, notes = [], []
    for ri, (pair, attack, budget, eps_tex) in enumerate(ROWS):
        tgt = cfg['difficulty'][a.model][attack][pair]
        dirs = {}
        for sel, _lab in SELS:
            base, dpp = ('random', False) if sel == 'random' else ('ours', True)
            ns = make_ns(a, pair, attack, base, budget, dpp, tgt)
            dirs[sel] = os.path.join(a.out_dir, FU.build_run_name(ns))

        # the pinned set: exactly what aug.sh writes and defense.py consumes
        have = None
        for sel, _lab in SELS:
            ts = set(DEF.cached_targets(dirs[sel])) if os.path.isdir(dirs[sel]) else set()
            have = ts if have is None else (have & ts)
        pinned = sorted(have or [])
        if a.num_targets:
            pinned = pinned[:a.num_targets]
        keep = set(pinned) if pinned else None
        if not pinned:
            notes.append('%s / %s / b%g: no paired targets'
                         % (pair, ATTACK_TEX[attack], budget))

        for sel, sel_lab in SELS:
            cells = []
            for aug, _alab in AUGS:
                if aug == 'none':
                    path = os.path.join(dirs[sel], 'results.csv')
                else:
                    tag = 'none+aug-%s' % aug
                    path = os.path.join(a.defense_out_dir,
                                        '%s__def-%s' % (os.path.basename(dirs[sel]), tag),
                                        'results.csv')
                s = score(path, keep, a.num_victims)
                if s is None:
                    cells.append('--')
                    continue
                asr, cta, nt, ntr = s
                cells.append(r'%.1f \small{(%.2f)}' % (asr, cta))
                # a cell scored on fewer trials than the protocol asks for is
                # still a number, and would sit in the table looking finished --
                # say so for EVERY column, not just No Aug.
                want = len(pinned) * (a.num_victims or 0)
                if nt != len(pinned) or (want and ntr != want):
                    notes.append('INCOMPLETE %s / %s / b%g / %s / %s: %d target(s), '
                                 '%d trial(s), wanted %d x %d'
                                 % (pair, ATTACK_TEX[attack], budget, sel_lab,
                                    _alab, nt, ntr, len(pinned), a.num_victims))
            body.append('%s & %s &\n%s & %s & %s \\\\'
                        % (PAIR_TEX[pair], ATTACK_TEX[attack], eps_tex, sel_lab,
                           ' & '.join(cells)))
        if ri == len(ROWS) - 1:
            pass                                  # no separator after the last row
        elif ri == len(ROWS) // 2 - 1:
            body.append('\n\\midrule\n')          # between the two class pairs
        elif ri % 2 == 1:
            body.append('\n\\addlinespace')       # between the GM and FC blocks
        else:
            body.append('')

    tex = r"""%% generated by make_aug_table.py -- do not hand-edit, regenerate instead
%%
%% No Aug. is recomputed from the undefended runs in %s, restricted to the
%% first %d paired target(s) and victim ids < %d, i.e. exactly the trials the
%% augmented runs use. The other three columns come from %s.
%%
%% The GM rows are at 2e-2, not the 1e-2 of the original draft: there are no DPP
%% gradmatch poisons on disk at 1e-2 (dog-bird 2 targets, frog-airplane 0).
\begin{table}[t]
\centering
\caption{
Robustness of base selection to victim-side data augmentation.
Poisons are optimized once and kept fixed; only the victim-training
augmentation is changed.  All experiments use %s.
Entries report ASR (\%%), with clean test accuracy in parentheses.
}
\label{tab:augmentation-robustness}
\small
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.10}
\resizebox{\textwidth}{!}{
\begin{tabular}{lllclcccc}
\toprule
\textbf{Pair} &
\textbf{Attack} &
\textbf{$\epsilon$} &
\textbf{Selection} &
\textbf{No Aug.} &
\textbf{Crop+Flip} &
\textbf{RandAugment} &
\textbf{Cutout} \\
\midrule

%s

\bottomrule
\end{tabular}
}
\end{table}
""" % (a.out_dir, a.num_targets, a.num_victims,
       a.defense_out_dir, a.model, '\n'.join(body).strip())

    with open(a.out, 'w') as f:
        f.write(tex)
    print('wrote %s' % a.out)
    filled = sum(c.count(r'\small') for c in body)
    print('%d of %d cells filled' % (filled, len(ROWS) * len(SELS) * len(AUGS)))
    for n in dict.fromkeys(notes):
        print('  note: %s' % n)


if __name__ == '__main__':
    main()
