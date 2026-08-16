#!/usr/bin/env python
"""
summarize_aug.py -- read the augmentation sweep off defense_result/ as a table.

One table per (model, attack, class_pair, budget, defense): rows are the base
SELECTION, columns are the victim-training AUGMENTATION, and each cell is the
attack success rate over the targets, plus the victim's clean test accuracy.

The comparison is only meaningful if every cell was measured on the same target
images, so by default a config is restricted to the targets that appear in EVERY
(selection, aug) cell of it, and anything dropped is reported. --unpaired turns
that off and scores each cell on whatever it has.

    python summarize_aug.py
    python summarize_aug.py --model ResNet20BN --attack fc --csv aug_table.csv
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

AUG_ORDER = ['none', 'standard', 'randaug', 'cutout', 'dsa']
SEL_ORDER = ['random', 'ours', 'dpp2', 'dpp1', 'mmr', 'filter', 'pca']


def _order(vals, pref):
    known = [v for v in pref if v in vals]
    return known + sorted(v for v in vals if v not in pref)


def read_rows(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '*', 'results.csv'))):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if not (r.get('target_idx') and r.get('success')):
                    continue
                # 'aug' postdates the first defense runs; they were all unaugmented
                r['aug'] = r.get('aug') or 'none'
                r['_dir'] = os.path.basename(os.path.dirname(path))
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--defense_out_dir', default='defense_result')
    p.add_argument('--model', default=None)
    p.add_argument('--attack', default=None)
    p.add_argument('--class_pair', default=None)
    p.add_argument('--defense', default=None, help='filter on the defense tag')
    p.add_argument('--unpaired', action='store_true',
                   help='score every cell on its own targets instead of on the '
                        'intersection (the numbers stop being comparable)')
    p.add_argument('--csv', default=None, help='also write the cells here')
    a = p.parse_args()

    rows = read_rows(a.defense_out_dir)
    if not rows:
        raise SystemExit('no results.csv under %s' % a.defense_out_dir)

    # the defense tag carries the aug too; strip it so the defense stays readable
    def def_only(r):
        bits = [b for b in r['defense'].split('+') if not b.startswith('aug-')]
        return '+'.join(bits) or 'none'

    keep = lambda r: all(v is None or r[k] == v for k, v in
                         [('model', a.model), ('attack', a.attack),
                          ('class_pair', a.class_pair)])
    rows = [r for r in rows if keep(r)]
    if a.defense is not None:
        rows = [r for r in rows if def_only(r).startswith(a.defense)]
    if not rows:
        raise SystemExit('nothing left after the filters')

    # (config) -> (sel, aug) -> target -> [success per victim]
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ctas = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cfg = (r['model'], r['attack'], r['class_pair'], float(r['budget']),
               def_only(r))
        key = (r['sel'], r['aug'])
        cells[cfg][key][int(r['target_idx'])].append(int(r['success']))
        ctas[cfg][key].append(float(r['clean_test_acc']))

    out_rows = []
    for cfg in sorted(cells):
        model, attack, pair, budget, defense = cfg
        per = cells[cfg]
        sels = _order({k[0] for k in per}, SEL_ORDER)
        augs = _order({k[1] for k in per}, AUG_ORDER)

        common, note = None, ''
        if not a.unpaired:
            for s in sels:
                for g in augs:
                    ts = set(per.get((s, g), {}))
                    common = ts if common is None else (common & ts)
            common = common or set()
            allt = set()
            for v in per.values():
                allt |= set(v)
            missing = len(allt) - len(common)
            if missing:
                note = '  (%d of %d targets dropped: not in every cell)' % (
                    missing, len(allt))
            if not common:
                print('%s / %s / %s / b%g / defense %s: no target is in every '
                      'cell -- rerun the missing ones, or --unpaired\n'
                      % (model, attack, pair, budget, defense))
                continue

        print('=== %s / %s / %s / budget %g / defense %s ===%s'
              % (model, attack, pair, budget, defense, note))
        head = '%-10s' % 'sel' + ''.join('%16s' % g for g in augs)
        print(head)
        for s in sels:
            line = '%-10s' % s
            for g in augs:
                d = per.get((s, g))
                if not d:
                    line += '%16s' % '-'
                    continue
                ts = sorted(common) if common is not None else sorted(d)
                ts = [t for t in ts if t in d]
                asr = 100.0 * float(np.mean([np.mean(d[t]) for t in ts]))
                cta = 100.0 * float(np.mean(ctas[cfg][(s, g)]))
                ntr = sum(len(d[t]) for t in ts)
                line += '%16s' % ('%.1f%% (%.1f)' % (asr, cta))
                out_rows.append({'model': model, 'attack': attack,
                                 'class_pair': pair, 'budget': budget,
                                 'defense': defense, 'sel': s, 'aug': g,
                                 'asr': asr, 'cta': cta,
                                 'num_targets': len(ts), 'num_trials': ntr})
            print(line)
        n = len(sorted(common)) if common is not None else None
        print('   cell = ASR%% (clean test acc %%)%s\n'
              % ('   over %d paired target(s)' % n if n is not None else ''))

    if a.csv and out_rows:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print('wrote %s (%d cells)' % (a.csv, len(out_rows)))


if __name__ == '__main__':
    main()
