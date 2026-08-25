#!/usr/bin/env python
"""How much of the SVHN row of tab:cross-dataset is on disk, and what it says.

    python appendix/svhn_status.py count     -> one integer, runs finished of 20
    python appendix/svhn_status.py report    -> per-pair ASR, the four cells,
                                                the LaTeX row, what is missing

Used by svhn-cells.sh: `count` drives the resume loop, `report` is step 4. Reads
files only, never touches a GPU, so it is safe to run on a dead allocation.

A run counts as finished on the same criterion the driver skips it by -- the
==== ASR line in its log -- so the two can never disagree about what is left.

Cell = the unweighted mean over the five pairs of each pair's ASR, which is how
the CIFAR-10 and CIFAR-100 rows of the same table were computed (CIFAR-100 GM:
(0+0+60+0+0)/5 = 12.0, matching the published cell).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'ours_result')
PAIRFILE = os.path.join(ROOT, 'target_sets', 'xdata_pairs_SVHN.json')
DATASET, MODEL, BUDGET, SEED = 'SVHN', 'ConvNetBN', '0.002', '42'

# (column label, attack, base flag) -- the four cells the table asks for.
CELLS = [('GM', 'gradmatch', 'random'), ('GM', 'gradmatch', 'ours'),
         ('SAPA', 'sapa', 'random'), ('SAPA', 'sapa', 'ours')]


def run_name(attack, base, pair):
    """Must match the two `run` calls in appendix/ap2-cifar100.sh exactly."""
    sn = '_worst0.05' if attack == 'sapa' else ''
    stem = '%s_%s_%s_%s_%s_b%s_eps8_seed%s' % (
        DATASET, MODEL, attack, base, pair, BUDGET, SEED)
    if base == 'ours':
        return '%s_lam1_cosine_seldpp2%s_ce5' % (stem, sn)
    return '%s%s_ce5' % (stem, sn)


def pairs():
    if not os.path.exists(PAIRFILE):
        return []
    with open(PAIRFILE) as f:
        return [p['class_pair'] for p in json.load(f)['pairs']]


def asr(attack, base, pair):
    """Per-pair ASR in percent, or None if that run has not finished.

    The log line is the authority on doneness; summary.json is where the number
    comes from. A run whose log has the line but no readable summary is a real
    inconsistency, so it reports as unfinished rather than as a silent zero.
    """
    name = run_name(attack, base, pair)
    d = os.path.join(OUT_DIR, name)
    try:
        with open(os.path.join(d, 'log.txt')) as f:
            if ('==== %s : ASR = ' % name) not in f.read():
                return None
        with open(os.path.join(d, 'summary.json')) as f:
            s = json.load(f)
    except (OSError, ValueError):
        return None
    return 100.0 * s['asr_mean']


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'report'
    ps = pairs()
    grid = {(a, b, p): asr(a, b, p) for _, a, b in CELLS for p in ps}
    done = sum(v is not None for v in grid.values())

    if mode == 'count':
        print(done)
        return

    if not ps:
        print('  no %s -- run step 1 first.' % os.path.relpath(PAIRFILE, ROOT))
        return

    print('  %d of 20 runs finished.  ASR (%%) per pair, 5 victims each:' % done)
    print('')
    print('  %-9s %-8s %s' % ('pair', 'attack', '  '.join('%7s' % x for x in ('Random', 'DPP'))))
    for p in ps:
        for label, a, _ in CELLS[::2]:
            row = []
            for base in ('random', 'ours'):
                v = grid[(a, base, p)]
                row.append('%7s' % ('--' if v is None else '%.1f' % v))
            print('  %-9s %-8s %s' % (p, label, '  '.join(row)))
    print('')

    missing = [run_name(a, b, p) for _, a, b in CELLS for p in ps
               if grid[(a, b, p)] is None]
    if missing:
        print('  %d run(s) still missing -- rerun this script with a GPU:' % len(missing))
        for m in missing:
            print('    %s' % m)
        print('')
        print('  The cells below average only the pairs that finished. They are')
        print('  NOT the table values: the row needs all five pairs, and partial')
        print('  means move a lot (in the CIFAR-100 row 3 of 5 pairs were 0%).')
        print('')

    cells = {}
    for label, a, b in CELLS:
        got = [grid[(a, b, p)] for p in ps if grid[(a, b, p)] is not None]
        cells[(a, b)] = (sum(got) / len(got)) if got else None

    def fmt(v):
        return '--' if v is None else '%.1f' % v

    for label, a in (('GM', 'gradmatch'), ('SAPA', 'sapa')):
        r, o = cells[(a, 'random')], cells[(a, 'ours')]
        d = '--' if (r is None or o is None) else '%+.1f' % (o - r)
        print('  SVHN %-5s Random %-6s DPP %-6s Delta %s'
              % (label, fmt(r), fmt(o), d))

    if not missing:
        print('')
        print('  latex/appendix.tex, tab:cross-dataset -- replace the two SVHN rows:')
        for label, a in (('GM', 'gradmatch'), ('SAPA', 'sapa')):
            r, o = cells[(a, 'random')], cells[(a, 'ours')]
            print('SVHN \\small{(ConvNetBN)}          & %-4s & %s & %s & %+.1f \\\\'
                  % (label, fmt(r), fmt(o), o - r))


if __name__ == '__main__':
    main()
