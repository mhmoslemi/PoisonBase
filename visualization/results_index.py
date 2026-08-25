#!/usr/bin/env python
"""
visualization/results_index.py -- a tidy index of every victim trial the sweeps
have already run, built from <out_dir>/<run_name>/results.csv.

No torch, no GPU, no dataset: this reads csv files and nothing else. A full scan
of ~650 run directories takes a couple of seconds, and the result is cached in
figs/results_index.csv so the figures that use it start instantly.

Why a parser and not just the csv columns
-----------------------------------------
results.csv records base = random | ours, but Greedy and DPP are BOTH 'ours' --
what separates them is the run directory name that final_update.build_run_name
produced ('..._lam1_cosine' vs '..._lam1_cosine_seldpp2'). The selection label
therefore has to come from the directory name, and so does everything else that
names a configuration but is not a csv column (sharpness sigma, craft ensemble,
target-difficulty tag, cross-architecture selector, ...).

Configuration key
-----------------
Two runs are the SAME configuration when they agree on
    dataset, model, attack, class pair, budget, epsilon, seed, and every
    remaining run-name tag that is not part of the selection rule
and differ only in how the bases were chosen. That is exactly the comparison the
paper's tables make, so Random / Greedy / DPP rows can be paired target by
target -- and, inside a target, victim by victim.

Legacy directories written by the older naming scheme (FC/GRAD/budget...) are
skipped, and so is any results.csv whose header is not the current one; both are
counted and reported rather than silently mixed in.
"""

import argparse
import collections
import csv
import math
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO_ROOT, 'ours_result')

# CIFAR10_ConvNetBN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2_ce5
RUN_RE = re.compile(
    r'^(?P<dataset>[A-Za-z0-9]+)_(?P<model>[A-Za-z0-9]+)_(?P<attack>fc|gradmatch|sapa)_'
    r'(?P<base>random|ours)_(?P<pair>[a-z0-9_]+-[a-z0-9_]+)_b(?P<budget>[0-9.]+)_'
    r'eps(?P<eps>[0-9p]+)_seed(?P<seed>\d+)(?P<rest>.*)$')

# run-name tokens that describe the SELECTION RULE (everything else is config)
SEL_TOKEN = re.compile(
    r'^(lam[0-9.]+|cosine|l2|seldpp[0-9.]+|selmmr[0-9.]+|selfilter[0-9.]+|'
    r'selpca[0-9.]+|sel(first|bottom|grand|el2n|boundary|pixel|featsim|relevance)|'
    r'top\d+)$')

REQUIRED = {'target_idx', 'victim_id', 'success'}
FIELDS = ['run_dir', 'dataset', 'model', 'attack', 'pair', 'budget', 'eps', 'seed',
          'tags', 'selection', 'lam', 'dist', 'target_idx', 'victim_id', 'success',
          'clean_test_acc', 'clean_asr', 'target_score', 'craft_obj', 'num_poisons']


def parse_run_name(name):
    """-> dict(config fields, selection, lam, dist) or None for a legacy name."""
    m = RUN_RE.match(name)
    if not m:
        return None
    rest = [t for t in m.group('rest').split('_') if t]
    sel_toks = [t for t in rest if SEL_TOKEN.match(t)]
    tags = [t for t in rest if not SEL_TOKEN.match(t)]

    if m.group('base') == 'random':
        selection = 'random'
    else:
        dpp = [t for t in sel_toks if t.startswith('seldpp')]
        other = [t for t in sel_toks
                 if t.startswith(('selmmr', 'selfilter', 'selpca', 'top'))
                 or (t.startswith('sel') and not t.startswith('seldpp'))]
        if dpp:
            selection = 'dpp' + dpp[0][len('seldpp'):]     # e.g. dpp2, dpp0.5
        elif other:
            selection = other[0]                            # a ladder / ablation rule
        else:
            selection = 'greedy'                            # plain top-m by score
    return dict(
        dataset=m.group('dataset'), model=m.group('model'), attack=m.group('attack'),
        pair=m.group('pair'), budget=float(m.group('budget')), eps=m.group('eps'),
        seed=int(m.group('seed')), tags='_'.join(tags), selection=selection,
        lam=next((t[3:] for t in sel_toks if t.startswith('lam')), '1'),
        dist=next((t for t in sel_toks if t in ('cosine', 'l2')), 'cosine'))


def config_key(rec):
    """Everything except the selection rule."""
    return (rec['dataset'], rec['model'], rec['attack'], rec['pair'],
            float(rec['budget']), rec['eps'], int(rec['seed']), rec['tags'])


def _f(row, key, lo=None, hi=None):
    try:
        v = float(row[key])
    except (TypeError, ValueError, KeyError):
        return ''
    if not math.isfinite(v):
        return ''
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return ''            # a legacy csv with shifted columns, not a measurement
    return v


def scan(results_dir=DEFAULT_RESULTS, verbose=True):
    """Every usable victim trial as a flat list of dicts."""
    if not os.path.isdir(results_dir):
        raise SystemExit('results directory not found: %s' % results_dir)
    out, skipped, malformed, empty = [], 0, 0, 0
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name, 'results.csv')
        if not os.path.exists(path):
            continue
        cfg = parse_run_name(name)
        if cfg is None:
            skipped += 1
            continue
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            if not REQUIRED <= set(rd.fieldnames or []):
                malformed += 1
                continue
            n0 = len(out)
            for r in rd:
                try:
                    t, v, s = int(r['target_idx']), int(r['victim_id']), int(r['success'])
                except (TypeError, ValueError):
                    malformed += 1
                    break
                rec = dict(cfg)
                rec.update(run_dir=name, target_idx=t, victim_id=v, success=s,
                           clean_test_acc=_f(r, 'clean_test_acc', 0.2, 1.0),
                           clean_asr=_f(r, 'clean_asr', 0.0, 100.0),
                           target_score=_f(r, 'target_score', 0.0, 1.0),
                           craft_obj=_f(r, 'craft_obj', 0.0, 1e6),
                           num_poisons=_f(r, 'num_poisons', 0, 1e6))
                out.append(rec)
            if len(out) == n0:
                empty += 1
    if verbose:
        print('  scanned %s: %d trials, %d legacy dirs skipped, %d malformed, '
              '%d empty' % (results_dir, len(out), skipped, malformed, empty))
    return out


def load(cache_path, results_dir=DEFAULT_RESULTS, refresh=False, verbose=True):
    """Read the cached index, or scan and write it."""
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, newline='') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r['budget'] = float(r['budget'])
            r['seed'] = int(r['seed'])
            r['target_idx'] = int(r['target_idx'])
            r['victim_id'] = int(r['victim_id'])
            r['success'] = int(r['success'])
            for k in ('clean_test_acc', 'clean_asr', 'target_score', 'craft_obj',
                      'num_poisons'):
                r[k] = float(r[k]) if r[k] not in ('', None) else ''
        if verbose:
            print('  loaded %d trials from %s (delete it or pass --refresh to '
                  'rescan)' % (len(rows), cache_path))
        return rows
    rows = scan(results_dir, verbose=verbose)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    with open(cache_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    if verbose:
        print('  wrote %s' % cache_path)
    return rows


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def by_target(rows, selections=None, lam='1', dist='cosine'):
    """{(config_key, selection, target): {victim_id: success}}.

    lam / dist restrict the 'ours' family to the paper's default selector; the
    lambda- and distance-ablation runs live in the same directory tree and would
    otherwise be silently averaged in. Random is unaffected by either.
    """
    out = collections.defaultdict(dict)
    for r in rows:
        sel = r['selection']
        if selections is not None and sel not in selections:
            continue
        if sel != 'random' and (str(r['lam']) != lam or r['dist'] != dist):
            continue
        out[(config_key(r), sel, r['target_idx'])][r['victim_id']] = r['success']
    return out


def paired_targets(rows, a='random', b='dpp2', min_victims=3, lam='1', dist='cosine'):
    """One record per (configuration, target) that BOTH selections ran.

    Strictly paired: the ASR of each arm is computed over the victim ids the two
    arms have in common, so a target is never compared across different victim
    seeds or different victim counts.
    """
    tab = by_target(rows, selections={a, b}, lam=lam, dist=dist)
    seen = collections.defaultdict(dict)
    for (cfg, sel, t), vic in tab.items():
        seen[(cfg, t)][sel] = vic
    out = []
    for (cfg, t), arms in sorted(seen.items(), key=lambda kv: str(kv[0])):
        if a not in arms or b not in arms:
            continue
        shared = sorted(set(arms[a]) & set(arms[b]))
        if len(shared) < min_victims:
            continue
        asr_a = 100.0 * sum(arms[a][v] for v in shared) / len(shared)
        asr_b = 100.0 * sum(arms[b][v] for v in shared) / len(shared)
        out.append(dict(dataset=cfg[0], model=cfg[1], attack=cfg[2], pair=cfg[3],
                        budget=cfg[4], eps=cfg[5], seed=cfg[6], tags=cfg[7],
                        target_idx=t, n_victims=len(shared),
                        asr_a=asr_a, asr_b=asr_b, gain=asr_b - asr_a,
                        sel_a=a, sel_b=b))
    return out


def attach_target_meta(rows, points, a='random'):
    """Add the clean-model quantities a point inherits from its target/run.

    target_score is the clean ensemble's softmax probability of y_adv for that
    target (how nearly the clean model already gets it wrong) and clean_asr is
    the clean victims' agreement with y_adv; both are properties of the TARGET,
    not of a selection, so either arm may supply them. craft_obj is the crafting
    objective the run reported and IS per selection, so it is stored per arm.
    """
    meta = {}
    obj = collections.defaultdict(dict)
    for r in rows:
        k = (config_key(r), r['target_idx'])
        if r['target_score'] != '' and k not in meta:
            meta[k] = dict(target_score=r['target_score'], clean_asr=r['clean_asr'])
        if r['craft_obj'] != '':
            obj[k][r['selection']] = r['craft_obj']
        if r['clean_test_acc'] != '':
            obj[k].setdefault('_cta', {})[r['selection']] = r['clean_test_acc']
    for p in points:
        k = ((p['dataset'], p['model'], p['attack'], p['pair'], p['budget'],
              p['eps'], p['seed'], p['tags']), p['target_idx'])
        m = meta.get(k, {})
        p['target_score'] = m.get('target_score', '')
        p['clean_asr'] = m.get('clean_asr', '')
        o = obj.get(k, {})
        p['craft_obj_a'] = o.get(p['sel_a'], '')
        p['craft_obj_b'] = o.get(p['sel_b'], '')
        cta = o.get('_cta', {})
        p['cta_a'] = cta.get(p['sel_a'], '')
        p['cta_b'] = cta.get(p['sel_b'], '')
    return points


def add_index_args(p, default_out):
    g = p.add_argument_group('results index (no gpu, no dataset)')
    g.add_argument('--results_dir', default=DEFAULT_RESULTS,
                   help='where the attack runs live (read-only)')
    g.add_argument('--index_csv', default=os.path.join(default_out, 'results_index.csv'),
                   help='cached tidy index; delete it or pass --refresh to rescan')
    g.add_argument('--refresh', action='store_true',
                   help='rescan the run directories instead of using the cache')
    g.add_argument('--selection_b', default='dpp2',
                   help="the 'ours' arm: dpp2 (alpha=2, the paper default), dpp1, "
                        'greedy, ...')
    g.add_argument('--selection_a', default='random', help='the baseline arm')
    g.add_argument('--min_victims', type=int, default=3,
                   help='drop a (config, target) with fewer shared victim seeds')
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_index_args(p, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs'))
    args = p.parse_args()
    rows = load(args.index_csv, args.results_dir, refresh=True)
    pts = paired_targets(rows, args.selection_a, args.selection_b, args.min_victims)
    print('  %s vs %s: %d paired (config, target) points across %d configurations'
          % (args.selection_a, args.selection_b, len(pts),
             len({(p['model'], p['attack'], p['pair'], p['budget'], p['tags'])
                  for p in pts})))
    wins = sum(1 for p in pts if p['gain'] > 0)
    ties = sum(1 for p in pts if p['gain'] == 0)
    print('  %s better on %d, tied on %d, worse on %d'
          % (args.selection_b, wins, ties, len(pts) - wins - ties))


if __name__ == '__main__':
    main()
