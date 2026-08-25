"""Choose the utility-matched defense strength from what the calibration left on disk.

appendix.tex's rule: the STRONGEST setting whose clean test accuracy is within two
points of the undefended model, decided on clean data only. defense_tag() puts the
strength in the run-dir name (epic-s<subset>, friends-...-e<eps>), so every swept
setting has its own directory and its own logged "defended clean CTA" -- which means
the choice can be recomputed here rather than eyeballed.

Strength ordering: EPIC keeps less of the training set as --epic_subset_size falls,
so smaller is stronger. FRIENDS adds more noise as --noise_eps rises, so larger is
stronger.

    python appendix/pick_defense.py --defense epic       # prints the value
    python appendix/pick_defense.py --defense friends --verbose
"""
import argparse, glob, os, re, sys

TOL = 2.0          # percentage points of clean accuracy


def clean_cta(logpath):
    v = None
    for l in open(logpath, errors='ignore'):
        m = re.search(r'defended clean CTA = ([\d.]+)', l)
        if m:
            v = float(m.group(1)) * 100
    return v


def undefended():
    """Reference: ConvNetBN clean victims, as logged by any of its runs."""
    best = None
    for p in glob.glob('ours_result/CIFAR10_ConvNetBN_*/log.txt'):
        for l in open(p, errors='ignore'):
            m = re.search(r'clean baseline CTA = ([\d.]+)', l)
            if m:
                best = float(m.group(1)) * 100
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--defense', choices=['epic', 'friends'], required=True)
    p.add_argument('--model', default='ConvNetBN')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()

    ref = undefended()
    if ref is None:
        sys.exit('pick_defense: no clean baseline found for %s' % a.model)

    pat = 'epic-s([\\d.]+)' if a.defense == 'epic' else 'friends-[^-]*-e([\\d.]+)'
    found = {}
    for d in glob.glob('defense_result/CIFAR10_%s_*__def-*' % a.model):
        m = re.search(pat, os.path.basename(d))
        lg = os.path.join(d, 'log.txt')
        if not m or not os.path.exists(lg):
            continue
        c = clean_cta(lg)
        if c is not None:
            found.setdefault(float(m.group(1)), c)

    if not found:
        sys.exit('pick_defense: no %s calibration runs on disk yet -- run ap5-a.sh first'
                 % a.defense)

    # strongest first: EPIC ascending subset size is weaker, FRIENDS ascending eps is stronger
    order = sorted(found) if a.defense == 'epic' else sorted(found, reverse=True)
    ok = [v for v in order if found[v] >= ref - TOL]
    if a.verbose:
        print('# undefended %s clean CTA = %.2f, band >= %.2f' % (a.model, ref, ref - TOL), file=sys.stderr)
        for v in order:
            print('#   %-8s clean CTA %.2f  %s'
                  % (v, found[v], 'within band' if found[v] >= ref - TOL else 'too weak a model'),
                  file=sys.stderr)
    if not ok:
        sys.exit('pick_defense: no %s setting stays within %.1f points of %.2f -- widen the '
                 'sweep in ap5-a.sh' % (a.defense, TOL, ref))
    print(ok[0])


if __name__ == '__main__':
    main()
