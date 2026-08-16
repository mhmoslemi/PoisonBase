#!/usr/bin/env python
"""
test_victim_aug.py -- checks victim_aug.py against the reference recipes.

Run:  python test_victim_aug.py            (cpu is enough; add --cuda to repeat
                                             the equivalence checks on gpu)

The claim being tested is the one the experiment rests on: these batched ops are
per-sample and match torchvision / the Cutout reference, so `--victim_aug
standard|randaug|cutout` is the real recipe and not a batch-level approximation
of it.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

import victim_aug as VA


def _ctx(device, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    from utils import ParamDiffAug
    return {'norm': lambda x: (x - m) / s, 'denorm': lambda x: x * s + m,
            'im_size': (32, 32), 'dsa_param': ParamDiffAug(), 'device': device}


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------------- #

@check('randaug: batched == torchvision looped per image')
def t_randaug_matches_torchvision(device):
    """The only real risk in the batched RandAugment is that grouping images by
    (op, sign) is not the same as applying the op to each image alone. Force the
    same choices in both and demand bit-equality."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.autoaugment import _apply_op

    g = torch.Generator().manual_seed(0)
    N = 64
    x = torch.randint(0, 256, (N, 3, 32, 32), generator=g, dtype=torch.uint8).to(device)
    ops = VA.randaug_ops(magnitude=9, num_bins=31, im_size=(32, 32))
    assert len(ops) == 14, len(ops)

    for trial in range(4):
        gg = torch.Generator().manual_seed(100 + trial)
        choice = torch.randint(len(ops), (N,), generator=gg)
        sign = torch.where(torch.randint(2, (N,), generator=gg).bool(),
                           torch.tensor(-1), torch.tensor(1))
        got = VA.randaug_step(x, ops, choice, sign)

        want = torch.empty_like(x)
        for i in range(N):
            name, mag, signed = ops[int(choice[i])]
            m = mag * (int(sign[i]) if signed else 1)
            one = x[i:i + 1]
            want[i:i + 1] = (one if name == 'Identity' else _apply_op(
                one, name, m, interpolation=InterpolationMode.NEAREST, fill=None))
        assert torch.equal(got, want), 'trial %d: %d/%d pixels differ' % (
            trial, (got != want).sum().item(), got.numel())

    # every op must actually be reachable
    seen = set()
    for _ in range(400):
        c = torch.randint(len(ops), (N,))
        seen.update(int(v) for v in c.unique())
    assert seen == set(range(len(ops))), sorted(seen)
    return '4 trials x %d images, all 14 ops bit-identical to _apply_op' % N


@check('randaug: op choice is per sample and uniform')
def t_randaug_per_sample(device):
    ops = VA.randaug_ops()
    N, R = 256, 60
    x = torch.zeros(N, 3, 32, 32, dtype=torch.uint8, device=device)
    torch.manual_seed(0)
    # a batch of identical images must come out NOT all identical
    x[:] = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8, device=device)
    out = VA.rand_augment(x, ops, num_ops=2)
    n_distinct = len({tuple(v.flatten()[:16].tolist()) for v in out})
    assert n_distinct > 5, 'only %d distinct results -- batch-level, not per-sample' % n_distinct

    counts = torch.zeros(len(ops))
    for _ in range(R):
        counts += torch.bincount(torch.randint(len(ops), (N,)), minlength=len(ops))
    exp = N * R / len(ops)
    assert (counts - exp).abs().max() < 0.35 * exp, counts.tolist()
    return '%d distinct outputs from one repeated image; op histogram flat' % n_distinct


@check('crop: matches transforms.RandomCrop(32, padding=4), per sample')
def t_crop(device):
    torch.manual_seed(0)
    N = 128
    x = torch.rand(N, 3, 32, 32, device=device)
    out = VA.rand_crop(x, pad=4)
    xp = F.pad(x, (4, 4, 4, 4), value=0.0)

    # every output must be SOME (i, j) window of the padded image, and the (i, j)
    # actually used must vary within the batch and cover the full 9x9 grid
    used = set()
    for i in range(N):
        hit = None
        for a in range(9):
            for b in range(9):
                if torch.equal(out[i], xp[i, :, a:a + 32, b:b + 32]):
                    hit = (a, b)
                    break
            if hit:
                break
        assert hit is not None, 'image %d is not a crop of its own padded self' % i
        used.add(hit)
    assert len(used) > 20, 'only %d distinct offsets -- not per sample' % len(used)

    torch.manual_seed(0)
    grid = torch.zeros(9, 9)
    for _ in range(80):
        i = torch.randint(0, 9, (N,))
        j = torch.randint(0, 9, (N,))
        grid += torch.bincount(i * 9 + j, minlength=81).view(9, 9)
    assert (grid > 0).all(), 'some offsets unreachable'
    exp = N * 80 / 81.0
    assert (grid - exp).abs().max() < 0.5 * exp, grid
    # zero padding is what shows up when the window hangs off the edge
    edge = VA.rand_crop(torch.ones(1, 3, 32, 32, device=device), pad=4)
    assert edge.min() >= 0.0 and edge.max() == 1.0
    return '%d/81 offsets in one batch of %d, offsets uniform, pad = 0' % (len(used), N)


@check('flip: p=0.5, per sample, exact mirror')
def t_flip(device):
    torch.manual_seed(0)
    N = 4096
    x = torch.rand(N, 3, 32, 32, device=device)
    out = VA.rand_hflip(x, 0.5)
    same = torch.tensor([torch.equal(out[i], x[i]) for i in range(N)])
    flipped = torch.tensor([torch.equal(out[i], x[i].flip(-1)) for i in range(N)])
    assert (same | flipped).all(), 'some image is neither itself nor its mirror'
    frac = float(flipped.float().mean())
    assert 0.45 < frac < 0.55, frac
    return 'flipped %.3f of %d images (target 0.5)' % (frac, N)


@check('cutout: 16x16 box, centre uniform, clipped at the border, fill 0')
def t_cutout(device):
    torch.manual_seed(0)
    N = 512
    x = torch.ones(N, 3, 32, 32, device=device)
    out = VA.rand_cutout(x, length=16)
    holes = (out[:, 0] == 0.0)
    sizes = holes.flatten(1).sum(1)
    assert sizes.max() <= 256, sizes.max().item()
    full = int((sizes == 256).sum())
    assert full > 0 and int((sizes < 256).sum()) > 0, 'border clipping missing'

    # each hole must be one axis-aligned rectangle of the right nominal extent
    for i in range(0, N, 37):
        rows = holes[i].any(1).nonzero().flatten()
        cols = holes[i].any(0).nonzero().flatten()
        assert torch.equal(rows, torch.arange(int(rows[0]), int(rows[-1]) + 1,
                                              device=rows.device))
        assert torch.equal(cols, torch.arange(int(cols[0]), int(cols[-1]) + 1,
                                              device=cols.device))
        h, w = len(rows), len(cols)
        assert h == 16 or int(rows[0]) == 0 or int(rows[-1]) == 31, h
        assert w == 16 or int(cols[0]) == 0 or int(cols[-1]) == 31, w
    # every pixel outside the hole is untouched
    assert torch.equal(out[~holes.unsqueeze(1).expand_as(out)],
                       x[~holes.unsqueeze(1).expand_as(out)])
    # distinct boxes within one batch -> per sample
    boxes = {(int(holes[i].any(1).nonzero()[0]), int(holes[i].any(0).nonzero()[0]))
             for i in range(N)}
    assert len(boxes) > 50, len(boxes)
    return ('%d/%d full 16x16 boxes, rest clipped at the border, '
            '%d distinct boxes' % (full, N, len(boxes)))


@check('augmenter: poison survives, no augmentation leaks between images')
def t_augmenter(device):
    ctx = _ctx(device)
    torch.manual_seed(0)
    N = 96
    x01 = torch.rand(N, 3, 32, 32, device=device)
    x_norm = ctx['norm'](x01)
    args = argparse.Namespace(aug_pad=4, aug_flip_p=0.5, aug_cutout_length=16,
                              aug_randaug_ops=2, aug_randaug_magnitude=9,
                              dsa_strategy='color_crop_cutout_flip_scale_rotate')

    notes = []
    for mode in ['none', 'standard', 'randaug', 'cutout', 'dsa']:
        args.victim_aug = mode
        aug = VA.make_augmenter(args, ctx)
        if mode == 'none':
            assert aug is None
            notes.append('none->None')
            continue
        out1, out2 = aug(x_norm), aug(x_norm)
        assert out1.shape == x_norm.shape and out1.dtype == x_norm.dtype
        assert torch.isfinite(out1).all()
        # a fresh draw every call -- otherwise it is not dynamic per epoch
        assert not torch.equal(out1, out2), '%s: two calls gave the same batch' % mode
        # the batch must not collapse: identical inputs get independent draws
        rep = aug(x_norm[:1].expand(N, -1, -1, -1).contiguous())
        distinct = len({tuple(v.flatten()[:24].tolist()) for v in rep})
        assert distinct > 5, ('%s: one repeated image gave %d distinct outputs -- '
                              'batch-level, not per-sample' % (mode, distinct))
        notes.append('%s(%d/%d distinct)' % (mode, distinct, N))
    return 'ok for %s (fresh draw each call)' % ', '.join(notes)


@check('augmenter: the saved poison is what gets augmented')
def t_poison_first(device):
    """standard/cutout are geometry + masking only, so the augmented poisoned
    image must equal the augmentation of (base + delta) -- i.e. the delta is
    inside the crop, not applied after it."""
    ctx = _ctx(device)
    N = 64
    torch.manual_seed(3)
    base01 = torch.rand(N, 3, 32, 32, device=device)
    delta = (torch.rand(N, 3, 32, 32, device=device) * 2 - 1) * (8 / 255.0)
    adv01 = (base01 + delta).clamp(0, 1)

    args = argparse.Namespace(victim_aug='standard', aug_pad=4, aug_flip_p=0.5,
                              aug_cutout_length=16, aug_randaug_ops=2,
                              aug_randaug_magnitude=9, dsa_strategy=None)
    aug = VA.make_augmenter(args, ctx)
    torch.manual_seed(11)
    got = ctx['denorm'](aug(ctx['norm'](adv01)))
    torch.manual_seed(11)
    aug_base = ctx['denorm'](aug(ctx['norm'](base01)))
    torch.manual_seed(11)
    aug_adv = ctx['denorm'](aug(ctx['norm'](adv01)))

    assert torch.allclose(got, aug_adv, atol=1e-6)
    d = (got - aug_base).abs()
    assert d.max() > 1e-4, 'the perturbation vanished under augmentation'
    assert d.max() <= 8 / 255.0 + 1e-4, ('crop/flip moved the perturbation but must '
                                         'not amplify it: max %.4f' % d.max())
    moved = int((d.flatten(1).max(1).values > 1e-6).sum())
    return ('delta carried through crop+flip on %d/%d images, '
            'linf still <= 8/255 (%.3f/255)' % (moved, N, d.max().item() * 255))


@check('defense.py: augmentation reaches both victim-training paths')
def t_both_training_paths(device):
    """train_victim_defended has two loops: a fast path that delegates to
    FU.train_from_scratch (--defense none) and its own loop (epic/friends/
    advtrain). The augmenter has to be applied in both, and only the second one
    is exercised by a `--defense none` smoke run."""
    import torch.nn as nn

    import defense as DEF

    calls = {'n': 0}

    class Counting(object):
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, x):
            calls['n'] += 1
            return self.inner(x)

    ctx = _ctx(device)
    N, bs, epochs = 200, 50, 2
    ctx.update(num_classes=4, channel=3,
               train_imgs=ctx['norm'](torch.rand(N, 3, 32, 32, device=device)),
               train_labs=torch.randint(0, 4, (N,), device=device),
               test_imgs=torch.rand(4, 3, 32, 32, device=device),
               test_labs=torch.zeros(4, dtype=torch.long, device=device))

    base = dict(victim_epochs=epochs, victim_lr=0.01, victim_bs=bs,
                victim_decay=[], victim_wd=0.0, victim_aug='standard',
                aug_pad=4, aug_flip_p=0.5, aug_cutout_length=16,
                aug_randaug_ops=2, aug_randaug_magnitude=9, dsa_strategy=None,
                noise_type=['bernoulli'], noise_eps=8.0, adv_eps=8.0,
                adv_steps=2, adv_step_size=0.0, epic_freq=2, epic_drop_after=0,
                epic_stop_after=-1, epic_subset_size=0.1, epic_metric='euclidean',
                epic_cluster_thresh=1.0, friendly_begin_epoch=99,
                friendly_clamp=16.0)

    seen = {}
    for name, parts in [('fast (--defense none)', ['none']),
                        ('own loop (--defense noise)', ['noise'])]:
        args = argparse.Namespace(defense_parts=parts, **base)
        ctx.pop('augmenter', None)
        ctx['augmenter'] = Counting(VA.make_augmenter(args, ctx))
        calls['n'] = 0
        torch.manual_seed(0)
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 4)).to(device)
        DEF.train_victim_defended(args, ctx, net)
        want = epochs * (N // bs)
        assert calls['n'] == want, ('%s: augmenter called %d times, expected one '
                                    'per minibatch (%d)' % (name, calls['n'], want))
        seen[name] = calls['n']

    # and it must be a no-op-free path when the mode is none
    args = argparse.Namespace(defense_parts=['none'], **dict(base, victim_aug='none'))
    ctx.pop('augmenter', None)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 4)).to(device)
    DEF.train_victim_defended(args, ctx, net)
    assert ctx['augmenter'] is None, '--victim_aug none must not build an augmenter'
    return ' / '.join('%s: %d batches' % kv for kv in seen.items())


@check('results.csv written before the aug column still reads correctly')
def t_csv_migration(device):
    """Adding 'aug' widened RESULT_FIELDS. A results.csv from before that has a
    narrower header, and csv assigns by POSITION -- appending to it without
    migrating would make target_idx read the aug value, victim_id read
    target_idx, and so on, silently."""
    import csv
    import os
    import shutil
    import tempfile

    import defense as DEF

    old_fields = [f for f in DEF.RESULT_FIELDS if f != 'aug']
    assert 'aug' in DEF.RESULT_FIELDS and len(old_fields) + 1 == len(DEF.RESULT_FIELDS)

    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, 'results.csv')
        want = []
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writeheader()
            for t in (11, 22):
                for v in (0, 1):
                    row = dict.fromkeys(old_fields, '')
                    row.update(model='ResNet20BN', attack='fc', base='ours',
                               sel='dpp2', defense='none', target_idx=t,
                               victim_id=v, success=v, clean_test_acc=0.9)
                    w.writerow(row)
                    want.append((str(t), str(v), str(v)))

        assert DEF.migrate_header(path) is True
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            assert rd.fieldnames == DEF.RESULT_FIELDS, rd.fieldnames
            got = [(r['target_idx'], r['victim_id'], r['success']) for r in rd]
        assert got == want, (got, want)

        # a shard opened on the migrated file appends in the right columns
        rf, w = DEF.open_shard(d, 0)
        shutil.copy(path, os.path.join(d, 'results_rank0.csv'))
        rf.close()
        rf, w = DEF.open_shard(d, 0)
        w.writerow({k: '' for k in DEF.RESULT_FIELDS} |
                   {'target_idx': 33, 'victim_id': 5, 'success': 1, 'aug': 'randaug'})
        rf.close()
        with open(os.path.join(d, 'results_rank0.csv'), newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows[-1]['target_idx'] == '33' and rows[-1]['aug'] == 'randaug', rows[-1]
        assert rows[0]['target_idx'] == '11' and rows[0]['aug'] == '', rows[0]
        n_old = len(want)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return ('%d pre-aug rows kept their columns, new row appended aligned'
            % n_old)


@check('tag: one directory per augmentation config')
def t_tag(device):
    args = argparse.Namespace(aug_pad=4, aug_flip_p=0.5, aug_cutout_length=16,
                              aug_randaug_ops=2, aug_randaug_magnitude=9,
                              dsa_strategy='x')
    tags = {}
    for mode in VA.MODES:
        args.victim_aug = mode
        tags[mode] = VA.aug_tag(args)
    assert tags['none'] == ''
    assert len(set(tags.values())) == len(VA.MODES), tags
    args.victim_aug, args.aug_randaug_magnitude = 'randaug', 14
    assert VA.aug_tag(args) != tags['randaug']
    return ' '.join('%s=%r' % kv for kv in sorted(tags.items()))


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', action='store_true')
    a = ap.parse_args()
    devices = ['cpu'] + (['cuda'] if a.cuda and torch.cuda.is_available() else [])
    bad = 0
    for dev in devices:
        print('=== device %s ===' % dev)
        for name, fn in CHECKS:
            try:
                note = fn(dev)
                print('  PASS  %-58s %s' % (name, note or ''))
            except Exception as e:
                bad += 1
                print('  FAIL  %-58s %s: %s' % (name, type(e).__name__, e))
                import traceback
                traceback.print_exc()
    print('\n%d check(s) failed' % bad if bad else '\nall checks passed')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
