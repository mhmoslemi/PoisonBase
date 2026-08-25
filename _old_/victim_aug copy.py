#!/usr/bin/env python
"""
victim_aug.py

Victim-training data augmentation for the poison-REPLAY experiments (defense.py).

The point of the experiment this exists for: the poisons are already on disk and
are NOT re-optimized. defense.py writes the saved perturbation into the training
image once, before training starts (run_trial -> train_imgs[base_idx] =
norm(x_adv01)), and the augmentation here is then applied on top of the poisoned
image, freshly sampled for every batch of every epoch. So the pipeline is

    base image  ->  + saved delta  ->  augment(new draw each epoch)  ->  network

which is exactly what a victim who augments would do to a poisoned dataset: the
victim cannot see the perturbation, it just augments whatever it was given.

Modes (--victim_aug):

  none      no augmentation. The MetaPoison victim protocol the attacks used.

  standard  RandomCrop(32, padding=4) + RandomHorizontalFlip(p=0.5).

  randaug   standard, then RandAugment(num_ops=2, magnitude=9).

  cutout    standard, then one random 16x16 Cutout region.

  dsa       the DiffAugment / DSA strategy that was already wired up here
            (--dsa_strategy, default color_crop_cutout_flip_scale_rotate).
            Kept so the old `--victim_aug` flag keeps meaning what it meant.

Everything is GPU resident and manually batched in this codebase -- there is no
DataLoader and no PIL image anywhere -- so every op below is a batched tensor op
with PER-SAMPLE randomness. That is the part worth being careful about: applying
one sampled transform to a whole batch is a materially weaker augmentation than
torchvision's per-image sampling, and it is the easy thing to get wrong here.

Fidelity to the torchvision recipes:

  * crop / flip run in [0,1] pixel space before normalization, and the crop pads
    with 0 (black), which is what transforms.RandomCrop(32, padding=4) does.

  * RandAugment runs on uint8, like torchvision's (it requires uint8 for
    Posterize/Equalize). It is exactly torchvision's op list and magnitude table,
    read off the installed torchvision rather than hardcoded, and the ops are
    applied through torchvision's own _apply_op. Per-sample sampling is done by
    grouping: torchvision's RandAugment magnitude is a CONSTANT (only the op and
    its sign are random per image), so every image that drew the same (op, sign)
    at a given step gets the identical transform and can be sent through
    _apply_op in one call. That makes the batched version distributionally
    IDENTICAL to looping torchvision over the images one at a time -- see
    test_victim_aug.py, which checks it against exactly that loop -- at ~24 op
    calls per step instead of one per image.

    Note the uint8 round trip quantizes the poison to the 1/255 grid (the saved
    deltas are continuous). That is inherent to the standard recipe, and the
    <=0.5/255 rounding is small next to eps=8/255.

  * Cutout is applied AFTER normalization with fill 0, and its box is centred on
    a uniformly random pixel and clipped at the border (so an edge box covers
    less than 16x16). Both match DeVries & Taylor's reference implementation.
"""

import torch
import torch.nn.functional as F

from _old_.utils import DiffAugment

MODES = ['none', 'standard', 'randaug', 'cutout', 'dsa']


# --------------------------------------------------------------------------- #
# crop / flip / cutout: batched, one independent draw per image
# --------------------------------------------------------------------------- #

def rand_crop(x, pad=4):
    """RandomCrop(H, padding=pad) with zero padding, per sample."""
    N, C, H, W = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
    di = torch.randint(0, 2 * pad + 1, (N,), device=dev)
    dj = torch.randint(0, 2 * pad + 1, (N,), device=dev)
    n = torch.arange(N, device=dev).view(N, 1, 1, 1)
    c = torch.arange(C, device=dev).view(1, C, 1, 1)
    r = di.view(N, 1, 1, 1) + torch.arange(H, device=dev).view(1, 1, H, 1)
    s = dj.view(N, 1, 1, 1) + torch.arange(W, device=dev).view(1, 1, 1, W)
    return xp[n, c, r, s]


def rand_hflip(x, p=0.5):
    """RandomHorizontalFlip(p), per sample."""
    N = x.shape[0]
    flip = torch.rand(N, device=x.device) < p
    return torch.where(flip.view(N, 1, 1, 1), x.flip(-1), x)


def rand_cutout(x, length=16, fill=0.0):
    """One Cutout box per sample, centre uniform over the image, clipped at the
    border. DeVries & Taylor's Cutout, which runs after normalization with
    fill 0 -- so `x` is expected NORMALIZED here, not in [0,1]."""
    N, _C, H, W = x.shape
    dev = x.device
    cy = torch.randint(0, H, (N, 1, 1), device=dev)
    cx = torch.randint(0, W, (N, 1, 1), device=dev)
    y1, y2 = (cy - length // 2).clamp(0, H), (cy + length // 2).clamp(0, H)
    x1, x2 = (cx - length // 2).clamp(0, W), (cx + length // 2).clamp(0, W)
    rows = torch.arange(H, device=dev).view(1, H, 1)
    cols = torch.arange(W, device=dev).view(1, 1, W)
    hole = ((rows >= y1) & (rows < y2) & (cols >= x1) & (cols < x2)).unsqueeze(1)
    return torch.where(hole, torch.full_like(x, fill), x)


# --------------------------------------------------------------------------- #
# RandAugment: torchvision's ops and magnitudes, applied per sample by grouping
# --------------------------------------------------------------------------- #

def randaug_ops(magnitude=9, num_bins=31, im_size=(32, 32)):
    """torchvision's RandAugment space at this magnitude -> [(name, mag, signed)].

    Read off the installed torchvision so the table can never drift from what
    transforms.RandAugment(num_ops, magnitude) would actually do.
    """
    from torchvision.transforms.autoaugment import RandAugment as _TVRandAugment
    space = _TVRandAugment(num_magnitude_bins=num_bins)._augmentation_space(
        num_bins, tuple(im_size))
    return [(name, float(mags[magnitude].item()) if mags.ndim > 0 else 0.0,
             bool(signed)) for name, (mags, signed) in space.items()]


def randaug_step(x_u8, ops, choice, sign, interpolation=None, fill=None):
    """One RandAugment op per sample: image i gets ops[choice[i]], signed sign[i].

    `choice` / `sign` are CPU int tensors of length N (kept on the cpu so the
    grouping never syncs the gpu). Images sharing an (op, sign) share the exact
    same transform -- the magnitude is a constant in RandAugment -- so each group
    goes through torchvision's _apply_op in a single batched call.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.autoaugment import _apply_op
    if interpolation is None:
        interpolation = InterpolationMode.NEAREST

    out = x_u8.clone()
    choice, sign = choice.cpu(), sign.cpu()
    for k, (name, mag, signed) in enumerate(ops):
        if name == 'Identity':
            continue
        for s in ((1, -1) if signed else (1,)):
            hit = (choice == k)
            if signed:
                hit = hit & (sign == s)
            idx = hit.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            idx = idx.to(out.device)
            out.index_copy_(0, idx, _apply_op(
                out.index_select(0, idx), name, mag * s,
                interpolation=interpolation, fill=fill))
    return out


def rand_augment(x_u8, ops, num_ops=2, interpolation=None, fill=None):
    """RandAugment(num_ops, magnitude) on a uint8 batch, sampled per image."""
    N = x_u8.shape[0]
    n = len(ops)
    for _ in range(num_ops):
        choice = torch.randint(n, (N,))
        # torchvision: `if signed and torch.randint(2, (1,)): magnitude *= -1`
        sign = torch.where(torch.randint(2, (N,)).bool(),
                           torch.tensor(-1), torch.tensor(1))
        x_u8 = randaug_step(x_u8, ops, choice, sign, interpolation, fill)
    return x_u8


# --------------------------------------------------------------------------- #
# the augmenter handed to the victim training loop
# --------------------------------------------------------------------------- #

class VictimAugmenter(object):
    """normalized batch -> augmented normalized batch, a fresh draw every call.

    Held per process and called once per minibatch, so the same image is
    augmented differently in every epoch.
    """

    def __init__(self, mode, ctx, pad=4, flip_p=0.5, cutout_length=16,
                 randaug_num_ops=2, randaug_magnitude=9, randaug_bins=31,
                 dsa_strategy=None):
        if mode not in MODES:
            raise ValueError('unknown --victim_aug %r (pick from %s)'
                             % (mode, '/'.join(MODES)))
        if mode == 'dsa' and not dsa_strategy:
            raise ValueError('--victim_aug dsa needs a --dsa_strategy')
        self.mode = mode
        self.norm, self.denorm = ctx['norm'], ctx['denorm']
        self.pad, self.flip_p = pad, flip_p
        self.cutout_length = cutout_length
        self.randaug_num_ops = randaug_num_ops
        self.dsa_strategy, self.dsa_param = dsa_strategy, ctx['dsa_param']
        self.ops = (randaug_ops(randaug_magnitude, randaug_bins, ctx['im_size'])
                    if mode == 'randaug' else None)

    def __repr__(self):
        if self.mode == 'dsa':
            return 'VictimAugmenter(dsa: %s)' % self.dsa_strategy
        if self.mode == 'none':
            return 'VictimAugmenter(none)'
        s = 'crop%d+flip%g' % (self.pad, self.flip_p)
        if self.mode == 'randaug':
            s += ' -> randaug(n=%d, m from %d ops)' % (self.randaug_num_ops,
                                                       len(self.ops))
        elif self.mode == 'cutout':
            s += ' -> cutout%d' % self.cutout_length
        return 'VictimAugmenter(%s: %s)' % (self.mode, s)

    def __call__(self, x_norm):
        if self.mode == 'none':
            return x_norm
        if self.mode == 'dsa':
            return DiffAugment(x_norm, self.dsa_strategy, param=self.dsa_param)

        # crop/flip belong in pixel space: RandomCrop pads with black, and
        # RandAugment is a uint8 recipe.
        x01 = self.denorm(x_norm).clamp(0.0, 1.0)
        if self.mode == 'randaug':
            x = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)
            x = rand_hflip(rand_crop(x, self.pad), self.flip_p)
            x = rand_augment(x, self.ops, self.randaug_num_ops)
            x01 = x.to(x_norm.dtype).div_(255.0)
        else:
            x01 = rand_hflip(rand_crop(x01, self.pad), self.flip_p)

        out = self.norm(x01)
        if self.mode == 'cutout':
            # Cutout's reference implementation masks after Normalize
            out = rand_cutout(out, self.cutout_length)
        return out


def make_augmenter(args, ctx):
    """None when --victim_aug none, so the caller can skip the call entirely."""
    mode = getattr(args, 'victim_aug', 'none') or 'none'
    if mode == 'none':
        return None
    return VictimAugmenter(
        mode, ctx, pad=args.aug_pad, flip_p=args.aug_flip_p,
        cutout_length=args.aug_cutout_length,
        randaug_num_ops=args.aug_randaug_ops,
        randaug_magnitude=args.aug_randaug_magnitude,
        dsa_strategy=args.dsa_strategy)


def aug_tag(args):
    """Short name for the augmentation CONFIG, for run dirs and cache keys."""
    mode = getattr(args, 'victim_aug', 'none') or 'none'
    if mode == 'none':
        return ''
    if mode == 'dsa':
        return 'aug-dsa'
    tag = 'aug-%s' % mode
    if args.aug_pad != 4 or args.aug_flip_p != 0.5:
        tag += '-p%d-f%g' % (args.aug_pad, args.aug_flip_p)
    if mode == 'randaug' and (args.aug_randaug_ops != 2
                              or args.aug_randaug_magnitude != 9):
        tag += '-n%dm%d' % (args.aug_randaug_ops, args.aug_randaug_magnitude)
    if mode == 'cutout' and args.aug_cutout_length != 16:
        tag += '-L%d' % args.aug_cutout_length
    return tag


def add_args(p):
    """The --victim_aug knobs, shared by anything that trains a victim."""
    p.add_argument('--victim_aug', nargs='?', const='dsa', default='none',
                   choices=MODES,
                   help='victim-training augmentation, applied on top of the '
                        'already-poisoned image, resampled every epoch. Bare '
                        '--victim_aug still means dsa, as it did before.')
    p.add_argument('--aug_pad', type=int, default=4,
                   help='RandomCrop padding for standard/randaug/cutout')
    p.add_argument('--aug_flip_p', type=float, default=0.5)
    p.add_argument('--aug_cutout_length', type=int, default=16)
    p.add_argument('--aug_randaug_ops', type=int, default=2,
                   help="RandAugment num_ops")
    p.add_argument('--aug_randaug_magnitude', type=int, default=9,
                   help='RandAugment magnitude')
    return p
