"""ASD architecture adapter, using the official three-stage training algorithm.

ASD: https://github.com/KuofengGao/ASD (ASD.py, model/loss.py,
utils/trainer/semi.py, pinned commit 8bcb10b345fa39f239e6c211e8190ec27d88efbd).
See README.md for explicit architecture adaptations.
"""
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import atomic_json, pause_if_requested, save_torch


def augment(x):
    b, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect').permute(0, 2, 3, 1)
    dy = torch.randint(9, (b, 1, 1), device=x.device)
    dx = torch.randint(9, (b, 1, 1), device=x.device)
    x = padded[torch.arange(b, device=x.device)[:, None, None],
               dy + torch.arange(h, device=x.device)[None, :, None],
               dx + torch.arange(w, device=x.device)[None, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(b, 1, 1, 1, device=x.device) < .5
    return torch.where(flip, x.flip(-1), x).contiguous()


def sce_loss(logits, labels):
    prob = logits.softmax(1).clamp(1e-7, 1)
    onehot = F.one_hot(labels, logits.shape[1]).float().clamp(1e-4, 1)
    return .1 * F.cross_entropy(logits, labels, reduction='none') - (prob*onehot.log()).sum(1)


@torch.no_grad()
def losses(net, x, y, sce=False, augmented=False, bs=256):
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        z = net(augment(x[i:i+bs]) if augmented else x[i:i+bs])
        out.append(sce_loss(z, y[i:i+bs]) if sce else
                   F.cross_entropy(z, y[i:i+bs], reduction='none'))
    return torch.cat(out)


def interleave(chunks):
    b, n = len(chunks[0]), len(chunks)
    sizes = [b//n]*n
    for i in range(b-sum(sizes)):
        sizes[-i-1] += 1
    offsets = np.cumsum([0]+sizes).tolist()
    blocks = [[x[offsets[i]:offsets[i+1]] for i in range(n)] for x in chunks]
    for i in range(1, n):
        blocks[0][i], blocks[i][i] = blocks[i][i], blocks[0][i]
    return [torch.cat(p) for p in blocks]


def split_asd(loss, labels, trusted, epoch, meta_loss=None):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if epoch < 60:
        mask[trusted] = True
        count = (epoch//5)*10
        for c in range(10):
            candidates = ((labels == c) & ~mask).nonzero().flatten()
            chosen = candidates[torch.argsort(loss[candidates], stable=True)[:count]]
            mask[chosen] = True
    else:
        score = loss if epoch < 90 else loss - meta_loss
        mask[torch.argsort(score, stable=True)[:len(labels)//2]] = True
    return mask


def meta_parameters(net):
    # The official ResNet18 adapts its last two stages and classifier.
    if hasattr(net, 'layer3'):
        modules = [net.layer2, net.layer3, net.classifier]
    else:
        pools = [i for i, m in enumerate(net.features)
                 if isinstance(m, (torch.nn.AvgPool2d, torch.nn.MaxPool2d))]
        assert len(pools) >= 2
        modules = [net.features[pools[-3]+1:] if len(pools) >= 3 else net.features,
                   net.classifier]
    for p in net.parameters():
        p.requires_grad_(False)
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


class PoolBatches:
    def __init__(self, indices, size=64):
        self.indices, self.size = indices, size
        self.perm = indices[:0]
        self.pos = 0
        assert len(indices) >= size

    def next(self):
        if self.pos + self.size > len(self.perm):
            self.perm = self.indices[torch.randperm(len(self.indices), device=self.indices.device)]
            self.pos = 0
        result = self.perm[self.pos:self.pos+self.size]
        self.pos += self.size
        return result


def mixmatch_epoch(net, opt, x, y, labeled, epoch, steps=1024):
    xp = PoolBatches(labeled.nonzero().flatten())
    up = PoolBatches((~labeled).nonzero().flatten())
    net.train()
    total = 0.
    for step in range(steps):
        ix, iu = xp.next(), up.next()
        xi, u1, u2 = augment(x[ix]), augment(x[iu]), augment(x[iu])
        xt = F.one_hot(y[ix], 10).float()
        with torch.no_grad():
            p = (net(u1).softmax(1)+net(u2).softmax(1))/2
            p = p.square()
            ut = p/p.sum(1, keepdim=True)
        inputs = torch.cat([xi, u1, u2])
        targets = torch.cat([xt, ut, ut])
        coefficient = float(np.random.beta(.75, .75))
        coefficient = max(coefficient, 1-coefficient)
        permutation = torch.randperm(len(inputs), device=x.device)
        mixed = coefficient*inputs + (1-coefficient)*inputs[permutation]
        target = coefficient*targets + (1-coefficient)*targets[permutation]
        logits = interleave([net(z) for z in interleave(list(mixed.split(64)))])
        lx = -(logits[0].log_softmax(1)*target[:64]).sum(1).mean()
        lu = (torch.cat(logits[1:]).softmax(1)-target[64:]).square().mean()
        loss = lx + 15*min((epoch+step/steps)/120, 1)*lu
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach())
    return total/steps


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def train_asd(net, x, y, trusted, directory, epochs=120, steps=1024):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.pt'
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    start, history = 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt'])
        start, history = state['epoch'], state['history']
        restore_rng(state['rng'])
        print(f'ASD resume: {start}/{epochs} epochs complete in {directory}', flush=True)
        del state
    for ep in range(start, epochs):
        before = time.monotonic()
        score = losses(net, x, y, sce=True, augmented=True)
        meta_score = None
        if ep >= 90:
            meta = copy.deepcopy(net)
            meta_opt = torch.optim.Adam(meta_parameters(meta), lr=.015)
            meta.train()
            perm = torch.randperm(len(y), device=x.device)
            for ids in perm.split(128):
                meta_opt.zero_grad(set_to_none=True)
                F.cross_entropy(meta(augment(x[ids])), y[ids]).backward()
                meta_opt.step()
            meta_score = losses(meta, x, y, sce=True, augmented=True)
            del meta, meta_opt
        labeled = split_asd(score, y, trusted, ep, meta_score)
        mean_loss = mixmatch_epoch(net, opt, x, y, labeled, ep, steps)
        history.append(dict(epoch=ep+1, labeled=int(labeled.sum()), loss=mean_loss,
                            seconds=time.monotonic()-before))
        save_torch(checkpoint, dict(net=net.state_dict(), opt=opt.state_dict(),
                   epoch=ep+1, history=history, rng=rng_state()))
        atomic_json(directory/'training.json', history)
        print('ASD', ep+1, '/', epochs, history[-1], flush=True)
        pause_if_requested()
    return net.eval()


