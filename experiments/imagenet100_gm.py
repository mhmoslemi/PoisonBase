#!/usr/bin/env python3
"""One appendix setting: ImageNet-100, ImageNet ResNet18, GM, RAND/M/BASIS."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_update as fu

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
METHODS = ('random', 'minus-m', 'basis')
PROTOCOL = dict(dataset='ImageNet-100', class_seed=0, classes=100,
                model='torchvision_resnet18_weights_none', epochs=90, batch_size=128,
                lr=0.1, momentum=0.9, weight_decay=1e-4, decay=[30, 60],
                train_crop=224, eval_resize=256, eval_crop=224,
                attack='gradmatch', budget=0.002, epsilon=8/255,
                craft_steps=250, craft_alpha=0.0039216, restarts=8,
                surrogates=3, targets=8, victims=6, lambda_margin=1,
                base_dist='cosine', poison_geometry='native_center_field_v1')
STOP = False


class Paused(Exception):
    pass


def request_stop(*_):
    global STOP
    STOP = True


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def load(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)


def seed(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)


def locate_data(base):
    for path in (base / 'imagenet', base / 'imagenet1k', base / 'ILSVRC2012', base):
        if (path / 'train').is_dir() and (path / 'val').is_dir():
            return path.resolve()
    raise RuntimeError(f'ImageNet-1K not found under {base}. Expected '
                       'train/<WNID> and val/<WNID> directories; '
                       'set IMAGENET_ROOT to their parent directory.')


def prepare(data_root, output):
    data_root = locate_data(data_root)
    available = sorted(p.name for p in (data_root / 'train').iterdir()
                       if p.is_dir() and p.name.startswith('n'))
    if len(available) != 1000:
        raise RuntimeError(f'Expected 1000 ImageNet-1K train class directories, found {len(available)}')
    classes = sorted(random.Random(0).sample(available, 100))
    poison_class, target_class = random.Random(0).sample(classes, 2)
    manifest = dict(protocol=PROTOCOL, classes=classes, data_root=str(data_root),
                    poison_class=poison_class, target_class=target_class)
    for split in ('train', 'val'):
        records = []
        for label, name in enumerate(classes):
            directory = data_root / split / name
            images = sorted(p for p in directory.rglob('*')
                            if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
            if not images:
                raise RuntimeError(f'No images in {directory}; validation images must be organized by WNID')
            records.extend([str(p.relative_to(data_root)), label] for p in images)
        manifest[split] = records
    manifest['num_poisons'] = fu.rho_to_m(PROTOCOL['budget'], len(manifest['train']))
    identity = {k: v for k, v in manifest.items() if k != 'data_root'}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != manifest['fingerprint']:
        raise RuntimeError('Existing ImageNet experiment has a different protocol or file list')
    atomic_json(path, manifest)
    (output / 'classes_seed0.txt').write_text('\n'.join(classes) + '\n')
    print(f'Prepared 100 classes, {len(manifest["train"])} train images, '
          f'{manifest["num_poisons"]} poisons, pair {poison_class} -> {target_class}', flush=True)


def pixels(path):
    with Image.open(path) as image:
        return TF.to_tensor(image.convert('RGB'))


def normalize(x):
    return TF.normalize(x, MEAN, STD)


def eval_pixels(x):
    return TF.center_crop(TF.resize(x, 256, antialias=True), [224, 224])


def native_poison(clean, delta):
    """Inject a bounded 224-grid perturbation into the original image geometry.

    The grid occupies the evaluation center crop in resize-short-side-256
    coordinates. Bilinear transport to the original resolution preserves its
    L-infinity bound. Both crafting and victim training use this same map,
    before their respective preprocessing; original image dimensions are kept.
    """
    h, w = clean.shape[-2:]
    rh, rw = (256, int(w * 256 / h)) if h <= w else (int(h * 256 / w), 256)
    top, left = int(round((rh - 224) / 2)), int(round((rw - 224) / 2))
    field = F.pad(delta, (left, rw - 224 - left, top, rh - 224 - top))
    field = TF.resize(field, [h, w], antialias=True)
    return (clean + field).clamp(0, 1)


class ImageNetResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None, num_classes=100)

    def embed(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return n.avgpool(x).flatten(1)

    def forward(self, x):
        return self.net.fc(self.embed(x))


class Images(Dataset):
    def __init__(self, root, records, training=False, seed_value=0, epoch=0, poison=None):
        self.root, self.records, self.training = root, records, training
        self.seed, self.epoch = seed_value, epoch
        self.poison = {} if poison is None else dict(zip(poison['indices'], poison['delta']))
        self.augment = transforms.Compose([transforms.RandomResizedCrop(224, antialias=True),
                                             transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        relative, label = self.records[index]
        x = pixels(self.root / relative)
        if index in self.poison:
            x = native_poison(x, self.poison[index])
        if self.training:
            # Per-example/epoch randomness survives worker count and batch resume.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    ((self.seed * 1000003 + self.epoch) * 1000003 + index) % (2**63))
                x = self.augment(x)
        else:
            x = eval_pixels(x)
        return normalize(x), label


@torch.no_grad()
def evaluate(net, dataset, device, workers):
    net.eval()
    correct = total = 0
    for x, y in DataLoader(dataset, batch_size=128, num_workers=workers,
                            pin_memory=str(device).startswith('cuda')):
        correct += (net(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += len(y)
        if STOP:
            raise Paused()
    return correct / total


def train_model(path, manifest, data_root, device, workers, seed_value, poison=None):
    seed(seed_value)
    net = ImageNetResNet18().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    state = load(path, device) if path.exists() else None
    epoch, offset = 0, 0
    if state:
        if state['fingerprint'] != manifest['fingerprint'] or state['seed'] != seed_value:
            raise RuntimeError(f'incompatible training checkpoint: {path}')
        net.load_state_dict(state['model'])
        if state.get('complete'):
            return net.eval(), state['accuracy']
        opt.load_state_dict(state['optimizer'])
        epoch, offset = state['epoch'], state['offset']
    def checkpoint(ep, position, complete=False, accuracy=None):
        save(path, dict(fingerprint=manifest['fingerprint'], seed=seed_value,
                        model=net.state_dict(), optimizer=opt.state_dict() if not complete else None,
                        epoch=ep, offset=position, complete=complete, accuracy=accuracy))
    n = len(manifest['train'])
    for ep in range(epoch, 90):
        net.train()
        for group in opt.param_groups:
            group['lr'] = 0.1 * (0.1 ** (int(ep >= 30) + int(ep >= 60)))
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed_value + ep)).tolist()
        dataset = Images(data_root, manifest['train'], True, seed_value, ep, poison)
        loader = DataLoader(dataset, batch_size=128, sampler=order[offset:],
                            num_workers=workers, pin_memory=str(device).startswith('cuda'))
        completed = offset
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            completed += len(y)
            if STOP or (completed // 128) % 250 == 0:
                checkpoint(ep, completed)
            if STOP:
                raise Paused()
        checkpoint(ep + 1, 0)
        offset = 0
        print(f'{path.name}: epoch {ep + 1}/90 complete', flush=True)
    accuracy = evaluate(net, Images(data_root, manifest['val']), device, workers)
    checkpoint(90, 0, True, accuracy)
    return net.eval(), accuracy


def surrogates(output, manifest, device):
    nets = []
    for i in range(3):
        state = load(output / 'models' / f'surrogate_{i}.pt', device)
        if not state.get('complete') or state['fingerprint'] != manifest['fingerprint']:
            raise RuntimeError('all three clean surrogate models must be complete')
        net = ImageNetResNet18().to(device)
        net.load_state_dict(state['model'])
        nets.append(net.eval())
    return nets


@torch.no_grad()
def pin_targets(manifest, data_root, output, device):
    nets = surrogates(output, manifest, device)
    label = manifest['classes'].index(manifest['target_class'])
    adv = manifest['classes'].index(manifest['poison_class'])
    candidates = [i for i, (_, y) in enumerate(manifest['val']) if y == label]
    random.Random(0).shuffle(candidates)
    chosen = []
    for idx in candidates:
        x = normalize(eval_pixels(pixels(data_root / manifest['val'][idx][0]))).to(device)
        probability = torch.stack([n(x[None]).softmax(1)[0] for n in nets]).mean(0)
        if int(probability.argmax()) == label:
            chosen.append(dict(index=idx, path=manifest['val'][idx][0],
                               surrogate_p_adv=float(probability[adv])))
        if len(chosen) == 8:
            break
        if STOP:
            raise Paused()
    if len(chosen) != 8:
        raise RuntimeError('fewer than eight correctly classified targets in the fixed target class')
    atomic_json(output / 'targets.json', dict(fingerprint=manifest['fingerprint'], targets=chosen))


class CraftView:
    def __init__(self, originals, canonical, device):
        self.originals, self.canonical, self.device = originals, canonical, device

    def __call__(self, proposed, start, end):
        views = []
        for local, index in enumerate(range(start, end)):
            delta = proposed[local] - self.canonical[index]
            clean = self.originals[index].to(self.device)
            views.append(normalize(eval_pixels(native_poison(clean, delta))))
        return torch.stack(views)


def craft_target(method, target, manifest, data_root, directory, nets, device):
    final = directory / f'poison_{target["index"]}.pt'
    if final.exists():
        return load(final)
    idx = target['index']
    adv = manifest['classes'].index(manifest['poison_class'])
    target_class = manifest['classes'].index(manifest['target_class'])
    x_t = normalize(eval_pixels(pixels(data_root / target['path']))).to(device)
    candidates = [i for i, (_, y) in enumerate(manifest['train']) if y == adv]
    count = manifest['num_poisons']
    if not 0 < count <= len(candidates):
        raise RuntimeError('poison count must fit the fixed adversarial-class pool')
    seed_value = 42 * 100003 + idx
    seed(seed_value)
    bases_file = directory / f'bases_{idx}.json'
    if bases_file.exists():
        selected = json.loads(bases_file.read_text())
    elif method == 'random':
        selected = random.Random(seed_value).sample(candidates, count)
        atomic_json(bases_file, selected)
    else:
        images = torch.stack([normalize(eval_pixels(pixels(data_root / manifest['train'][i][0])))
                              for i in candidates]).to(device)
        labels = torch.full((len(candidates),), adv, device=device)
        if method == 'minus-m':
            chosen = fu.select_base_components(nets, images, labels, x_t, adv, count,
                                               device, 'minus-m', batch_size=32,
                                               base_dist='cosine', target_class=target_class)
        else:
            chosen = fu.select_base_ours(nets, images, labels, x_t, adv, count, 1.0,
                                        device, base_dist='cosine', bs=32)
        selected = [candidates[i] for i in chosen.tolist()]
        atomic_json(bases_file, selected)
        del images, labels
    if STOP:
        raise Paused()
    originals = [pixels(data_root / manifest['train'][i][0]) for i in selected]
    base = torch.stack([eval_pixels(x) for x in originals]).to(device)
    progress = directory / f'craft_{idx}.pt'
    resume = load(progress, device) if progress.exists() else None
    last_saved = time.monotonic()
    def checkpoint(state):
        nonlocal last_saved
        if STOP or time.monotonic() - last_saved >= 600:
            save(progress, state)
            last_saved = time.monotonic()
        if STOP:
            raise Paused()
    # Exact micro-batching, signed Adam, 250 steps and 8 restarts, as in the
    # existing GM runner. The custom view applies native-image poisons before
    # evaluation preprocessing; victim training injects those same fields.
    seed(seed_value)
    crafted, objective = fu.craft_gradmatch(
        nets, base, x_t, adv, normalize, 8/255, 0.0039216, 250, 8, device,
        dsa_strategy=fu.DSA_DEFAULT, dsa_param=fu.ParamDiffAug(),
        lowmem=True, chunk=2, poison_transform=CraftView(originals, base, device),
        resume_state=resume, checkpoint_callback=checkpoint)
    delta = (crafted - base).detach().cpu()
    linf = max(float((native_poison(x, d) - x).abs().max())
               for x, d in zip(originals, delta))
    if linf > 8/255 + 1e-6:
        raise RuntimeError(f'native-pixel perturbation exceeds the bound: {linf}')
    poison = dict(indices=selected, delta=delta, objective=objective, native_linf=linf,
                  fingerprint=manifest['fingerprint'], target=idx, method=method)
    save(final, poison)
    progress.unlink(missing_ok=True)
    return poison


def experiment(method, manifest, data_root, output, device, workers):
    target_data = json.loads((output / 'targets.json').read_text())
    if target_data['fingerprint'] != manifest['fingerprint']:
        raise RuntimeError('target set does not match this dataset/protocol')
    targets = target_data['targets']
    directory = output / method
    directory.mkdir(parents=True, exist_ok=True)
    nets = surrogates(output, manifest, device)
    adv = manifest['classes'].index(manifest['poison_class'])
    for target in targets:
        poison = craft_target(method, target, manifest, data_root, directory, nets, device)
        for victim in range(6):
            result_path = directory / f'target_{target["index"]}_victim_{victim}.json'
            if result_path.exists():
                continue
            checkpoint = directory / f'victim_{target["index"]}_{victim}.pt'
            model, accuracy = train_model(checkpoint, manifest, data_root, device, workers,
                                           42 * 100000 + target['index'] * 100 + victim,
                                           poison)
            x = normalize(eval_pixels(pixels(data_root / target['path']))).to(device)
            with torch.no_grad():
                pred = int(model(x[None]).argmax(1))
            atomic_json(result_path, dict(method=method, target_idx=target['index'],
                        victim_id=victim, success=int(pred == adv), prediction=pred,
                        clean_test_acc=accuracy, realized_linf=poison['native_linf'],
                        num_poisons=manifest['num_poisons']))
            del model
            # Final metrics and poisons remain; completed victim weights are
            # unnecessary for this appendix experiment and consume many GB.
            checkpoint.unlink(missing_ok=True)
            torch.cuda.empty_cache()
            if STOP:
                raise Paused()
    rows = [json.loads((directory / f'target_{t["index"]}_victim_{v}.json').read_text())
            for t in targets for v in range(6)]
    with (directory / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_target = [np.mean([r['success'] for r in rows if r['target_idx'] == t['index']]) for t in targets]
    atomic_json(directory / 'summary.json', dict(method=method, num_targets=8,
                num_victims=6, num_trials=48, asr_mean=float(np.mean(per_target)),
                asr_std=float(np.std(per_target)),
                clean_test_acc=float(np.mean([r['clean_test_acc'] for r in rows])),
                protocol=PROTOCOL, fingerprint=manifest['fingerprint']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'surrogate', 'targets', 'experiment'])
    parser.add_argument('--id', default='0')
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    if args.phase == 'prepare':
        prepare(args.data_root, args.output)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('This ImageNet experiment requires a CUDA GPU')
    manifest = json.loads((args.output / 'manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL:
        raise RuntimeError('saved experiment protocol differs from the runner')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda:0'
    if args.phase == 'surrogate':
        if args.id not in ('0', '1', '2'):
            raise ValueError('surrogate ID must be 0, 1, or 2')
        train_model(args.output / 'models' / f'surrogate_{args.id}.pt', manifest,
                    args.data_root, device, args.workers, 1042 + int(args.id))
    elif args.phase == 'targets':
        pin_targets(manifest, args.data_root, args.output, device)
    else:
        if args.id not in METHODS:
            raise ValueError('experiment method must be random, minus-m, or basis')
        experiment(args.id, manifest, args.data_root, args.output, device, args.workers)


if __name__ == '__main__':
    try:
        main()
    except Paused:
        print('Checkpoint saved; requesting another allocation.', flush=True)
        sys.exit(75)
