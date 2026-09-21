#!/usr/bin/env python3
"""Small, explicitly labeled TinyImageNet appendix comparison (not ImageNet-100)."""
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import imagenet100_gm as run
from experiments.tinyimagenet20_data import inspect_archive

PROTOCOL = dict(run.PROTOCOL, dataset='TinyImageNet-20 (200 train images/class)',
                classes=20, train_per_class=200, subset_seed=0,
                model='resnet18_small_stem_weights_none', small_stem=True,
                epochs=30, decay=[15, 25], train_crop=64, eval_resize=64, eval_crop=64,
                train_augmentation='random_crop_64_padding_4_horizontal_flip',
                poison_geometry='native_64_v1', craft_chunk=8)


def prepare(archive, output):
    archive, blob, classes = inspect_archive(archive)
    output.mkdir(parents=True, exist_ok=True)
    data_root = (output / 'data').resolve()
    poison_class, target_class = random.Random(0).sample(classes, 2)
    manifest = dict(protocol=PROTOCOL, classes=classes, data_root=str(data_root),
                    source_archive=str(archive), poison_class=poison_class,
                    target_class=target_class, source_indices={})
    image_hash = hashlib.sha256()
    for split in ('train', 'val'):
        records, source_indices = [], []
        labels = blob[f'labels_{split}']
        for label, name in enumerate(classes):
            original_label = blob['classes'].index(name)
            indices = (labels == original_label).nonzero().flatten().tolist()
            # Independent per-class seed, fixed before model training/results.
            if split == 'train':
                indices = sorted(random.Random(original_label).sample(indices, 200))
            for index in indices:
                relative = f'{split}/{name}/{index}.png'
                array = blob[f'images_{split}'][index].permute(1, 2, 0).numpy()
                image_hash.update(array.tobytes())
                records.append([relative, label])
                source_indices.append(index)
        manifest[split] = records
        manifest['source_indices'][split] = source_indices
    manifest['pixels_sha256'] = image_hash.hexdigest()
    manifest['num_poisons'] = run.fu.rho_to_m(PROTOCOL['budget'], len(manifest['train']))
    identity = {k: v for k, v in manifest.items() if k not in ('data_root', 'source_archive')}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != manifest['fingerprint']:
        raise RuntimeError('Existing TinyImageNet output has a different protocol or data; use a new output directory')
    for split in ('train', 'val'):
        for (relative, _), index in zip(manifest[split], manifest['source_indices'][split]):
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            array = blob[f'images_{split}'][index].permute(1, 2, 0).numpy()
            # Lossless uint8 export; never allocate the full dataset in float.
            temp = path.with_suffix('.tmp')
            Image.fromarray(array).save(temp, format='PNG', compress_level=1)
            temp.replace(path)
    path = output / 'manifest.json'
    run.atomic_json(path, manifest)
    (output / 'classes_seed0.txt').write_text('\n'.join(classes) + '\n')
    print(f'Prepared TinyImageNet-20: {len(manifest["train"])} train / '
          f'{len(manifest["val"])} validation images, {manifest["num_poisons"]} poisons; '
          f'pair {poison_class} -> {target_class}', flush=True)


if __name__ == '__main__':
    try:
        run.main(protocol=PROTOCOL, prepare_fn=prepare)
    except run.Paused:
        print('Checkpoint saved; requesting another allocation.', flush=True)
        sys.exit(75)
