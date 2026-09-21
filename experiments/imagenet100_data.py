#!/usr/bin/env python3
"""Read-only ImageNet layout validation, also usable before sbatch submission."""
import argparse
from pathlib import Path
import random
import sys

IMAGE_SUFFIXES = {'.jpeg', '.jpg', '.png'}


def locate_data(base):
    base = Path(base)
    for path in (base / 'imagenet', base / 'imagenet1k', base / 'ILSVRC2012', base):
        if (path / 'train').is_dir() and (path / 'val').is_dir():
            return path.resolve()
    raise RuntimeError(
        f'ImageNet-1K is not installed at {base}. Expected train/<WNID> and '
        'val/<WNID> folders. Set IMAGENET_ROOT to their parent directory. '
        'CIFAR and TinyImageNet files cannot be used for this experiment.')


def inspect_layout(base):
    root = locate_data(base)
    available = sorted(p.name for p in (root / 'train').iterdir()
                       if p.is_dir() and p.name.startswith('n'))
    if len(available) != 1000:
        raise RuntimeError(f'Expected 1000 ImageNet-1K training classes at {root}, found {len(available)}')
    classes = sorted(random.Random(0).sample(available, 100))
    for split in ('train', 'val'):
        for name in classes:
            directory = root / split / name
            if not directory.is_dir() or not any(
                    p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                    for p in directory.rglob('*')):
                raise RuntimeError(f'Missing or empty ImageNet class directory: {directory}')
    return root, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args()
    try:
        root, classes = inspect_layout(args.data_root)
    except (OSError, RuntimeError) as exc:
        print(f'ImageNet data check failed: {exc}\nNo jobs were submitted.', file=sys.stderr)
        return 1
    print(f'ImageNet data verified: 100 seeded classes have train/val files at {root}', file=sys.stderr)
    print(root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
