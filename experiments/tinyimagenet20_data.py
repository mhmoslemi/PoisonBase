#!/usr/bin/env python3
"""Validate the existing TinyImageNet tensor archive without expanding it."""
import argparse
from pathlib import Path
import random
import sys

import torch


def inspect_archive(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'tinyimagenet.pt'
    if not path.is_file():
        raise RuntimeError(f'TinyImageNet archive not found: {path}')
    blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    classes = blob['classes']
    if len(classes) != 200 or len(set(classes)) != 200:
        raise RuntimeError('Expected the 200-class TinyImageNet archive')
    selected = sorted(random.Random(0).sample(sorted(classes), 20))
    for split, minimum in [('train', 200), ('val', 8)]:
        images, labels = blob[f'images_{split}'], blob[f'labels_{split}']
        if images.dtype != torch.uint8 or tuple(images.shape[1:]) != (3, 64, 64):
            raise RuntimeError(f'Expected uint8 NCHW 64x64 {split} images')
        if labels.ndim != 1 or len(labels) != len(images):
            raise RuntimeError(f'Invalid {split} labels')
        for name in selected:
            if int((labels == classes.index(name)).sum()) < minimum:
                raise RuntimeError(f'Not enough {split} samples for {name}')
    return path.resolve(), blob, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        path, _, _ = inspect_archive(args.archive)
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f'TinyImageNet data check failed: {exc}\nNo jobs were submitted.', file=sys.stderr)
        return 1
    print('TinyImageNet verified: will use 20 classes x 200 training images at 64x64.', file=sys.stderr)
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
