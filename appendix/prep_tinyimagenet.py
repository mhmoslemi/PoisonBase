"""Build the tinyimagenet.pt that utils.get_dataset expects.

utils.get_dataset('TinyImageNet', path) loads <path>/tinyimagenet.pt and reads
'classes', 'images_train', 'labels_train', 'images_val', 'labels_val'. Nothing in
the repository ever created that file, so the cross-dataset experiment cannot run
until this is done once.

    # if the node has no internet, fetch the zip on the login node first:
    #   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    python appendix/prep_tinyimagenet.py --src tiny-imagenet-200.zip
    python appendix/prep_tinyimagenet.py --src /path/to/tiny-imagenet-200   # extracted
    python appendix/prep_tinyimagenet.py                                    # download here

Images are stored as uint8 (N, 3, 64, 64); get_dataset does the /255 and the
per-channel normalisation itself. That keeps the file ~1.2 GB instead of ~5 GB.

Class names are the WordNet ids, which is what --class_pair takes, e.g.
    --class_pair n01443537-n01629819   (adversarial-target)
"""
import argparse, io, os, sys, zipfile

import numpy as np
import torch
from PIL import Image

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def _img(fp):
    im = Image.open(fp).convert('RGB')
    if im.size != (64, 64):
        im = im.resize((64, 64), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1)


def from_zip(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    wnids = sorted({n.split('/')[2] for n in names if '/train/' in n and n.endswith('.JPEG')})
    idx = {w: i for i, w in enumerate(wnids)}
    tr_x, tr_y = [], []
    for n in names:
        if '/train/' in n and n.endswith('.JPEG'):
            tr_x.append(_img(io.BytesIO(z.read(n))))
            tr_y.append(idx[n.split('/')[2]])
    ann = [l.split('\t') for l in z.read(
        [n for n in names if n.endswith('val_annotations.txt')][0]).decode().splitlines() if l]
    lab = {a[0]: idx[a[1]] for a in ann}
    va_x, va_y = [], []
    for n in names:
        if '/val/images/' in n and n.endswith('.JPEG'):
            va_x.append(_img(io.BytesIO(z.read(n))))
            va_y.append(lab[os.path.basename(n)])
    return wnids, tr_x, tr_y, va_x, va_y


def from_dir(root):
    wnids = sorted(d for d in os.listdir(os.path.join(root, 'train')))
    idx = {w: i for i, w in enumerate(wnids)}
    tr_x, tr_y = [], []
    for w in wnids:
        d = os.path.join(root, 'train', w, 'images')
        for f in sorted(os.listdir(d)):
            tr_x.append(_img(os.path.join(d, f)))
            tr_y.append(idx[w])
    lab = {}
    with open(os.path.join(root, 'val', 'val_annotations.txt')) as f:
        for l in f:
            a = l.split('\t')
            lab[a[0]] = idx[a[1]]
    va_x, va_y = [], []
    d = os.path.join(root, 'val', 'images')
    for f in sorted(os.listdir(d)):
        va_x.append(_img(os.path.join(d, f)))
        va_y.append(lab[f])
    return wnids, tr_x, tr_y, va_x, va_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=None, help='tiny-imagenet-200.zip or the extracted dir')
    p.add_argument('--out', default='/home/mmoslem3/scratch/data/tinyimagenet.pt')
    a = p.parse_args()

    src = a.src
    if src is None:
        src = os.path.join(os.path.dirname(a.out), 'tiny-imagenet-200.zip')
        if not os.path.exists(src):
            print('downloading %s -> %s' % (URL, src))
            import urllib.request
            try:
                urllib.request.urlretrieve(URL, src)
            except Exception as e:
                sys.exit('download failed (%s).\nCompute nodes usually have no internet: '
                         'fetch it on the login node with\n  wget %s\nthen rerun with '
                         '--src <the zip>' % (e, URL))

    print('reading %s ...' % src)
    if os.path.isdir(src):
        wnids, tr_x, tr_y, va_x, va_y = from_dir(src)
    else:
        wnids, tr_x, tr_y, va_x, va_y = from_zip(src)

    blob = {'classes': wnids,
            'images_train': torch.stack(tr_x),
            'labels_train': torch.tensor(tr_y, dtype=torch.long),
            'images_val': torch.stack(va_x),
            'labels_val': torch.tensor(va_y, dtype=torch.long)}
    assert blob['images_train'].shape[1:] == (3, 64, 64), blob['images_train'].shape
    print('  train %s  val %s  classes %d'
          % (tuple(blob['images_train'].shape), tuple(blob['images_val'].shape), len(wnids)))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(blob, a.out)
    print('wrote %s (%.1f GB)' % (a.out, os.path.getsize(a.out) / 2**30))
    print('first classes: %s' % ', '.join(wnids[:6]))


if __name__ == "__main__":
    main()
