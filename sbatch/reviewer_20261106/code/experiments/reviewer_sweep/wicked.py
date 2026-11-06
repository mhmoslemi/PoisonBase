"""Wicked Oddities selectors, adapted to fixed clean-label GM poison crafting."""
import copy
import io
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import to_tensor, resize

from defenses import save_torch
from protocol import atomic_json
from wo_resnet import ResNet18

MEAN = (.4914, .4822, .4465)
STD = (.247, .243, .261)


def knn_outlier(features, k=50):
    features = F.normalize(features, dim=1)
    values = []
    for start in range(0, len(features), 256):
        sim = features[start:start+256] @ features.T
        sim[torch.arange(len(sim), device=sim.device),
            torch.arange(start, start+len(sim), device=sim.device)] = -torch.inf
        values.append(1-sim.topk(k, dim=1).values.mean(1))
    return torch.cat(values)


def pretrained_scores(ctx, output):
    before = time.monotonic()
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(ctx['device']).eval()
    net.fc = torch.nn.Identity()
    ids = (ctx['train_labs'] == 5).nonzero().flatten()
    mean = torch.tensor((.485, .456, .406), device=ctx['device'])[None, :, None, None]
    std = torch.tensor((.229, .224, .225), device=ctx['device'])[None, :, None, None]
    features = []
    with torch.no_grad():
        for idx in ids.split(32):
            x = ctx['denorm'](ctx['train_imgs'][idx]).clamp(0, 1)
            x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False, antialias=True)
            features.append(net((x-mean)/std))
        score = knn_outlier(torch.cat(features), 50)
    save_torch(output, dict(indices=ids.cpu(), scores=score.cpu(),
               representation='supervised ImageNet ResNet50 IMAGENET1K_V1', k=50,
               seconds=time.monotonic()-before))


def tiny_tensors(archive, device, train):
    """Decode the author's first 100 sorted classes once, directly from the archive."""
    with zipfile.ZipFile(archive) as z:
        classes = sorted(z.read('tiny-imagenet-200/wnids.txt').decode().split())[:100]
        mapping = {name:i for i,name in enumerate(classes)}
        if train:
            members = [(n, mapping[n.split('/')[2]]) for n in z.namelist()
                       if n.startswith('tiny-imagenet-200/train/') and n.endswith('.JPEG')
                       and n.split('/')[2] in mapping]
        else:
            members = [('tiny-imagenet-200/val/images/'+parts[0], mapping[parts[1]])
                       for line in z.read('tiny-imagenet-200/val/val_annotations.txt').decode().splitlines()
                       if (parts := line.split())[1] in mapping]
        images = torch.empty((len(members), 3, 32, 32), device=device)
        labels = torch.tensor([label for _, label in members], device=device)
        for start in range(0, len(members), 256):
            batch = []
            for name, _ in members[start:start+256]:
                with Image.open(io.BytesIO(z.read(name))) as im:
                    batch.append(to_tensor(resize(im.convert('RGB'), [32,32])))
            images[start:start+len(batch)] = torch.stack(batch).to(device)
    return images, labels


def ood_augment(x):
    b, _, h, w = x.shape
    padded = F.pad(x, (5,5,5,5)).permute(0,2,3,1)
    dy = torch.randint(11,(b,1,1),device=x.device)
    dx = torch.randint(11,(b,1,1),device=x.device)
    x = padded[torch.arange(b,device=x.device)[:,None,None],
               dy+torch.arange(h,device=x.device)[None,:,None],
               dx+torch.arange(w,device=x.device)[None,None,:]].permute(0,3,1,2)
    angle = (torch.rand(b,device=x.device)*20-10)*np.pi/180
    theta = torch.zeros(b,2,3,device=x.device)
    theta[:,0,0] = theta[:,1,1] = angle.cos()
    theta[:,0,1] = -angle.sin(); theta[:,1,0] = angle.sin()
    x = F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),
                      mode='nearest',align_corners=False)
    return torch.where(torch.rand(b,1,1,1,device=x.device)<.5,x.flip(-1),x)


def ood_scores(ctx, archive, validation_ids, directory):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    output = directory/'wo_ood.pt'
    if output.exists():
        return
    before = time.monotonic()
    device = ctx['device']
    ids = (ctx['train_labs']==5).nonzero().flatten()
    tx,ty = tiny_tensors(archive,device,True)
    x = torch.cat([tx,ctx['denorm'](ctx['train_imgs'][ids]).clamp(0,1)])
    y = torch.cat([ty,torch.full_like(ids,100)])
    del tx,ty
    vx,vy = tiny_tensors(archive,device,False)
    vi = validation_ids[ctx['test_labs'][validation_ids] == 5]
    vx = torch.cat([vx,ctx['denorm'](ctx['test_imgs'][vi]).clamp(0,1)])
    vy = torch.cat([vy,torch.full_like(vi,100)])
    mean = torch.tensor(MEAN,device=device)[None,:,None,None]
    std = torch.tensor(STD,device=device)[None,:,None,None]
    net = ResNet18(num_classes=101,input_size=32).to(device)
    opt = torch.optim.SGD(net.parameters(),lr=.01,momentum=.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
    start,best,best_state = 0,-1.,None
    checkpoint = directory/'ood_checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location=device,weights_only=False)
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt'])
        scheduler.load_state_dict(state['scheduler'])
        start,best,best_state = state['epoch'],state['best'],state['best_state']
        from defenses import restore_rng
        restore_rng(state['rng'])
    for ep in range(start,100):
        net.train()
        for idx in torch.randperm(len(y),device=device).split(128):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net((ood_augment(x[idx])-mean)/std),y[idx]).backward()
            opt.step()
        scheduler.step(); net.eval()
        with torch.no_grad():
            correct = sum(int((net((vx[i:i+128]-mean)/std).argmax(1)==vy[i:i+128]).sum())
                          for i in range(0,len(vy),128))
        acc = correct/len(vy)
        if acc > best:
            best,best_state = acc,copy.deepcopy(net.state_dict())
        if (ep+1)%5==0 or ep==99:
            from defenses import rng_state
            save_torch(checkpoint,dict(net=net.state_dict(),opt=opt.state_dict(),
                       scheduler=scheduler.state_dict(),epoch=ep+1,best=best,
                       best_state=best_state,rng=rng_state()))
            print('WO OOD epoch',ep+1,'validation accuracy',acc,flush=True)
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        scores = torch.cat([F.cross_entropy(net((ctx['denorm'](ctx['train_imgs'][b]).clamp(0,1)-mean)/std),
                                  torch.full_like(b,100),reduction='none') for b in ids.split(128)])
    save_torch(output,dict(indices=ids.cpu(),scores=scores.cpu(),epochs=100,
               ood_classes=100,validation_accuracy=best,seconds=time.monotonic()-before))
    atomic_json(directory/'ood_protocol.json',dict(model='authors myresnet18',
                train='all clean dog train images + first 100 sorted TinyImageNet classes',
                validation='TinyImageNet validation + reserved 100 CIFAR10 dog validation images',
                optimizer='SGD lr=.01 momentum=.9 wd=.0005 cosine100',
                augmentation='crop padding5, rotation10, horizontal flip; GPU tensor adapter'))
