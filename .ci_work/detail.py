import json,re
ROOT='/home/mmoslem3/scratch/attack_if'
H=json.load(open(ROOT+'/.ci_work/harvest.json'))
pat=re.compile(r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_(fc|gradmatch|sapa)_(random|ours)_(dog-bird|frog-airplane)_b([0-9.]+)_eps8_seed42(.*)$')
for name in sorted(H):
    m=pat.match(name)
    if not m: continue
    model,atk,base,pair,bud,rest=m.groups()
    toks=[t for t in rest.split('_') if t]
    t=set(toks)-{'ce5','worst0.05','lam1','cosine'}
    tg=[x for x in t if x.startswith('tgt')]
    t={x for x in t if not x.startswith('tgt')}
    if base=='random':
        if t: continue
        arm='RAND'
    elif t=={'jacw1'}: arm='GRAFT+'
    elif not t: arm='GRAFT'
    else: continue
    d=H[name]
    obs=sum(len(v) for v in d.values())
    vc=sorted({len(v) for v in d.values()})
    print(f"{model:11s} {pair:14s} {atk:9s} b{bud:<6s} {arm:7s} {tg[0] if tg else 'tgt?':7s} ntgt={len(d):2d} obs={obs:3d} vic={vc}")
