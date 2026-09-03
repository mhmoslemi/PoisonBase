import json,re,os
ROOT='/home/mmoslem3/scratch/attack_if'
H=json.load(open(ROOT+'/.ci_work/harvest.json'))
pat=re.compile(r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_(fc|gradmatch|sapa)_(random|ours)_(dog-bird|frog-airplane)_b([0-9.]+)_eps8_seed42(.*)$')
recs={}
for name,tg in H.items():
    m=pat.match(name)
    if not m: continue
    model,atk,base,pair,bud,rest=m.groups()
    toks=[t for t in rest.split('_') if t]
    recs[name]=dict(model=model,atk=atk,base=base,pair=pair,bud=float(bud),toks=toks,
                    ntgt=len(tg),nvic=min((len(v) for v in tg.values()),default=0))
json.dump(recs,open(ROOT+'/.ci_work/recs.json','w'))
# canonical token sets
def arm(r):
    t=set(r['toks'])
    t.discard('ce5'); t.discard('worst0.05'); t.discard('lam1'); t.discard('cosine')
    t={x for x in t if not x.startswith('tgt')}
    if r['base']=='random':
        return 'RAND' if not t else None
    if t=={'jacw1'}: return 'GRAFT+'
    if not t: return 'GRAFT'
    return None
from collections import defaultdict
tbl=defaultdict(dict)
for n,r in recs.items():
    a=arm(r)
    if not a: continue
    key=(r['model'],r['atk'],r['pair'],r['bud'])
    tbl[key].setdefault(a,[]).append((n,r['ntgt'],r['nvic']))
for M in ['ConvNetBN','ResNet20BN','VGG13BN']:
  for P in ['dog-bird','frog-airplane']:
    for A in ['fc','gradmatch','sapa']:
      for B in [0.001,0.002,0.005,0.01,0.02,0.04]:
        v=tbl.get((M,A,P,B),{})
        s=[]
        for a in ['RAND','GRAFT','GRAFT+']:
            e=v.get(a)
            s.append(f"{a}:" + (",".join(f"{nt}x{nv}" for _,nt,nv in e) if e else "-"))
        print(f"{M:11s} {P:14s} {A:9s} b{B:<6}  " + "  ".join(s))
