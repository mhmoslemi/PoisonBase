import json,re,math,random
from collections import defaultdict
ROOT='/home/mmoslem3/scratch/attack_if'
H=json.load(open(ROOT+'/.ci_work/harvest.json'))
pat=re.compile(r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_(fc|gradmatch|sapa)_(random|ours)_(dog-bird|frog-airplane)_b([0-9.]+)_eps8_seed42(.*)$')
cand=defaultdict(list)
for name,tg in H.items():
    m=pat.match(name)
    if not m: continue
    model,atk,base,pair,bud,rest=m.groups()
    toks=set(t for t in rest.split('_') if t)-{'ce5','worst0.05','lam1','cosine'}
    tgt=[t for t in toks if t.startswith('tgt')]
    toks={t for t in toks if not t.startswith('tgt')}
    if base=='random':
        if toks: continue
        arm='RAND'
    elif toks=={'jacw1'}: arm='GRAFT+'
    elif not toks: arm='GRAFT'
    else: continue
    d={int(t):{int(k):v for k,v in vv.items()} for t,vv in tg.items()}
    cand[(model,atk,pair,float(bud),arm)].append((name,tgt[0] if tgt else '', d))

def pick(key):
    c=cand.get(key)
    if not c: return None
    return max(c,key=lambda x:(len(x[2]), sum(len(v) for v in x[2].values())))

def rate(d):
    return {t:sum(v.values())/len(v) for t,v in d.items() if v}

def mean_se(r):
    vals=list(r.values()); n=len(vals)
    if n==0: return None,None,0
    m=sum(vals)/n
    se=(math.sqrt(sum((x-m)**2 for x in vals)/(n-1))/math.sqrt(n)) if n>1 else 0.0
    return m*100, se*100, n

out={}
for M in ['ConvNetBN','ResNet20BN','VGG13BN']:
  for P in ['dog-bird','frog-airplane']:
    for A in ['fc','gradmatch','sapa']:
      for B in [0.001,0.002,0.005,0.01,0.02,0.04]:
        row={}
        for arm in ['RAND','GRAFT','GRAFT+']:
            p=pick((M,A,P,B,arm))
            row[arm]=(p[0],rate(p[2])) if p else None
        out[f"{M}|{A}|{P}|{B}"]=row
        r=row['RAND']; g=row['GRAFT+']
        rm=mean_se(r[1]) if r else (None,None,0)
        gm=mean_se(g[1]) if g else (None,None,0)
        f=lambda x: f"{x:.1f}" if x is not None else " -- "
        print(f"{M:11s} {P:14s} {A:9s} b{B:<6} RAND={f(rm[0])}({f(rm[1])})n{rm[2]:2d}  GRAFTJ={f(gm[0])}({f(gm[1])})n{gm[2]:2d}")
json.dump({k:{a:(v[0],{str(t):x for t,x in v[1].items()}) if v else None for a,v in row.items()} for k,row in out.items()},
          open(ROOT+'/.ci_work/picked.json','w'))
