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

# canonical tgt per (model,atk,pair): most common tgt among GRAFT+ runs
canon={}
for (M,A,P,B,arm),lst in cand.items():
    if arm!='GRAFT+': continue
    c=canon.setdefault((M,A,P),defaultdict(int))
    for _,t,d in lst: c[t]+=len(d)
canon={k:max(v,key=v.get) for k,v in canon.items()}

def get(M,A,P,B,arm,tgt):
    lst=[x for x in cand.get((M,A,P,B,arm),[]) if x[1]==tgt]
    if not lst: return None
    return max(lst,key=lambda x:(len(x[2]),sum(len(v) for v in x[2].values())))

def rates(d): return {t:sum(v.values())/len(v) for t,v in d.items() if v}
def ms(r):
    v=list(r.values()); n=len(v)
    if n==0: return None,None,0
    m=sum(v)/n
    se=(math.sqrt(sum((x-m)**2 for x in v)/(n-1))/math.sqrt(n)) if n>1 else 0.0
    return m*100,se*100,n

BUD=[0.001,0.002,0.005,0.01,0.02,0.04]
res={}
for M in ['ConvNetBN','ResNet20BN','VGG13BN']:
  for P in ['dog-bird','frog-airplane']:
    for A in ['fc','gradmatch','sapa']:
      tgt=canon.get((M,A,P))
      for B in BUD:
        r=get(M,A,P,B,'RAND',tgt); g=get(M,A,P,B,'GRAFT+',tgt)
        e={'tgt':tgt,'rand':None,'graft':None,'ci':None,'npair':0,'nrand':0,'ngraft':0}
        if r:
            rr=rates(r[2])
            # restrict rand to canonical 10 if it accidentally merged extra targets
            e['rand']=ms(rr); e['rand_name']=r[0]; e['rand_rates']=rr
        if g:
            gr=rates(g[2]); e['graft']=ms(gr); e['graft_name']=g[0]; e['graft_rates']=gr
        if r and g:
            common=sorted(set(e['rand_rates'])&set(e['graft_rates']))
            e['npair']=len(common)
            if common:
                d=[e['graft_rates'][t]-e['rand_rates'][t] for t in common]
                rng=random.Random(20260901)
                n=len(d); reps=[]
                for _ in range(10000):
                    s=[d[rng.randrange(n)] for _ in range(n)]
                    reps.append(sum(s)/n*100)
                reps.sort()
                lo=reps[int(0.025*len(reps))]; hi=reps[int(0.975*len(reps))-1]
                e['ci']=(lo,hi); e['delta']=sum(d)/n*100
        res[f"{M}|{A}|{P}|{B}"]=e
json.dump(res,open(ROOT+'/.ci_work/final.json','w'))
for M in ['ConvNetBN','ResNet20BN','VGG13BN']:
  for P in ['dog-bird','frog-airplane']:
    for A in ['fc','gradmatch','sapa']:
      for B in BUD:
        e=res[f"{M}|{A}|{P}|{B}"]
        r=e['rand'] or (None,None,0); g=e['graft'] or (None,None,0)
        f=lambda x:f"{x:5.1f}" if x is not None else "  -- "
        ci=f"[{e['ci'][0]:.1f}, {e['ci'][1]:.1f}]" if e['ci'] else "--"
        print(f"{M:11s} {P:14s} {A:9s} b{B:<6} {e['tgt']:6s} R={f(r[0])}({f(r[1])})n{r[2]:2d} G+={f(g[0])}({f(g[1])})n{g[2]:2d} pair={e['npair']:2d} CI={ci}")
