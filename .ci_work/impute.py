import json,math,random
from collections import defaultdict
ROOT='/home/mmoslem3/scratch/attack_if'
def load(p):
    raw=json.load(open(p))
    return {k:{int(t):{int(a):b for a,b in v.items()} for t,v in d.items()} for k,d in raw.items()}
C=load(ROOT+'/.ci_work/csvd.json'); L=load(ROOT+'/.ci_work/logd.json')
def merge(a,b):
    o=defaultdict(dict)
    for t,v in a.items(): o[t].update(v)
    for t,v in b.items(): o[t].update(v)
    return dict(o)
def variant(n,vk):
    c=C.get(n,{}); l=L.get(n,{})
    return {'csv':c,'log':l,'csv_first':merge(l,c),'log_first':merge(c,l)}[vk]
RS=json.load(open(ROOT+'/.ci_work/rand_sel.json'))
for key in ['VGG13BN|sapa|dog-bird|0.04','VGG13BN|sapa|frog-airplane|0.04']:
    n,vk,tgt=RS[key]
    d=variant(n,vk)
    rr={t:sum(v.values())/len(v) for t,v in d.items() if v}
    ts=sorted(rr)
    print(key, n, 'ntgt',len(ts))
    print('  rand per-target:',[round(rr[t],3) for t in ts])
    dl=[1.0-rr[t] for t in ts]
    rng=random.Random(20260901); N=len(dl); reps=[]
    for _ in range(20000):
        reps.append(sum(dl[rng.randrange(N)] for _ in range(N))/N*100)
    reps.sort()
    lo=reps[int(0.025*len(reps))]; hi=reps[int(0.975*len(reps))-1]
    print(f"  Delta={sum(dl)/N*100:.1f}  CI=[{lo:.1f}, {hi:.1f}]")
