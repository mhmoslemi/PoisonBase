import json,re,math
from collections import defaultdict
ROOT='/home/mmoslem3/scratch/attack_if'
def load(p):
    raw=json.load(open(p))
    return {k:{int(t):{int(a):b for a,b in v.items()} for t,v in d.items()} for k,d in raw.items()}
C=load(ROOT+'/.ci_work/csvd.json'); L=load(ROOT+'/.ci_work/logd.json')
F=json.load(open(ROOT+'/.ci_work/full_tex.json'))
names=set(C)|set(L)
def merge(a,b):
    out=defaultdict(dict)
    for t,v in a.items(): out[t].update(v)
    for t,v in b.items(): out[t].update(v)
    return dict(out)
variants={n:{'csv':C.get(n,{}),'log':L.get(n,{}),
             'csv_first':merge(L.get(n,{}),C.get(n,{})),
             'log_first':merge(C.get(n,{}),L.get(n,{}))} for n in names}
pat=re.compile(r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_(fc|gradmatch|sapa)_(random|ours)_(dog-bird|frog-airplane)_b([0-9.]+)_eps8_seed42(.*)$')
def ms(d):
    r=[sum(v.values())/len(v) for v in d.values() if v]
    n=len(r)
    if n==0: return None
    m=sum(r)/n
    se=(math.sqrt(sum((x-m)**2 for x in r)/(n-1))/math.sqrt(n)) if n>1 else 0.0
    return m*100,se*100,n
cell=defaultdict(list)
for n in names:
    m=pat.match(n)
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
    cell[(model,atk,pair,float(bud),arm)].append((n,tgt[0] if tgt else ''))
sel={}; miss=[]
for M in ['ConvNetBN','ResNet20BN','VGG13BN']:
  for A in ['fc','gradmatch','sapa']:
    for P in ['dog-bird','frog-airplane']:
      for B in [0.001,0.002,0.005,0.01,0.02,0.04]:
        k=f"{M}|{A}|{P}|{B}"
        tv=F.get(k+"|greedy_J")
        best=None
        for n,tgt in cell.get((M,A,P,B,'GRAFT+'),[]):
            for vk,d in variants[n].items():
                s=ms(d)
                if not s: continue
                if tv is not None and abs(round(s[0],1)-tv)<0.05:
                    sc=(s[2], vk=='log_first', vk=='csv_first')
                    if best is None or sc>best[0]: best=(sc,n,vk,tgt,s)
        if best: sel[k]=(best[1],best[2],best[3])
        else: miss.append((k,tv,[(n,tgt,{vk:(round(ms(d)[0],1),ms(d)[2]) for vk,d in variants[n].items() if ms(d)}) for n,tgt in cell.get((M,A,P,B,'GRAFT+'),[])]))
json.dump(sel,open(ROOT+'/.ci_work/graft_sel.json','w'))
print("matched",len(sel),"/108")
for k,tv,c in miss:
    print("MISS",k,"full.tex greedy_J =",tv)
    for n,tgt,d in c: print("     ",n,d)
