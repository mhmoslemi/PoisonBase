import json,re,math,random
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
GS=json.load(open(ROOT+'/.ci_work/graft_sel.json'))
TEX=json.load(open(ROOT+'/.ci_work/tex_rand.json'))
# fallback selections for unmatched cells: best (most targets, log_first)
pat=re.compile(r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_(fc|gradmatch|sapa)_(random|ours)_(dog-bird|frog-airplane)_b([0-9.]+)_eps8_seed42(.*)$')
cell=defaultdict(list)
for n in set(C)|set(L):
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
def ms(d):
    r=[sum(v.values())/len(v) for v in d.values() if v]
    n=len(r)
    if n==0: return None
    m=sum(r)/n
    se=(math.sqrt(sum((x-m)**2 for x in r)/(n-1))/math.sqrt(n)) if n>1 else 0.0
    return m*100,se*100,n
def rates(d): return {t:sum(v.values())/len(v) for t,v in d.items() if v}

rows=[]
for M in ['ConvNetBN','ResNet20BN','VGG13BN']:
  for P in ['dog-bird','frog-airplane']:
    for A in ['fc','gradmatch','sapa']:
      for B in [0.001,0.002,0.005,0.01,0.02,0.04]:
        k=f"{M}|{A}|{P}|{B}"
        g=GS.get(k); r=RS.get(k)
        note=[]
        gd=variant(g[0],g[1]) if g else None
        gtgt=g[2] if g else None
        if gd is None:
            cands=cell.get((M,A,P,B,'GRAFT+'),[])
            best=None
            for n,tg in cands:
                d=variant(n,'log_first'); s=ms(d)
                if s and (best is None or s[2]>best[0]): best=(s[2],n,d,tg)
            if best: gd=best[2]; gtgt=best[3]; note.append('graft+ unmatched-to-full.tex')
        if r: rd=variant(r[0],r[1]); rtgt=r[2]
        else:
            rd=None; rtgt=None
            cands=[c for c in cell.get((M,A,P,B,'RAND'),[]) if (gtgt is None or c[1]==gtgt)] or cell.get((M,A,P,B,'RAND'),[])
            best=None
            for n,tg in cands:
                d=variant(n,'log_first'); s=ms(d)
                if s and (best is None or s[2]>best[0]): best=(s[2],n,d,tg)
            if best: rd=best[2]; rtgt=best[3]; note.append('rand reconstruction != printed')
        gs=ms(gd) if gd else None
        rs=ms(rd) if rd else None
        ci=None; npair=0; delta=None
        if gd and rd:
            gr=rates(gd); rr=rates(rd)
            common=sorted(set(gr)&set(rr))
            npair=len(common)
            if npair>=2:
                dl=[gr[t]-rr[t] for t in common]
                rng=random.Random(20260901); n=len(dl); reps=[]
                for _ in range(20000):
                    reps.append(sum(dl[rng.randrange(n)] for _ in range(n))/n*100)
                reps.sort()
                ci=(reps[int(0.025*len(reps))],reps[int(0.975*len(reps))-1])
                delta=sum(dl)/n*100
        rows.append(dict(key=k,M=M,P=P,A=A,B=B,tex_rand=TEX[k],
                         rand=rs,graft=gs,ci=ci,delta=delta,npair=npair,
                         rtgt=rtgt,gtgt=gtgt,note=note))
json.dump(rows,open(ROOT+'/.ci_work/rows.json','w'))
for e in rows:
    f=lambda x:f"{x:6.1f}" if x is not None else "   -- "
    ci=f"[{e['ci'][0]:.1f}, {e['ci'][1]:.1f}]" if e['ci'] else "--"
    g=e['graft'] or (None,None,0)
    print(f"{e['M']:11s} {e['P']:14s} {e['A']:9s} b{e['B']:<6} texR={e['tex_rand'][0]:>5s} "
          f"G+={f(g[0])}({f(g[1])})n{g[2]:2d} tgtR={str(e['rtgt']):6s} tgtG={str(e['gtgt']):6s} pair={e['npair']:2d} "
          f"D={f(e['delta'])} CI={ci} {';'.join(e['note'])}")
