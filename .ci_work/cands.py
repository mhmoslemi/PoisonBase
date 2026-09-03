import json,re,math
from collections import defaultdict
ROOT='/home/mmoslem3/scratch/attack_if'
H=json.load(open(ROOT+'/.ci_work/harvest.json'))
pat=re.compile(r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_(fc|gradmatch|sapa)_(random|ours)_(dog-bird|frog-airplane)_b([0-9.]+)_eps8_seed42(.*)$')
import sys
want=sys.argv[1:]
for name in sorted(H):
    m=pat.match(name)
    if not m: continue
    model,atk,base,pair,bud,rest=m.groups()
    if base!='random': continue
    if want and not all(w in name for w in want): continue
    d=H[name]
    rates=[sum(v.values())/len(v) for v in d.values() if v]
    mm=sum(rates)/len(rates)*100 if rates else 0
    se=(math.sqrt(sum((x-mm/100)**2 for x in rates)/(len(rates)-1))/math.sqrt(len(rates))*100) if len(rates)>1 else 0
    print(f"{name}  n={len(d)} ASR={mm:.1f} SE={se:.1f} tg={sorted(int(t) for t in d)[:12]}")
