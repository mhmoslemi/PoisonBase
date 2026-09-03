import json,re
ROOT='/home/mmoslem3/scratch/attack_if'
rows=json.load(open(ROOT+'/.ci_work/rows.json'))
R={r['key']:r for r in rows}
models=['ConvNetBN','ResNet20BN','VGG13BN']
pairs=['dog-bird','frog-airplane']
buds=[0.001,0.002,0.005,0.01,0.02,0.04]
attacks=['fc','gradmatch','sapa']
SKIP={'VGG13BN|sapa|dog-bird|0.04','VGG13BN|sapa|frog-airplane|0.04'}
order=[]
for M in models:
    for P in pairs:
        for B in buds:
            for A in attacks:
                order.append(f"{M}|{A}|{P}|{B}")
def fmt(x):
    s=f"{x:.1f}"
    return "0.0" if s=="-0.0" else s
txt=open(ROOT+'/CI_table.tex').read()
tok=re.compile(r'\\asr\{\}\{\}|\\ci\{\[\]\}')
it=iter(order)
state={'k':None,'phase':0}
filled=0; skipped=[]
def rep(m):
    global filled
    s=m.group(0)
    if s=='\\asr{}{}':
        state['k']=next(it)
        e=R[state['k']]
        if state['k'] in SKIP or e['graft'] is None: return s
        filled+=1
        return "\\asr{%s}{%s}"%(fmt(e['graft'][0]),fmt(e['graft'][1]))
    else:
        e=R[state['k']]
        if state['k'] in SKIP or e['ci'] is None:
            skipped.append(state['k']); return s
        return "\\ci{[%s, %s]}"%(fmt(e['ci'][0]),fmt(e['ci'][1]))
new=tok.sub(rep,txt)
open(ROOT+'/CI_table.tex','w').write(new)
print("filled asr:",filled,"left empty:",sorted(set(skipped)))
