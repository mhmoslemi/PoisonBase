import re,json
ROOT='/home/mmoslem3/scratch/attack_if'
txt=open(ROOT+'/full.tex').read()
body=txt.split('\\midrule',1)[1]
blocks=[b for b in body.split('\\midrule')]
models=['ConvNetBN','ResNet20BN','VGG13BN']
attacks=['fc','gradmatch','sapa']
pairs=['dog-bird','frog-airplane']
buds=[0.001,0.002,0.005,0.01,0.02,0.04]
arms=['Random','greedy','DPP','greedy_J','DPP_J']
out={}
bi=0
for mi,blk in enumerate(blocks):
    if '\\score' not in blk: continue
    # rows separated by \\ ; each row starts with & N &
    rows=re.split(r'\\\\',blk)
    rows=[r for r in rows if '\\score' in r]
    if len(rows)!=6: print('warn',mi,len(rows))
    for ri,row in enumerate(rows):
        cells=re.findall(r'\\score(?:TT|B)\{([-\d.]+)\}\{([\d.]+)\}|(--)',row)
        vals=[]
        for a,b,c in cells:
            vals.append(None if c else float(a))
        if len(vals)!=30: print('warnlen',models[mi],ri,len(vals)); continue
        for pi in range(2):
            for ai in range(3):
                for si in range(5):
                    idx=pi*15+ai*5+si
                    out[f"{models[mi]}|{attacks[ai]}|{pairs[pi]}|{buds[ri]}|{arms[si]}"]=vals[idx]
json.dump(out,open(ROOT+'/.ci_work/full_tex.json','w'))
print(len(out))
for k in sorted(out)[:8]: print(k,out[k])
