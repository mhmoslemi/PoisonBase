import re,json
ROOT='/home/mmoslem3/scratch/attack_if'
txt=open(ROOT+'/CI_table.tex').read()
# split into tables
tabs=re.split(r'\\begin\{table\*\}',txt)[1:]
models=['ConvNetBN','ResNet20BN','VGG13BN']
attacks=['fc','gradmatch','sapa']
rhos=[1,2,5,10,20,40]
budmap={1:0.001,2:0.002,5:0.005,10:0.01,20:0.02,40:0.04}
out={}
for mi,tab in enumerate(tabs):
    body=tab.split('\\midrule',1)[1]
    blocks=body.split('\\midrule')
    pairs=['dog-bird','frog-airplane']
    for bi,blk in enumerate(blocks[:2]):
        rows=[r for r in blk.split('\\\\') if '\\asr' in r]
        assert len(rows)==6, (mi,bi,len(rows))
        for ri,row in enumerate(rows):
            asrs=re.findall(r'\\asr\{([^}]*)\}\{([^}]*)\}',row)
            assert len(asrs)==6, row
            for ai in range(3):
                rand=asrs[ai*2]
                out[f"{models[mi]}|{attacks[ai]}|{pairs[bi]}|{budmap[rhos[ri]]}"]=(rand[0],rand[1])
json.dump(out,open(ROOT+'/.ci_work/tex_rand.json','w'))
print(len(out))
for k in list(out)[:6]: print(k,out[k])
