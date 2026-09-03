import os, re, csv, json, glob
from collections import defaultdict

ROOT='/home/mmoslem3/scratch/attack_if'
data=defaultdict(lambda: defaultdict(dict))

RES=os.path.join(ROOT,'ours_result')
ncsv=0
for d in sorted(os.listdir(RES)):
    p=os.path.join(RES,d,'results.csv')
    if not os.path.isfile(p): continue
    with open(p) as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames or 'target_idx' not in rd.fieldnames: continue
        for r in rd:
            try:
                data[d][int(r['target_idx'])][int(r['victim_id'])]=int(r['success'])
            except (ValueError,TypeError,KeyError): pass
    ncsv+=1

run_re=re.compile(r'=== run start: (\S+) on ')
vic_re=re.compile(r'\[t(\d+) v(\d+)/(\d+)\]\s+(SUCCESS|fail)')
files=[]
for pat in ['sbatch/logs/*','sbatch/logs2/*','*.txt','results/*.txt','ours_result/logs/**/*']:
    files+= [f for f in glob.glob(os.path.join(ROOT,pat), recursive=True) if os.path.isfile(f)]
files += glob.glob(os.path.join(RES,'*','log.txt'))
nlog=0
for f in files:
    try:
        txt=open(f,errors='ignore').read()
    except Exception: continue
    if 'run start:' not in txt: continue
    cur=None
    for line in txt.splitlines():
        m=run_re.search(line)
        if m: cur=m.group(1); continue
        m=vic_re.search(line)
        if m and cur:
            t=int(m.group(1)); v=int(m.group(2))-1
            data[cur][t][v]=1 if m.group(4)=='SUCCESS' else 0
    nlog+=1

out={k:{str(t):{str(a):b for a,b in vv.items()} for t,vv in v.items()} for k,v in data.items()}
json.dump(out,open(os.path.join(ROOT,'.ci_work/harvest.json'),'w'))
print("csv dirs:",ncsv,"log files parsed:",nlog,"runs total:",len(data))
full=[k for k,v in data.items() if len(v)==10 and all(len(x)>=6 for x in v.values())]
print("runs 10x>=6:",len(full))
