import os, re, csv, json, glob
from collections import defaultdict
ROOT='/home/mmoslem3/scratch/attack_if'
RES=os.path.join(ROOT,'ours_result')

csvdata=defaultdict(lambda: defaultdict(dict))
for d in sorted(os.listdir(RES)):
    p=os.path.join(RES,d,'results.csv')
    if not os.path.isfile(p): continue
    with open(p) as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames or 'target_idx' not in rd.fieldnames: continue
        for r in rd:
            try: csvdata[d][int(r['target_idx'])][int(r['victim_id'])]=int(r['success'])
            except (ValueError,TypeError,KeyError): pass

logdata=defaultdict(lambda: defaultdict(dict))
run_re=re.compile(r'=== run start: (\S+) on ')
vic_re=re.compile(r'\[t(\d+) v(\d+)/(\d+)\]\s+(SUCCESS|fail)')
files=[]
for pat in ['sbatch/logs/*','sbatch/logs2/*','*.txt','results/*.txt','ours_result/logs/**/*']:
    files+=[f for f in glob.glob(os.path.join(ROOT,pat),recursive=True) if os.path.isfile(f)]
files+=glob.glob(os.path.join(RES,'*','log.txt'))
for f in files:
    try: txt=open(f,errors='ignore').read()
    except Exception: continue
    if 'run start:' not in txt: continue
    cur=None
    for line in txt.splitlines():
        m=run_re.search(line)
        if m: cur=m.group(1); continue
        m=vic_re.search(line)
        if m and cur:
            logdata[cur][int(m.group(1))][int(m.group(2))-1]=1 if m.group(4)=='SUCCESS' else 0

merged={}
src={}
for k in set(csvdata)|set(logdata):
    if csvdata.get(k):
        merged[k]=csvdata[k]; src[k]='csv'
    else:
        merged[k]=logdata[k]; src[k]='log'
json.dump({k:{str(t):{str(a):b for a,b in v.items()} for t,v in d.items()} for k,d in merged.items()},
          open(ROOT+'/.ci_work/harvest.json','w'))
json.dump(src,open(ROOT+'/.ci_work/src.json','w'))
print('runs',len(merged),'csv-backed',sum(1 for v in src.values() if v=='csv'),'log-only',sum(1 for v in src.values() if v=='log'))
