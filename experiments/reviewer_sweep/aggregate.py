"""Validate the fixed experiment matrix and write tables and paired comparisons."""
import csv
import json
from pathlib import Path
import numpy as np
from protocol import cells, TARGETS, REPEATS, atomic_json


def aggregate(root):
    root=Path(root)
    rows=[]; missing=[]; bycell={}
    for cell in cells():
        trials=[]
        for target in TARGETS:
            for repeat in REPEATS:
                file=root/'cells'/f"{cell['id']:03d}"/f't{target}_r{repeat}'/'result.json'
                if not file.exists():
                    missing.append(str(file.relative_to(root))); continue
                result=json.loads(file.read_text())
                assert result['cell']==cell and result['target']==target and result['repeat']==repeat
                assert result['num_poisons']==round(50000*cell['budget'])
                assert result['linf']<=8/255+1e-6
                trials.append(result)
        bycell[cell['id']]=trials
        if len(trials)==30:
            row=dict(cell,asr=np.mean([r['success'] for r in trials]),
                     clean_accuracy=np.mean([r['clean_accuracy'] for r in trials]),
                     clean_asr=np.mean([r['clean_target_prediction']==5 for r in trials]),
                     n_trials=len(trials))
            rows.append(row)
    atomic_json(root/'completion.json',dict(completed_cells=len(rows),expected_cells=120,
                completed_trials=sum(map(len,bycell.values())),expected_trials=3600,missing=missing))
    if rows:
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    comparisons=[]
    rng=np.random.default_rng(9243)
    for cell in cells():
        if cell['selector']!='basis' or len(bycell[cell['id']])!=30:
            continue
        for comparator in ('random','margin','similarity','wo_pretrained','wo_ood'):
            other=next((c for c in cells() if c['selector']==comparator and
                        all(c[k]==cell[k] for k in ('model','budget','fraction','pool','defense'))),None)
            if other is None or len(bycell[other['id']])!=30:
                continue
            a={(r['target'],r['repeat']):r['success'] for r in bycell[cell['id']]}
            b={(r['target'],r['repeat']):r['success'] for r in bycell[other['id']]}
            differences=np.array([np.mean([a[t,r]-b[t,r] for r in REPEATS]) for t in TARGETS])
            bootstrap=differences[rng.integers(0,10,(10000,10))].mean(1)
            comparisons.append(dict(cell_id=cell['id'],comparator=comparator,
                delta_asr=float(differences.mean()),ci_low=float(np.quantile(bootstrap,.025)),
                ci_high=float(np.quantile(bootstrap,.975)),target_wins=int((differences>0).sum()),
                target_ties=int((differences==0).sum()),target_losses=int((differences<0).sum())))
    atomic_json(root/'paired_comparisons.json',comparisons)
    if missing:
        raise RuntimeError(f'{len(missing)} of 3600 trials missing; see completion.json')
    print('Verified all 120 configurations and 3600 evaluations.',flush=True)
