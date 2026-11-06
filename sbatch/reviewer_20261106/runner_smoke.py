"""Small integration run; all synthetic/check artifacts stay in calibration."""
import os
import json
import torch
import run

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark=True
run.OUT=run.ROOT/'reviewer_20261106_result'/'calibration'/os.environ['SLURM_JOB_ID']/'runner'
original_craft=run.fu.craft_selected_bases

def short_craft(*args,**kwargs):
    return original_craft(*args,**dict(kwargs,steps=3,restarts=1))

run.fu.craft_selected_bases=short_craft
run.ordinary=lambda net,ctx: run.fu.train_from_scratch(net,ctx['train_imgs'],ctx['train_labs'],2,.1,128,[],ctx['device'])
for model,cell_id in [('ConvNetBN',0),('ResNet20BN',18)]:
    for repeat in range(3):
        run.control(model,repeat,'plain')
    run.run_cell(cell_id,0)
    rows=list((run.OUT/'cells'/f'{cell_id:03d}').glob('t*_r*/result.json'))
    assert len(rows)==6
    assert all(json.loads(p.read_text())['num_poisons']==50 for p in rows)
    run.run_cell(cell_id,0)
    assert len(list((run.OUT/'cells'/f'{cell_id:03d}').glob('t*_r*/result.json')))==6
run.atomic_json(run.ROOT/'reviewer_20261106_result'/'runner_checks_passed.json',dict(job=os.environ['SLURM_JOB_ID']))
print('BOTH RUNNERS, PAIRED CONTROLS, POISON CACHE, AND RESUME PASSED',flush=True)
