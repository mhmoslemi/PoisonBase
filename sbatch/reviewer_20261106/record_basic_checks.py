"""Record passed checks while Telltale jobs remain explicitly held in Slurm."""
import json
import os
from pathlib import Path
from protocol import atomic_json, source_fingerprint
root=Path(os.environ['REVIEW_ROOT'])
out=root/'reviewer_20261106_result'
for name in ('resume_checks_passed.json','runner_checks_passed.json'):
    assert json.loads((out/name).read_text())['job']=='2161523'
log=(root/'sbatch/reviewer_20261106/logs/allocated_check_v2.out').read_text()
assert 'CALIBRATION ConvNetBN' in log and 'CALIBRATION ResNet20BN' in log
result=dict(source_fingerprint=source_fingerprint(),job=os.environ['SLURM_JOB_ID'],
    checks='math, both architectures, real GM, ASD including meta/resume, paired runners and caches',
    detector_pending=True)
atomic_json(out/'basic_checks_passed.json',result)
if not (out/'checks_passed.json').exists():
    atomic_json(out/'checks_passed.json',result)
print('NON-TELLTALE CHECKS RECORDED',flush=True)
