#!/usr/bin/env python3
import ast
from pathlib import Path
import subprocess
from generate_jobs import JOBS, body

HERE = Path(__file__).resolve().parent
expected = {f'job_{phase}_{identifier}.sh' for phase, identifier in JOBS}
if {p.name for p in HERE.glob('job_*.sh')} != expected:
    raise SystemExit('Expected exactly eight setup/method job files')
for phase, identifier in JOBS:
    path = HERE / f'job_{phase}_{identifier}.sh'
    if path.read_text() != body(phase, identifier):
        raise SystemExit(f'Job differs from the pinned configuration: {path}')
for path in HERE.glob('*.sh'):
    subprocess.run(['bash', '-n', str(path)], check=True)
for path in [*HERE.glob('*.py'), HERE.parents[1] / 'experiments/imagenet100_gm.py']:
    ast.parse(path.read_text(), filename=str(path))
print('Validated eight ImageNet job templates; 03:20:00; account chosen by submitter.')
