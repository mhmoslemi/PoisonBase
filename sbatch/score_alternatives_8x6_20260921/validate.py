#!/usr/bin/env python3
"""Validate the whole batch without GPUs, PyTorch, data, or submission."""
import ast
import csv
from pathlib import Path
import subprocess
from generate_jobs import filename, job_text
from run_cell import cells, SELECTORS, SETTINGS, TARGETS

HERE = Path(__file__).resolve().parent


def main():
    expected = cells()
    if len(expected) != 40 or len({(c['attack'], c['budget'], c['selector']) for c in expected}) != 40:
        raise SystemExit('expected 40 distinct experiments')
    if {(c['attack'], c['budget']) for c in expected} != set(SETTINGS):
        raise SystemExit('unexpected attack/budget setting')
    if {c['selector'] for c in expected} != set(SELECTORS):
        raise SystemExit('unexpected selector')
    if any(len(t) != 8 or len(set(t)) != 8 for t in TARGETS.values()):
        raise SystemExit('target sets must contain exactly eight distinct IDs')
    if {p.name for p in HERE.glob('job_*.sh')} != {filename(c) for c in expected}:
        raise SystemExit('job files do not match the experiment grid')
    for cell in expected:
        path = HERE / filename(cell)
        if path.read_text() != job_text(cell):
            raise SystemExit(f'job differs from pinned protocol: {path.name}')
    with (HERE / 'manifest.tsv').open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    desired = [dict(c, num_targets='8', num_victims='6', job_file=filename(c)) for c in expected]
    if rows != desired:
        raise SystemExit('manifest does not match the jobs')
    for path in HERE.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    print('Validated 40 jobs: ConvNetBN, 5 new selectors, 8 targets x 6 victims, aip-boyuwang, 03:20:00.')


if __name__ == '__main__':
    main()
