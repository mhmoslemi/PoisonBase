#!/usr/bin/env python3
"""Validate the attack_if copy and its shared runtime without GPUs or data."""
import ast
import csv
from pathlib import Path
import subprocess
import sys
from generate_jobs import job_text, source_rows, SHARED

HERE = Path(__file__).resolve().parent


def main():
    # Also validate the canonical configuration source and shared shell runtime.
    subprocess.run([sys.executable, str(SHARED / 'validate.py')], check=True)
    expected = source_rows()
    with (HERE / 'manifest.tsv').open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if rows != expected:
        raise SystemExit('attack_if manifest must exactly match the original 40 experiments')
    if {p.name for p in HERE.glob('job_*.sh')} != {r['job_file'] for r in expected}:
        raise SystemExit('expected exactly the same 40 individual experiment jobs')
    for row in expected:
        path = HERE / row['job_file']
        if path.read_text() != job_text(row):
            raise SystemExit(f'job differs from attack_if settings: {path.name}')
    for path in HERE.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in (SHARED / 'run_cell.py', HERE / '_job_common.sh'):
        if not path.is_file():
            raise SystemExit(f'missing runtime: {path}')
    print('Validated 40 attack_if jobs: 8 targets x 6 victims, aip-boyuwang, 03:20:00; data=/home/mmoslem3/scratch/data.')


if __name__ == '__main__':
    main()
