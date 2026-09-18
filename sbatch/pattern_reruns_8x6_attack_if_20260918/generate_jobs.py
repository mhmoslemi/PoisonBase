#!/usr/bin/env python3
"""Create the attack_if-path profile of the 72 pattern rerun jobs."""

from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "sbatch" / "pattern_reruns_8x6_20260918"


def main() -> None:
    source_jobs = sorted(SOURCE.glob("job_*.sh"))
    if len(source_jobs) != 72:
        raise RuntimeError(f"expected 72 PoisonBase jobs, found {len(source_jobs)}")
    for old in OUT.glob("job_*.sh"):
        old.unlink()

    for source in source_jobs:
        text = source.read_text()
        text = text.replace("#SBATCH --job-name=pr", "#SBATCH --job-name=ar")
        text = text.replace(
            "/home/mmoslem3/scratch/PoisonBase/sbatch/logs/pr",
            "/home/mmoslem3/scratch/attack_if/sbatch/logs/ar")
        text = text.replace(
            'export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"',
            'export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"\n'
            'export ROOT="${ROOT:-$SOURCE_ROOT}"')
        text = text.replace(
            'export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"',
            'export DATA_ROOT="${DATA_ROOT:-/home/mmoslem3/scratch/data}"')
        text = text.replace(
            'export RESULT_ROOT="${RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}"',
            'export RESULT_ROOT="${RESULT_ROOT:-$ROOT/pattern_rerun_8x6_result}"\n'
            'export RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/attack_if_pattern_rerun_8x6}"')
        text = text.replace(' victims=6"', ' victims=6 profile=attack_if"')
        destination = OUT / source.name
        destination.write_text(text)
        destination.chmod(0o755)

    manifest = (SOURCE / "manifest.tsv").read_text()
    lines = manifest.splitlines()
    lines[0] += "\tprofile"
    lines[1:] = [line + "\tattack_if" for line in lines[1:]]
    (OUT / "manifest.tsv").write_text("\n".join(lines) + "\n")
    print("generated 72 attack_if jobs")


if __name__ == "__main__":
    main()
