#!/usr/bin/env python3
"""Repair only missing Vulcan augmentation poison caches, then submit.

One Vulcan SSH process remains open for the check, upload, and submission.
If anything is missing, one Killarney SSH process downloads all missing caches.
SSH multiplexing is explicitly disabled.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parent
KILLARNEY_HOST = os.environ.get(
    "KILLARNEY_HOST", "mmoslem3@killarney.alliancecan.ca"
)
VULCAN_HOST = os.environ.get("VULCAN_HOST", "mmoslem3@vulcan.alliancecan.ca")
KILLARNEY_ROOT = os.environ.get(
    "KILLARNEY_ROOT", "/home/mmoslem3/scratch/attack_if"
)
VULCAN_ROOT = os.environ.get(
    "VULCAN_ROOT", "/home/mmoslem3/scratch/PoisonBase"
)

SSH_OPTIONS = [
    "-o", "ControlMaster=no",
    "-o", "ControlPersist=no",
    "-o", "ControlPath=none",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=20",
]
SENTINEL = "__END_MISSING__"


def die(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def exports(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(
            r"export ([A-Z0-9_]+)=(?:'([^']*)'|([^\s]+))", line
        )
        if match:
            values[match.group(1)] = match.group(2) or match.group(3)
    return values


def required_poison_paths() -> list[str]:
    wrapper_dir = LOCAL_ROOT / "sbatch" / "vulcan_remaining_20260827"
    template_dir = LOCAL_ROOT / "sbatch" / "augment_extra"
    wrappers = sorted(wrapper_dir.glob("augment_*_resume.sh"))
    if len(wrappers) != 21:
        die(f"expected 21 augmentation wrappers; found {len(wrappers)}")

    runs: set[str] = set()
    for wrapper in wrappers:
        text = wrapper.read_text()
        match = re.search(
            r"^source .*/sbatch/augment_extra/([^/\s]+\.sh)$",
            text,
            re.MULTILINE,
        )
        if not match:
            die(f"source template missing from {wrapper}")
        template = template_dir / match.group(1)
        if not template.is_file():
            die(f"template missing: {template}")
        values = exports(template)
        for key in ("RUN_RANDOM", "RUN_GREEDY", "RUN_DPP2", "RUN_DPP025", "RUN_DPP01"):
            if not values.get(key):
                die(f"{key} missing from {template}")
            runs.add(values[key])

    if len(runs) != 25:
        die(f"expected 25 distinct poison runs; found {len(runs)}")
    return [f"ours_result/{run}/poison_cache" for run in sorted(runs)]


VULCAN_REMOTE = r"""
set -eu
root='/home/mmoslem3/scratch/PoisonBase'
manifest=$(mktemp)
cleanup_manifest() { rm -f "$manifest"; }
trap cleanup_manifest EXIT HUP INT TERM

while IFS= read -r artifact_path; do
    [ "$artifact_path" = '__END_MANIFEST__' ] && break
    printf '%s\n' "$artifact_path" >> "$manifest"
    count=$(find "$root/$artifact_path" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l | tr -d '[:space:]')
    if [ "$count" -lt 5 ]; then
        printf 'MISSING\t%s\n' "$artifact_path"
    else
        printf 'Vulcan present: %s (%s perturbations)\n' "$artifact_path" "$count" >&2
    fi
done
printf '__END_MISSING__\n'

mkdir -p "$root"
tar -xf - -C "$root"

while IFS= read -r artifact_path; do
    count=$(find "$root/$artifact_path" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l | tr -d '[:space:]')
    [ "$count" -ge 5 ] || {
        printf 'ERROR: still missing %s after upload (%s perturbations)\n' "$artifact_path" "$count" >&2
        exit 1
    }
done < "$manifest"

printf 'Vulcan verification complete: all 25 poison caches are ready.\n'
cd "$root"
sh submit_vulcan-remaining-augmentation.sh
"""


KILLARNEY_REMOTE = r"""
set -eu
cd '/home/mmoslem3/scratch/attack_if'
while IFS= read -r artifact_path; do
    [ -d "$artifact_path" ] || {
        printf 'ERROR: Killarney cache missing: %s\n' "$artifact_path" >&2
        exit 1
    }
    count=$(find "$artifact_path" -maxdepth 1 -type f -name 'delta_*.pt' | wc -l | tr -d '[:space:]')
    [ "$count" -ge 5 ] || {
        printf 'ERROR: Killarney has only %s perturbations in %s\n' "$count" "$artifact_path" >&2
        exit 1
    }
    du -sh "$artifact_path" >&2
    printf '%s\n' "$artifact_path"
done | tar --checkpoint=500 --checkpoint-action=dot -cf - -T -
printf '\nKillarney missing-cache stream complete.\n' >&2
"""


def copy_submit_files(stage: Path) -> None:
    sbatch_dst = stage / "sbatch"
    sbatch_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        LOCAL_ROOT / "sbatch" / "vulcan_remaining_20260827",
        sbatch_dst / "vulcan_remaining_20260827",
        dirs_exist_ok=True,
    )
    shutil.copy2(
        LOCAL_ROOT / "submit_vulcan-remaining-augmentation.sh", stage
    )


def tar_command(stage: Path) -> list[str]:
    gtar = shutil.which("gtar")
    if gtar:
        return [
            gtar,
            "--checkpoint=500",
            "--checkpoint-action=dot",
            "-cf", "-",
            "-C", str(stage),
            ".",
        ]
    return ["tar", "-cf", "-", "-C", str(stage), "."]


def run() -> int:
    paths = required_poison_paths()
    print("Authenticate to Vulcan once...", flush=True)
    vulcan = subprocess.Popen(
        ["ssh", *SSH_OPTIONS, VULCAN_HOST, VULCAN_REMOTE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert vulcan.stdin is not None
    assert vulcan.stdout is not None

    try:
        manifest = "".join(f"{path}\n" for path in paths)
        manifest += "__END_MANIFEST__\n"
        vulcan.stdin.write(manifest.encode())
        vulcan.stdin.flush()

        missing: list[str] = []
        while True:
            raw = vulcan.stdout.readline()
            if not raw:
                code = vulcan.poll()
                die(f"Vulcan connection ended before its manifest (status {code})")
            line = raw.decode(errors="replace").rstrip("\n")
            if line == SENTINEL:
                break
            if line.startswith("MISSING\t"):
                missing.append(line.split("\t", 1)[1])
            elif line:
                print(line, flush=True)

        print(
            f"Vulcan reports {len(missing)} missing poison cache(s).",
            flush=True,
        )

        with tempfile.TemporaryDirectory(prefix="augment-repair.") as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()

            if missing:
                archive = Path(tmp) / "killarney-missing.tar"
                print("Authenticate to Killarney once...", flush=True)
                with archive.open("wb") as output:
                    result = subprocess.run(
                        ["ssh", *SSH_OPTIONS, KILLARNEY_HOST, KILLARNEY_REMOTE],
                        input=("".join(f"{path}\n" for path in missing)).encode(),
                        stdout=output,
                        stderr=None,
                        check=False,
                    )
                if result.returncode:
                    die(f"Killarney transfer failed with status {result.returncode}")
                with tarfile.open(archive, "r:") as bundle:
                    bundle.extractall(stage)

                for path in missing:
                    count = len(list((stage / path).glob("delta_*.pt")))
                    if count < 5:
                        die(f"downloaded only {count} perturbations for {path}")

            copy_submit_files(stage)
            payload = sum(
                path.stat().st_size
                for path in stage.rglob("*")
                if path.is_file()
            )
            print(f"Uploading {payload / (1024 ** 2):.1f} MiB to Vulcan...", flush=True)

            tar_proc = subprocess.Popen(
                tar_command(stage),
                stdout=subprocess.PIPE,
                stderr=None,
            )
            assert tar_proc.stdout is not None
            shutil.copyfileobj(tar_proc.stdout, vulcan.stdin, length=1024 * 1024)
            tar_proc.stdout.close()
            if tar_proc.wait() != 0:
                die("local tar process failed")

        vulcan.stdin.close()
        for raw in vulcan.stdout:
            sys.stdout.write(raw.decode(errors="replace"))
            sys.stdout.flush()
        code = vulcan.wait()
        if code:
            die(f"Vulcan verification/submission failed with status {code}")
        print("Augmentation repair and submission completed.", flush=True)
        return 0
    except BaseException:
        if vulcan.poll() is None:
            vulcan.terminate()
            try:
                vulcan.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vulcan.kill()
        raise


if __name__ == "__main__":
    raise SystemExit(run())

