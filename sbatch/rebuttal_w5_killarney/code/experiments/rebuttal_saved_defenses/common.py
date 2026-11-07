"""Atomic checkpoints and cooperative Slurm timeout handling."""
import json
import os
import signal
import time
from pathlib import Path

STOP_REQUESTED = False


class Paused(RuntimeError):
    pass


def install_stop_handlers():
    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f'Signal {signum}: will checkpoint at the next boundary.', flush=True)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)


def stopping():
    deadline = float(os.environ.get('DEFENSE_DEADLINE', 'inf'))
    return STOP_REQUESTED or time.time() >= deadline


def pause_if_requested():
    if stopping():
        raise Paused('Progress saved; rerun the submission script to continue.')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.environ.get("SLURM_JOB_ID", "local")}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.environ.get("SLURM_JOB_ID", "local")}.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)
