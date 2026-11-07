"""Resume one r2 pretraining part while printing every completed epoch."""
import argparse
import os
from pathlib import Path
import sys
from datetime import datetime


code = Path(os.environ["REVIEW_CODE"])
sys.path.insert(0, str(code / "experiments" / "reviewer_sweep"))
sys.path.insert(0, str(code))
import run as sweep_run  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, choices=(1, 2, 3), required=True)
args = parser.parse_args()

pool = Path(os.environ["RESUME_POOL"])
missing = [i for i in range(args.part * 5, (args.part + 1) * 5)
           if not (pool / f"net_{i}.pt").exists()]
pending = iter(missing)
original_train = sweep_run.fu.train_from_scratch


def train_with_progress(*train_args, **train_kwargs):
    net_id = next(pending)

    def progress(epoch, total):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] net_{net_id}: epoch {epoch}/{total} complete", flush=True)

    train_kwargs["progress_callback"] = progress
    return original_train(*train_args, **train_kwargs)


print(f"Part {args.part}: missing checkpoints {missing or 'none'}", flush=True)
sweep_run.fu.train_from_scratch = train_with_progress
sweep_run.pretrain("ResNet20BN", 0.5, 2, args.part)
