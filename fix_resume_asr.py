#!/usr/bin/env python3
"""
Fix per-target ASR/CTA lines in a concatenated resume log.

Why they are wrong: final.py prints

    target %d: ASR=... CTA=... +/- ...

from `succ`/`ctas`, which only hold the trials run in *that* process. When a
run is resumed part-way through a target, the resumed process prints the
average over the victims it ran (e.g. v3..v6) and silently drops the victims
the earlier process already did (v1, v2). The run-level `==== ... ====` line
is not affected -- it is rebuilt from results.csv, which keeps every trial.

This script rebuilds each per-target line from every `[tN vK/M]` trial line
belonging to the same run name anywhere in the file, and rewrites it in place
(the original is kept as <file>.bak).

    python fix_resume_asr.py resume_raw.txt [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import OrderedDict

RUN_RE = re.compile(r"=== run start: (\S+) on ")
TRIAL_RE = re.compile(r"\[t(\d+) v(\d+)/(\d+)\] (SUCCESS|fail) \(pred=\S+\) CTA=([\d.]+)")
SUMM_RE = re.compile(r"^(.*?\btarget )(\d+)(: ASR=)([\d.]+)(% CTA=)([\d.]+)( \+/- )([\d.]+)\s*$")


def mean(v):
    return sum(v) / len(v)


def pstd(v):                      # np.std default: population, not sample
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def collect_trials(lines):
    """(run, target) -> {victim: (success, cta)}, over the whole file."""
    trials, run, dups = OrderedDict(), None, []
    for i, ln in enumerate(lines):
        m = RUN_RE.search(ln)
        if m:
            run = m.group(1)
            continue
        m = TRIAL_RE.search(ln)
        if m and run:
            tgt, vi = int(m.group(1)), int(m.group(2))
            d = trials.setdefault((run, tgt), {})
            rec = (m.group(4) == "SUCCESS", float(m.group(5)))
            if vi in d:
                if d[vi] != rec:
                    dups.append((i + 1, run, tgt, vi, d[vi], rec))
            else:
                d[vi] = rec
    return trials, dups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    args = ap.parse_args()

    with open(args.path) as f:
        lines = f.read().splitlines()

    trials, dups = collect_trials(lines)
    for ln_no, run, tgt, vi, old, new in dups:
        print(f"[warn] line {ln_no}: {run} t{tgt} v{vi} logged twice with "
              f"different results {old} vs {new}; kept the first")

    run, fixed = None, 0
    for i, ln in enumerate(lines):
        m = RUN_RE.search(ln)
        if m:
            run = m.group(1)
            continue
        m = SUMM_RE.match(ln)
        if not (m and run):
            continue
        tgt = int(m.group(2))
        d = trials.get((run, tgt))
        if not d:
            print(f"[warn] line {i+1}: no trials found for {run} t{tgt}, left alone")
            continue

        oks = [v[0] for v in d.values()]
        ctas = [v[1] for v in d.values()]
        # Same formatting as final.py, so a correct line is byte-identical.
        new = (f"{m.group(1)}{tgt}{m.group(3)}{100.0 * mean(oks):.1f}"
               f"{m.group(5)}{mean(ctas):.4f}{m.group(7)}{pstd(ctas):.4f}")
        if new != ln:
            print(f"line {i+1}: {len(d)} trials (victims "
                  f"{','.join(str(v) for v in sorted(d))})")
            print(f"  -  {ln.strip()}")
            print(f"  +  {new.strip()}")
            lines[i] = new
            fixed += 1

    if fixed and not args.dry_run:
        shutil.copyfile(args.path, args.path + ".bak")
        with open(args.path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nfixed {fixed} line(s) in {args.path} (backup: {args.path}.bak)")
    else:
        print(f"\n{fixed} line(s) need fixing"
              + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
