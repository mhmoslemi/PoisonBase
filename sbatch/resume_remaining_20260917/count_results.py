#!/usr/bin/env python3
"""Print the number of unique completed (target, victim) evaluations."""

import csv
import glob
import os
import sys


def main() -> None:
    recursive = False
    arguments = sys.argv[1:]
    if arguments[:1] == ["--recursive"]:
        recursive = True
        arguments = arguments[1:]
    if len(arguments) != 1:
        raise SystemExit("usage: count_results.py [--recursive] RUN_DIRECTORY")

    run_dir = arguments[0]
    pairs = set()
    if recursive:
        pattern = os.path.join(run_dir, "**", "results*.csv")
    else:
        pattern = os.path.join(run_dir, "results*.csv")
    for path in glob.glob(pattern, recursive=recursive):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("target_idx") or row.get("victim_id") is None:
                    continue
                pairs.add((int(row["target_idx"]), int(row["victim_id"])))
    print(len(pairs))


if __name__ == "__main__":
    main()
