#!/usr/bin/env bash
#
# The SVHN row of latex/appendix.tex tab:cross-dataset. Nothing else.
#
#   sh svhn-cells.sh
#
#   step 1  download SVHN and sample the five digit pairs                 ~2 min
#   step 2  train 20 ConvNetBN surrogates on SVHN            (cached)      ~0 min
#   step 3  20 attack runs: 5 pairs x {GM, SAPA} x {Random, DPP}          ~3 h
#   step 4  report: per-pair ASR, the four cells, the LaTeX row           ~0 min
#
# ~3.5 h on an L40S. Skips what is done, resumes what was killed, takes no arguments.
#
# RESUMING. These jobs die when the GPU allocation goes away, which happens
# often enough that the first attempt is not expected to finish. Rerun the same
# command; it is idempotent at three levels:
#
#   * a finished run is skipped on the ==== ASR line in its log;
#   * a run killed mid-flight resumes per (target, victim) trial from
#     results_rank*.csv, so only the unfinished victims are retrained;
#   * step 3 makes up to MAX_PASSES passes over the queue, so one run dying
#     no longer strands the runs queued behind it -- the next pass picks them
#     up. A pass that adds nothing stops the loop instead of spinning.
#
# Step 0b is the part that actually needed fixing. A hard kill leaves .lock
# behind, and a live lock does not make the next attempt wait -- final_update.py
# returns immediately and exits 0, so the driver reads that as success and the
# cell ends up silently empty. LOCK_STALE_S clears it after two hours; step 0b
# clears it now, but only when the owning process is really gone.
#
# Step 4 never runs the GPU. Run `sh svhn-cells.sh` on a dead allocation and it
# reports how far along the row is and what is still missing.
#
# ConvNetBN, not ResNet18BN. The CIFAR-100 row had to change the backbone because
# 100 classes at 500 images each is a different problem; SVHN is 32x32 with ten
# classes like CIFAR-10, so keeping ConvNetBN makes the SVHN row a clean
# dataset-only comparison against the CIFAR-10 row rather than a dataset-and-
# backbone one.
#
# SVHN's train split is 73,257 images, so the same 2e-3 budget buys 147 poisons
# a target here against CIFAR-10's 100 -- the budget is a fraction, and that is
# the quantity the table holds fixed.
#
# VICTIM LEARNING RATE. The first pass ran at the CIFAR lr of 0.1 and ~14% of
# victim runs collapsed to a single-class predictor: CTA lands on 0.196, which is
# SVHN's majority-class rate, and the same victim index collapses across attacks
# for a given pair. A collapsed victim can never register a hit on the target, so
# it biases ASR down and drags mean CTA from 0.91 to 0.76. This runs at 0.01,
# where SVHN trains stably. Because victim_lr is not part of the run name, step 0
# moves any directory trained at 0.1 aside first -- otherwise the new run would
# resume the collapsed trials and silently mix two learning rates in one cell.

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

# Number of passes step 3 makes over the 20-run queue. Each pass skips what is
# already on disk, so a pass only costs what is left.
MAX_PASSES="${MAX_PASSES:-4}"

# No GPU is no longer fatal: step 4 is a report over files on disk, and after a
# crash the first thing you want is to see how far the row got. Steps 1-3 are
# skipped instead of the script exiting.
NO_GPU=""
if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    NO_GPU=1
    echo "svhn-cells.sh: no CUDA device visible -- get a GPU allocation to make"
    echo "               progress. Skipping to step 4 to report what is on disk."; }
fi

echo "########## step 0: retire the lr=0.1 runs ##########"
# victim_lr is not in the run name, so these have to be moved, not overwritten.
# Renames only -- logs, CSVs and poison caches are kept under the .lr0.1 suffix.
RETIRED=0
for D in ours_result/SVHN_*; do
    case "$D" in *.lr0.1) continue;; esac
    [ -d "$D" ] || continue
    grep -q '"victim_lr": 0.1' "$D/log.txt" 2>/dev/null || continue
    if [ -n "${DRY_RUN:-}" ]; then echo "    would retire: $(basename "$D")"; RETIRED=$((RETIRED+1)); continue; fi
    mv "$D" "$D.lr0.1" && echo "    retired: $(basename "$D")" && RETIRED=$((RETIRED+1))
done
[ "$RETIRED" = 0 ] && echo "    nothing to retire"
echo "    (undo:  for d in ours_result/*.lr0.1; do mv \"\$d\" \"\${d%.lr0.1}\"; done)"

echo
echo "########## step 0b: clear locks left by killed runs ##########"
# .lock holds "<node>:<pid>". Removing one that a live process still holds would
# let two writers into the same run dir, where they delete each other's
# results_rank*.csv -- so the pid is checked before anything is removed.
#
# ps -p, not kill -0: kill -0 fails with EPERM on a process owned by someone
# else, which is indistinguishable from "no such process", so a live run would
# read as dead and have its lock pulled out from under it. ps -p answers the
# question that is actually being asked.
#
# A lock written on another node cannot be checked that way at all, and neither
# can one whose contents are not "<node>:<pid>", so those go by age -- 30 min,
# against a single run of about 9 min.
sweep_locks() {
    SWEPT=0
    for L in ours_result/SVHN_*/.lock; do
        [ -f "$L" ] || continue
        R=$(basename "$(dirname "$L")")
        WHO=$(cat "$L" 2>/dev/null)
        NODE=${WHO%%:*}
        PID=${WHO##*:}
        # Unparseable contents fall through to the age test. Two separate
        # checks: "has a colon at all", then "the pid part is all digits".
        case "$WHO" in *:*) ;; *) NODE="" ;; esac
        case "$PID" in ''|*[!0-9]*) NODE="" ;; esac
        if [ -n "$NODE" ] && [ "$NODE" = "$(uname -n)" ]; then
            if ps -p "$PID" >/dev/null 2>&1; then
                echo "    live, left alone: $R ($WHO)"; continue
            fi
        elif [ -z "$(find "$L" -mmin +30 2>/dev/null)" ]; then
            echo "    not ours to judge and under 30 min old, left alone: $R ($WHO)"; continue
        fi
        if [ -n "${DRY_RUN:-}" ]; then echo "    would clear: $R ($WHO)"; SWEPT=$((SWEPT+1)); continue; fi
        rm -f "$L" && echo "    cleared stale lock: $R ($WHO)" && SWEPT=$((SWEPT+1))
    done
    [ "$SWEPT" = 0 ] && echo "    no stale locks"
    return 0
}
sweep_locks

echo
echo "########## step 1: SVHN + the five digit pairs ##########"
if [ -n "$NO_GPU" ]; then echo "    (no GPU, skipped)"; else
if [ -z "${DRY_RUN:-}" ]; then
python - <<'PYEOF'
import sys
sys.path.insert(0, '/home/mmoslem3/scratch/attack_if')
from utils import get_dataset
ch, im, nc, mean, std, dtr, dte, cn, _ = (get_dataset('SVHN', '/home/mmoslem3/scratch/data') + (None,)*9)[:9]
print('  SVHN ready: %d train, %d test, %d classes, %s' % (len(dtr), len(dte), nc, im))
PYEOF
fi
python appendix/pick_pairs.py --dataset SVHN --data_path /home/mmoslem3/scratch/data \
    --num_pairs 5 --seed 42 --out target_sets/xdata_pairs_SVHN.json
fi

echo
echo "########## step 2: 20 ConvNetBN surrogates on SVHN ##########"
if [ -n "$NO_GPU" ]; then echo "    (no GPU, skipped)"; else
DATASET=SVHN MODEL=ConvNetBN BUDGET=0.002 VICTIM_LR=0.01 STEP=surrogates sh appendix/ap2-cifar100.sh || exit 1
fi

echo
echo "########## step 3: 20 attack runs ##########"
# ap2-cifar100.sh exits on the first run that fails, which used to leave every
# run queued behind it untouched -- one dead allocation halfway down the list
# cost the whole remainder. Passes fix that: each one restarts the queue and
# skips what finished, so the failure only costs the run it happened in. The
# loop stops on a pass that adds nothing, which is what a persistent failure
# (no disk, a real bug) looks like as opposed to a preempted allocation.
if [ -n "$NO_GPU" ]; then echo "    (no GPU, skipped)"; else
PASS=1
PREV=-1
while : ; do
    DONE=$(python appendix/svhn_status.py count)
    if [ "$DONE" -ge 20 ]; then
        echo "--- all 20 runs on disk"; break
    fi
    if [ -n "${DRY_RUN:-}" ] && [ "$PASS" -gt 1 ]; then break; fi
    if [ "$DONE" -le "$PREV" ]; then
        echo "--- pass $((PASS-1)) finished at $DONE/20 and added nothing."
        echo "    Not a preemption -- read the log of a run listed as missing below."
        break
    fi
    if [ "$PASS" -gt "$MAX_PASSES" ]; then
        echo "--- stopping at $DONE/20 after $MAX_PASSES passes (raise MAX_PASSES to go on)."
        break
    fi
    echo "--- pass $PASS of at most $MAX_PASSES: $DONE/20 runs already on disk"
    PREV="$DONE"
    [ "$PASS" -gt 1 ] && sweep_locks
    DATASET=SVHN MODEL=ConvNetBN BUDGET=0.002 VICTIM_LR=0.01 STEP=runs sh appendix/ap2-cifar100.sh
    PASS=$((PASS+1))
done
fi

echo
echo "########## step 4: the SVHN row ##########"
python appendix/svhn_status.py report

echo
echo "=== svhn-cells.sh finished ==="
