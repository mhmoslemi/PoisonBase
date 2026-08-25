#!/usr/bin/env bash
#
# appendix.tex, tab:utility-defense -- the epic half.
#
# Self-contained. It reads the calibration ap5-a left on disk and picks the
# strength itself (strongest setting whose clean accuracy is within two points of
# the undefended ConvNetBN), so there is nothing to type. Override with
# EPIC_KEEP=<value> if you want a specific one.
#
# ConvNetBN, eps=2e-2, GM and SAPA, dog-bird and frog-airplane, Random vs DPP.
# 8 runs. Poisons are replayed from ours_result -- all 8 sets are already there.
#
#   sh appendix/ap5-b.sh

set -u

cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap5-b.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

# Calibrated on clean data by ap5-a.sh: subset 0.05 gives 80.33 against an
# undefended 81.05 (-0.72), the strongest setting inside the two-point band.
# 0.1 is already -2.76. Override with EPIC_KEEP=<value>.
VAL="${EPIC_KEEP:-0.05}"
if [ -z "$VAL" ]; then
    VAL=$(python appendix/pick_defense.py --defense epic --verbose) || {
        echo
        echo "No usable epic calibration on disk yet. Run ap5-a.sh first (it sweeps the"
        echo "strengths on clean data); if it has already run and every setting is still"
        echo "outside the two-point band, widen the sweep or set EPIC_KEEP=<value> by hand."
        exit 1; }
fi
echo "=== epic at strength $VAL (clean-utility matched) ==="

# The difficulty degree the b0.02 poisons were crafted under. defense.sh normally
# reads it from sweep_config.json, but that file only records the (model, attack,
# pair) combos table.tex reports and has NO ConvNetBN/sapa entry -- without
# TARGET_SELECT the four SAPA rows die with "no difficulty for ConvNetBN / sapa".
# SAPA reused the GM targets for each pair, so the degrees below match the crafts
# already on disk.
# PAIRS lets appendix/final shard this by class pair; both by default.
for PAIR in ${PAIRS:-dog-bird frog-airplane}; do
    case "$PAIR" in
        dog-bird)      TSEL=70;;
        frog-airplane) TSEL=35;;
    esac
    for ATT in gradmatch sapa; do
        echo "--- $PAIR | $ATT | epic (tgt$TSEL)"
        env MODEL=ConvNetBN ATTACK="$ATT" CLASS_PAIR="$PAIR" BUDGETS="0.02" \
            SELS="random dpp" DEFENSES="epic" NUM_VICTIMS=5 NUM_TARGETS=5 \
            TARGET_SELECT="$TSEL" \
            EPIC_SUBSET="$VAL" sh ./defense.sh || exit 1
    done
done

echo "=== ap5-b.sh finished ==="
