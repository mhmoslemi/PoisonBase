#!/usr/bin/env bash
#
# cv0-1 -- PREREQUISITE for cv1.sh and cv3.sh. Recrafts the Random arm of the
# ConvNetBN fc/gradmatch sweep for the dog-bird pair.
#
# cv0-1 (dog-bird) and cv0-2 (frog-airplane) are the two halves of the old cv0.sh,
# split so they can run on two GPUs at once: 5.7 h each instead of 11.5 h in
# sequence. They touch disjoint run directories, so they are safe to run
# together and in any order.
#
# WHY THIS EXISTS: the augmentation protocol REPLAYS saved poisons, and pins its
# targets to the intersection of what Random and DPP have cached. On disk today:
#
#     ConvNetBN gradmatch/fc, every cell:   DPP = 10 targets,  Random = 0
#
# so aug.sh reports "no target has poisons under every selection" and skips
# every budget. The ConvNet Random numbers in tab:sweep_asr_clean_minus_cta are
# real -- their runs happened -- but those directories were cleaned and only
# raw.txt still has the logs. The perturbations themselves are gone, so they
# have to be recrafted before anything can be replayed through an augmented
# victim. (The ResNet20BN table did not need this: its Random caches survived.)
#
# WHAT IT DOES: crafts the 10 pinned targets under --base random for the 4 cells
# of this pair, and trains victims 0..3 on each.
#
# --num_victims 4, NOT 1 and not the sweep's 6. make_aug_table.py builds the
# No Aug. column by filtering these very results to the first 5 paired targets
# and victim_id < 4, so the Random arm needs those four victims or that column
# comes out over 5 trials against the DPP arm's 20 -- unpaired, and the
# generator flags it INCOMPLETE. Victims 4 and 5 are never read, so they are
# not trained. The surrogate and clean-victim caches these runs read are
# already on disk. Targets are pinned to
# target_sets/ConvNetBN_<attack>_dog-bird.json -- the same 10 the DPP arm
# attacked, which is what makes the two arms comparable.
#
# COST (craft s/target measured from this project's own ConvNet logs:
# fc 2 s at b0.002 / 12 s at b0.01; gradmatch 172 s at b0.002 / 1080 s at b0.02;
# victim ~50 s):
#   fc        b0.002   10 x    2 s craft + 40 x 50 s victim =  0.56 h
#   fc        b0.01    10 x   12 s craft + 40 x 50 s victim =  0.59 h
#   gradmatch b0.002   10 x  172 s craft + 40 x 50 s victim =  1.03 h
#   gradmatch b0.02    10 x 1080 s craft + 40 x 50 s victim =  3.56 h
#   --------------------------------------------------------------
#   L40S ~5.7 h    H100 ~3.2 h  (at ~1.8x; extrapolated)
#
# gradmatch at b0.02 is ~62% of that on its own -- if the allocation is short,
# CELLS="1 2" gets both fc cells done in ~1.2 h, which is all cv3/cv3 need.
#
# Nothing already cached is redone -- a rerun after a wall-clock kill resumes.
#
#   sh cv0-1.sh
#   CELLS="1 2" sh cv0-1.sh     # run only some cells (1..4, cheapest first)

set -u

CELLS="${CELLS:-1 2 3 4}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "cv0-1.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

# "<attack> <budget> <difficulty label>" -- cheapest first
cell_1="fc        0.002  50"
cell_2="fc        0.01   50"
cell_3="gradmatch 0.002  70"
cell_4="gradmatch 0.02   70"

for c in $CELLS; do
    eval "spec=\${cell_$c:-}"
    [ -n "$spec" ] || { echo "cv0-1.sh: no such cell '$c' (expected 1..4)"; exit 1; }
    # shellcheck disable=SC2086
    set -- $spec
    attack=$1; bud=$2; tg=$3

    echo "=== cv0-1 cell $c | ConvNetBN / $attack / dog-bird / b$bud | crafting Random arm ==="
    python final_update.py \
        --dataset CIFAR10 --data_path /home/mmoslem3/scratch/data --seed 42 \
        --cache_dir ./cache --out_dir ours_result \
        --model ConvNetBN --attack "$attack" --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$bud" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 --craft_batch 256 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 10 --target_select "$tg" \
        --target_idx_file "target_sets/ConvNetBN_${attack}_dog-bird.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
done

echo "=== cv0-1.sh finished (cells: $CELLS) ==="
echo "when cv0-1 and cv0-2 are both done, run cv1.sh and cv3.sh"
