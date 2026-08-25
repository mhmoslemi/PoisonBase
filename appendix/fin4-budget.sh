#!/usr/bin/env bash
#
# 4 of 5 -- app-base.tex tab:selection-ladder-budget (20 cells + the derived row).
#
# GM with ConvNetBN on bird->dog, five selection rules (Bottom-m, First-m, Random,
# Pixel distance, DPP) across four poison budgets. The 2e-3 column is exactly the
# GM/ConvNetBN column of fin3-ladder.sh, so run fin3 first and those five skip
# themselves; only the other three budgets are new. ~24 h.
#
#   sh appendix/fin4-budget.sh
#   DRY_RUN=1 sh appendix/fin4-budget.sh
#
# TWO THINGS IN THE DRAFT THIS SCRIPT CANNOT SATISFY AS WRITTEN:
#
#   1. "we select each base set once and reuse it across eps" -- not possible when
#      eps is the poison budget, because the number of poisons N_p = eps * N_total
#      is 100 / 250 / 500 / 2000 across these four columns. The base SETS therefore
#      cannot be identical; what is held fixed is the selection RULE. The caption
#      needs rewording, or the experiment needs N_p pinned (--budget 0 --num_poisons
#      100) so the columns really do share a base set.
#   2. "only the admissible perturbation neighborhood changes across columns" and
#      "when the perturbation budget is small, the clean base more strongly restricts
#      the region accessible to the poison optimizer" describe an L-infinity radius,
#      not a poison fraction. Everywhere else in the paper eps is the poison budget
#      (appendix.tex: "ConvNetBN, eps = 5e-3" for b0.005 runs), and this script
#      follows that. If the radius is what you meant, swap --budget for --epsilon;
#      build_run_name now keeps sub-1/255 radii in separate directories, which it
#      did not before -- 2e-3 and 5e-3 both used to render 'eps1' and would have
#      shared a poison_cache.

set -u

DATA_PATH=/home/mmoslem3/scratch/data
SEED=42
NT=5
NV=5
MODEL=ConvNetBN
ATT=gradmatch
PAIR=dog-bird
DRY_RUN="${DRY_RUN:-}"
# Sharding knobs, used by appendix/final. Defaults cover the whole table.
BUDGETS="${BUDGETS:-0.002 0.005 0.01 0.04}"
CRITS="${CRITS:-random bottom first pixel dpp}"

cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "fin4: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

IDX="target_sets/ladder_${MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --model "$MODEL" --pair "$PAIR" \
        --target_select random --num_targets "$NT" --num_victims "$NV" \
        --out "$IDX" || exit 1
fi

run() {
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "ours_result/$TAG/log.txt" 2>/dev/null; then
        echo "--- done already: $TAG"; return 0; fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $* --target_idx_file $IDX"; return 0; fi
    python final_update.py "$@" --target_idx_file "$IDX" || exit 1
}

# cheapest budget first, so a short allocation still lands whole columns
for BUDGET in $BUDGETS; do

    COMMON="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
        --cache_dir ./cache --out_dir ours_result --pair_order poison-target \
        --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 --clean_baseline \
        --target_select random --num_targets $NT --num_victims $NV \
        --model $MODEL --class_pair $PAIR --budget $BUDGET --attack $ATT"

    echo "########## budget $BUDGET ##########"

    for C in $CRITS; do
        case "$C" in
            random)
                LABEL="b$BUDGET | Random"
                run "CIFAR10_${MODEL}_${ATT}_random_${PAIR}_b${BUDGET}_eps8_seed${SEED}_ce5" \
                    $COMMON --base random
                ;;
            dpp)
                LABEL="b$BUDGET | DPP"
                run "CIFAR10_${MODEL}_${ATT}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_seldpp2_ce5" \
                    $COMMON --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0
                ;;
            *)
                LABEL="b$BUDGET | $C"
                run "CIFAR10_${MODEL}_${ATT}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_sel${C}_ce5" \
                    $COMMON --base ours --base_dist cosine --lambda_margin 1.0 --sel_criterion $C
                ;;
        esac
    done
done

echo "=== fin4-budget.sh finished ==="
