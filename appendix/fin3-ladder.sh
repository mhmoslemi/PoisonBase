#!/usr/bin/env bash
#
# 3 of 5 -- app-base.tex tab:selection-ladder (44 cells).
#
# 11 selection rules x {ConvNetBN, ResNet20BN} x {FC, GM} on bird->dog
# (--class_pair dog-bird) at a poison budget of 2e-3, 5 pinned targets, 5 victims.
#
# Eight of the eleven rules did not exist before this table and are now
# --sel_criterion in final_update.py: first, bottom, grand, el2n, boundary, pixel,
# featsim, relevance. Random is --base random, Greedy is --base ours, DPP adds
# --sel_dpp. Each rule gets its own run directory via the _sel<name> suffix, so
# nothing shares a poison_cache.
#
# All 20 surrogates are already cached for both architectures, so no run here
# pays surrogate cost. ~35 h; rerun until finished, completed runs skip themselves.
#
#   sh appendix/fin3-ladder.sh
#   DRY_RUN=1 sh appendix/fin3-ladder.sh
#
# NOTE ON NOTATION: this treats the caption's "eps = 2e-3" as the POISON BUDGET,
# matching how every other table in the paper reads eps (appendix.tex says
# "ConvNetBN, eps = 5e-3" for runs whose directories are b0.005). If you meant the
# L-infinity radius instead, swap --budget 0.002 for --epsilon 0.002 below and pick
# a budget.

set -u

DATA_PATH=/home/mmoslem3/scratch/data
SEED=42
NT=5
NV=5
BUDGET=0.002
PAIR=dog-bird
DRY_RUN="${DRY_RUN:-}"
# Sharding knobs, used by appendix/final. Defaults cover the whole table.
# CRITS is in table order; "random", "greedy" and "dpp" are handled specially,
# the rest map straight onto --sel_criterion.
MODELS="${MODELS:-ConvNetBN ResNet20BN}"
ATTACKS="${ATTACKS:-fc gradmatch}"
CRITS="${CRITS:-bottom first random grand el2n boundary pixel featsim relevance greedy dpp}"

cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "fin3: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

run() {   # $1 = run-dir name, $2.. = flags
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "ours_result/$TAG/log.txt" 2>/dev/null; then
        echo "--- done already: $TAG"; return 0; fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $* --target_idx_file $IDX"; return 0; fi
    python final_update.py "$@" --target_idx_file "$IDX" || exit 1
}

for MODEL in $MODELS; do

    IDX="target_sets/ladder_${MODEL}_${PAIR}.json"
    if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
        python appendix/pin_targets.py --model "$MODEL" --pair "$PAIR" \
            --target_select random --num_targets "$NT" --num_victims "$NV" \
            --out "$IDX" || exit 1
    fi

    COMMON="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
        --cache_dir ./cache --out_dir ours_result --pair_order poison-target \
        --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 --clean_baseline \
        --target_select random --num_targets $NT --num_victims $NV \
        --model $MODEL --class_pair $PAIR --budget $BUDGET"

    for ATT in $ATTACKS; do
        for C in $CRITS; do
            case "$C" in
                random)
                    LABEL="$MODEL | $ATT | Random"
                    run "CIFAR10_${MODEL}_${ATT}_random_${PAIR}_b${BUDGET}_eps8_seed${SEED}_ce5" \
                        $COMMON --attack $ATT --base random
                    ;;
                greedy)
                    LABEL="$MODEL | $ATT | Greedy"
                    run "CIFAR10_${MODEL}_${ATT}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_ce5" \
                        $COMMON --attack $ATT --base ours --base_dist cosine --lambda_margin 1.0
                    ;;
                dpp)
                    LABEL="$MODEL | $ATT | DPP"
                    run "CIFAR10_${MODEL}_${ATT}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_seldpp2_ce5" \
                        $COMMON --attack $ATT --base ours --base_dist cosine --lambda_margin 1.0 \
                        --sel_dpp --sel_alpha 2.0
                    ;;
                *)
                    LABEL="$MODEL | $ATT | $C"
                    run "CIFAR10_${MODEL}_${ATT}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_sel${C}_ce5" \
                        $COMMON --attack $ATT --base ours --base_dist cosine \
                        --lambda_margin 1.0 --sel_criterion $C
                    ;;
            esac
        done
    done
done

echo "=== fin3-ladder.sh finished ==="
