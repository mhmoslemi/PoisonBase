#!/usr/bin/env bash
#
# Everything still missing from the paper's tables, in one file.
#
# table.tex, cross_arch.tex, defense_table.tex and aug_table.tex are complete, and
# so is the alpha ablation. What is left is 13 cells of ablation.tex:
#
#   tab:lambda-ablation        12 cells
#   tab:ensemble-size-ablation  1 cell   (ConvNetBN 5e-3, K=5)
#
# appendix.tex is deliberately NOT covered -- those runs live in appendix/, and the
# stale copies under extra/ are not part of any current table.
#
# Two of the 13 are partly on disk from allocations that died, and resume rather
# than restart:
#   ResNet20BN 5e-3 lambda=100 Greedy   16/20 trials, 4/5 crafts
#   ConvNetBN  5e-3 K=5                 14/20 trials, 4/5 crafts
#
# PROTOCOL: 5 targets x 4 victims = 20 trials, targets pinned to the first 5 of
# target_sets/<model>_gradmatch_dog-bird.json in file order -- the same protocol
# every filled cell of ablation.tex already uses.
#
# COST on an L40S: ~12.2 h, from craft and victim rates measured in these very runs
# (ConvNet 242 s/craft and 50 s/trial at 5e-3, 519 s/craft at 1e-2; ResNet20 472 s
# and 130 s at 5e-3). Ordered cheapest first, so the short ones land early.
#
# Resumable: a run killed on the wall clock picks up at its first missing trial, so
# just run this file again. Completed runs are skipped in a second.
#
#   sh finish.sh
#   DRY_RUN=1 sh finish.sh    # print the commands and stop

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
NT="${NUM_TARGETS:-5}"
NV="${NUM_VICTIMS:-4}"
DRY_RUN="${DRY_RUN:-}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "finish.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

run() {   # $1 = run tag, $2.. = the flags that differ between cells
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "--- already complete: $TAG"
        return 0
    fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then
        echo "    python final_update.py $* (tag $TAG)"
        return 0
    fi
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model "$M" --attack sapa --base ours \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$B" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --base_dist cosine \
        --sharp_mode worst --sharp_sigma 0.05 \
        --num_surrogates "$NSUR" --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets "$NT" --target_select "$TGT" \
        --target_idx_file "target_sets/${M}_gradmatch_dog-bird.json" \
        --num_victims "$NV" --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline "$@" || exit 1
}

M=ConvNetBN; B=0.005; TGT=70; NSUR=20
LABEL="ConvNetBN b0.005 | K=5  (~0.15 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2_K5_worst0.05_ce5_tgt70 \
    --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 --sel_K 5

M=ResNet20BN; B=0.005; TGT=14; NSUR=20
LABEL="ResNet20BN b0.005 | lambda=100 Greedy  (~0.28 h)"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam100_cosine_worst0.05_ce5_tgt14 \
    --lambda_margin 100

M=ConvNetBN; B=0.005; TGT=70; NSUR=20
LABEL="ConvNetBN b0.005 | lambda=4 Greedy  (~0.61 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam4_cosine_worst0.05_ce5_tgt70 \
    --lambda_margin 4

M=ConvNetBN; B=0.005; TGT=70; NSUR=20
LABEL="ConvNetBN b0.005 | lambda=100 Greedy  (~0.61 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam100_cosine_worst0.05_ce5_tgt70 \
    --lambda_margin 100

M=ConvNetBN; B=0.005; TGT=70; NSUR=20
LABEL="ConvNetBN b0.005 | lambda=100 DPP  (~0.61 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam100_cosine_seldpp2_worst0.05_ce5_tgt70 \
    --lambda_margin 100 --sel_dpp --sel_alpha 2.0

M=ConvNetBN; B=0.01; TGT=70; NSUR=20
LABEL="ConvNetBN b0.01 | lambda=2 DPP  (~1.00 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam2_cosine_seldpp2_worst0.05_ce5_tgt70 \
    --lambda_margin 2 --sel_dpp --sel_alpha 2.0

M=ConvNetBN; B=0.01; TGT=70; NSUR=20
LABEL="ConvNetBN b0.01 | lambda=4 Greedy  (~1.00 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam4_cosine_worst0.05_ce5_tgt70 \
    --lambda_margin 4

M=ConvNetBN; B=0.01; TGT=70; NSUR=20
LABEL="ConvNetBN b0.01 | lambda=4 DPP  (~1.00 h)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam4_cosine_seldpp2_worst0.05_ce5_tgt70 \
    --lambda_margin 4 --sel_dpp --sel_alpha 2.0

M=ResNet20BN; B=0.005; TGT=14; NSUR=20
LABEL="ResNet20BN b0.005 | lambda=0 Greedy  (~1.38 h)"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam0_cosine_worst0.05_ce5_tgt14 \
    --lambda_margin 0

M=ResNet20BN; B=0.005; TGT=14; NSUR=20
LABEL="ResNet20BN b0.005 | lambda=0 DPP  (~1.38 h)"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam0_cosine_seldpp2_worst0.05_ce5_tgt14 \
    --lambda_margin 0 --sel_dpp --sel_alpha 2.0

M=ResNet20BN; B=0.005; TGT=14; NSUR=20
LABEL="ResNet20BN b0.005 | lambda=0.5 Greedy  (~1.38 h)"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam0.5_cosine_worst0.05_ce5_tgt14 \
    --lambda_margin 0.5

M=ResNet20BN; B=0.005; TGT=14; NSUR=20
LABEL="ResNet20BN b0.005 | lambda=0.5 DPP  (~1.38 h)"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam0.5_cosine_seldpp2_worst0.05_ce5_tgt14 \
    --lambda_margin 0.5 --sel_dpp --sel_alpha 2.0

M=ResNet20BN; B=0.005; TGT=14; NSUR=20
LABEL="ResNet20BN b0.005 | lambda=2 Greedy  (~1.38 h)"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam2_cosine_worst0.05_ce5_tgt14 \
    --lambda_margin 2

echo "=== finish.sh done -- all 13 remaining ablation.tex cells ==="
