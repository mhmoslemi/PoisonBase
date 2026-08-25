#!/usr/bin/env bash
#
# appendix.tex, tab:cross-dataset -- chunk 4 of 4.
#
# the remaining 19 TinyImageNet attack runs -- needs several allocations, rerun until it finishes
#
# ~70.1 h, from measured rates: a ResNet18BN TinyImageNet clean victim is 2327 s,
# a surrogate 2792 s (60 epochs vs the victim's 50), a craft at N_p=100 is 1666 s,
# so one attack run (1 craft + 5 victims) is 3.69 h.
#
# Every unit is skipped if already on disk, so a killed job just needs this same
# file rerun. Needs --mem=32G.
#
#   sh appendix/ap2-4.sh

set -u

DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
NV=5
TINY_MODEL=ResNet18BN
DRY_RUN="${DRY_RUN:-}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap2-4.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi
[ -s "$DATA_PATH/tinyimagenet.pt" ] || {
    echo "TinyImageNet not prepared -- python appendix/prep_tinyimagenet.py --src tiny-imagenet-200.zip"; exit 1; }

run() {
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "--- done already: $TAG"; return 0; fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $*"; return 0; fi
    python final_update.py "$@" || exit 1
}

TINY_BASE="--dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR \
    --model $TINY_MODEL --attack gradmatch --base random \
    --class_pair n01443537-n01629819 --pair_order poison-target \
    --budget 0.001 --epsilon 0.0313725 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_victims $NV --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0"
SDIR="$CACHE_DIR/surrogates/TinyImageNet_${TINY_MODEL}_60ep_lr0.1_bs128_seed$SEED"

echo
echo "########## TinyImageNet attack runs ##########"

PAIR=n01443537-n01629819
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | DPP"
run "TinyImageNet_${TINY_MODEL}_gradmatch_ours_n01443537-n01629819_b0.001_eps8_seed42_lam1_cosine_seldpp2_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0  --target_idx_file "$IDX"

PAIR=n01443537-n01629819
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | RANDOM"
run "TinyImageNet_${TINY_MODEL}_sapa_random_n01443537-n01629819_b0.001_eps8_seed42_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base random --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01443537-n01629819
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | DPP"
run "TinyImageNet_${TINY_MODEL}_sapa_ours_n01443537-n01629819_b0.001_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01641577-n01644900
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | RANDOM"
run "TinyImageNet_${TINY_MODEL}_gradmatch_random_n01641577-n01644900_b0.001_eps8_seed42_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base random  --target_idx_file "$IDX"

PAIR=n01641577-n01644900
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | DPP"
run "TinyImageNet_${TINY_MODEL}_gradmatch_ours_n01641577-n01644900_b0.001_eps8_seed42_lam1_cosine_seldpp2_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0  --target_idx_file "$IDX"

PAIR=n01641577-n01644900
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | RANDOM"
run "TinyImageNet_${TINY_MODEL}_sapa_random_n01641577-n01644900_b0.001_eps8_seed42_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base random --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01641577-n01644900
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | DPP"
run "TinyImageNet_${TINY_MODEL}_sapa_ours_n01641577-n01644900_b0.001_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01698640-n01742172
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | RANDOM"
run "TinyImageNet_${TINY_MODEL}_gradmatch_random_n01698640-n01742172_b0.001_eps8_seed42_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base random  --target_idx_file "$IDX"

PAIR=n01698640-n01742172
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | DPP"
run "TinyImageNet_${TINY_MODEL}_gradmatch_ours_n01698640-n01742172_b0.001_eps8_seed42_lam1_cosine_seldpp2_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0  --target_idx_file "$IDX"

PAIR=n01698640-n01742172
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | RANDOM"
run "TinyImageNet_${TINY_MODEL}_sapa_random_n01698640-n01742172_b0.001_eps8_seed42_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base random --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01698640-n01742172
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | DPP"
run "TinyImageNet_${TINY_MODEL}_sapa_ours_n01698640-n01742172_b0.001_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01768244-n01770081
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | RANDOM"
run "TinyImageNet_${TINY_MODEL}_gradmatch_random_n01768244-n01770081_b0.001_eps8_seed42_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base random  --target_idx_file "$IDX"

PAIR=n01768244-n01770081
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | DPP"
run "TinyImageNet_${TINY_MODEL}_gradmatch_ours_n01768244-n01770081_b0.001_eps8_seed42_lam1_cosine_seldpp2_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0  --target_idx_file "$IDX"

PAIR=n01768244-n01770081
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | RANDOM"
run "TinyImageNet_${TINY_MODEL}_sapa_random_n01768244-n01770081_b0.001_eps8_seed42_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base random --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01768244-n01770081
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | DPP"
run "TinyImageNet_${TINY_MODEL}_sapa_ours_n01768244-n01770081_b0.001_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01774384-n01774750
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | RANDOM"
run "TinyImageNet_${TINY_MODEL}_gradmatch_random_n01774384-n01774750_b0.001_eps8_seed42_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base random  --target_idx_file "$IDX"

PAIR=n01774384-n01774750
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | GRADMATCH | DPP"
run "TinyImageNet_${TINY_MODEL}_gradmatch_ours_n01774384-n01774750_b0.001_eps8_seed42_lam1_cosine_seldpp2_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack gradmatch --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0  --target_idx_file "$IDX"

PAIR=n01774384-n01774750
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | RANDOM"
run "TinyImageNet_${TINY_MODEL}_sapa_random_n01774384-n01774750_b0.001_eps8_seed42_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base random --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

PAIR=n01774384-n01774750
IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
        --pair "$PAIR" --target_select random --num_targets 1 --num_victims $NV --out "$IDX" || exit 1
fi
LABEL="TinyImageNet | $PAIR | SAPA | DPP"
run "TinyImageNet_${TINY_MODEL}_sapa_ours_n01774384-n01774750_b0.001_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5" \
    --dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model "$TINY_MODEL" --class_pair "$PAIR" --budget 0.001 \
    --attack sapa --base ours --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 --sharp_mode worst --sharp_sigma 0.05 --target_idx_file "$IDX"

echo "=== ap2-4.sh finished ==="
