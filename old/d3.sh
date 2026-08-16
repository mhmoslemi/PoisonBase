#!/usr/bin/env bash
#
# Resume shard 3 of 4 for the killed dpp (SEL_ALPHA=2.0) sweep in dpp_crash.txt.
#
# Every flag below is copied verbatim from the "args:" line each of these runs
# logged before it was killed, so the resumed run is bit-identical to the one
# that died -- note --num_surrogates 20 (NOT the 5 in sel_dpp.sh) and
# --base_dist cosine --sel_dpp --sel_alpha 2.0.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: each run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.  Nothing already computed is redone,
# and no surrogate or clean victim is retrained (the cache keys are unchanged).
#
# Work left in this shard (measured from the timings in dpp_crash.txt):
#   ResNet20BN  b0.01   21 victim trials + 3 crafts   ~0.81 h
#   ResNet20BN  b0.02   19 victim trials + 3 crafts   ~0.71 h
#   VGG13BN     b0.001   7 victim trials + 1 craft    ~0.22 h
#   ---------------------------------------------------------------
#   estimated total: 1.75 h        (the other 3 shards: 1.70-1.75 h)
#
# Run the four shards on four separate GPUs:  sh d1.sh & sh d2.sh & ...

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# These runs are ~1.7 h of GPU work; on a CPU-only node they would silently
# crawl for days, which is how the earlier batch was lost.
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "d3.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

run () {   # run <model> <target_select> <budget>
    echo "=== d3 | dpp alpha=2.0 | $1 / fc / frog-airplane | budget $3 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model "$1" --attack fc --base ours \
        --class_pair frog-airplane --pair_order poison-target \
        --budget "$3" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 --craft_batch 256 \
        --base_dist cosine --lambda_margin 1.0 \
        --sel_dpp --sel_alpha 2.0 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 10 --target_select "$2" \
        --target_idx_file "target_sets/$1_fc_frog-airplane.json" \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline
}

run ResNet20BN  2 0.01
run ResNet20BN  2 0.02
run VGG13BN     3 0.001

echo "=== d3.sh finished ==="
