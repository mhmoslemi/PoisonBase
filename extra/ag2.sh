#!/usr/bin/env bash
#
# ag2 -- ConvNet RandAugment, Random arm, both budgets (the two killed runs)
#
# Cells this fills in aug_table.tex:
#
#     ConvNetBN  b0.002  Random  RandAugment  13/20 done,  7 left
#     ConvNetBN  b0.01   Random  RandAugment   9/20 done, 11 left
#
# Both died on the Slurm wall clock (jobs 4832996 and 4832999, CANCELLED
# 20:02-20:03), not on an error, so every finished trial stays. Note the trials
# on disk include rows still sitting in results_rank0.csv that were never merged
# into results.csv -- defense.py merges them on the next run and resumes after.
#
# Protocol is the table's: the 5 pinned targets x victims 0-3 = 20 trials per
# cell. Nothing is crafted -- the saved perturbations are replayed from
# ours_result and the augmentation is resampled on top of them every epoch.
# aug.sh passes no --no_resume, so a partly-finished cell picks up where it left.
#
# COST:
#   18 trials x 359 s = 1.80 h.
#
#   359 s/trial is measured from these runs' own logs, ON AN H100. RandAugment is
#   augmentation-bound rather than GPU-bound (the same victim under Crop+Flip is
#   38 s), so do not expect an H100 to be much faster than an L40S here -- budget
#   the same ~1.8 h either way.
#
#   sh ag2.sh
#   DRY_RUN=1 sh ag2.sh    # print the defense.py commands and stop

set -u

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# --- pinning guard ------------------------------------------------------------
# aug.sh REWRITES target_sets/aug_ConvNetBN_fc_dog-bird_b<b>.json on every run,
# from the Random-DPP intersection of what is cached in ours_result, and takes
# its first 5. The Crop+Flip cells already in the table were scored on
# [2540, 3725, 3875, 5169, 5705]; if the intersection ever changed, this shard
# would silently score a different 5 and the row would stop being paired.
python - <<'PYCHK' || exit 1
import argparse, json, os, sys
import final_update as FU, defense as DEF
SCORED = [2540, 3725, 3875, 5169, 5705]
tgt = json.load(open('sweep_config.json'))['difficulty']['ConvNetBN']['fc']['dog-bird']
def ns(base, b, dpp):
    return argparse.Namespace(dataset='CIFAR10', model='ConvNetBN', attack='fc', base=base,
        class_pair='dog-bird', budget=b, epsilon=0.0313725, seed=42, lambda_margin=1.0,
        base_dist='cosine', sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=dpp,
        sel_pool=3.0, sel_mu=1.0, sel_alpha=2.0, fc_mode='sample', sharp_mode='worst',
        sharp_sigma=0.05, sharp_samples=20, craft_ensemble=5, target_select=int(tgt))
for b in [0.002, 0.01]:
    have = None
    for base, dpp in [('random', False), ('ours', True)]:
        d = os.path.join('ours_result', FU.build_run_name(ns(base, b, dpp)))
        ts = set(DEF.cached_targets(d)) if os.path.isdir(d) else set()
        have = ts if have is None else (have & ts)
    five = sorted(have)[:5]
    print('  b%g -> %d paired, first 5 = %s' % (b, len(have), five))
    if five != SCORED:
        sys.exit('  ABORT: b%g pins %s, but the finished cells were scored on %s'
                 % (b, five, SCORED))
PYCHK

MODEL=ConvNetBN ATTACK=fc CLASS_PAIR="dog-bird" \
    BUDGETS="0.002 0.01" SELS="random" PAIR_SELS="random dpp" \
    AUGS="randaug" NUM_TARGETS=5 NUM_VICTIMS=4 \
    sh ./aug.sh || exit 1

echo "=== ag2.sh finished ==="
