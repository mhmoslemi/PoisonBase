#!/usr/bin/env bash
#
# ca2 -- ConvNetBN / FC / dog-bird / $2{\times}10^{-3}$ / DPP
#
# Fills these 3 cells of aug_table.tex (the ConvNetBN block):
#
#     dog--bird & FC & $2{\times}10^{-3}$ & DPP & (No Aug.) & <Crop+Flip> <RandAugment> <Cutout>
#
# Protocol is the table's: 5 targets x 4 victims = 20 trials per cell, targets
# pinned over BOTH selections, No Aug. not rerun (make_aug_table.py recomputes it
# from ours_result over the same targets and victims).
#
# Nothing is crafted -- the saved perturbations are replayed from ours_result and
# the augmentation is resampled on top of them every epoch.
#
# COST: 3 augmentations x 20 trials x ~53 s (measured from the ConvNet aug runs
# already in defense_result) = ~0.9 h, minus whatever is already on disk:
#     standard    4/20 done
#     randaug     0/20 done
#     cutout      0/20 done
#   L40S ~0.8 h    H100 ~0.5 h  (at ~1.8x; extrapolated)
#
# aug.sh passes no --no_resume, so a partly-finished cell resumes.
#
#   sh ca2.sh
#   DRY_RUN=1 sh ca2.sh    # print the defense.py commands and stop

set -u

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# --- pinning guard ------------------------------------------------------------
# aug.sh pins the SORTED Random-DPP intersection and takes its first 5. This
# shard runs ONE selection, so it must still pin over BOTH (PAIR_SELS below) or
# it would score a different 5 than the other half of the row. Check that the
# Random arm is fully crafted before starting -- a partial arm changes the
# intersection and therefore which 5 targets are scored.
python - <<'PYCHK' || exit 1
import argparse, os, sys, json
import final_update as FU, defense as DEF
tgt = json.load(open('sweep_config.json'))['difficulty']['ConvNetBN']['fc']['dog-bird']
def ns(base, dpp):
    return argparse.Namespace(dataset='CIFAR10', model='ConvNetBN', attack='fc', base=base,
        class_pair='dog-bird', budget=0.002, epsilon=0.0313725, seed=42, lambda_margin=1.0,
        base_dist='cosine', sel_filter=False, sel_pca=False, sel_mmr=False, sel_dpp=dpp,
        sel_pool=3.0, sel_mu=1.0, sel_alpha=2.0, fc_mode='sample', sharp_mode='worst',
        sharp_sigma=0.05, sharp_samples=20, craft_ensemble=5, target_select=int(tgt))
have = {}
for sel, (base, dpp) in [('random', ('random', False)), ('dpp', ('ours', True))]:
    d = os.path.join('ours_result', FU.build_run_name(ns(base, dpp)))
    have[sel] = set(DEF.cached_targets(d)) if os.path.isdir(d) else set()
n = len(have['random'] & have['dpp'])
print('  b0.002  random=%d dpp=%d -> %d paired' % (len(have['random']), len(have['dpp']), n))
if n < len(have['dpp']) or n < 5:
    sys.exit('ca2: the Random arm is not fully crafted for b0.002 (%d of %d paired). '
             'Run cv0-1.sh first.' % (n, len(have['dpp'])))
PYCHK

MODEL=ConvNetBN ATTACK=fc CLASS_PAIR="dog-bird" \
    BUDGETS="0.002" SELS="dpp" PAIR_SELS="random dpp" \
    AUGS="standard randaug cutout" NUM_TARGETS=5 NUM_VICTIMS=4 \
    sh ./aug.sh || exit 1

echo "=== ca2.sh finished ==="
