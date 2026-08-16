#!/usr/bin/env bash
#
# cv1 -- ConvNetBN augmentation shard: GM / dog-bird
#
# Fills these rows of aug_table_convnet.tex (Random and DPP side by side, both
# budgets, all three augmentations):
#
#     dog--bird & GM & $2{\times}10^{-3}$ & Random/DPP & .. & <Crop+Flip> <RandAugment> <Cutout>
#     dog--bird & GM & $2{\times}10^{-2}$ & Random/DPP & .. & <Crop+Flip> <RandAugment> <Cutout>
#
# RUN cv0-1.sh FIRST (it is the half that covers this pair). The Random arm has no cached poisons for this combo, and
# aug.sh pins targets to the Random-DPP intersection -- without it this shard
# reports "no target has poisons under every selection" and does nothing.
# The check below fails fast rather than letting that look like success.
#
# Same protocol as the ResNet20BN table: 5 targets x 4 victims = 20 trials per
# cell, targets pinned over BOTH selections, No Aug. not rerun (it is recomputed
# from ours_result by make_aug_table.py over those same targets and victims).
#
# COST: 2 budgets x 3 augs x 2 selections = 12 runs x 20 trials x ~54 s
#       (ConvNet victim ~50 s undefended, x1.08 for augmented, as measured on
#        ResNet) -> L40S ~3.6 h    H100 ~2.0 h  (at ~1.8x; extrapolated)
#
#   sh cv1.sh
#   DRY_RUN=1 sh cv1.sh    # print the defense.py commands and stop

set -u

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python - <<'PYCHK' || exit 1
import argparse, os, sys, json
import final_update as FU, defense as DEF
tgt = json.load(open('sweep_config.json'))['difficulty']['ConvNetBN']['gradmatch']['dog-bird']
def ns(base, b, dpp):
    return argparse.Namespace(dataset='CIFAR10', model='ConvNetBN', attack='gradmatch',
        base=base, class_pair='dog-bird', budget=b, epsilon=0.0313725, seed=42,
        lambda_margin=1.0, base_dist='cosine', sel_filter=False, sel_pca=False,
        sel_mmr=False, sel_dpp=dpp, sel_pool=3.0, sel_mu=1.0, sel_alpha=2.0,
        fc_mode='sample', sharp_mode='worst', sharp_sigma=0.05, sharp_samples=20,
        craft_ensemble=5, target_select=int(tgt))
bad = []
for b in (0.002, 0.02):
    have = {}
    for sel, (base, dpp) in [('random', ('random', False)), ('dpp', ('ours', True))]:
        d = os.path.join('ours_result', FU.build_run_name(ns(base, b, dpp)))
        have[sel] = set(DEF.cached_targets(d)) if os.path.isdir(d) else set()
    n = len(have['random'] & have['dpp'])
    print('  b%-6g random=%2d dpp=%2d -> %2d paired' % (b, len(have['random']), len(have['dpp']), n))
    # must be the FULL dpp set, not just >=5. aug.sh pins the SORTED intersection
    # and takes its first 5; crafting runs in the target_sets file order, which is
    # not sorted order, so a partially-crafted Random arm yields a different
    # 'first 5' than the finished one -- the shard would score a target set that
    # no later regeneration reproduces.
    if n < len(have['dpp']) or n < 5:
        bad.append(b)
if bad:
    sys.exit('cv1: budget(s) %s do not have the full paired target set yet -- let '
             'cv0-1.sh finish crafting the Random arm first. aug.sh pins the SORTED '
             'intersection and takes its first 5, and crafting runs in target-file '
             'order, so a partial Random arm makes this shard score a target set the '
             'finished table never reproduces.' % bad)
PYCHK

MODEL=ConvNetBN ATTACK=gradmatch CLASS_PAIR="dog-bird" \
    BUDGETS="0.002 0.02" SELS="random dpp" PAIR_SELS="random dpp" \
    AUGS="standard randaug cutout" NUM_TARGETS=5 NUM_VICTIMS=4 \
    sh ./aug.sh || exit 1

echo "=== cv1.sh finished ==="
