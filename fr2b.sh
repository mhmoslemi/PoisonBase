#!/usr/bin/env bash
#
# tab:defense-robustness -- FRIENDS / GM / frog-airplane / $2{\times}10^{-2}$ / Random only
#
# Shard 2 of 3, split out of the old fr2.sh (which was 4 runs / ~12.8 h in one
# file). Together fr2a + fr2b + fr2c fill the four FRIENDS cells of the two
# frog-airplane GM rows:
#
#     frog--airplane & GM & $2{\times}10^{-3}$ & .. & .. & .. & .. & <Random> & <DPP>   <- fr2a
#     frog--airplane & GM & $2{\times}10^{-2}$ & .. & .. & .. & .. & <Random> & ..      <- fr2b
#     frog--airplane & GM & $2{\times}10^{-2}$ & .. & .. & .. & .. & ..       & <DPP>   <- fr2c
#
# Nothing is crafted -- the saved perturbations are replayed from ours_result/.
# The No Defense columns are not run here; they are recomputed from the same 5
# targets and victims 0-3.
#
# PROTOCOL: 5 targets x 4 victims = 20 trials per run, matching everything
# already in defense_result.txt (NUM_VICTIMS=4 is passed explicitly --
# defense.sh's own default is 6, which would be a different protocol).
#
# COST, from rates measured in defense_result.txt: FRIENDS is 517 s/trial plus a
# 61 s friendly-noise pass per trial = 578 s.
#   1 run(s) x 20 trials x 578 s
#   L40S ~3.2 h      H100 ~1.8 h  (at ~1.8x; extrapolated, not measured)
#
# defense.py resumes from results.csv, so a wall-clock kill just needs a rerun.
#
#   sh fr2b.sh
#   DRY_RUN=1 sh fr2b.sh    # print the defense.py command and stop

sh "$(dirname "$0")/preflight_cuda.sh" || exit 1

# the guard below imports final_update/defense, so it needs the venv and the
# project dir. preflight_cuda.sh sources the venv too, but in its own `sh`
# process, so that activation does not reach this shell.
source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# --- pinning guard ------------------------------------------------------------
# defense.sh pins the targets to the INTERSECTION over whatever SELS it is given
# and writes them to a shared per-combo file,
# target_sets/def_ResNet20BN_gradmatch_frog-airplane_b0.02.json. This shard runs
# ONE selection, so that file would be rewritten with just this selection's
# targets. That is only safe while Random and DPP have identical cached target
# sets -- which they do today (10 = 10, intersection 10). Check it rather than
# assume it: if someone crafts extra poisons for one selection later, the two
# cells of this row would silently stop being paired.
python - <<'PYCHK' || exit 1
import argparse, os, sys
import final_update as FU, defense as DEF
def ns(base, dpp):
    return argparse.Namespace(dataset='CIFAR10', model='ResNet20BN', attack='gradmatch',
        base=base, class_pair='frog-airplane', budget=0.02, epsilon=0.0313725, seed=42,
        lambda_margin=1.0, base_dist='cosine', sel_filter=False, sel_pca=False,
        sel_mmr=False, sel_dpp=dpp, sel_pool=3.0, sel_mu=1.0, sel_alpha=2.0,
        fc_mode='sample', sharp_mode='worst', sharp_sigma=0.05, sharp_samples=20,
        craft_ensemble=5, target_select=10)
sets = {}
for sel, (base, dpp) in [('random', ('random', False)), ('dpp', ('ours', True))]:
    d = os.path.join('ours_result', FU.build_run_name(ns(base, dpp)))
    sets[sel] = sorted(DEF.cached_targets(d)) if os.path.isdir(d) else []
if sets['random'] != sets['dpp']:
    sys.exit('pinning guard FAILED: random has %d cached targets, dpp has %d, and '
             'they are not the same set. Run both selections in ONE defense.sh '
             'call (SELS="random dpp") so the intersection is pinned once.'
             % (len(sets['random']), len(sets['dpp'])))
print('pinning guard ok: random and dpp share the same %d targets' % len(sets['random']))
PYCHK

MODEL="${MODEL:-ResNet20BN}" \
ATTACK=gradmatch \
CLASS_PAIR="frog-airplane" \
BUDGETS="0.02" \
SELS="random" \
DEFENSES="friends" \
NUM_VICTIMS="${NUM_VICTIMS:-4}" \
sh "$(dirname "$0")/defense.sh" || exit 1

echo "=== fr2b.sh finished ==="
