#!/usr/bin/env bash
#
# Everything the paper still needs, in one file.
#
#   sh finish.sh
#
# Run it on every GPU allocation you get. It skips whatever is already on disk,
# resumes whatever was killed mid-flight, and prints a status table at the end
# saying what is left. When that table is empty the experiments are done.
#
# Work is ordered cheapest-first, so a short allocation still closes whole cells
# rather than leaving six things half done. Two cells (GM and SAPA at 4e-2 on the
# greedy grid) take ~6.9 h each and will not fit one allocation -- rerun this file
# and they pick up per target.
#
# Nothing here is destructive and nothing takes arguments. Safe to run in parallel
# on several nodes: each run directory is lock-guarded, so a second process that
# reaches a directory already in progress stands down instead of corrupting it.
#
# What it fills, and where the numbers land:
#   latex/appendix.tex    tab:augmentation-aware   SAPA RandAugment DPP
#                         tab:cross-dataset        CIFAR-100 GM row
#                         tab:computational-cost   surrogate peak memory
#   latex/app-base.tex    tab:selection-ladder     5 remaining ResNet cells
#                         tab:selection-ladder-budget  1e-2 Bottom-m + derived row
#   latex/greedy_table.tex  15 remaining Greedy cells
#
# After a run: python3 <scratchpad>/mk_greedy_table.py regenerates greedy_table.tex.
# The other three .tex files are edited by hand from the numbers in the logs.

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "finish.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

# ---------------------------------------------------------------- 1. minutes ---
echo
echo "##################### 1. short leftovers #####################"

# tab:computational-cost, the surrogate peak-memory cell. profile_selection.py
# cannot supply it: it reads training time off the cache without ever training.
python appendix/surrogate_mem.py --model ConvNetBN || echo "  (memory probe failed; not fatal)"

# tab:augmentation-aware, the last cell. Resumes from its banked trials.
STEP=ra_sapa SELS=dpp sh appendix/ap4-augaware.sh

# ---------------------------------------------------------------- 2. ~1-3 h ---
echo
echo "##################### 2. selection ladder + budget #####################"

# tab:selection-ladder, the ResNet20BN cells still open. fin3 skips finished rules.
MODELS=ResNet20BN ATTACKS=fc       CRITS="boundary dpp"              sh appendix/fin3-ladder.sh
MODELS=ResNet20BN ATTACKS=gradmatch CRITS="bottom el2n relevance"    sh appendix/fin3-ladder.sh

# tab:selection-ladder-budget, the 1e-2 column's last rule.
BUDGETS=0.01 CRITS=bottom sh appendix/fin4-budget.sh

# ---------------------------------------------------------------- 3. ~5 h ----
echo
echo "##################### 3. CIFAR-100 cross-dataset #####################"

# tab:cross-dataset needs the GM/DPP arm on all five instances. The 20 surrogates
# are cached, so this reads them rather than retraining.
STEP=runs sh appendix/ap2-cifar100.sh

# ---------------------------------------------------------------- 4. ~40 h ---
echo
echo "##################### 4. greedy grid #####################"

# latex/greedy_table.tex. Budgets run cheapest-first inside greedy-convnet.sh, so
# a short allocation lands whole low-budget cells before touching the 4e-2 ones.
ATTACKS=gradmatch BUDGETS="0.001 0.002 0.005 0.01" sh appendix/greedy-convnet.sh
ATTACKS=sapa      BUDGETS="0.001 0.002 0.005 0.01" sh appendix/greedy-convnet.sh
ATTACKS=gradmatch BUDGETS="0.02 0.04"              sh appendix/greedy-convnet.sh
ATTACKS=sapa      BUDGETS="0.02 0.04"              sh appendix/greedy-convnet.sh

# ---------------------------------------------------------------- status -----
echo
echo "##################### what is still missing #####################"
python - <<'PYSTATUS'
import os, csv, glob, re
D = "/home/mmoslem3/scratch/attack_if"

def trials(rd, nv=6, nt=10):
    seen = {}
    for f in sorted(glob.glob(os.path.join(rd, "results*.csv"))):
        try:
            for r in csv.DictReader(open(f)):
                t, v, s = r.get("target_idx"), r.get("victim_id"), r.get("success")
                if not (t and v and s): continue
                if not re.fullmatch(r"\d+", t.strip()): continue
                if not re.fullmatch(r"\d+", v.strip()): continue
                if s.strip() not in ("0", "1", "0.0", "1.0"): continue
                seen[(int(t), int(v))] = 1
        except Exception:
            pass
    ts = sorted({k[0] for k in seen})[:nt]
    return sum(1 for k in seen if k[0] in ts and k[1] < nv)

left = []

# --- greedy grid (10 targets x 6 victims)
TGT = {("fc","dog-bird"):"50",("fc","frog-airplane"):"20",("gradmatch","dog-bird"):"70",
       ("gradmatch","frog-airplane"):"35",("sapa","dog-bird"):"70",("sapa","frog-airplane"):"35"}
for (att, pair), tgt in sorted(TGT.items()):
    sh = "_worst0.05" if att == "sapa" else ""
    for b in ("0.001","0.002","0.005","0.01","0.02","0.04"):
        n = "CIFAR10_ConvNetBN_%s_ours_%s_b%s_eps8_seed42_lam1_cosine%s_ce5_tgt%s" % (att,pair,b,sh,tgt)
        d = trials(os.path.join(D, "ours_result", n))
        if d < 60: left.append(("greedy  %-9s %-14s b%-6s" % (att, pair, b), "%d/60" % d))

# --- ladder + budget (5 targets x 5 victims)
LAD = [("ResNet20BN","fc","0.002","selboundary"), ("ResNet20BN","fc","0.002","seldpp2"),
       ("ResNet20BN","gradmatch","0.002","selbottom"), ("ResNet20BN","gradmatch","0.002","selel2n"),
       ("ResNet20BN","gradmatch","0.002","selrelevance"), ("ConvNetBN","gradmatch","0.01","selbottom")]
for model, att, b, tag in LAD:
    n = "CIFAR10_%s_%s_ours_dog-bird_b%s_eps8_seed42_lam1_cosine_%s_ce5" % (model, att, b, tag)
    d = trials(os.path.join(D, "ours_result", n), nv=5, nt=5)
    if d < 25: left.append(("ladder  %-9s %-14s b%-6s %s" % (att, model, b, tag[3:]), "%d/25" % d))

# --- cifar100 GM/DPP (1 target x 5 victims)
for p in ("sea-willow_tree","plain-bicycle","wardrobe-lawn_mower","bottle-road","sunflower-cattle"):
    n = "CIFAR100_ResNet18BN_gradmatch_ours_%s_b0.002_eps8_seed42_lam1_cosine_seldpp2_ce5" % p
    d = trials(os.path.join(D, "ours_result", n), nv=5, nt=1)
    if d < 5: left.append(("xdata   GM dpp  %-22s" % p, "%d/5" % d))

# --- table 13 last cell (5 x 5)
n = "CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2_craftnoaug_worst0.05_ce5__def-none+aug-randaug"
d = trials(os.path.join(D, "defense_result", n), nv=5, nt=5)
if d < 25: left.append(("augaware SAPA RandAugment dpp", "%d/25" % d))

if not left:
    print("  nothing left -- every experiment in the paper is on disk.")
else:
    for what, prog in left:
        print("  %-52s %s" % (what, prog))
    print("\n  %d unit(s) remaining. Rerun this file on the next allocation." % len(left))
PYSTATUS

echo "=== finish.sh finished ==="
