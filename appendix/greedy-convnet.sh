#!/usr/bin/env bash
#
# latex/greedy_table.tex -- the ConvNetBN Greedy column, on the main-table protocol.
#
# Greedy is --base ours with the pointwise score only: cosine base distance and
# lambda 1, no --sel_dpp and no --sel_criterion. Everything else is copied from the
# runs behind table.tex, so a Greedy cell is paired trial-for-trial with the Random
# and DPP cells above it: same pinned targets, same six victim seeds, same craft.
#
# SELECTION SURROGATES. The published Random/DPP cells were selected over K=20
# surrogates. Twelve Greedy cells already on disk were selected over K=5 (one over
# K=15), and K is not part of the run name -- so re-running them at K=20 would load
# the old K=5 poison cache and silently keep the old selection. This script refuses
# to touch such a directory and prints the mv you would need. Deciding whether to
# rerun them is a judgement call about the table, not something a script should make.
#
#   sh appendix/greedy-convnet.sh                              # everything missing
#   ATTACKS=sapa sh appendix/greedy-convnet.sh                 # one attack
#   BUDGETS="0.001 0.002" sh appendix/greedy-convnet.sh        # the cheap columns
#   PAIRS=frog-airplane BUDGETS=0.04 sh appendix/greedy-convnet.sh
#   SURROGATES=5 sh appendix/greedy-convnet.sh                 # match the old cells instead
#   DRY_RUN=1 sh appendix/greedy-convnet.sh
#
# Budgets run cheapest-first so a short allocation lands whole columns. Everything
# resumes: finished cells are skipped, and a killed cell keeps its banked trials and
# its already-crafted targets. A 4e-2 GM/SAPA cell is ~6.9 h on an L40S and will not
# fit one allocation -- rerun the same line and it picks up where it stopped.

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

DATA_PATH=/home/mmoslem3/scratch/data
CACHE_DIR=./cache
OUT_DIR=ours_result
SEED=42
MODEL=ConvNetBN
NT=10
NV=6
SURROGATES="${SURROGATES:-20}"

ATTACKS="${ATTACKS:-fc gradmatch sapa}"
PAIRS="${PAIRS:-dog-bird frog-airplane}"
BUDGETS="${BUDGETS:-0.001 0.002 0.005 0.01 0.02 0.04}"
DRY_RUN="${DRY_RUN:-}"

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "greedy-convnet.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

COMMON="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --model $MODEL \
    --pair_order poison-target --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 --craft_ensemble 5 \
    --num_surrogates $SURROGATES --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --num_targets $NT --num_victims $NV \
    --base ours --base_dist cosine --lambda_margin 1.0"

echo "=== ConvNetBN Greedy, main protocol (${NT} targets x ${NV} victims), K=$SURROGATES ==="
echo "    attacks : $ATTACKS"
echo "    pairs   : $PAIRS"
echo "    budgets : $BUDGETS"
echo

for B in $BUDGETS; do
echo "########## budget $B ##########"
for ATT in $ATTACKS; do
for PAIR in $PAIRS; do

    # the pinned target set and difficulty degree behind table.tex. SAPA reuses
    # GradMatch's, which is what the published SAPA rows were run against.
    case "$ATT" in
        fc)   IDXATT=fc ;;
        *)    IDXATT=gradmatch ;;
    esac
    case "$PAIR" in
        dog-bird)      case "$IDXATT" in fc) TS=50 ;; *) TS=70 ;; esac ;;
        frog-airplane) case "$IDXATT" in fc) TS=20 ;; *) TS=35 ;; esac ;;
    esac
    IDX="target_sets/${MODEL}_${IDXATT}_${PAIR}.json"
    case "$ATT" in sapa) SH="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05" ;;
                   *)    SH=""; SN="" ;; esac

    TAG="CIFAR10_${MODEL}_${ATT}_ours_${PAIR}_b${B}_eps8_seed${SEED}_lam1_cosine${SN}_ce5_tgt${TS}"
    LOG="$OUT_DIR/$TAG/log.txt"

    # K first, and for finished cells too: a complete cell selected over a different
    # K is not comparable with the Random/DPP rows above it, and its poison cache is
    # keyed by the run name only, so a rerun here would silently keep that selection.
    if [ -s "$LOG" ]; then
        HAVE=$(grep -o '"num_surrogates": [0-9]*' "$LOG" | head -1 | tr -d 'a-z_": ')
        if [ -n "$HAVE" ] && [ "$HAVE" != "$SURROGATES" ]; then
            echo "!!! skipped $ATT | $PAIR | b$B -- on disk it was selected over K=$HAVE, not K=$SURROGATES."
            echo "    To recraft at K=$SURROGATES:  mv $OUT_DIR/$TAG $OUT_DIR/$TAG.K$HAVE"
            echo "    To keep it instead, rerun this script with SURROGATES=$HAVE."
            continue
        fi
    fi

    # the summary line alone is not proof of a full cell: some of these directories
    # hold a finished 5x4 ablation run under the same name. Count the trials.
    DONE=$(python - "$OUT_DIR/$TAG" <<'PYCHK'
import sys, os, csv, glob, re
rd = sys.argv[1]; seen = {}
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
ts = sorted({k[0] for k in seen})[:10]
print(sum(1 for k in seen if k[0] in ts and k[1] < 6))
PYCHK
)
    if [ "${DONE:-0}" -ge 60 ]; then
        echo "--- done already: $ATT | $PAIR | b$B  (60/60)"; continue; fi
    [ "${DONE:-0}" -gt 0 ] && echo "    (resuming from $DONE/60 banked trials)"

    echo "=== $ATT | $PAIR | b$B  (tgt$TS) ==="
    if [ -n "$DRY_RUN" ]; then
        echo "    python final_update.py $COMMON --class_pair $PAIR --attack $ATT $SH \\"
        echo "        --budget $B --target_select $TS --target_idx_file $IDX"
        continue
    fi
    python final_update.py $COMMON --class_pair "$PAIR" --attack "$ATT" $SH \
        --budget "$B" --target_select "$TS" --target_idx_file "$IDX" || exit 1

done
done
done

echo "=== greedy-convnet.sh finished ==="
