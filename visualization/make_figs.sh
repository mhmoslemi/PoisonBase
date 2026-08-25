#!/usr/bin/env bash
#
# make_figs.sh -- build every paper figure in visualization/ with one command.
#
#     sh visualization/make_figs.sh                # figures 1, 2, 3, 4
#     FIGS="1 2" sh visualization/make_figs.sh     # only some of them
#     DRY_RUN=1 sh visualization/make_figs.sh      # print the commands, run nothing
#     PLOT_ONLY=1 sh visualization/make_figs.sh    # redraw from the saved .npz (cpu ok)
#     FORCE=1 sh visualization/make_figs.sh        # redo figures whose pdf exists
#
# GPU: YES for the extraction pass of every figure (gradients, surrogate
# ensembles, poison crafting, victim training). PLOT_ONLY=1 needs no GPU -- it
# only reads the .npz files and the dataset. ALLOW_CPU=1 lifts the check, but
# figure 3 on cpu is not worth starting.
#
# Rough cost on one L40S, at the paper settings:
#     fig 1   ~5 min     1000 full-parameter gradients + one K=20 selector pass
#     fig 2   ~15 min    10 targets x (K=20 features + Greedy + DPP) + one PCA
#     fig 3   HOURS      30 crafts (5 targets x 3 selections x 2 attacks)
#                        at 8 restarts x 250 steps -- the dominant cost by far
#     fig 4   ~40 min    2 victims x 50 epochs, poisons loaded from the runs
#     fig 5   ~45 min    same two victims, plus per-poison gradient alignment
#                        (the EPIC-style dynamics figure)
#
# Everything is resumable: each figure writes a .npz next to its pdf, and a
# figure whose pdf already exists is skipped unless FORCE=1.

set -u

FIGS="${FIGS:-1 2 3 4 5}"
DRY_RUN="${DRY_RUN:-}"
FORCE="${FORCE:-}"
PLOT_ONLY="${PLOT_ONLY:-}"
ALLOW_CPU="${ALLOW_CPU:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-$HERE/figs}"
LOGS="$OUT/logs"

# same dataset location every script in this repo uses
DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
[ -d "$DATA_PATH" ] || DATA_PATH="$REPO/data"

MODEL="${MODEL:-ConvNetBN}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird}"       # the paper's bird -> dog
SEED="${SEED:-42}"
SEL_K="${SEL_K:-20}"
BUDGET="${BUDGET:-0.005}"                  # the appendix's "eps = 5e-3"
CACHE_DIR="${CACHE_DIR:-$REPO/cache}"
RESULT_DIR="${RESULT_DIR:-$REPO/ours_result}"

# one of the five pinned bird->dog targets of the reduced appendix protocol
# (target_sets/appx_broad_ConvNetBN_dog-bird.json). NOT chosen by attack success.
TARGET="${TARGET:-3741}"
# a surrogate outside the K = 20 selector ensemble; the cached pool holds 50
HELDOUT="${HELDOUT:-$CACHE_DIR/surrogates/${MODEL}_60ep_lr0.1_bs128_seed${SEED}/net_${SEL_K}.pt}"
VICTIM_ID="${VICTIM_ID:-0}"
FIG4_ATTACK="${FIG4_ATTACK:-gradmatch}"

cd "$REPO" || exit 1
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /home/mmoslem3/ENV/bin/activate ]; then
    # shellcheck disable=SC1091
    . /home/mmoslem3/ENV/bin/activate
fi

say() { printf '%s\n' "$*"; }
die() { printf '!! %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# preflight: everything these scripts LOAD must already exist. They never train
# a surrogate, never craft a poison outside figure 3, and never invent a file.
# --------------------------------------------------------------------------- #

say "=== preflight ==="
say "    repo        $REPO"
say "    data        $DATA_PATH"
say "    cache       $CACHE_DIR"
say "    out         $OUT"
say "    figures     $FIGS"

[ -d "$DATA_PATH" ] || die "dataset directory not found: $DATA_PATH (set DATA_PATH=...)"

if [ -z "$DRY_RUN" ] && [ -z "$PLOT_ONLY" ] && [ -z "$ALLOW_CPU" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        say "!! no CUDA device visible on $(hostname) -- these figures extract"
        say "   gradients, surrogate features, poisons and victims, all on gpu."
        say "   get an allocation first, e.g."
        say "     salloc --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=4 \\"
        say "            --mem=32G --time=12:00:00"
        say "   PLOT_ONLY=1 redraws from the saved .npz without a gpu."
        say "   (ALLOW_CPU=1 bypasses this check)"
        exit 1
    }
fi

SURR_DIR="$CACHE_DIR/surrogates/${MODEL}_60ep_lr0.1_bs128_seed${SEED}"
K_LAST=$((SEL_K - 1))
for i in 0 "$K_LAST"; do
    [ -s "$SURR_DIR/net_$i.pt" ] || \
        die "selector surrogate missing: $SURR_DIR/net_$i.pt -- these scripts load, never train"
done
case " $FIGS " in *" 1 "*)
    [ -s "$HELDOUT" ] || die "held-out checkpoint missing: $HELDOUT (set HELDOUT=...)"
    # the real guard (path + parameter fingerprint) is inside the python script;
    # this one just fails in 1 s instead of after loading the dataset
    HO_ABS="$(cd "$(dirname "$HELDOUT")" && pwd)/$(basename "$HELDOUT")"
    I=0
    while [ "$I" -lt "$SEL_K" ]; do
        if [ "$HO_ABS" = "$SURR_DIR/net_$I.pt" ]; then
            die "HELDOUT is selector surrogate net_$I -- figure 1 needs a model the selector never saw"
        fi
        I=$((I + 1))
    done
    ;;
esac

TGT_APPX="$REPO/target_sets/appx_broad_${MODEL}_${CLASS_PAIR}.json"
TGT_MAIN="$REPO/target_sets/${MODEL}_gradmatch_${CLASS_PAIR}.json"
case " $FIGS " in *" 2 "*) [ -s "$TGT_MAIN" ] || die "target list missing: $TGT_MAIN" ;; esac
case " $FIGS " in *" 3 "*) [ -s "$TGT_APPX" ] || die "target list missing: $TGT_APPX" ;; esac

mkdir -p "$OUT" "$LOGS"

COMMON="--dataset CIFAR10 --data_path $DATA_PATH --model $MODEL \
    --class_pair $CLASS_PAIR --pair_order poison-target --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $RESULT_DIR --sel_K $SEL_K \
    --lambda_margin 1.0 --base_dist cosine --sel_alpha 2.0 --out $OUT"

MODE=""
[ -n "$PLOT_ONLY" ] && MODE="--plot_only"

# run <stem> <script> <args...>
run() {
    STEM="$1"; SCRIPT="$2"; shift 2
    if [ -s "$OUT/$STEM.pdf" ] && [ -z "$FORCE" ] && [ -z "$PLOT_ONLY" ]; then
        say "--- already built, skipping: $OUT/$STEM.pdf   (FORCE=1 to redo)"
        return 0
    fi
    say ""
    say "=== $STEM ==="
    if [ -n "$DRY_RUN" ]; then
        say "    python -u $HERE/$SCRIPT $*"
        return 0
    fi
    T0=$(date +%s)
    # the exit status is carried through the pipe by hand: `sh` has no pipefail
    ( python -u "$HERE/$SCRIPT" "$@" 2>&1; echo $? > "$LOGS/.$STEM.status" ) \
        | tee "$LOGS/$STEM.log"
    ST="$(cat "$LOGS/.$STEM.status" 2>/dev/null || echo 1)"
    rm -f "$LOGS/.$STEM.status"
    [ "$ST" = 0 ] || die "$SCRIPT failed with status $ST (see $LOGS/$STEM.log)"
    say "    done in $(( ($(date +%s) - T0) / 60 )) min -> $OUT/$STEM.{pdf,png,csv,npz}"
}

# --------------------------------------------------------------------------- #

for F in $FIGS; do
case "$F" in

1)  run fig_intro_base_heterogeneity fig_intro_base_heterogeneity.py \
        $COMMON $MODE \
        --target_index "$TARGET" \
        --heldout_checkpoint "$HELDOUT" \
        --num_candidates "${NUM_CANDIDATES:-1000}"
    ;;

2)  # The display target must belong to figure 2's OWN target list (the 10 main
    # bird->dog targets), which is not the appendix's 5-target set that TARGET
    # comes from. Use TARGET when it happens to be in both -- 8252 is -- and
    # otherwise let the script fall back to the first target of the list.
    FIG2_TGT="${TGT_FILE_FIG2:-$TGT_MAIN}"
    DISP_FLAG=""
    if [ -n "${DISPLAY_TARGET:-}" ]; then
        DISP_FLAG="--display_target $DISPLAY_TARGET"
    elif grep -qE "^[[:space:]]*$TARGET,?[[:space:]]*$" "$FIG2_TGT" 2>/dev/null; then
        DISP_FLAG="--display_target $TARGET"
    fi
    run fig_method_greedy_vs_dpp_geometry fig_method_greedy_vs_dpp_geometry.py \
        $COMMON $MODE \
        --budget "$BUDGET" \
        --target_idx_file "$FIG2_TGT" \
        $DISP_FLAG
    ;;

3)  run fig_exp_optimization_trajectories fig_exp_optimization_trajectories.py \
        $COMMON $MODE \
        --budget "$BUDGET" \
        --target_idx_file "$TGT_APPX" \
        --attacks ${ATTACKS:-gradmatch sapa} \
        --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts "${RESTARTS:-8}" --craft_ensemble 5
    ;;

4)  # optional figure: it REPLAYS poisons the attack runs already saved, so it
    # is skipped (not failed) when those runs are not on disk yet
    MISS=""
    for D in "CIFAR10_${MODEL}_${FIG4_ATTACK}_random_${CLASS_PAIR}_b${BUDGET}_eps8_seed${SEED}" \
             "CIFAR10_${MODEL}_${FIG4_ATTACK}_ours_${CLASS_PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_seldpp2"; do
        ls -d "$RESULT_DIR/${D}"*"ce5" >/dev/null 2>&1 || MISS="$MISS $D"
    done
    if [ -n "$MISS" ]; then
        say ""
        say "--- figure 4 skipped: no saved poisons for$MISS"
        say "    run the attacks first (appendix/ap1-broad.sh builds exactly these),"
        say "    then:  FIGS=4 sh visualization/make_figs.sh"
        continue
    fi
    run fig_exp_representation_dynamics fig_exp_representation_dynamics.py \
        $COMMON $MODE \
        --budget "$BUDGET" --epsilon 0.0313725 --craft_ensemble 5 \
        --attack "$FIG4_ATTACK" --target_index "$TARGET" --victim_id "$VICTIM_ID" \
        --target_select random
    ;;

5)  # EPIC-style dynamics: replays saved poisons, so it skips when they are absent
    MISS=""
    for D in "CIFAR10_${MODEL}_${FIG4_ATTACK}_random_${CLASS_PAIR}_b${BUDGET}_eps8_seed${SEED}" \
             "CIFAR10_${MODEL}_${FIG4_ATTACK}_ours_${CLASS_PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_seldpp2"; do
        ls -d "$RESULT_DIR/${D}"*"ce5" >/dev/null 2>&1 || MISS="$MISS $D"
    done
    if [ -n "$MISS" ]; then
        say ""
        say "--- figure 5 skipped: no saved poisons for$MISS"
        continue
    fi
    run fig_poison_dynamics fig_poison_dynamics.py \
        $COMMON $MODE \
        --budget "$BUDGET" --epsilon 0.0313725 --craft_ensemble 5 \
        --attack "$FIG4_ATTACK" --target_index "$TARGET" --victim_id "$VICTIM_ID" \
        --target_select random --align_threshold "${ALIGN_THRESHOLD:-0}"
    ;;

*)  die "unknown figure '$F' (expected 1 2 3 4 5)" ;;
esac
done

say ""
say "=== make_figs.sh finished ==="
ls -1 "$OUT"/*.pdf 2>/dev/null | sed 's/^/    /'
