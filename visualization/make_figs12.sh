#!/usr/bin/env bash
#
# make_figs12.sh -- rebuild ONLY the two selection figures (1 and 2), the revised
# three-way versions that include Random.
#
#     sh visualization/make_figs12.sh                  # both, from scratch
#     TARGET=8252 sh visualization/make_figs12.sh      # the one target that is in
#                                                      # BOTH the main-10 and the
#                                                      # appendix-5 target lists
#     PLOT_ONLY=1 sh visualization/make_figs12.sh      # redraw from the .npz, no gpu
#     FIGS=2 sh visualization/make_figs12.sh           # just figure 2
#     DRY_RUN=1 sh visualization/make_figs12.sh        # print the commands only
#
# Always rebuilds (no "already built" skip): this script exists to iterate on the
# two figures. Nothing here trains a model or crafts a poison -- figure 1 measures
# gradients on a held-out checkpoint, figure 2 only runs the three selectors.
#
# GPU: yes for the extraction (~4 min for figure 1 over the whole 5000-candidate
# pool, ~15 min for figure 2 over 10 targets). PLOT_ONLY=1 needs none.
#
# WHICH SCENARIO (read off the paper's own tables)
#   table.tex           the main sweep compares Random vs DPP only. The largest,
#                       cleanest gaps are ConvNet / dog-bird (bird->dog):
#                       GM 1.7 -> 50.0 and FC 13.3 -> 83.3 ASR at 5e-3.
#   ablation.tex        the alpha / lambda ablations are the only place Greedy
#                       appears next to DPP: ConvNetBN SAPA at 5e-3 gives
#                       Random 10.0, Greedy 60.0, DPP(alpha=2) 55.0 -- i.e. the
#                       big effect is Random -> target-conditioned selection,
#                       and Greedy vs DPP is a wash on ConvNetBN.
#   app-base.tex        the selection ladder is ConvNetBN, bird->dog, 5 targets,
#                       at 2e-3, and does carry both Greedy and DPP rows.
#
# So the defaults below are ConvNetBN / dog-bird / budget 5e-3, and the figures
# are framed as Random vs (Greedy, DPP). To match the ladder table instead:
#   BUDGET=0.002 TGT_FILE_FIG2=target_sets/ladder_ConvNetBN_dog-bird.json \
#       sh visualization/make_figs12.sh
# A smaller budget is also where Greedy and DPP have the most room to differ:
# at m=250 they picked 247 of the same 250 bases, which the figure now reports.
#
# What changed vs the first version
#   fig 1  Random / Greedy / DPP overlaid on the utility distribution and on the
#          score-vs-utility scatter; symlog axes (U is heavy tailed: ~half the
#          pool sits inside |U| < 0.01, which a linear axis renders as one bar);
#          rho reported on the score-INDEPENDENT sample only, plus the rho of the
#          two score terms separately; panel (c) shows Random bases too and
#          annotates inside the frames so nothing is clipped.
#   fig 2  Random added to all three panels, so the quality/redundancy trade-off
#          has a reference point; three C_S heatmaps on shared colour limits;
#          |S_greedy AND S_dpp| computed, printed and written to the csv.

set -u

FIGS="${FIGS:-1 2}"
DRY_RUN="${DRY_RUN:-}"
PLOT_ONLY="${PLOT_ONLY:-}"
ALLOW_CPU="${ALLOW_CPU:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-$HERE/figs}"
LOGS="$OUT/logs"

DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
[ -d "$DATA_PATH" ] || DATA_PATH="$REPO/data"

MODEL="${MODEL:-ConvNetBN}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird}"       # the paper's bird -> dog
SEED="${SEED:-42}"
SEL_K="${SEL_K:-20}"
BUDGET="${BUDGET:-0.005}"                  # m = 250
SEL_ALPHA="${SEL_ALPHA:-2.0}"
CACHE_DIR="${CACHE_DIR:-$REPO/cache}"
RESULT_DIR="${RESULT_DIR:-$REPO/ours_result}"

# 8252 is the only bird->dog target that appears in BOTH the main sweep's 10
# (target_sets/ConvNetBN_gradmatch_dog-bird.json) and the appendix / ladder 5
# (appx_broad, ladder), so all figures can show the same instance.
TARGET="${TARGET:-8252}"                   # fig 1's target; also fig 2's display
                                           # target when it is in fig 2's list
HELDOUT="${HELDOUT:-$CACHE_DIR/surrogates/${MODEL}_60ep_lr0.1_bs128_seed${SEED}/net_${SEL_K}.pt}"
NUM_CANDIDATES="${NUM_CANDIDATES:-0}"      # 0 = the whole 5000-candidate pool

cd "$REPO" || exit 1
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /home/mmoslem3/ENV/bin/activate ]; then
    # shellcheck disable=SC1091
    . /home/mmoslem3/ENV/bin/activate
fi

say() { printf '%s\n' "$*"; }
die() { printf '!! %s\n' "$*" >&2; exit 1; }

say "=== preflight ==="
say "    repo        $REPO"
say "    data        $DATA_PATH"
say "    out         $OUT"
say "    figures     $FIGS   (target $TARGET, budget $BUDGET, K=$SEL_K, alpha=$SEL_ALPHA)"
say "    scenario    ConvNetBN / $CLASS_PAIR -- the paper's main pair; Random vs Greedy vs DPP"
if [ -n "$PLOT_ONLY" ]; then
    say "    PLOT_ONLY   re-drawing from the cached .npz -- no gpu, no dataset"
fi

[ -d "$DATA_PATH" ] || die "dataset directory not found: $DATA_PATH (set DATA_PATH=...)"

if [ -z "$DRY_RUN" ] && [ -z "$PLOT_ONLY" ] && [ -z "$ALLOW_CPU" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        say "!! no CUDA device visible on $(hostname) -- figure 1 measures full"
        say "   parameter gradients and figure 2 embeds 5000 candidates x $SEL_K nets."
        say "   PLOT_ONLY=1 redraws from the saved .npz without a gpu."
        say "   (ALLOW_CPU=1 bypasses this check)"
        exit 1
    }
fi

SURR_DIR="$CACHE_DIR/surrogates/${MODEL}_60ep_lr0.1_bs128_seed${SEED}"
K_LAST=$((SEL_K - 1))
for i in 0 "$K_LAST"; do
    [ -s "$SURR_DIR/net_$i.pt" ] || \
        die "selector surrogate missing: $SURR_DIR/net_$i.pt -- this script loads, never trains"
done

case " $FIGS " in *" 1 "*)
    [ -s "$HELDOUT" ] || die "held-out checkpoint missing: $HELDOUT (set HELDOUT=...)"
    HO_ABS="$(cd "$(dirname "$HELDOUT")" && pwd)/$(basename "$HELDOUT")"
    I=0
    while [ "$I" -lt "$SEL_K" ]; do
        [ "$HO_ABS" = "$SURR_DIR/net_$I.pt" ] && \
            die "HELDOUT is selector surrogate net_$I -- figure 1 needs a model the selector never saw"
        I=$((I + 1))
    done
    ;;
esac

TGT_MAIN="$REPO/target_sets/${MODEL}_gradmatch_${CLASS_PAIR}.json"
TGT_APPX="$REPO/target_sets/appx_broad_${MODEL}_${CLASS_PAIR}.json"
FIG2_TGT="${TGT_FILE_FIG2:-$TGT_MAIN}"
case " $FIGS " in *" 2 "*) [ -s "$FIG2_TGT" ] || die "target list missing: $FIG2_TGT" ;; esac

mkdir -p "$OUT" "$LOGS"

COMMON="--dataset CIFAR10 --data_path $DATA_PATH --model $MODEL \
    --class_pair $CLASS_PAIR --pair_order poison-target --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $RESULT_DIR --sel_K $SEL_K \
    --lambda_margin 1.0 --base_dist cosine --sel_alpha $SEL_ALPHA --out $OUT"

MODE=""
[ -n "$PLOT_ONLY" ] && MODE="--plot_only"

run() {
    STEM="$1"; SCRIPT="$2"; shift 2
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

for F in $FIGS; do
case "$F" in

1)  run fig_intro_base_heterogeneity fig_intro_base_heterogeneity.py \
        $COMMON $MODE \
        --target_index "$TARGET" \
        --heldout_checkpoint "$HELDOUT" \
        --budget "$BUDGET" \
        --num_candidates "$NUM_CANDIDATES" \
        --cache_images "${CACHE_IMAGES:-16}" \
        --linthresh "${LINTHRESH:-0.01}"
    ;;

2)  # the display target must be in FIGURE 2's list, which is not the appendix
    # 5-target set TARGET usually comes from; 8252 is in both
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

*)  die "unknown figure '$F' for this script (expected 1 or 2)" ;;
esac
done

say ""
say "=== make_figs12.sh finished ==="
for S in fig_intro_base_heterogeneity fig_method_greedy_vs_dpp_geometry; do
    [ -s "$OUT/$S.pdf" ] && say "    $OUT/$S.pdf"
done
