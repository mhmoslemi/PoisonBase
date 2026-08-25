#!/usr/bin/env bash
#
# make_figs_fast.sh -- the results-only figures. NO GPU, NO DATASET, ~10 seconds.
#
#     sh visualization/make_figs_fast.sh              # all three
#     FIGS=paired sh visualization/make_figs_fast.sh  # just one
#     REFRESH=1 sh visualization/make_figs_fast.sh    # rescan ours_result/ first
#     ATTACK=sapa sh visualization/make_figs_fast.sh  # panel (a) of 'anatomy'
#     DRY_RUN=1 sh visualization/make_figs_fast.sh
#
# These read only <out_dir>/*/results.csv -- the 20k victim trials the sweeps have
# already run -- through visualization/results_index.py, which caches a tidy index
# in figs/results_index.csv. They import matplotlib and nothing else heavy, so
# they run on a login node.
#
#   paired    fig_results_paired_scatter        one point per attacked target:
#             Random ASR vs DPP ASR on the same victim seeds. The point: base
#             selection helps target by target, not only in the mean.
#   budget    fig_results_budget_efficiency     how much more poison Random needs
#             to reach DPP's ASR. The point: selection is worth several times the
#             perturbation budget.
#   anatomy   fig_results_mechanism_anatomy     crafting objective vs ASR, gain vs
#             target difficulty, and clean accuracy. The point: the optimization
#             really does get easier, and it costs no accuracy.

set -u

FIGS="${FIGS:-paired budget anatomy}"
DRY_RUN="${DRY_RUN:-}"
REFRESH="${REFRESH:-}"
ATTACK="${ATTACK:-gradmatch}"
SELECTION_B="${SELECTION_B:-dpp2}"
SELECTION_A="${SELECTION_A:-random}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-$HERE/figs}"
LOGS="$OUT/logs"
RESULT_DIR="${RESULT_DIR:-$REPO/ours_result}"

cd "$REPO" || exit 1
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /home/mmoslem3/ENV/bin/activate ]; then
    # shellcheck disable=SC1091
    . /home/mmoslem3/ENV/bin/activate
fi

say() { printf '%s\n' "$*"; }
die() { printf '!! %s\n' "$*" >&2; exit 1; }

[ -d "$RESULT_DIR" ] || die "results directory not found: $RESULT_DIR"
mkdir -p "$OUT" "$LOGS"

COMMON="--results_dir $RESULT_DIR --index_csv $OUT/results_index.csv --out $OUT \
    --selection_a $SELECTION_A --selection_b $SELECTION_B"
[ -n "$REFRESH" ] && COMMON="$COMMON --refresh"

say "=== results-only figures (no gpu) ==="
say "    results   $RESULT_DIR"
say "    out       $OUT"
say "    arms      $SELECTION_A  vs  $SELECTION_B"
say "    figures   $FIGS"

run() {
    STEM="$1"; SCRIPT="$2"; shift 2
    say ""
    say "=== $STEM ==="
    if [ -n "$DRY_RUN" ]; then say "    python -u $HERE/$SCRIPT $*"; return 0; fi
    ( python -u "$HERE/$SCRIPT" "$@" 2>&1; echo $? > "$LOGS/.$STEM.status" ) \
        | tee "$LOGS/$STEM.log"
    ST="$(cat "$LOGS/.$STEM.status" 2>/dev/null || echo 1)"
    rm -f "$LOGS/.$STEM.status"
    [ "$ST" = 0 ] || die "$SCRIPT failed with status $ST (see $LOGS/$STEM.log)"
}

for F in $FIGS; do
case "$F" in
    paired)  run fig_results_paired_scatter    fig_results_paired_scatter.py    $COMMON ;;
    budget)  run fig_results_budget_efficiency fig_results_budget_efficiency.py $COMMON ;;
    anatomy) run fig_results_mechanism_anatomy fig_results_mechanism_anatomy.py $COMMON --attack "$ATTACK" ;;
    *) die "unknown figure '$F' (expected: paired | budget | anatomy)" ;;
esac
done

say ""
say "=== make_figs_fast.sh finished ==="
for S in fig_results_paired_scatter fig_results_budget_efficiency fig_results_mechanism_anatomy; do
    [ -s "$OUT/$S.pdf" ] && say "    $OUT/$S.pdf"
done
