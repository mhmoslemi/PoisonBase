#!/usr/bin/env bash
# Runtime adapter for the defense-extra alpha sweep. It reuses the established
# one-cell staging/sync implementation while also importing the retained H200
# alpha=0.25 attack and defense shards when they exist.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
ALPHA025_SNAPSHOT="$SOURCE_ROOT/last_night_H200_2026-08-26"

export SBATCH_COMMON_LIBRARY_ONLY=1
source "$SOURCE_ROOT/sbatch/_job_common.sh"
unset SBATCH_COMMON_LIBRARY_ONLY

stage_alpha025_attack_if_present() {
    local run_name="$1"
    [ "${EXTRA_ALPHA:-}" = 0.25 ] || return 0
    copy_dir_if_present \
        "$ALPHA025_SNAPSHOT/ours_result_alpha025_live/$run_name" \
        "$RUN_ROOT/ours_result/$run_name"
}

stage_alpha025_defense_if_present() {
    local def_name="$1"
    [ "${EXTRA_ALPHA:-}" = 0.25 ] || return 0
    copy_dir_if_present \
        "$ALPHA025_SNAPSHOT/defense_result/$def_name" \
        "$RUN_ROOT/defense_result/$def_name"
}

# Override the library's attack staging only to overlay the retained H200 state.
stage_attack_job() {
    local target_degree run_name budget
    target_degree="$(cfg_target)"
    copy_dir_if_present \
        "$SOURCE_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42" \
        "$RUN_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
    copy_dir_if_present \
        "$SOURCE_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    for budget in $BUDGETS; do
        run_name="$(attack_run_name "$SELECT" "$budget" "$target_degree")"
        ATTACK_RUN_NAMES+=("$run_name")
        stage_alpha025_attack_if_present "$run_name"
        copy_dir_if_present "$SOURCE_ROOT/ours_result/$run_name" \
                            "$RUN_ROOT/ours_result/$run_name"
    done
}

# Defense staging imports the matching poison cache and any partial defense
# shards before overlaying newer copies from the main result trees.
stage_defense_job() {
    local target_degree run_name def_name def_tag budget selection
    local def_target jac_tag="" jacobian_tag
    target_degree="$(cfg_target)"
    def_tag="$(defense_tag "$DEFENSES")"
    copy_dir_if_present \
        "$SOURCE_ROOT/cache/defended_victims/${MODEL}_${def_tag}_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/defended_victims/${MODEL}_${def_tag}_50ep_lr0.1_bs125_wd0_seed42"

    if [ "${USE_JACOBIAN_SCORE:-0}" = 1 ]; then
        for selection in $SELS; do
            if [ "$selection" != random ]; then
                jacobian_tag="$(fmt_g "${JACOBIAN_WEIGHT:-1.0}")"
                jac_tag="_jacw${jacobian_tag}"
            fi
        done
    fi
    for budget in $BUDGETS; do
        def_target="def_${MODEL}_${ATTACK}_${CLASS_PAIR}_b${budget}${jac_tag}.json"
        DEF_TARGET_FILES+=("$def_target")
        copy_file_if_present "$SOURCE_ROOT/target_sets/$def_target" \
                             "$RUN_ROOT/target_sets"
        for selection in $SELS; do
            run_name="$(attack_run_name "$selection" "$budget" "$target_degree")"
            stage_alpha025_attack_if_present "$run_name"
            copy_dir_if_present "$SOURCE_ROOT/ours_result/$run_name" \
                                "$RUN_ROOT/ours_result/$run_name"
            def_name="${run_name}__def-${def_tag}"
            DEFENSE_RUN_NAMES+=("$def_name")
            stage_alpha025_defense_if_present "$def_name"
            copy_dir_if_present "$SOURCE_ROOT/defense_result/$def_name" \
                                "$RUN_ROOT/defense_result/$def_name"
        done
    done
}

main "$@"
