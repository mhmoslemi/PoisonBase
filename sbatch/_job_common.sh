#!/usr/bin/env bash
# Shared runtime for the generated PoisonBase SLURM jobs. This file is sourced
# by each sbatch/attack/*.sh and sbatch/defense/*.sh job.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
LEGACY_SOURCE_ROOT="${LEGACY_SOURCE_ROOT:-}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-/home/mmoslem3/scratch/data}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
RUN_ROOT="$SLURM_TMPDIR/attack_if"
LOCAL_DATA_ROOT="$SLURM_TMPDIR/data"

ATTACK_RUN_NAMES=()
DEFENSE_RUN_NAMES=()
DEF_TARGET_FILES=()
SYNCED=0
STEP_PID=""

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

copy_dir_if_present() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
    else
        say "stage: optional directory absent: $src"
    fi
}

copy_file_if_present() {
    local src="$1" dst_dir="$2"
    if [ -f "$src" ]; then
        mkdir -p "$dst_dir"
        rsync -a "$src" "$dst_dir/"
    else
        say "stage: optional file absent: $src"
    fi
}

copy_legacy_dir_if_present() {
    local relative="$1" dst="$2"
    [ -n "$LEGACY_SOURCE_ROOT" ] || return 0
    [ "$LEGACY_SOURCE_ROOT" != "$SOURCE_ROOT" ] || return 0
    [ -d "$LEGACY_SOURCE_ROOT/$relative" ] || return 0
    say "stage: importing legacy partials from $LEGACY_SOURCE_ROOT/$relative"
    copy_dir_if_present "$LEGACY_SOURCE_ROOT/$relative" "$dst"
}

copy_legacy_file_if_present() {
    local relative="$1" dst_dir="$2"
    [ -n "$LEGACY_SOURCE_ROOT" ] || return 0
    [ "$LEGACY_SOURCE_ROOT" != "$SOURCE_ROOT" ] || return 0
    [ -f "$LEGACY_SOURCE_ROOT/$relative" ] || return 0
    say "stage: importing legacy file from $LEGACY_SOURCE_ROOT/$relative"
    copy_file_if_present "$LEGACY_SOURCE_ROOT/$relative" "$dst_dir"
}

cfg_target() {
    local lookup_attack="$ATTACK"
    [ "$lookup_attack" = sapa ] && lookup_attack=gradmatch
    if [ -n "${TARGET_SELECT:-}" ]; then
        printf '%s\n' "$TARGET_SELECT"
        return
    fi
    python - "$RUN_ROOT/sweep_config.json" "$MODEL" "$lookup_attack" "$CLASS_PAIR" <<'PY'
import json, sys
path, model, attack, pair = sys.argv[1:]
with open(path) as handle:
    cfg = json.load(handle)
print(cfg['difficulty'][model][attack][pair])
PY
}

fmt_g() {
    python - "$1" <<'PY'
import sys
print('%g' % float(sys.argv[1]))
PY
}

attack_run_name() {
    local selection="$1" budget="$2" target_degree="$3"
    local base=random name alpha_tag jacobian_tag
    [ "$selection" != random ] && base=ours
    name="CIFAR10_${MODEL}_${ATTACK}_${base}_${CLASS_PAIR}_b${budget}_eps8_seed42"
    if [ "$base" = ours ]; then
        name+="_lam1_cosine"
        case "$selection" in
            dpp)
                alpha_tag="$(fmt_g "${SEL_ALPHA:-2.0}")"
                name+="_seldpp${alpha_tag}"
                ;;
            exact) name+="_selexactgigt" ;;
            a-mr)  name+="_selAminusMR" ;;
            minus-m) name+="_selMinusM" ;;
            r) name+="_selR" ;;
            a) name+="_selA" ;;
            a-minus-m) name+="_selAminusM" ;;
            a-plus-r) name+="_selAplusR" ;;
            minus-m-times-r) name+="_selMinusMtimesR" ;;
        esac
        if [ "${USE_JACOBIAN_SCORE:-0}" = 1 ] && \
           [ "$selection" != exact ] && [ "$selection" != a-mr ] && \
           [ "$selection" != minus-m ] && [ "$selection" != r ] && \
           [ "$selection" != a ] && [ "$selection" != a-minus-m ] && \
           [ "$selection" != a-plus-r ] && \
           [ "$selection" != minus-m-times-r ]; then
            jacobian_tag="$(fmt_g "${JACOBIAN_WEIGHT:-1.0}")"
            name+="_jacw${jacobian_tag}"
        fi
    fi
    if [ "$ATTACK" = sapa ]; then
        name+="_${SHARP_MODE:-worst}${SHARP_SIGMA:-0.05}"
    fi
    name+="_ce5_tgt${target_degree}"
    printf '%s\n' "$name"
}

defense_tag() {
    case "$1" in
        epic) printf '%s\n' 'epic-s0.1-f2-d10' ;;
        friends) printf '%s\n' 'friends-friendly.bernoulli-e8-p5-clp16' ;;
        *) die "generated jobs only support the epic/friends tags in run_defense.txt (got $1)" ;;
    esac
}

stage_code_and_data() {
    local required=(final_update.py networks.py utils.py sweep_config.json)
    local file
    [ "$JOB_KIND" = attack ] && required+=(sel_dpp.sh)
    [ "$JOB_KIND" = defense ] && required+=(defense.sh defense.py victim_aug.py)
    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$RUN_ROOT/cache" \
             "$RUN_ROOT/ours_result" "$RUN_ROOT/defense_result" \
             "$RUN_ROOT/target_sets"
    for file in "${required[@]}"; do
        [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
    done

    [ -d "$PERSIST_DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 input missing: $PERSIST_DATA_ROOT/cifar-10-batches-py"
    rsync -a "$PERSIST_DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    local target_attack="$ATTACK"
    if [ "$target_attack" = sapa ] && \
       [ ! -s "$SOURCE_ROOT/target_sets/${MODEL}_${target_attack}_${CLASS_PAIR}.json" ]; then
        target_attack=gradmatch
    fi
    copy_file_if_present \
        "$SOURCE_ROOT/target_sets/${MODEL}_${target_attack}_${CLASS_PAIR}.json" \
        "$RUN_ROOT/target_sets"
}

stage_attack_job() {
    local target_degree run_name budget
    target_degree="$(cfg_target)"
    copy_legacy_dir_if_present \
        "cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42" \
        "$RUN_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
    copy_dir_if_present \
        "$SOURCE_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42" \
        "$RUN_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
    copy_legacy_dir_if_present \
        "cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    copy_dir_if_present \
        "$SOURCE_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    for budget in $BUDGETS; do
        run_name="$(attack_run_name "$SELECT" "$budget" "$target_degree")"
        ATTACK_RUN_NAMES+=("$run_name")
        copy_legacy_dir_if_present "ours_result/$run_name" \
                                   "$RUN_ROOT/ours_result/$run_name"
        copy_dir_if_present "$SOURCE_ROOT/ours_result/$run_name" \
                            "$RUN_ROOT/ours_result/$run_name"
    done
}

stage_defense_job() {
    local target_degree run_name def_name def_tag budget selection def_target jac_tag="" jacobian_tag
    target_degree="$(cfg_target)"
    def_tag="$(defense_tag "$DEFENSES")"
    copy_legacy_dir_if_present \
        "cache/defended_victims/${MODEL}_${def_tag}_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/defended_victims/${MODEL}_${def_tag}_50ep_lr0.1_bs125_wd0_seed42"
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
        copy_legacy_file_if_present "target_sets/$def_target" "$RUN_ROOT/target_sets"
        copy_file_if_present "$SOURCE_ROOT/target_sets/$def_target" "$RUN_ROOT/target_sets"
        for selection in $SELS; do
            run_name="$(attack_run_name "$selection" "$budget" "$target_degree")"
            copy_legacy_dir_if_present "ours_result/$run_name" \
                                       "$RUN_ROOT/ours_result/$run_name"
            copy_dir_if_present "$SOURCE_ROOT/ours_result/$run_name" \
                                "$RUN_ROOT/ours_result/$run_name"
            def_name="${run_name}__def-${def_tag}"
            DEFENSE_RUN_NAMES+=("$def_name")
            copy_legacy_dir_if_present "defense_result/$def_name" \
                                       "$RUN_ROOT/defense_result/$def_name"
            copy_dir_if_present "$SOURCE_ROOT/defense_result/$def_name" \
                                "$RUN_ROOT/defense_result/$def_name"
        done
    done
}

sync_cache_dir() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    rsync -a --ignore-existing --exclude='*.tmp' "$src/" "$dst/"
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    say "sync: preserving outputs in $SOURCE_ROOT"
    local name file
    if [ "$JOB_KIND" = attack ]; then
        for name in "${ATTACK_RUN_NAMES[@]}"; do
            [ -d "$RUN_ROOT/ours_result/$name" ] || continue
            mkdir -p "$SOURCE_ROOT/ours_result/$name"
            rsync -a --exclude='.lock' --exclude='*.tmp' \
                "$RUN_ROOT/ours_result/$name/" "$SOURCE_ROOT/ours_result/$name/"
        done
        sync_cache_dir \
            "$RUN_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42" \
            "$SOURCE_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed42"
        sync_cache_dir \
            "$RUN_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42" \
            "$SOURCE_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    else
        for name in "${DEFENSE_RUN_NAMES[@]}"; do
            [ -d "$RUN_ROOT/defense_result/$name" ] || continue
            mkdir -p "$SOURCE_ROOT/defense_result/$name"
            rsync -a --exclude='.lock' --exclude='*.tmp' \
                "$RUN_ROOT/defense_result/$name/" "$SOURCE_ROOT/defense_result/$name/"
        done
        for file in "${DEF_TARGET_FILES[@]}"; do
            [ -f "$RUN_ROOT/target_sets/$file" ] || continue
            mkdir -p "$SOURCE_ROOT/target_sets"
            rsync -a "$RUN_ROOT/target_sets/$file" "$SOURCE_ROOT/target_sets/"
        done
        local def_tag
        def_tag="$(defense_tag "$DEFENSES")"
        sync_cache_dir \
            "$RUN_ROOT/cache/defended_victims/${MODEL}_${def_tag}_50ep_lr0.1_bs125_wd0_seed42" \
            "$SOURCE_ROOT/cache/defended_victims/${MODEL}_${def_tag}_50ep_lr0.1_bs125_wd0_seed42"
    fi
    say "sync: complete"
}

handle_signal() {
    local signal="$1"
    say "signal: received $signal; stopping the job step before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

main() {
    [ "${JOB_KIND:-}" = attack ] || [ "${JOB_KIND:-}" = defense ] || \
        die 'JOB_KIND must be attack or defense'
    [ -n "${SLURM_TMPDIR:-}" ] || die 'SLURM_TMPDIR is unset; submit this file with sbatch'

    # Killarney provides Environment Modules in batch shells, while Vulcan's
    # non-login batch shell may not define `module`.  The Vulcan virtualenv is
    # self-contained, so module loading is useful when available but optional.
    if command -v module >/dev/null 2>&1; then
        module load python/3.11.5 cuda/12.6 cudnn
    else
        say "environment modules unavailable; using $PYTHON_ENV directly"
    fi
    source "$PYTHON_ENV/bin/activate"

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT

    stage_code_and_data
    if [ "$JOB_KIND" = attack ]; then
        stage_attack_job
    else
        stage_defense_job
    fi

    export PROJECT_ROOT="$RUN_ROOT"
    export DATA_PATH="$LOCAL_DATA_ROOT"
    export CACHE_DIR="$RUN_ROOT/cache"
    export OUT_DIR="$RUN_ROOT/ours_result"
    export DEF_OUT_DIR="$RUN_ROOT/defense_result"
    export PYTHON_ENV

    say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
    say "work: $RUN_ROOT"
    say "config: $ORIGINAL_COMMAND"
    if [ "$JOB_KIND" = attack ]; then
        say "protocol: attack=${NUM_TARGETS:-8} targets x ${NUM_VICTIMS:-5} victims"
    else
        say "protocol: defense=${NUM_TARGETS:-7} targets x ${NUM_VICTIMS:-5} victims"
    fi
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    if [ "$JOB_KIND" = attack ]; then
        srun --ntasks=1 sh "$RUN_ROOT/sel_dpp.sh" &
    else
        srun --ntasks=1 sh "$RUN_ROOT/defense.sh" &
    fi
    STEP_PID=$!
    set +e
    wait "$STEP_PID"
    local status=$?
    set -e
    STEP_PID=""
    sync_outputs
    trap - EXIT
    exit "$status"
}

if [ "${SBATCH_COMMON_LIBRARY_ONLY:-0}" != 1 ]; then
    main "$@"
fi
