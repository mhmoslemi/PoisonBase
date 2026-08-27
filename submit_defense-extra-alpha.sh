#!/bin/sh
# Submit every unfinished alpha=0.25 cell, then all alpha=0.1 cells on Vulcan.
# Each sbatch is one table cell. Defense cells depend on their matching
# poison-generation cell unless that alpha=0.25 attack is already complete.
set -eu

ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
JOB_DIR="$ROOT/sbatch/defense_extra_alpha"
PRECOMPUTE_DIR="$ROOT/sbatch/cross_vulcan_precompute"
mkdir -p "$ROOT/sbatch/logs"

JOB_COUNT=$(find "$JOB_DIR" -maxdepth 1 -type f -name 'defextra_*.sh' -print | wc -l | tr -d ' ')
[ "$JOB_COUNT" -eq 63 ] || {
    echo "ERROR: expected 63 defense-extra one-cell jobs; found $JOB_COUNT" >&2
    exit 1
}

if [ "${DRY_RUN:-0}" != 1 ]; then
    case "$(hostname -s)" in
        vulcan*) ;;
        *)
            echo "ERROR: submit_defense-extra-alpha.sh must run on vulcan.alliancecan.ca" >&2
            echo "Current host: $(hostname -s)" >&2
            exit 1
            ;;
    esac
fi

cache_file_count() {
    if [ ! -d "$1" ]; then printf '0\n'; return 0; fi
    find "$1" -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | wc -l | tr -d ' '
}

ensure_cache() {
    cache_job="$1"
    surrogate_cache=$(sed -n 's/^export SURROGATE_CACHE=//p' "$cache_job")
    victim_cache=$(sed -n 's/^export VICTIM_CACHE=//p' "$cache_job")
    cache_job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$cache_job")
    surrogate_count=$(cache_file_count "$ROOT/cache/surrogates/$surrogate_cache")
    victim_count=$(cache_file_count "$ROOT/cache/clean_victims/$victim_cache")
    if [ "$surrogate_count" -ge 20 ] && [ "$victim_count" -ge 5 ]; then
        printf 'cache ready: %s (%s surrogates, %s victims)\n' \
            "$cache_job_name" "$surrogate_count" "$victim_count" >&2
        return 0
    fi

    active=$(squeue -h -u "${USER:-mmoslem3}" -n "$cache_job_name" -o '%A' | sort -u | head -n 1)
    if [ -n "$active" ]; then
        case "$active" in *[!0-9]*) echo "ERROR: invalid active job ID: $active" >&2; exit 1 ;; esac
        printf 'cache prerequisite: reusing %s job %s\n' "$cache_job_name" "$active" >&2
        printf '%s\n' "$active"
        return 0
    fi

    raw=$(sbatch --parsable --account="$ACCOUNT" "$cache_job")
    cache_id=${raw%%;*}
    case "$cache_id" in *[!0-9]*|'') echo "ERROR: invalid sbatch output: $raw" >&2; exit 1 ;; esac
    printf 'cache prerequisite: %s -> %s\n' "$(basename "$cache_job")" "$cache_id" >&2
    printf '%s\n' "$cache_id"
}

prepare_cifar10() {
    cifar=$(find "$ROOT/data" -type d -name cifar-10-batches-py -print -quit 2>/dev/null)
    [ -n "$cifar" ] && return 0
    [ -f "$ENV_ACTIVATE" ] || {
        echo "ERROR: missing environment activation script: $ENV_ACTIVATE" >&2
        exit 1
    }
    mkdir -p "$ROOT/data"
    (
        . "$ENV_ACTIVATE"
        python - "$ROOT/data" <<'PY'
import sys
from torchvision.datasets import CIFAR10
CIFAR10(sys.argv[1], train=True, download=True)
CIFAR10(sys.argv[1], train=False, download=True)
PY
    )
}

verify_saved_prerequisites() {
    for run_name in \
        CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt50 \
        CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_seldpp0.25_worst0.05_ce5_tgt14; do
        run_dir=""
        best_count=0
        for candidate in \
            "$ROOT/last_night_H200_2026-08-26/ours_result_alpha025_live/$run_name" \
            "$ROOT/ours_result/$run_name"; do
            [ -d "$candidate" ] || continue
            poison_count=$(find "$candidate/poison_cache" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l | tr -d ' ')
            if [ "$poison_count" -gt "$best_count" ]; then
                best_count="$poison_count"
                run_dir="$candidate"
            fi
            [ "$poison_count" -ge 5 ] && break
        done
        [ "$best_count" -ge 5 ] || {
            echo "ERROR: saved alpha=0.25 prerequisite has only $best_count/5 poison deltas: $run_name" >&2
            exit 1
        }
        echo "saved poison prerequisite ready: $run_dir ($best_count deltas)" >&2
    done
}

CACHE_DEPENDENCIES=""
if [ "${DRY_RUN:-0}" = 1 ]; then
    CACHE_DEPENDENCIES=VULCAN_CACHE_JOBS
else
    prepare_cifar10
    verify_saved_prerequisites
    for cache_job in \
        "$PRECOMPUTE_DIR/cross_precompute_convnet.sh" \
        "$PRECOMPUTE_DIR/cross_precompute_resnet20.sh" \
        "$PRECOMPUTE_DIR/cross_precompute_vgg13.sh"; do
        [ -f "$cache_job" ] || {
            echo "ERROR: missing cache prerequisite script: $cache_job" >&2
            exit 1
        }
        cache_id=$(ensure_cache "$cache_job")
        [ -z "$cache_id" ] || CACHE_DEPENDENCIES="${CACHE_DEPENDENCIES:+$CACHE_DEPENDENCIES:}$cache_id"
    done
fi

submit_attack() {
    label="$1"
    file="$2"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch --account=%s --dependency=afterok:%s %s\n' \
            "$ACCOUNT" "$CACHE_DEPENDENCIES" "$JOB_DIR/$file" >&2
        id="DRY_${file%.sh}"
    elif [ -n "$CACHE_DEPENDENCIES" ]; then
        raw=$(sbatch --parsable --account="$ACCOUNT" \
            --dependency="afterok:$CACHE_DEPENDENCIES" "$JOB_DIR/$file")
        id=${raw%%;*}
    else
        raw=$(sbatch --parsable --account="$ACCOUNT" "$JOB_DIR/$file")
        id=${raw%%;*}
    fi
    printf "Submitted %s as %s\n" "$label" "$id" >&2
    printf "%s\n" "$id"
}

submit_defense() {
    label="$1"
    dependency="$2"
    file="$3"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        if [ -n "$dependency" ]; then
            printf 'sbatch --account=%s --dependency=afterok:%s %s\n' \
                "$ACCOUNT" "$dependency" "$JOB_DIR/$file"
        else
            printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$JOB_DIR/$file"
        fi
        return 0
    elif [ -n "$dependency" ]; then
        raw=$(sbatch --parsable --account="$ACCOUNT" \
            --dependency="afterok:$dependency" "$JOB_DIR/$file")
    else
        raw=$(sbatch --parsable --account="$ACCOUNT" "$JOB_DIR/$file")
    fi
    id=${raw%%;*}
    printf "Submitted %s as %s\n" "$label" "$id"
}

printf '\nSubmitting defense-extra alpha=0.25\n'
submit_defense '0.25 config 01 EPIC' "" 'defextra_a025_001_convnet_bp_b001_epic.sh'
submit_defense '0.25 config 01 FRIENDS' "" 'defextra_a025_002_convnet_bp_b001_friends.sh'
attack_a025_02=$(submit_attack '0.25 config 02 No Defense' 'defextra_a025_003_convnet_bp_b002_nodef.sh')
submit_defense '0.25 config 02 EPIC' "$attack_a025_02" 'defextra_a025_004_convnet_bp_b002_epic.sh'
submit_defense '0.25 config 02 FRIENDS' "$attack_a025_02" 'defextra_a025_005_convnet_bp_b002_friends.sh'
attack_a025_03=$(submit_attack '0.25 config 03 No Defense' 'defextra_a025_006_convnet_gm_b002_nodef.sh')
submit_defense '0.25 config 03 EPIC' "$attack_a025_03" 'defextra_a025_007_convnet_gm_b002_epic.sh'
submit_defense '0.25 config 03 FRIENDS' "$attack_a025_03" 'defextra_a025_008_convnet_gm_b002_friends.sh'
attack_a025_05=$(submit_attack '0.25 config 05 No Defense' 'defextra_a025_009_resnet20_bp_b001_nodef.sh')
submit_defense '0.25 config 05 EPIC' "$attack_a025_05" 'defextra_a025_010_resnet20_bp_b001_epic.sh'
submit_defense '0.25 config 05 FRIENDS' "$attack_a025_05" 'defextra_a025_011_resnet20_bp_b001_friends.sh'
attack_a025_06=$(submit_attack '0.25 config 06 No Defense' 'defextra_a025_012_resnet20_gm_b0002_nodef.sh')
submit_defense '0.25 config 06 EPIC' "$attack_a025_06" 'defextra_a025_013_resnet20_gm_b0002_epic.sh'
submit_defense '0.25 config 06 FRIENDS' "$attack_a025_06" 'defextra_a025_014_resnet20_gm_b0002_friends.sh'
attack_a025_07=$(submit_attack '0.25 config 07 No Defense' 'defextra_a025_015_resnet20_gm_b001_nodef.sh')
submit_defense '0.25 config 07 EPIC' "$attack_a025_07" 'defextra_a025_016_resnet20_gm_b001_epic.sh'
submit_defense '0.25 config 07 FRIENDS' "$attack_a025_07" 'defextra_a025_017_resnet20_gm_b001_friends.sh'
submit_defense '0.25 config 08 FRIENDS' "" 'defextra_a025_018_resnet20_sapa_b001_friends.sh'
attack_a025_09=$(submit_attack '0.25 config 09 No Defense' 'defextra_a025_019_vgg13_gm_b0005_nodef.sh')
submit_defense '0.25 config 09 EPIC' "$attack_a025_09" 'defextra_a025_020_vgg13_gm_b0005_epic.sh'
submit_defense '0.25 config 09 FRIENDS' "$attack_a025_09" 'defextra_a025_021_vgg13_gm_b0005_friends.sh'
attack_a025_10=$(submit_attack '0.25 config 10 No Defense' 'defextra_a025_022_vgg13_gm_b001_nodef.sh')
submit_defense '0.25 config 10 EPIC' "$attack_a025_10" 'defextra_a025_023_vgg13_gm_b001_epic.sh'
submit_defense '0.25 config 10 FRIENDS' "$attack_a025_10" 'defextra_a025_024_vgg13_gm_b001_friends.sh'
attack_a025_12=$(submit_attack '0.25 config 12 No Defense' 'defextra_a025_025_vgg13_sapa_b001_nodef.sh')
submit_defense '0.25 config 12 EPIC' "$attack_a025_12" 'defextra_a025_026_vgg13_sapa_b001_epic.sh'
submit_defense '0.25 config 12 FRIENDS' "$attack_a025_12" 'defextra_a025_027_vgg13_sapa_b001_friends.sh'
printf '\nSubmitting defense-extra alpha=0.1\n'
attack_a01_01=$(submit_attack '0.1 config 01 No Defense' 'defextra_a01_028_convnet_bp_b001_nodef.sh')
submit_defense '0.1 config 01 EPIC' "$attack_a01_01" 'defextra_a01_029_convnet_bp_b001_epic.sh'
submit_defense '0.1 config 01 FRIENDS' "$attack_a01_01" 'defextra_a01_030_convnet_bp_b001_friends.sh'
attack_a01_02=$(submit_attack '0.1 config 02 No Defense' 'defextra_a01_031_convnet_bp_b002_nodef.sh')
submit_defense '0.1 config 02 EPIC' "$attack_a01_02" 'defextra_a01_032_convnet_bp_b002_epic.sh'
submit_defense '0.1 config 02 FRIENDS' "$attack_a01_02" 'defextra_a01_033_convnet_bp_b002_friends.sh'
attack_a01_03=$(submit_attack '0.1 config 03 No Defense' 'defextra_a01_034_convnet_gm_b002_nodef.sh')
submit_defense '0.1 config 03 EPIC' "$attack_a01_03" 'defextra_a01_035_convnet_gm_b002_epic.sh'
submit_defense '0.1 config 03 FRIENDS' "$attack_a01_03" 'defextra_a01_036_convnet_gm_b002_friends.sh'
attack_a01_04=$(submit_attack '0.1 config 04 No Defense' 'defextra_a01_037_convnet_sapa_b002_nodef.sh')
submit_defense '0.1 config 04 EPIC' "$attack_a01_04" 'defextra_a01_038_convnet_sapa_b002_epic.sh'
submit_defense '0.1 config 04 FRIENDS' "$attack_a01_04" 'defextra_a01_039_convnet_sapa_b002_friends.sh'
attack_a01_05=$(submit_attack '0.1 config 05 No Defense' 'defextra_a01_040_resnet20_bp_b001_nodef.sh')
submit_defense '0.1 config 05 EPIC' "$attack_a01_05" 'defextra_a01_041_resnet20_bp_b001_epic.sh'
submit_defense '0.1 config 05 FRIENDS' "$attack_a01_05" 'defextra_a01_042_resnet20_bp_b001_friends.sh'
attack_a01_06=$(submit_attack '0.1 config 06 No Defense' 'defextra_a01_043_resnet20_gm_b0002_nodef.sh')
submit_defense '0.1 config 06 EPIC' "$attack_a01_06" 'defextra_a01_044_resnet20_gm_b0002_epic.sh'
submit_defense '0.1 config 06 FRIENDS' "$attack_a01_06" 'defextra_a01_045_resnet20_gm_b0002_friends.sh'
attack_a01_07=$(submit_attack '0.1 config 07 No Defense' 'defextra_a01_046_resnet20_gm_b001_nodef.sh')
submit_defense '0.1 config 07 EPIC' "$attack_a01_07" 'defextra_a01_047_resnet20_gm_b001_epic.sh'
submit_defense '0.1 config 07 FRIENDS' "$attack_a01_07" 'defextra_a01_048_resnet20_gm_b001_friends.sh'
attack_a01_08=$(submit_attack '0.1 config 08 No Defense' 'defextra_a01_049_resnet20_sapa_b001_nodef.sh')
submit_defense '0.1 config 08 EPIC' "$attack_a01_08" 'defextra_a01_050_resnet20_sapa_b001_epic.sh'
submit_defense '0.1 config 08 FRIENDS' "$attack_a01_08" 'defextra_a01_051_resnet20_sapa_b001_friends.sh'
attack_a01_09=$(submit_attack '0.1 config 09 No Defense' 'defextra_a01_052_vgg13_gm_b0005_nodef.sh')
submit_defense '0.1 config 09 EPIC' "$attack_a01_09" 'defextra_a01_053_vgg13_gm_b0005_epic.sh'
submit_defense '0.1 config 09 FRIENDS' "$attack_a01_09" 'defextra_a01_054_vgg13_gm_b0005_friends.sh'
attack_a01_10=$(submit_attack '0.1 config 10 No Defense' 'defextra_a01_055_vgg13_gm_b001_nodef.sh')
submit_defense '0.1 config 10 EPIC' "$attack_a01_10" 'defextra_a01_056_vgg13_gm_b001_epic.sh'
submit_defense '0.1 config 10 FRIENDS' "$attack_a01_10" 'defextra_a01_057_vgg13_gm_b001_friends.sh'
attack_a01_11=$(submit_attack '0.1 config 11 No Defense' 'defextra_a01_058_vgg13_sapa_b0005_nodef.sh')
submit_defense '0.1 config 11 EPIC' "$attack_a01_11" 'defextra_a01_059_vgg13_sapa_b0005_epic.sh'
submit_defense '0.1 config 11 FRIENDS' "$attack_a01_11" 'defextra_a01_060_vgg13_sapa_b0005_friends.sh'
attack_a01_12=$(submit_attack '0.1 config 12 No Defense' 'defextra_a01_061_vgg13_sapa_b001_nodef.sh')
submit_defense '0.1 config 12 EPIC' "$attack_a01_12" 'defextra_a01_062_vgg13_sapa_b001_epic.sh'
submit_defense '0.1 config 12 FRIENDS' "$attack_a01_12" 'defextra_a01_063_vgg13_sapa_b001_friends.sh'
