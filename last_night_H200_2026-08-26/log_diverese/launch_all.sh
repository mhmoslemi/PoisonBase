#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ubuntu/mohammad/PoisonBase
VENV=/home/ubuntu/mohammad/.venv
DATA=/home/ubuntu/mohammad/data
LOGDIR="$ROOT/log_diverese"
ATTACK_OUT="$ROOT/ours_result_alpha025_live"
DEFENSE_OUT="$ROOT/defense_result"
CACHE="$ROOT/cache_alpha025_live"
GEN="$LOGDIR/sel_dpp_generate.sh"
STATUS="$LOGDIR/status.tsv"

mkdir -p "$LOGDIR" "$ATTACK_OUT" "$DEFENSE_OUT" "$CACHE"
printf 'timestamp\trun\tphase\tstate\trc\tdetail\n' > "$STATUS"
printf 'started_at=%s\nhost=%s\nroot=%s\ngpus=1,2,3\nmax_concurrent_configs=3\nattack_targets=5\nattack_verification_victims=1\ndefense_targets=5\ndefense_victims=4\nsel_alpha=0.25\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$ROOT" > "$LOGDIR/run_metadata.txt"

status() {
  local run=$1 phase=$2 state=$3 rc=$4 detail=${5:-}
  detail=${detail//$'\t'/ }
  detail=${detail//$'\n'/ }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" "$phase" "$state" "$rc" "$detail" >> "$STATUS"
}

run_one() {
  local id=$1 name=$2 model=$3 attack=$4 pair=$5 budget=$6 target=$7
  local run="${id}_${name}"
  local logfile="$LOGDIR/${run}.log"
  local attack_rc defense_rc delta_count run_dir
  : > "$logfile"
  status "$run" pipeline started - "model=$model attack=$attack pair=$pair budget=$budget target=${target:-auto}"
  {
    echo "[$(date --iso-8601=seconds)] START $run"
    echo "GPU policy: physical 1,2,3; this configuration uses all three logical devices"
    echo "Attack generation: alpha=0.25, 5 targets, 1 verification victim"
    echo "Defense evaluation: EPIC + FRIENDS, 5 targets x 4 victims"
    echo
    status "$run" attack running - "generating alpha=0.25 perturbations"
    env PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1,2,3 \
      PROJECT_ROOT="$ROOT" PYTHON_ENV="$VENV" DATA_PATH="$DATA" \
      OUT_DIR="$ATTACK_OUT" CACHE_DIR="$CACHE" \
      USE_JACOBIAN_SCORE=0 CLASS_PAIR="$pair" MODEL="$model" ATTACK="$attack" \
      BUDGETS="$budget" SELECT=dpp SEL_ALPHA=0.25 \
      NUM_TARGETS=5 NUM_VICTIMS=1 TARGET_SELECT="$target" \
      bash "$GEN"
    attack_rc=$?
    echo "[$(date --iso-8601=seconds)] attack exit=$attack_rc"
    if [ "$attack_rc" -ne 0 ]; then
      status "$run" attack failed "$attack_rc" "generator exited nonzero"
      status "$run" pipeline failed "$attack_rc" "defense skipped"
      return "$attack_rc"
    fi

    mapfile -t dirs < <(find "$ATTACK_OUT" -mindepth 1 -maxdepth 1 -type d \
      -name "CIFAR10_${model}_${attack}_ours_${pair}_b${budget}_*seldpp0.25*" | sort)
    if [ "${#dirs[@]}" -ne 1 ]; then
      echo "ERROR: expected one attack directory, found ${#dirs[@]}"
      printf '%s\n' "${dirs[@]}"
      status "$run" attack failed 90 "expected one attack directory; found ${#dirs[@]}"
      status "$run" pipeline failed 90 "defense skipped"
      return 90
    fi
    run_dir=${dirs[0]}
    delta_count=$(find "$run_dir/poison_cache" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l)
    echo "Attack cache: $run_dir ($delta_count perturbation files)"
    if [ "$delta_count" -lt 5 ]; then
      echo "ERROR: attack completed without five usable perturbation files"
      status "$run" attack failed 91 "only $delta_count delta files"
      status "$run" pipeline failed 91 "defense skipped"
      return 91
    fi
    status "$run" attack complete 0 "$delta_count delta files"

    echo
    echo "[$(date --iso-8601=seconds)] starting defenses"
    status "$run" defense running - "epic friends"
    env PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1,2,3 \
      PROJECT_ROOT="$ROOT" PYTHON_ENV="$VENV" DATA_PATH="$DATA" \
      OUT_DIR="$ATTACK_OUT" DEF_OUT_DIR="$DEFENSE_OUT" CACHE_DIR="$CACHE" \
      USE_JACOBIAN_SCORE=0 CLASS_PAIR="$pair" MODEL="$model" ATTACK="$attack" \
      BUDGETS="$budget" SELS=dpp SEL_ALPHA=0.25 DEFENSES="epic friends" \
      NUM_TARGETS=5 NUM_VICTIMS=4 TARGET_SELECT="$target" \
      bash "$ROOT/defense.sh"
    defense_rc=$?
    echo "[$(date --iso-8601=seconds)] defense exit=$defense_rc"
    if [ "$defense_rc" -eq 0 ]; then
      status "$run" defense complete 0 "epic friends"
      status "$run" pipeline complete 0 "all requested work finished"
    else
      status "$run" defense failed "$defense_rc" "see $logfile"
      status "$run" pipeline failed "$defense_rc" "attack cache retained for resume"
    fi
    echo "[$(date --iso-8601=seconds)] END $run rc=$defense_rc"
    return "$defense_rc"
  } >> "$logfile" 2>&1
}

run_wave() {
  local wave=$1
  shift
  local pids=() specs=() spec pid rc
  status "wave_$wave" scheduler started - "$# configurations"
  for spec in "$@"; do
    IFS='|' read -r id name model attack pair budget target <<< "$spec"
    run_one "$id" "$name" "$model" "$attack" "$pair" "$budget" "$target" &
    pid=$!
    pids+=("$pid")
    specs+=("$id")
    status "${id}_${name}" scheduler launched - "pid=$pid wave=$wave"
  done
  rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  status "wave_$wave" scheduler complete "$rc" "all child processes exited"
  return 0
}

# Wave 1 deliberately has one configuration per model. This builds each model's
# shared network cache without cross-process races. Later waves safely reuse it.
run_wave 1 \
  '04|convnet_sapa_dog_bird_b0_02|ConvNetBN|sapa|dog-bird|0.02|70' \
  '08|resnet20_sapa_dog_bird_b0_01|ResNet20BN|sapa|dog-bird|0.01|14' \
  '11|vgg13_sapa_dog_bird_b0_005|VGG13BN|sapa|dog-bird|0.005|50'

run_wave 2 \
  '01|convnet_fc_dog_bird_b0_01|ConvNetBN|fc|dog-bird|0.01|' \
  '06|resnet20_gradmatch_dog_bird_b0_02|ResNet20BN|gradmatch|dog-bird|0.02|' \
  '12|vgg13_sapa_dog_bird_b0_01|VGG13BN|sapa|dog-bird|0.01|50'

run_wave 3 \
  '02|convnet_fc_dog_bird_b0_02|ConvNetBN|fc|dog-bird|0.02|' \
  '05|resnet20_gradmatch_dog_bird_b0_01|ResNet20BN|gradmatch|dog-bird|0.01|' \
  '09|vgg13_gradmatch_dog_bird_b0_005|VGG13BN|gradmatch|dog-bird|0.005|'

run_wave 4 \
  '03|convnet_gradmatch_dog_bird_b0_02|ConvNetBN|gradmatch|dog-bird|0.02|' \
  '07|resnet20_gradmatch_frog_airplane_b0_02|ResNet20BN|gradmatch|frog-airplane|0.02|' \
  '10|vgg13_gradmatch_dog_bird_b0_01|VGG13BN|gradmatch|dog-bird|0.01|'

awk -F '\t' 'NR==1 || ($3=="pipeline" && ($4=="complete" || $4=="failed"))' "$STATUS" > "$LOGDIR/final_summary.tsv"
status launcher scheduler all_done 0 "all four waves processed"
echo "[$(date --iso-8601=seconds)] launcher finished" 
