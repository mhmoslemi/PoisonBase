#!/usr/bin/env bash
set -u
ROOT=/home/ubuntu/mohammad/PoisonBase
LOGDIR="$ROOT/log_diverese/cross_k13"
STATUS="$LOGDIR/status.tsv"
VENV=/home/ubuntu/mohammad/.venv
DATA=/home/ubuntu/mohammad/data
CACHE="$ROOT/cache_cross_k13_live"
OUT="$ROOT/ours_result"
WRAPPER="$LOGDIR/cross_arch_k13.sh"
status() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" "$5" >> "$STATUS"; }
run_prefetch() {
  local id=$1 model=$2 selector=$3
  local item="x${id}_K1_fc_A${model}_S${selector}"
  local log="$LOGDIR/prefetch_${item}.log" rc
  status "prefetch_${item}" prefetch running - 'physical_gpus=4,5; protected by normal run lock'
  {
    echo "[$(date --iso-8601=seconds)] PREFETCH START $item"
    echo 'DPP alpha=2; Jacobian off; selector K=1; attack surrogates=20'
    echo "A=V=$model; S=$selector; attack=fc; physical GPUs=4,5"
    env PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5 \
      PROJECT_DIR="$ROOT" DATA_PATH="$DATA" OUT_DIR="$OUT" CACHE_DIR="$CACHE" \
      VENV_ACTIVATE="$VENV/bin/activate" MODELS="$model" SELECTOR_MODELS="$selector" \
      ATTACKS=fc SELECTIONS=dpp NUM_SURROGATES=20 CRAFT_ENSEMBLE=5 \
      NUM_TARGETS=5 NUM_VICTIMS=4 SEL_ALPHA=2 USE_JACOBIAN_SCORE=0 \
      RUN_MATCHED_DPP=0 SEL_K=1 DRY_RUN=0 \
      bash "$WRAPPER"
    rc=$?
    echo "[$(date --iso-8601=seconds)] PREFETCH END rc=$rc"
    exit "$rc"
  } > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    status "prefetch_${item}" prefetch complete 0 'original scheduler will resume/skip completed run'
  else
    status "prefetch_${item}" prefetch failed "$rc" "log=$log; original scheduler remains available to retry"
  fi
  return "$rc"
}
status prefetch_wave_2 scheduler started - 'x04-x06; combined with active x03 gives four trainings per physical GPU 4/5'
pids=()
run_prefetch 04 ResNet20BN VGG13BN & pids+=("$!")
run_prefetch 05 VGG13BN ConvNetBN & pids+=("$!")
run_prefetch 06 VGG13BN ResNet20BN & pids+=("$!")
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
status prefetch_wave_2 scheduler complete "$rc" 'three prefetched configurations exited'
exit "$rc"
