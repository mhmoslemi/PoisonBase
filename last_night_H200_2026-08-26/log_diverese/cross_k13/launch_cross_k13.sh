#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ubuntu/mohammad/PoisonBase
VENV=/home/ubuntu/mohammad/.venv
DATA=/home/ubuntu/mohammad/data
LOGDIR="$ROOT/log_diverese/cross_k13"
CACHE="$ROOT/cache_cross_k13_live"
OUT="$ROOT/ours_result"
WRAPPER="$LOGDIR/cross_arch_k13.sh"
STATUS="$LOGDIR/status.tsv"
PRIOR_PID_FILE="$ROOT/log_diverese/launcher.pid"

mkdir -p "$LOGDIR" "$CACHE" "$OUT"
printf 'timestamp\titem\tphase\tstate\trc\tdetail\n' > "$STATUS"
printf 'started_at=%s\nhost=%s\ninitial_gpus=4,5\nexpanded_gpus=1,2,3,4,5 after prior queue\nmax_experiments_per_gpu=3\nselection=dpp\nalpha=2\njacobian=off\nselector_K=1,3\nattack_surrogates=20\ntargets=5\nvictims=4\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" > "$LOGDIR/run_metadata.txt"

status() {
  local item=$1 phase=$2 state=$3 rc=$4 detail=${5:-}
  detail=${detail//$'\t'/ }
  detail=${detail//$'\n'/ }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$item" "$phase" "$state" "$rc" "$detail" >> "$STATUS"
}

# Snapshot every fully written checkpoint already produced by the first queue.
while IFS= read -r src; do
  rel=${src#"$ROOT/cache_alpha025_live/"}
  mkdir -p "$CACHE/$(dirname "$rel")"
  [ -s "$CACHE/$rel" ] || cp -p "$src" "$CACHE/$rel"
done < <(find "$ROOT/cache_alpha025_live" -type f -name 'net_*.pt' 2>/dev/null)

expected_path() {
  local model=$1 part=$2 id=$3
  if [ "$part" = surrogate ]; then
    printf '%s/surrogates/%s_60ep_lr0.1_bs128_seed42/net_%s.pt' "$CACHE" "$model" "$id"
  else
    printf '%s/clean_victims/%s_50ep_lr0.1_bs125_wd0_seed42/net_%s.pt' "$CACHE" "$model" "$id"
  fi
}

precompute_one() {
  local model=$1 part=$2 id=$3 gpu=$4
  local item="pre_${model}_${part}_${id}" log="$LOGDIR/pre_${model}_${part}_${id}.log" expected rc
  expected=$(expected_path "$model" "$part" "$id")
  if [ -s "$expected" ]; then
    status "$item" precompute cached 0 "$expected"
    return 0
  fi
  status "$item" precompute running - "physical_gpu=$gpu"
  {
    echo "[$(date --iso-8601=seconds)] model=$model part=$part id=$id physical_gpu=$gpu"
    env PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" \
      "$VENV/bin/python" "$ROOT/final_update.py" \
      --dataset CIFAR10 --data_path "$DATA" --seed 42 \
      --cache_dir "$CACHE" --out_dir "$OUT" --model "$model" \
      --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
      --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
      --victim_decay 40 --victim_wd 0.0 \
      --precompute_only --precompute_part "$part" --precompute_id "$id"
    rc=$?
    echo "[$(date --iso-8601=seconds)] exit=$rc"
    [ "$rc" -eq 0 ]
  } > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -s "$expected" ]; then
    status "$item" precompute complete 0 "physical_gpu=$gpu"
    return 0
  fi
  status "$item" precompute failed "${rc:-1}" "physical_gpu=$gpu log=$log"
  return "${rc:-1}"
}

status cache precompute started - "separate non-racing cache; max four trainings per GPU 4/5"
missing=()
for model in ConvNetBN ResNet20BN VGG13BN; do
  for id in $(seq 0 19); do
    [ -s "$(expected_path "$model" surrogate "$id")" ] || missing+=("$model|surrogate|$id")
  done
  for id in $(seq 0 3); do
    [ -s "$(expected_path "$model" victim "$id")" ] || missing+=("$model|victim|$id")
  done
done
status cache precompute inventory - "missing=${#missing[@]} seeded=$((72-${#missing[@]}))"

batch=0
for ((off=0; off<${#missing[@]}; off+=8)); do
  pids=()
  batch=$((batch+1))
  status "precompute_batch_$batch" scheduler started - "up to 8 jobs; four per physical GPU"
  for ((j=0; j<8 && off+j<${#missing[@]}; j++)); do
    IFS='|' read -r model part id <<< "${missing[off+j]}"
    gpu=$((4 + (j % 2)))
    precompute_one "$model" "$part" "$id" "$gpu" &
    pids+=("$!")
  done
  batch_rc=0
  for pid in "${pids[@]}"; do wait "$pid" || batch_rc=1; done
  status "precompute_batch_$batch" scheduler complete "$batch_rc" "children=${#pids[@]}"
done

cache_total=$(find "$CACHE" -type f -name 'net_*.pt' | wc -l)
if [ "$cache_total" -lt 72 ]; then
  status cache precompute failed 92 "expected 72 networks, found $cache_total; experiments not started"
  exit 92
fi
status cache precompute complete 0 "networks=$cache_total"

printf 'id\tK\tattack\tvictim_model\tselector_model\tbudget\talpha\tjacobian\ttargets\tvictims\n' > "$LOGDIR/manifest.tsv"
specs=()
id=0
for k in 1 3; do
  for attack in fc gradmatch; do
    for model in ConvNetBN ResNet20BN VGG13BN; do
      for selector in ConvNetBN ResNet20BN VGG13BN; do
        [ "$selector" = "$model" ] && continue
        id=$((id+1))
        printf '%02d\t%s\t%s\t%s\t%s\t0.005\t2\toff\t5\t4\n' "$id" "$k" "$attack" "$model" "$selector" >> "$LOGDIR/manifest.tsv"
        specs+=("$(printf '%02d' "$id")|$k|$attack|$model|$selector")
      done
    done
  done
done

select_gpus() {
  local prior=''
  [ -s "$PRIOR_PID_FILE" ] && prior=$(cat "$PRIOR_PID_FILE")
  if [ -n "$prior" ] && kill -0 "$prior" 2>/dev/null; then
    printf '4,5'
  else
    printf '1,2,3,4,5'
  fi
}

run_exp() {
  local id=$1 k=$2 attack=$3 model=$4 selector=$5 gpus=$6
  local item="x${id}_K${k}_${attack}_A${model}_S${selector}"
  local log="$LOGDIR/${item}.log" rc
  status "$item" experiment running - "physical_gpus=$gpus"
  {
    echo "[$(date --iso-8601=seconds)] START $item"
    echo "DPP alpha=2; Jacobian off; selector K=$k; attack surrogates=20"
    echo "A=V=$model; S=$selector; attack=$attack; physical GPUs=$gpus"
    env PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpus" \
      PROJECT_DIR="$ROOT" DATA_PATH="$DATA" OUT_DIR="$OUT" CACHE_DIR="$CACHE" \
      VENV_ACTIVATE="$VENV/bin/activate" MODELS="$model" SELECTOR_MODELS="$selector" \
      ATTACKS="$attack" SELECTIONS=dpp NUM_SURROGATES=20 CRAFT_ENSEMBLE=5 \
      NUM_TARGETS=5 NUM_VICTIMS=4 SEL_ALPHA=2 USE_JACOBIAN_SCORE=0 \
      RUN_MATCHED_DPP=0 SEL_K="$k" DRY_RUN=0 \
      bash "$WRAPPER"
    rc=$?
    echo "[$(date --iso-8601=seconds)] END rc=$rc"
    [ "$rc" -eq 0 ]
  } > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    status "$item" experiment complete 0 "physical_gpus=$gpus"
  else
    status "$item" experiment failed "$rc" "physical_gpus=$gpus log=$log"
  fi
  return "$rc"
}

wave=0
for ((off=0; off<${#specs[@]}; off+=3)); do
  wave=$((wave+1))
  gpus=$(select_gpus)
  pids=()
  status "experiment_wave_$wave" scheduler started - "physical_gpus=$gpus; configurations<=3"
  for ((j=0; j<3 && off+j<${#specs[@]}; j++)); do
    IFS='|' read -r id k attack model selector <<< "${specs[off+j]}"
    run_exp "$id" "$k" "$attack" "$model" "$selector" "$gpus" &
    pids+=("$!")
  done
  wave_rc=0
  for pid in "${pids[@]}"; do wait "$pid" || wave_rc=1; done
  status "experiment_wave_$wave" scheduler complete "$wave_rc" "physical_gpus=$gpus children=${#pids[@]}"
done

awk -F '\t' 'NR==1 || ($3=="experiment" && ($4=="complete" || $4=="failed"))' "$STATUS" > "$LOGDIR/final_summary.tsv"
status launcher scheduler all_done 0 "24 off-diagonal K1/K3 cells processed"
echo "[$(date --iso-8601=seconds)] cross K1/K3 launcher finished"
