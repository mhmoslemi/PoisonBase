#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
for relative in \
    random/conv_sapa_b001/draw_5/target_821/poison.pt \
    random/conv_sapa_b001/draw_5/target_821/victim_0/training.pt \
    first_m/res_gm_b002/draw_0/target_2065/poison.pt \
    first_m/res_gm_b002/draw_0/target_2065/victim_2/training.pt \
    random/res_gm_b002/draw_4/target_8740/crafting.pt \
    random/res_gm_b002/draw_4/target_6707/poison.pt \
    random/res_gm_b002/draw_4/target_6707/victim_1/training.pt; do
    [ -s "$ROOT/rebuttal_w4_result/$relative" ] || {
        printf 'ERROR: missing resume state: %s/rebuttal_w4_result/%s\n' "$ROOT" "$relative" >&2
        exit 1
    }
done
export W4_ACCOUNT=aip-bghaddar
exec sh "$ROOT/submit_w4_ghaddar.sh" rand_conv_sapa_b001_2 first_res_gm_b002 rand_res_gm_b002_2 rand_res_gm_b002_3
