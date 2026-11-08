#!/bin/sh
# Only the ten W4 tasks previously submitted as 2175937..2175946.
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)" || exit 1
export W4_ACCOUNT=aip-yiweilu
exec bash "$ROOT/submit_w4_ghaddar.sh" \
    rand_res_sapa_b001_3 \
    first_res_sapa_b002 \
    rand_res_sapa_b002_1 \
    rand_res_sapa_b002_2 \
    rand_res_sapa_b002_3 \
    first_res_sapa_b005 \
    rand_res_sapa_b005_1 \
    rand_res_sapa_b005_2 \
    rand_res_sapa_b005_3 \
    rand_res_sapa_b005_4
