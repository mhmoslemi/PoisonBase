#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec bash "$ROOT/experiments/rebuttal_tiny_gm/interactive_craft.sh" basis_lam0p1
