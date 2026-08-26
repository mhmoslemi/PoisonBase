#!/bin/sh
# Submit only the blank ResNet20/VGG13 budget-20/40 cells added to full.tex.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/sbatch/full_b20_b40/submit.sh"
