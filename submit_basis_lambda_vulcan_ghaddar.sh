#!/bin/sh
# Vulcan's 33-job half; shared launcher preserves the same scientific snapshot.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BASIS_SITE=vulcan exec bash "$SCRIPT_DIR/submit_basis_lambda_killarney_ghaddar.sh" "$@"
