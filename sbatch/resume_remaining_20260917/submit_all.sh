#!/usr/bin/env bash
# Submit every still-incomplete transfer and selector configuration.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/submit_transfer.sh"
bash "$HERE/submit_selectors.sh"
