#!/usr/bin/env bash
# Run official 30-env × 3-round eval for one checkpoint + config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

CHECKPOINT="${1:?Usage: $0 <checkpoint.pt> [baseline|variant]}"
VARIANT="${2:-variant}"

if [[ "${VARIANT}" == "baseline" ]]; then
  CONFIG="${BASELINE_CONFIG}"
else
  CONFIG="${VARIANT_CONFIG}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

cd "${EVAL_NAV_ROOT}"
export NEPHER_EVAL_IN_PROCESS=1
python scripts/evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --headless
