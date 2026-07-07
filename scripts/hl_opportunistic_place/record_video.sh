#!/usr/bin/env bash
# Record a long single-episode video for opportunistic in-bin placement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

EVAL_NAV_ROOT="$(cd "${PROJECT_ROOT}/../eval-nav" && pwd)"
CONFIG="${EVAL_NAV_ROOT}/configs/task-franka-pickplace-multibase-opportunistic-place-video.yaml"

CHECKPOINT="${CHECKPOINT:-${BASELINE_CHECKPOINT}}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${PROJECT_ROOT}/best_policy/best_policy.pt"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found. Set CHECKPOINT=... or sync best_policy." >&2
  exit 1
fi

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

echo "=== OpportunisticPlace video (up to 60 s / 1800 steps) ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Config:     ${CONFIG}"
echo "Vulkan ICD: ${VK_ICD_FILENAMES}"
echo

cd "${EVAL_NAV_ROOT}"
export NEPHER_EVAL_IN_PROCESS=1
python scripts/evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --headless

RUN_DIR="$(ls -td "${EVAL_NAV_ROOT}/logs/franka-pickplace-multibase-hl-opportunistic-place-video"/eval_run_* 2>/dev/null | head -1)"
if [[ -n "${RUN_DIR}" ]]; then
  echo
  echo "Video folder: ${RUN_DIR}/videos/eval/"
  ls -la "${RUN_DIR}/videos/eval/" 2>/dev/null || echo "(no MP4 yet — episode may have ended before flush)"
  echo
  echo "View in browser:"
  echo "  cd ${EVAL_NAV_ROOT}/video-viewer && ./start.sh"
fi
