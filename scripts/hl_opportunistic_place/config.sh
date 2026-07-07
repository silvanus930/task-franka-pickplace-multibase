#!/usr/bin/env bash
# Champion baseline gate for opportunistic-placement A/B eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_NAV_ROOT="$(cd "${PROJECT_ROOT}/../eval-nav" && pwd)"

# Frozen champion (5510) — do not regress below this.
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-logs/rsl_rl/franka_ll_ee_tracking/2026-07-07_19-01-58_hl_in_loop_v3_finetune/model_5510.pt}"
BASELINE_SCORE="${BASELINE_SCORE:-0.141}"
BASELINE_SUCCESSES="${BASELINE_SUCCESSES:-14}"

BASELINE_CONFIG="${EVAL_NAV_ROOT}/configs/task-franka-pickplace-multibase.yaml"
VARIANT_CONFIG="${EVAL_NAV_ROOT}/configs/task-franka-pickplace-multibase-opportunistic-place.yaml"
VARIANT_LOG_DIR="${EVAL_NAV_ROOT}/logs/franka-pickplace-multibase-hl-opportunistic-place"

GATE_MIN_SCORE="${GATE_MIN_SCORE:-${BASELINE_SCORE}}"
GATE_MIN_SUCCESSES="${GATE_MIN_SUCCESSES:-${BASELINE_SUCCESSES}}"
