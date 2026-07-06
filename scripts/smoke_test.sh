#!/usr/bin/env bash
# Smoke-test gate for Franka PickPlace Multibase.
#
# Usage (from anywhere, with Isaac Lab env active):
#   ./scripts/smoke_test.sh              # phases 0–4 (install → SafePlay)
#   ./scripts/smoke_test.sh --check-only # verify artifacts from a prior run
#   ./scripts/smoke_test.sh --full       # also run 30-env EnvHub benchmark
#   ./scripts/smoke_test.sh --skip-install
#
# Exit code 0 = all executed checks passed. Non-zero = at least one failure.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/franka_ll_ee_tracking"
BEST_POLICY="${PROJECT_ROOT}/best_policy/best_policy.pt"
EXPORT_PT="${PROJECT_ROOT}/best_policy/exported/ll_policy.pt"
EXPORT_ONNX="${PROJECT_ROOT}/best_policy/exported/ll_policy.onnx"
SMOKE_LOG="${PROJECT_ROOT}/logs/smoke_test.log"

CHECK_ONLY=0
FULL=0
SKIP_INSTALL=0

for arg in "$@"; do
  case "${arg}" in
    --check-only) CHECK_ONLY=1 ;;
    --full) FULL=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "${SMOKE_LOG}")"

declare -a PHASE_NAMES=()
declare -a PHASE_STATUS=()
declare -a PHASE_DETAIL=()

pass() {
  local name="$1"
  local detail="$2"
  PHASE_NAMES+=("${name}")
  PHASE_STATUS+=("PASS")
  PHASE_DETAIL+=("${detail}")
  echo "[PASS] ${name}: ${detail}"
}

fail() {
  local name="$1"
  local detail="$2"
  PHASE_NAMES+=("${name}")
  PHASE_STATUS+=("FAIL")
  PHASE_DETAIL+=("${detail}")
  echo "[FAIL] ${name}: ${detail}" >&2
}

run_cmd() {
  local logfile="$1"
  shift
  echo "+ $*" | tee -a "${logfile}"
  "$@" >>"${logfile}" 2>&1
}

latest_checkpoint() {
  find "${LOG_ROOT}" -name 'model_*.pt' -type f 2>/dev/null | sort | tail -1
}

grep_metrics_episodes() {
  local logfile="$1"
  local line
  line="$(grep -E '\[INFO\] Evaluation metrics:' "${logfile}" 2>/dev/null | tail -1 || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  if [[ "${line}" =~ episodes=([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# Phase 0 — install & registration
# ---------------------------------------------------------------------------
phase0_install() {
  local name="0_install_registration"
  if [[ "${SKIP_INSTALL}" -eq 1 ]]; then
    pass "${name}" "skipped (--skip-install)"
    return 0
  fi

  local log="${SMOKE_LOG}.phase0"
  : >"${log}"
  if ! run_cmd "${log}" python -m pip install -e source/franka_pickplace_multibase; then
    fail "${name}" "pip install -e failed (see ${log})"
    return 1
  fi
  if ! run_cmd "${log}" pip install nepher; then
    fail "${name}" "pip install nepher failed (see ${log})"
    return 1
  fi

  local reg_log="${SMOKE_LOG}.phase0.reg"
  if ! python - <<'PY' >"${reg_log}" 2>&1
import gymnasium as gym
import franka_pickplace_multibase  # noqa: F401

ids = sorted(k for k in gym.registry if "Nepher-Franka-PickPlace" in k)
print("Registered:", ids)
assert len(ids) == 5, f"expected 5 envs, got {len(ids)}: {ids}"
PY
  then
    fail "${name}" "gym registration check failed (see ${reg_log})"
    return 1
  fi

  pass "${name}" "package installed, 5 gym envs registered"
  return 0
}

# ---------------------------------------------------------------------------
# Phase 1 — smoke train
# ---------------------------------------------------------------------------
phase1_train() {
  local name="1_smoke_train"
  local log="${SMOKE_LOG}.phase1"
  : >"${log}"

  if ! run_cmd "${log}" python scripts/rsl_rl/train.py \
    --task=Nepher-Franka-PickPlace-LL-v0 \
    --headless \
    --num_envs 8 \
    --max_iterations 2; then
    fail "${name}" "train.py exited non-zero (see ${log})"
    return 1
  fi

  local ckpt
  ckpt="$(latest_checkpoint)"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    fail "${name}" "no model_*.pt under ${LOG_ROOT}"
    return 1
  fi

  if ! grep -q 'Logging experiment to:' "${log}"; then
    fail "${name}" "missing log-root confirmation in train output"
    return 1
  fi

  pass "${name}" "checkpoint written: ${ckpt}"
  return 0
}

# ---------------------------------------------------------------------------
# Phase 2 — LL play
# ---------------------------------------------------------------------------
phase2_ll_play() {
  local name="2_ll_play"
  local log="${SMOKE_LOG}.phase2"
  : >"${log}"

  if ! run_cmd "${log}" python scripts/rsl_rl/play.py \
    --task=Nepher-Franka-PickPlace-LL-Play-v0 \
    --headless \
    --num_envs 4 \
    --max_steps 200; then
    fail "${name}" "play.py (LL) exited non-zero (see ${log})"
    return 1
  fi

  if [[ ! -f "${BEST_POLICY}" ]]; then
    fail "${name}" "missing ${BEST_POLICY}"
    return 1
  fi
  if [[ ! -f "${EXPORT_PT}" || ! -f "${EXPORT_ONNX}" ]]; then
    fail "${name}" "missing exported policy under best_policy/exported/"
    return 1
  fi
  if ! grep -q 'Synced LL policy to best_policy' "${log}"; then
    fail "${name}" "checkpoint sync message not found (see ${log})"
    return 1
  fi

  local episodes
  episodes="$(grep_metrics_episodes "${log}" || true)"
  if [[ -z "${episodes}" || "${episodes}" -lt 1 ]]; then
    fail "${name}" "no completed episodes in metrics (see ${log})"
    return 1
  fi

  pass "${name}" "policy synced/exported, episodes=${episodes}"
  return 0
}

# ---------------------------------------------------------------------------
# Phase 3 — HL local play
# ---------------------------------------------------------------------------
phase3_hl_play() {
  local name="3_hl_local_play"
  local log="${SMOKE_LOG}.phase3"
  : >"${log}"

  if ! run_cmd "${log}" python scripts/rsl_rl/play.py \
    --task=Nepher-Franka-PickPlace-HL-Multibase-Play-v0 \
    --headless \
    --num_envs 2 \
    --max_episodes 4 \
    --max_steps 2000; then
    fail "${name}" "play.py (HL local) exited non-zero (see ${log})"
    return 1
  fi

  local episodes
  episodes="$(grep_metrics_episodes "${log}" || true)"
  if [[ -z "${episodes}" || "${episodes}" -lt 1 ]]; then
    fail "${name}" "no completed episodes in metrics (see ${log})"
    return 1
  fi

  pass "${name}" "HL local episodes completed: ${episodes}"
  return 0
}

# ---------------------------------------------------------------------------
# Phase 4 — EnvHub SafePlay
# ---------------------------------------------------------------------------
phase4_safeplay() {
  local name="4_envhub_safeplay"
  local log="${SMOKE_LOG}.phase4"
  : >"${log}"

  if ! run_cmd "${log}" python scripts/rsl_rl/play.py \
    --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0 \
    --headless \
    --max_episodes 5 \
    --max_steps 1500; then
    fail "${name}" "play.py (SafePlay) exited non-zero (see ${log})"
    return 1
  fi

  local episodes
  episodes="$(grep_metrics_episodes "${log}" || true)"
  if [[ -z "${episodes}" || "${episodes}" -lt 1 ]]; then
    fail "${name}" "no completed episodes in metrics (see ${log})"
    return 1
  fi

  pass "${name}" "SafePlay episodes completed: ${episodes}"
  return 0
}

# ---------------------------------------------------------------------------
# Phase 5 — EnvHub benchmark (optional)
# ---------------------------------------------------------------------------
phase5_benchmark() {
  local name="5_envhub_benchmark"
  local log="${SMOKE_LOG}.phase5"
  : >"${log}"

  if ! run_cmd "${log}" python scripts/rsl_rl/play.py \
    --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0 \
    --headless \
    --num_envs 30 \
    --max_episodes 90 \
    --max_steps 2000; then
    fail "${name}" "play.py (benchmark) exited non-zero (see ${log})"
    return 1
  fi

  local episodes
  episodes="$(grep_metrics_episodes "${log}" || true)"
  if [[ -z "${episodes}" || "${episodes}" -lt 1 ]]; then
    fail "${name}" "no completed episodes in metrics (see ${log})"
    return 1
  fi

  pass "${name}" "benchmark episodes completed: ${episodes}"
  return 0
}

# ---------------------------------------------------------------------------
# Check-only mode (no Isaac Sim re-run)
# ---------------------------------------------------------------------------
check_only() {
  phase0_install || true

  local ckpt
  ckpt="$(latest_checkpoint)"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    pass "1_smoke_train" "checkpoint exists: ${ckpt}"
  else
    fail "1_smoke_train" "no checkpoint under ${LOG_ROOT}"
  fi

  if [[ -f "${BEST_POLICY}" && -f "${EXPORT_PT}" && -f "${EXPORT_ONNX}" ]]; then
    pass "2_ll_play" "best_policy + exports present"
  else
    fail "2_ll_play" "missing best_policy or exports"
  fi

  for phase in phase2 phase3 phase4 phase5; do
    local log="${SMOKE_LOG}.${phase}"
    local label
    case "${phase}" in
      phase2) label="2_ll_play_metrics" ;;
      phase3) label="3_hl_local_play_metrics" ;;
      phase4) label="4_envhub_safeplay_metrics" ;;
      phase5) label="5_envhub_benchmark_metrics" ;;
    esac
    if [[ ! -f "${log}" ]]; then
      fail "${label}" "log missing: ${log}"
      continue
    fi
    local episodes
    episodes="$(grep_metrics_episodes "${log}" || true)"
    if [[ -n "${episodes}" && "${episodes}" -ge 1 ]]; then
      pass "${label}" "episodes=${episodes} in ${log}"
    else
      fail "${label}" "no Evaluation metrics line in ${log}"
    fi
  done
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
  echo
  echo "========== SMOKE TEST SUMMARY =========="
  local failed=0
  for i in "${!PHASE_NAMES[@]}"; do
    printf "  %-28s %s\n" "${PHASE_NAMES[$i]}" "${PHASE_STATUS[$i]}"
    printf "    %s\n" "${PHASE_DETAIL[$i]}"
    if [[ "${PHASE_STATUS[$i]}" == "FAIL" ]]; then
      failed=$((failed + 1))
    fi
  done
  echo "========================================"
  if [[ "${failed}" -eq 0 ]]; then
    echo "RESULT: ALL CHECKS PASSED"
    return 0
  fi
  echo "RESULT: ${failed} CHECK(S) FAILED"
  echo "Logs: ${SMOKE_LOG}.*"
  return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  echo "Project root: ${PROJECT_ROOT}"
  echo "Smoke log prefix: ${SMOKE_LOG}"

  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    check_only
    print_summary
    exit $?
  fi

  phase0_install
  phase1_train
  phase2_ll_play
  phase3_hl_play
  phase4_safeplay
  if [[ "${FULL}" -eq 1 ]]; then
    phase5_benchmark
  fi

  print_summary
}

main "$@"
