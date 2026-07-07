#!/usr/bin/env bash
# Full HL-in-loop pipeline: train → eval checkpoints → gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${SCRIPT_DIR}/results"

BASELINE_CKPT_NAME="$(basename "${BASELINE_CKPT}")"

echo ""
echo "################################################################"
echo "# HL-in-loop LL finetune pipeline (${HL_VERSION})"
echo "# Task: ${TASK}"
echo "# Gate : score >= ${BASELINE_SCORE} (${BASELINE_SUCCESSES}/90)"
if [[ -n "${GATE_MIN_SUCCESSES:-}" ]]; then
  echo "# Promote: >= ${GATE_MIN_SUCCESSES}/90 successes"
fi
echo "# Tip  : promote best *mid-run* ckpt, not the last one"
echo "################################################################"
echo ""

if [[ "${SKIP_BASELINE_EVAL:-0}" != "1" ]]; then
  echo "--- Step 0: baseline reference (${BASELINE_CKPT_NAME}) ---"
  "${SCRIPT_DIR}/eval.sh" "${BASELINE_CKPT}" "${RESULTS_TAG}_baseline_ref" || true
fi

echo "--- Step 1: train (${TRAIN_ITERS} iters from ${BASELINE_CKPT_NAME}, ${NUM_ENVS} envs) ---"
"${SCRIPT_DIR}/train.sh"

RUN_DIR="$(cat "${SCRIPT_DIR}/.last_run")"
mapfile -t CKPTS < <(ls -1 "${RUN_DIR}"/model_*.pt 2>/dev/null | sort -V)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoints in ${RUN_DIR}" >&2
  exit 1
fi

BEST_CKPT=""
BEST_SCORE="-1"
PREV_SCORE="-1"
GATE_LOG="${SCRIPT_DIR}/results/${RESULTS_TAG}_gate_summary.txt"
: > "${GATE_LOG}"

echo "--- Step 2: official eval on saved checkpoints ---" | tee -a "${GATE_LOG}"
for CKPT in "${CKPTS[@]}"; do
  if [[ "${SKIP_START_CKPT:-1}" == "1" && "$(basename "${CKPT}")" == "${BASELINE_CKPT_NAME}" ]]; then
    echo "" | tee -a "${GATE_LOG}"
    echo "Skipping start checkpoint copy ${BASELINE_CKPT_NAME}" | tee -a "${GATE_LOG}"
    continue
  fi

  TAG="${RESULTS_TAG}_$(basename "${CKPT}" .pt)"
  echo "" | tee -a "${GATE_LOG}"
  echo "Evaluating ${CKPT}" | tee -a "${GATE_LOG}"
  if "${SCRIPT_DIR}/eval.sh" "${CKPT}" "${TAG}"; then
    RESULT_JSON="${PROJECT_ROOT}/evaluation_result_${TAG}_$(basename "${CKPT}" .pt).json"
    SCORE="$(python -c "import json; print(json.load(open('${RESULT_JSON}'))['score'])")"
    echo "PASS ${CKPT} score=${SCORE}" | tee -a "${GATE_LOG}"
    if python -c "import sys; sys.exit(0 if float('${SCORE}') > float('${BEST_SCORE}') else 1)"; then
      BEST_SCORE="${SCORE}"
      BEST_CKPT="${CKPT}"
    fi
    if [[ "${PREV_SCORE}" != "-1" ]] && python -c "import sys; sys.exit(0 if float('${SCORE}') < float('${PREV_SCORE}') else 1)"; then
      echo "WARN: score regressed vs previous ckpt (${PREV_SCORE} -> ${SCORE})" | tee -a "${GATE_LOG}"
    fi
    PREV_SCORE="${SCORE}"
  else
    echo "FAIL ${CKPT} (below baseline ${BASELINE_SCORE})" | tee -a "${GATE_LOG}"
    PREV_SCORE="0"
  fi
done

echo "" | tee -a "${GATE_LOG}"
echo "============================================================" | tee -a "${GATE_LOG}"
if [[ -n "${BEST_CKPT}" ]]; then
  echo "BEST HL-in-loop (${HL_VERSION}): ${BEST_CKPT} (score=${BEST_SCORE})" | tee -a "${GATE_LOG}"
  echo "${BEST_CKPT}" > "${SCRIPT_DIR}/results/${RESULTS_TAG}_best_ckpt.txt"
  echo "${BEST_SCORE}" > "${SCRIPT_DIR}/results/${RESULTS_TAG}_best_score.txt"
else
  echo "No checkpoint beat baseline ${BASELINE_SCORE}. Keep ${BASELINE_CKPT_NAME}" | tee -a "${GATE_LOG}"
fi
echo "Gate log: ${GATE_LOG}" | tee -a "${GATE_LOG}"
echo "============================================================" | tee -a "${GATE_LOG}"
