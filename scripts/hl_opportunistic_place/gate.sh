#!/usr/bin/env bash
# Parse eval summary and compare against champion gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

SUMMARY="${1:?Usage: $0 <path/to/summary.txt>}"

if [[ ! -f "${SUMMARY}" ]]; then
  echo "Summary not found: ${SUMMARY}" >&2
  exit 1
fi

SCORE="$(grep -E '^Final Score:' "${SUMMARY}" | tail -1 | sed -E 's/.*Final Score: ([0-9.]+).*/\1/')"
SUCCESSES="$(grep -E '^\s+Successful:' "${SUMMARY}" | tail -1 | awk '{print $2}')"

echo "Result: score=${SCORE} successes=${SUCCESSES}/90"
echo "Gate:   score>=${GATE_MIN_SCORE} successes>=${GATE_MIN_SUCCESSES}"

python3 - <<PY
score = float("${SCORE}")
succ = int("${SUCCESSES}")
min_score = float("${GATE_MIN_SCORE}")
min_succ = int("${GATE_MIN_SUCCESSES}")
ok = score >= min_score and succ >= min_succ
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY
