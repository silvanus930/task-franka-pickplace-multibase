#!/usr/bin/env bash
# Start the eval video viewer (auto-discovers MP4s under eval-nav/logs and best_policy/videos).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-3080}"
HOST="${HOST:-0.0.0.0}"

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "[ERROR] Port ${PORT} is already in use."
  echo "        Try:  PORT=3000 $0"
  echo "        Or kill the old process:  fuser -k ${PORT}/tcp"
  exit 1
fi

echo "[EvalVideoViewer] Starting on http://${HOST}:${PORT}/"
echo "[EvalVideoViewer] Press Ctrl+C to stop."
exec python3 "${SCRIPT_DIR}/server.py" --host "${HOST}" --port "${PORT}"
