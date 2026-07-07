#!/usr/bin/env bash
# Option B: grasp-gated HL-in-loop v4 (strict terminations, 20-iter micro-finetune).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HL_VERSION=v4
exec "${SCRIPT_DIR}/run_pipeline.sh"
