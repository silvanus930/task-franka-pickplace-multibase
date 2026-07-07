#!/usr/bin/env bash
# Convenience wrapper: HL-in-loop v3 (recommended).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HL_VERSION=v3
exec "${SCRIPT_DIR}/run_pipeline.sh"
