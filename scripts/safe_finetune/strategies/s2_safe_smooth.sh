#!/usr/bin/env bash
# Strategy S2: slightly stronger smoothness penalties (targets CONTAINER_DISPLACED).
STRATEGY_ID="s2_safe_smooth"
STRATEGY_LABEL="SafeSmooth — conservative motion smoothing"
TASK="Nepher-Franka-PickPlace-LL-SafeSmooth-v0"
RUN_NAME="safe_smooth_finetune"
TARGET_FAILURE="early CONTAINER_DISPLACED (bin bumps)"
