#!/usr/bin/env bash
# Strategy S3: mild descend-before-close + shallow Z curriculum (targets placed=4/5).
STRATEGY_ID="s3_safe_shallow"
STRATEGY_LABEL="SafeShallow — descend before close at low Z"
TASK="Nepher-Franka-PickPlace-LL-SafeShallow-v0"
RUN_NAME="safe_shallow_finetune"
TARGET_FAILURE="placed=4/5 / DexCube obj=4 finger_miss"
