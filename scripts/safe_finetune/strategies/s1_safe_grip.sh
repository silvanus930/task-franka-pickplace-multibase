#!/usr/bin/env bash
# Strategy S1: mild grasp-contact shaping (targets obj=1 finger_miss).
STRATEGY_ID="s1_safe_grip"
STRATEGY_LABEL="SafeGrip — mild grasp-contact shaping"
TASK="Nepher-Franka-PickPlace-LL-SafeGrip-v0"
RUN_NAME="safe_grip_finetune"
TARGET_FAILURE="obj=1 finger_miss / hold_timeout at grasp"
