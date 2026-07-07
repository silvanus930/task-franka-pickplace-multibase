#!/usr/bin/env bash
# Strategy S4: mild S1 grip (0.3) + S2 smoothness (targets grasp + bin displacement).
STRATEGY_ID="s4_safe_combo"
STRATEGY_LABEL="SafeCombo — grip contact (0.3) + motion smoothing"
TASK="Nepher-Franka-PickPlace-LL-SafeCombo-v0"
RUN_NAME="safe_combo_finetune"
TARGET_FAILURE="grasp_miss / finger_miss + CONTAINER_DISPLACED"
