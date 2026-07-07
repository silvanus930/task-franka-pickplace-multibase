#!/usr/bin/env bash
# Strategy S5: S2++ smoothness + slow EE when gripper closed (targets CONTAINER_DISPLACED).
STRATEGY_ID="s5_safe_disp"
STRATEGY_LABEL="SafeDisp — strong smoothing + slow motion while closed"
TASK="Nepher-Franka-PickPlace-LL-SafeDisp-v0"
RUN_NAME="safe_disp_finetune"
TARGET_FAILURE="early CONTAINER_DISPLACED / bin bumps"
