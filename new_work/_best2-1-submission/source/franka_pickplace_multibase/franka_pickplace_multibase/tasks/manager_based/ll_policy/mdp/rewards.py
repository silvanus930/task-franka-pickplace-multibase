# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward terms for the LL policy.

EE-tracking rewards
  orientation_command_error_tanh  — tanh-kernel orientation tracking reward
  gripper_command_tracking        — soft reward for matching grip target

Standard position / orientation reward functions are re-exported from the
Isaac Lab reach task mdp (via the wildcard import in __init__.py):

  position_command_error          — L2 position penalty          (weight < 0)
  position_command_error_tanh     — tanh position reward         (weight > 0)
  orientation_command_error       — L2 orientation penalty       (weight < 0)

Design note
-----------
The LL policy is a *reactive executor*: it does not decide WHERE to go — that
is the High-Level policy's job.  It only needs to reach commanded poses and
match the commanded gripper state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward EE orientation tracking using a tanh kernel.

    Maps quaternion error magnitude (radians) through ``1 - tanh(error / std)``.
    Returns high reward (≈1) when aligned, smoothly decreasing to 0 with error.

    Args:
        std: Bandwidth in radians.  Use ~0.1–0.2 rad for fine orientation tracking.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]  # (N, 4), w-first quaternion in base frame
    # Rotate desired quat from base frame to world frame.
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    curr_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (N, 4)

    error = quat_error_magnitude(curr_quat_w, des_quat_w)  # (N,) in radians
    return 1.0 - torch.tanh(error / std)


def gripper_command_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    open_val: float = 0.04,
) -> torch.Tensor:
    """Soft reward for matching the gripper state to the commanded target.

    Computes the absolute difference between the current normalized opening
    fraction and the target (0 = open, 1 = close), then maps through
    ``1 - tanh(error / 0.2)`` so the reward is ≈1 when matched and ≈0 when
    fully opposite.

    Returns a (N,) tensor in [0, 1].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    finger_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2)
    current_fraction = finger_pos.mean(dim=-1) / open_val       # (N,) in [0, 1]

    # grip_target: 0.0 = open, 1.0 = close
    grip_target = env.command_manager.get_command(command_name).squeeze(-1)  # (N,)

    # Convert grip_target to the same "fraction" convention as current_fraction:
    # grip_target=0 (open) → target_fraction=1.0
    # grip_target=1 (close) → target_fraction=0.0
    target_fraction = 1.0 - grip_target  # (N,)

    error = (current_fraction - target_fraction).abs()  # (N,)
    return 1.0 - torch.tanh(error / 0.2)
