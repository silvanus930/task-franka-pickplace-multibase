# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the LL policy.

ee_pose_in_robot_base      — current EE pos + quat in robot base frame     (7D)
gripper_pos_normalized     — normalized gripper opening fraction [0, 1]     (1D)
grip_command_obs           — current grip target from GripperCommand         (1D)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_in_robot_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Current end-effector pose expressed in the robot base frame.

    Returns a (N, 7) tensor: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z].
    The quaternion convention follows Isaac Lab (w-first).

    This uses the same reference frame as ``UniformPoseCommandCfg`` so that the
    policy can directly compute the tracking error from the two tensors.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_id: int = asset_cfg.body_ids[0]

    ee_pos_w: torch.Tensor = robot.data.body_pos_w[:, body_id]   # (N, 3)
    ee_quat_w: torch.Tensor = robot.data.body_quat_w[:, body_id]  # (N, 4)

    # Transform world-frame EE pose into the robot root (base) frame.
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)  # (N, 7)


def gripper_pos_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_val: float = 0.04,
) -> torch.Tensor:
    """Normalized mean finger opening fraction in [0, 1].

    Each Franka finger joint spans [0.0, 0.04] m.  The mean of both joints
    is divided by ``open_val`` to produce a value in [0 = closed, 1 = open].

    Returns a (N, 1) tensor.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    # joint_ids contains both panda_finger_joint1 and panda_finger_joint2.
    finger_pos: torch.Tensor = robot.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2)
    mean_opening = finger_pos.mean(dim=-1, keepdim=True)                       # (N, 1)
    return (mean_opening / open_val).clamp(0.0, 1.0)


def grip_command_obs(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Binary gripper target from :class:`GripperCommand`.

    Returns a (N, 1) tensor with values in {0.0 (open), 1.0 (close)}.
    """
    return env.command_manager.get_command(command_name)  # (N, 1)
