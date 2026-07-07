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
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# HL classical planner ``Stage.GRASP`` index (close + hold before verify).
_HL_GRASP_STAGE = 2.0


def _hl_grasp_close_mask(
    env: ManagerBasedRLEnv,
    ee_pose_command_name: str,
    grip_command_name: str,
    grasp_stage: float = _HL_GRASP_STAGE,
) -> torch.Tensor:
    """True when HL is in GRASP and commands a closed gripper."""
    pose_term = env.command_manager.get_term(ee_pose_command_name)
    stage = pose_term.metrics["stage"]
    grip_close = env.command_manager.get_command(grip_command_name).squeeze(-1) > 0.5
    return (stage == grasp_stage) & grip_close


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


def gripper_command_tracking_grasp_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    grip_command_name: str,
    ee_pose_command_name: str = "ee_pose",
    open_val: float = 0.04,
    grasp_stage: float = _HL_GRASP_STAGE,
) -> torch.Tensor:
    """``gripper_command_tracking`` only during HL GRASP + close command."""
    rew = gripper_command_tracking(env, asset_cfg, grip_command_name, open_val=open_val)
    mask = _hl_grasp_close_mask(env, ee_pose_command_name, grip_command_name, grasp_stage)
    return torch.where(mask, rew, torch.zeros_like(rew))


def gripper_grasp_contact_shaping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    grip_command_name: str,
    ee_command_name: str,
    open_val: float = 0.04,
    contact_min: float = 0.010,
    contact_max: float = 0.035,
    empty_max: float = 0.004,
    grasp_z_threshold: float = 0.15,
    pose_error_threshold: float = 0.06,
) -> torch.Tensor:
    """Mild bonus for finger contact and penalty for empty closes at grasp poses."""
    robot: Articulation = env.scene[asset_cfg.name]
    finger_open = robot.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=-1)

    grip_close = env.command_manager.get_command(grip_command_name).squeeze(-1) > 0.5
    ee_cmd = env.command_manager.get_command(ee_command_name)
    cmd_z = ee_cmd[:, 2]

    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_cmd[:, :3]
    )
    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    pos_err = torch.norm(ee_pos_w - des_pos_w, dim=-1)

    at_grasp = (cmd_z <= grasp_z_threshold) & (pos_err < pose_error_threshold) & grip_close

    in_contact = (finger_open >= contact_min) & (finger_open <= contact_max)
    contact_rew = in_contact.float()
    empty_slam = finger_open < empty_max
    shaping = torch.where(empty_slam, -contact_rew.new_ones(()), contact_rew)

    return torch.where(at_grasp, shaping, torch.zeros_like(finger_open))


def gripper_grasp_contact_shaping_grasp_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    grip_command_name: str,
    ee_command_name: str,
    ee_pose_command_name: str = "ee_pose",
    open_val: float = 0.04,
    contact_min: float = 0.010,
    contact_max: float = 0.035,
    empty_max: float = 0.004,
    grasp_z_threshold: float = 0.15,
    pose_error_threshold: float = 0.06,
    grasp_stage: float = _HL_GRASP_STAGE,
) -> torch.Tensor:
    """Contact shaping only during HL GRASP (matches eval finger verify window)."""
    shaping = gripper_grasp_contact_shaping(
        env,
        asset_cfg,
        grip_command_name,
        ee_command_name,
        open_val=open_val,
        contact_min=contact_min,
        contact_max=contact_max,
        empty_max=empty_max,
        grasp_z_threshold=grasp_z_threshold,
        pose_error_threshold=pose_error_threshold,
    )
    mask = _hl_grasp_close_mask(env, ee_pose_command_name, grip_command_name, grasp_stage)
    return torch.where(mask, shaping, torch.zeros_like(shaping))


def no_close_while_high(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    grip_command_name: str,
    ee_command_name: str,
    grasp_z_threshold: float = 0.20,
    z_slack: float = 0.02,
    max_penalty_gap: float = 0.10,
) -> torch.Tensor:
    """Mild penalty for closing before the EE reaches the commanded grasp height."""
    robot: Articulation = env.scene[asset_cfg.name]

    grip_close = env.command_manager.get_command(grip_command_name).squeeze(-1) > 0.5
    ee_cmd = env.command_manager.get_command(ee_command_name)
    cmd_z = ee_cmd[:, 2]

    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_cmd[:, :3]
    )
    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    z_gap = (ee_pos_w[:, 2] - des_pos_w[:, 2]).clamp(min=0.0)

    active = (cmd_z <= grasp_z_threshold) & grip_close & (z_gap > z_slack)
    penalty = -(z_gap - z_slack).clamp(max=max_penalty_gap) / max_penalty_gap

    return torch.where(active, penalty, torch.zeros_like(z_gap))


def ee_speed_while_closed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    grip_command_name: str,
    vel_threshold: float = 0.12,
    low_z_world: float = 0.35,
    low_z_boost: float = 1.5,
) -> torch.Tensor:
    """Penalty for fast EE motion while the gripper is commanded closed.

    Empty-table proxy for HL bin bumps: after a close command the arm should
    translate slowly, especially at table height where the container sits.
    Returns a non-negative violation magnitude (apply a negative reward weight).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    grip_close = env.command_manager.get_command(grip_command_name).squeeze(-1) > 0.5

    ee_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
    speed = torch.norm(ee_vel_w, dim=-1)
    excess = (speed - vel_threshold).clamp(min=0.0)

    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    near_table = ee_pos_w[:, 2] < low_z_world
    boost = torch.where(near_table, excess.new_tensor(low_z_boost), excess.new_ones(()))

    violation = (excess / vel_threshold) * boost
    active = grip_close & (excess > 0.0)
    return torch.where(active, violation, torch.zeros_like(speed))
