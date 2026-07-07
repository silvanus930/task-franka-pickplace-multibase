# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative LL finetune environment variants (empty-table only).

Three safe strategies targeting the main eval failure modes without HL-in-loop
training or aggressive command / termination changes:

  S1 SafeGrip     — mild grasp-contact shaping (obj=1 finger_miss)
  S2 SafeSmooth   — slightly stronger smoothness penalties (bin displacement)
  S3 SafeShallow  — mild descend-before-close + existing shallow-Z curriculum (4/5)
  S4 SafeCombo    — S1 grip (0.3) + S2 smoothness (balanced tradeoff)
  S5 SafeDisp     — stronger smoothness + slow-EE-when-closed (bin displacement)
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import franka_pickplace_multibase.tasks.manager_based.ll_policy.mdp as mdp

from .ll_env_cfg import CurriculumCfg, LLEnvCfg, RewardsCfg


@configclass
class RewardsCfg_SafeGrip(RewardsCfg):
    """Baseline rewards + mild HL-aligned contact shaping at grasp height."""

    grip_contact_safe = RewTerm(
        func=mdp.gripper_grasp_contact_shaping,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand", joint_names=["panda_finger_joint.*"]),
            "grip_command_name": "grip_cmd",
            "ee_command_name": "ee_pose",
            "contact_min": 0.010,
            "contact_max": 0.035,
            "empty_max": 0.004,
            "grasp_z_threshold": 0.15,
            "pose_error_threshold": 0.06,
        },
    )


@configclass
class RewardsCfg_SafeSmooth(RewardsCfg):
    """Slightly higher smoothness penalties from step zero (conservative bump)."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg_SafeSmooth(CurriculumCfg):
    """Ramp smoothness a bit further than the baseline curriculum."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.035, "num_steps": 8_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.004, "num_steps": 8_000},
    )


@configclass
class RewardsCfg_SafeShallow(RewardsCfg):
    """Baseline rewards + mild penalty for closing above grasp Z."""

    no_close_high = RewTerm(
        func=mdp.no_close_while_high,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "grip_command_name": "grip_cmd",
            "ee_command_name": "ee_pose",
            "grasp_z_threshold": 0.18,
            "z_slack": 0.02,
            "max_penalty_gap": 0.08,
        },
    )


@configclass
class CurriculumCfg_SafeShallow(CurriculumCfg):
    """Keep shallow-Z sampling but bias slightly lower than baseline."""

    ee_pose_z_table = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.ee_pose.ranges.pos_z",
            "modify_fn": mdp.override_pose_z_range,
            "modify_params": {"z_range": (0.02, 0.12), "num_steps": 6_000},
        },
    )


@configclass
class RewardsCfg_SafeCombo(RewardsCfg):
    """S2 smoothness + mild S1 grip contact (conservative combo)."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    grip_contact_safe = RewTerm(
        func=mdp.gripper_grasp_contact_shaping,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand", joint_names=["panda_finger_joint.*"]),
            "grip_command_name": "grip_cmd",
            "ee_command_name": "ee_pose",
            "contact_min": 0.010,
            "contact_max": 0.035,
            "empty_max": 0.004,
            "grasp_z_threshold": 0.15,
            "pose_error_threshold": 0.06,
        },
    )


@configclass
class CurriculumCfg_SafeDisp(CurriculumCfg):
    """Ramp smoothness beyond S2 for displacement-focused finetune."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.045, "num_steps": 8_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.005, "num_steps": 8_000},
    )


@configclass
class RewardsCfg_SafeDisp(RewardsCfg):
    """S2++ smoothness + mild slow-motion penalty when gripper is closed."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.020)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.003,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    slow_ee_closed = RewTerm(
        func=mdp.ee_speed_while_closed,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "grip_command_name": "grip_cmd",
            "vel_threshold": 0.12,
            "low_z_world": 0.35,
            "low_z_boost": 1.5,
        },
    )


@configclass
class LLEnvCfg_SafeGripFinetune(LLEnvCfg):
    """S1: conservative grip-contact finetune on empty table."""

    rewards: RewardsCfg_SafeGrip = RewardsCfg_SafeGrip()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.debug_vis = False


@configclass
class LLEnvCfg_SafeSmoothFinetune(LLEnvCfg):
    """S2: conservative motion-smoothness finetune on empty table."""

    rewards: RewardsCfg_SafeSmooth = RewardsCfg_SafeSmooth()
    curriculum: CurriculumCfg_SafeSmooth = CurriculumCfg_SafeSmooth()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.debug_vis = False


@configclass
class LLEnvCfg_SafeShallowFinetune(LLEnvCfg):
    """S3: conservative shallow-grasp / descend-before-close finetune."""

    rewards: RewardsCfg_SafeShallow = RewardsCfg_SafeShallow()
    curriculum: CurriculumCfg_SafeShallow = CurriculumCfg_SafeShallow()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.debug_vis = False


@configclass
class LLEnvCfg_SafeComboFinetune(LLEnvCfg):
    """S4: mild grip shaping + motion smoothness finetune on empty table."""

    rewards: RewardsCfg_SafeCombo = RewardsCfg_SafeCombo()
    curriculum: CurriculumCfg_SafeSmooth = CurriculumCfg_SafeSmooth()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.debug_vis = False


@configclass
class LLEnvCfg_SafeDispFinetune(LLEnvCfg):
    """S5: displacement-focused smoothness finetune on empty table."""

    rewards: RewardsCfg_SafeDisp = RewardsCfg_SafeDisp()
    curriculum: CurriculumCfg_SafeDisp = CurriculumCfg_SafeDisp()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.debug_vis = False
