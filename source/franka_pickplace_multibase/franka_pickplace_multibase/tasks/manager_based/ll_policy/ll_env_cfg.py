# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-Level goal-conditioned EE tracking environment for Franka.

The LL policy receives a continuously resampled target end-effector pose and a
gripper command, and learns to:
  1. Move the EE to the commanded pose (position + orientation tracking).
  2. Open / close the gripper on command.

Action space (7D):
    arm_action    DifferentialIK delta pose  (Δx, Δy, Δz, Δrx, Δry, Δrz)  [6]
    gripper       binary gripper command                                      [1]

Observation space (41D):
    joint_pos          arm + finger joint positions (relative to default)   [9]
    joint_vel          arm + finger joint velocities                         [9]
    ee_pose_b          current EE pose (pos + quat) in robot base frame     [7]
    pose_command       target EE pose  (pos + quat) from command manager    [7]
    grip_command       target gripper state (0 = open, 1 = close)           [1]
    gripper_pos        current normalised gripper opening [0, 1]             [1]
    actions            last applied action                                   [7]

Commands:
    ee_pose     UniformPoseCommandCfg — resampled every 4 s mid-episode
    grip_cmd    GripperCommandCfg     — resampled only at episode reset
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import franka_pickplace_multibase.tasks.manager_based.ll_policy.mdp as mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


##
# Scene
##


@configclass
class LLSceneCfg(InteractiveSceneCfg):
    """Franka on a lab table."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # Stiff high-PD gains required for accurate IK tracking.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# Commands
##


@configclass
class CommandsCfg:
    """EE pose target and gripper target (both resampled mid-episode)."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Franka reachable workspace on the table
            pos_x=(0.25, 0.65),
            pos_y=(-0.40, 0.40),
            pos_z=(0.05, 0.55),
            # Orientation: gripper mostly pointing down with full yaw freedom.
            roll=(-0.3, 0.3),
            pitch=(2.8, math.pi),
            yaw=(-math.pi, math.pi),
        ),
    )

    # Resampled every 1–2 s so the LL policy learns open/close transitions needed by HL.
    grip_cmd = mdp.GripperCommandCfg(
        resampling_time_range=(1.0, 2.0),
        close_prob=0.5,
    )


##
# Actions
##


@configclass
class ActionsCfg:
    """IK-Rel 6D arm delta + binary 1D gripper = 7D total action."""

    arm_action: ActionTerm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action: ActionTerm = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


##
# Observations
##


@configclass
class ObservationsCfg:
    """41-dimensional policy observation (fully concatenated)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # --- Joint state (9D each: 7 arm + 2 fingers) ---
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # --- Current EE state in robot base frame (7D: pos + quat) ---
        ee_pose_b = ObsTerm(
            func=mdp.ee_pose_in_robot_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")},
        )

        # --- Commands (7D pose + 1D grip) ---
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )
        grip_command = ObsTerm(
            func=mdp.grip_command_obs,
            params={"command_name": "grip_cmd"},
        )

        # --- Gripper state (1D normalised opening fraction) ---
        gripper_pos = ObsTerm(
            func=mdp.gripper_pos_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])},
        )

        # --- Previous action (7D: 6 IK-Rel + 1 gripper) ---
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events
##


@configclass
class EventCfg:
    """Reset events executed at each episode boundary."""

    # Randomise only the 7 arm joints. Finger joints are reset separately to
    # their safe default opening so random scaling cannot request invalid widths.
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_finger_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    """Dense tracking rewards for EE pose and gripper state."""

    # ---- Position tracking ----
    ee_pos_tracking_coarse = RewTerm(
        func=mdp.position_command_error,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "command_name": "ee_pose",
        },
    )
    ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "command_name": "ee_pose",
        },
    )

    # ---- Orientation tracking ----
    ee_ori_tracking_coarse = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "command_name": "ee_pose",
        },
    )
    ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=1.5,
        params={
            "std": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "command_name": "ee_pose",
        },
    )

    # ---- Gripper state tracking ----
    grip_tracking = RewTerm(
        func=mdp.gripper_command_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "command_name": "grip_cmd",
        },
    )

    # ---- Smoothness penalties (ramped up by curriculum) ----
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Terminations
##


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Curriculum
##


@configclass
class CurriculumCfg:
    """Ramp smoothness penalties to encourage fluid motion over training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.03, "num_steps": 10_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.005, "num_steps": 10_000},
    )


##
# Environment configurations
##


@configclass
class LLEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for the LL EE-tracking policy."""

    scene: LLSceneCfg = LLSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 60 Hz physics, 30 Hz policy (decimation = 2).
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 60.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class LLEnvCfg_PLAY(LLEnvCfg):
    """Evaluation configuration: fewer envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
