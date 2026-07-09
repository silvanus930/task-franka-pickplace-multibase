# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-Level classical pick-and-place environment for Franka — container task.

The HL env extends the LL EE-tracking environment in three ways:

1. **Scene** – adds 5 varied YCB USD objects scattered on the table plus a
   static KLT bin container in one corner.

2. **Commands** – replaces the random ``UniformPoseCommand`` / ``GripperCommand``
   pair with ``HLPoseCommand`` + ``HLGripCommand``, driven by the
   ``PickPlacePlanner`` state machine.  Per-object grasp metadata (grasp Z,
   symmetry, yaw offset) is passed to the planner each step so it can
   correctly approach varied shapes.  The LL observation is structurally
   identical to training (same 41-D, same field names), so the frozen LL
   checkpoint runs via ``play.py`` unchanged.

3. **Events / Terminations** – scatters objects at random non-overlapping
   positions each episode and terminates when all objects are detected inside
   the container footprint.

Usage::

    python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-Play-v0
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import franka_pickplace_multibase.tasks.manager_based.hl_policy.mdp as mdp
from franka_pickplace_multibase.tasks.manager_based.hl_policy.mdp.object_assets import (
    CONTAINER_CFG,
    OBJECT_CATALOG,
    ContainerGeomCfg,
    make_container_rigid_cfg,
    make_object_rigid_cfg,
)
from franka_pickplace_multibase.tasks.manager_based.ll_policy.ll_env_cfg import LLEnvCfg, LLSceneCfg


##
# Geometry constants (exported for backward compat — EnvHub imports GOAL_POS_DEFAULT)
##

TABLE_SURFACE_Z: float = 0.03
GOAL_MARKER_THICKNESS: float = 0.002

# Container drop height = rim Z + a small drop offset (so LOWER targets just above the rim).
_C = CONTAINER_CFG
_DROP_Z_LOCAL: float = _C.rim_z + _C.drop_height_above_rim  # ~0.13 m local frame

# GOAL_POS_DEFAULT exported for EnvHub backward compat (centre of bin opening at drop Z).
GOAL_POS_DEFAULT: tuple[float, float, float] = (_C.pos[0], _C.pos[1], _DROP_Z_LOCAL)

# Spawn region for objects (robot-local XY).
# Wide enough so 5 objects can always be placed >= 10 cm from the container,
# regardless of where the container is sampled within its own range.
# Container range: x in [0.45, 0.62], y in [-0.25, -0.12].
# Objects are rejected if within interior + 10 cm clearance of the container,
# so the entire table front face is available and rejection sampling handles exclusion.
HL_SPAWN_POSE_RANGE: dict[str, tuple[float, float]] = {
    "x": (0.30, 0.75),        # within arm reach, in front of the robot
    "y": (-0.05, 0.30),       # positive-y side of the table, away from the typical bin area
    "z": (0.0, 0.0),
}


##
# Scene
##


@configclass
class HLSceneCfg(LLSceneCfg):
    """Extends LLSceneCfg with 5 varied graspable objects and a KLT bin container."""

    # ---- Graspable objects (one per catalog entry) ----
    object0: RigidObjectCfg = make_object_rigid_cfg("{ENV_REGEX_NS}/Object0", OBJECT_CATALOG[0])
    object1: RigidObjectCfg = make_object_rigid_cfg("{ENV_REGEX_NS}/Object1", OBJECT_CATALOG[1])
    object2: RigidObjectCfg = make_object_rigid_cfg("{ENV_REGEX_NS}/Object2", OBJECT_CATALOG[2])
    object3: RigidObjectCfg = make_object_rigid_cfg("{ENV_REGEX_NS}/Object3", OBJECT_CATALOG[3])
    object4: RigidObjectCfg = make_object_rigid_cfg("{ENV_REGEX_NS}/Object4", OBJECT_CATALOG[4])

    # ---- Dynamic KLT bin container (heavy rigid body; position randomised each episode) ----
    container: RigidObjectCfg = make_container_rigid_cfg("{ENV_REGEX_NS}/Container", CONTAINER_CFG)


##
# Helpers
##

# HLSceneCfg spawns the first 5 catalog entries (object0–object4).
# Slice here so commands, events, and terminations reference only objects
# that exist in the scene; OBJECT_CATALOG may have more entries for use by
# the typed-scenario envhub path (hl_env_cfg_envhub.py).
_NUM_SCENE_OBJECTS: int = 5
_SCENE_CATALOG = OBJECT_CATALOG[:_NUM_SCENE_OBJECTS]

_OBJECT_NAMES: list[str] = [obj.name for obj in _SCENE_CATALOG]  # ["object0".."object4"]
_M: int = len(_OBJECT_NAMES)

# Per-object grasp metadata lists (aligned with _OBJECT_NAMES).
_GRASP_Z_OFFSETS:   list[float] = [obj.grasp_z_offset              for obj in _SCENE_CATALOG]
_GRASP_SYMS:        list[float] = [obj.grasp_sym                   for obj in _SCENE_CATALOG]
# Use effective_grasp_yaw_offset() so objects with footprint_xy set automatically
# receive a π/2 rotation when their short axis is along local X, aligning the
# gripper fingers with the narrower dimension for a successful grasp.
_GRASP_YAW_OFFSETS: list[float] = [obj.effective_grasp_yaw_offset() for obj in _SCENE_CATALOG]
_FOOTPRINT_RADII:   list[float] = [obj.footprint_radius              for obj in _SCENE_CATALOG]

# Container drop marker config: visualise the bin opening as a blue rectangle.
_TABLE_HALF_X, _TABLE_HALF_Y = CONTAINER_CFG.table_interior_half_extents()
_MARKER_CFG = mdp.make_container_marker_cfg(
    half_x=_TABLE_HALF_X,
    half_y=_TABLE_HALF_Y,
    thickness=GOAL_MARKER_THICKNESS,
)


##
# Commands
##


@configclass
class HLCommandsCfg:
    """Command terms for the HL container task.

    Field names must match the LL observation terms (``"ee_pose"`` and
    ``"grip_cmd"``) so the frozen LL policy's 41-D observation is
    structurally identical to training.
    """

    ee_pose: mdp.HLPoseCommandCfg = mdp.HLPoseCommandCfg(
        # Objects in pick order (matched to HLSceneCfg attributes).
        cube_names=_OBJECT_NAMES,
        # All goals point to the container drop zone (overwritten per episode
        # by the scatter event, but a reasonable default for init-time sampling).
        goal_pos_defaults=[GOAL_POS_DEFAULT] * _M,
        table_surface_z=TABLE_SURFACE_Z,
        marker_thickness=GOAL_MARKER_THICKNESS,
        cube_size_xy=CONTAINER_CFG.interior_half_x * 2.0,
        goal_pose_visualizer_cfg=_MARKER_CFG,
        debug_vis=False,
        enable_log=True,
        log_interval=100,
        log_env_id=1,
        # Per-object grasp metadata.
        grasp_z_offsets=_GRASP_Z_OFFSETS,
        grasp_syms=_GRASP_SYMS,
        grasp_yaw_offsets=_GRASP_YAW_OFFSETS,
        # Container drop mode: goal Z = rim + offset; no yaw gate.
        container_drop=True,
        container_retract_xy_offset=_C.retract_xy_offset_table,
        # Carry height raised above the container rim (~11 cm + margin).
        carry_z=0.22,
        # Goal ranges: effectively zero (scatter event sets exact goals).
        ranges=mdp.HLPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    grip_cmd: mdp.HLGripCommandCfg = mdp.HLGripCommandCfg()


##
# Events
##


@configclass
class HLEventCfg:
    """HL reset events for the container task.

    Execution order (dict insertion order):
      1. reset_scene              – reset robot and all objects to defaults.
      2. reset_robot_joints       – randomise arm joint positions.
      3. reset_cube_and_goal_poses – scatter objects, set container goals.

    Note: HLEventCfg does NOT inherit from EventCfg to preserve reset ordering.
    """

    reset_scene: EventTerm = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    reset_robot_joints: EventTerm = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Attribute named "reset_cube_and_goal_poses" for EnvHub backward compatibility
    # (HLEnvCfg_Envhub replaces this attribute by name).
    reset_cube_and_goal_poses: EventTerm = EventTerm(
        func=mdp.reset_scattered_objects_into_container,
        mode="reset",
        params={
            "object_cfgs":                [SceneEntityCfg(name) for name in _OBJECT_NAMES],
            "footprint_radii":            _FOOTPRINT_RADII,
            "spawn_region":               HL_SPAWN_POSE_RANGE,
            "container_cfg":              SceneEntityCfg("container"),
            "container_pos_local":        (_C.pos[0], _C.pos[1], _C.pos[2]),
            "container_pos_range":        {"x": _C.pos_range_x, "y": _C.pos_range_y},
            "container_interior_half_x":  _C.interior_half_x,
            "container_interior_half_y":  _C.interior_half_y,
            "container_drop_z_local":     _DROP_Z_LOCAL,
            "container_clearance":        _C.object_clearance,
            "object_spacing":             _C.object_spacing,
            "pose_cmd_name":              "ee_pose",
        },
    )


##
# Terminations
##


@configclass
class HLTerminationsCfg:
    """Termination conditions for the HL container task."""

    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)

    # Success: all objects are detected inside the container (reads live container pose).
    # Attribute named "cube_at_goal" for EnvHub backward compatibility.
    cube_at_goal: DoneTerm = DoneTerm(
        func=mdp.objects_in_container,
        params={
            "object_cfgs":                [SceneEntityCfg(name) for name in _OBJECT_NAMES],
            "robot_cfg":                  SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "pose_cmd_name":              "ee_pose",
            "container_cfg":              SceneEntityCfg("container"),
            "container_pos_world_offset": (_C.pos[0], _C.pos[1], _C.pos[2]),
            "container_interior_half_x":  _C.interior_half_x,
            "container_interior_half_y":  _C.interior_half_y,
            "grip_open_threshold":        0.8,
            "success_dwell_s":            1.0,
            "enable_log":                 True,
        },
    )

    # Failure: any object fell off the table.
    # Attribute named "cube_fell" for EnvHub backward compatibility.
    # minimum_height is env-local Z (world Z - env_origin Z).  Table-top
    # objects live at local Z ~0.04–0.07 m; anything at or below 0.0 m has
    # left the table surface (lower shelf, table legs, or simulation floor).
    cube_fell: DoneTerm = DoneTerm(
        func=mdp.any_object_fell,
        params={
            "object_cfgs":   [SceneEntityCfg(name) for name in _OBJECT_NAMES],
            "minimum_height": 0.0,
        },
    )

    # Failure: gripper lost an object during LIFT or CARRY (dropped mid-transport).
    object_dropped: DoneTerm = DoneTerm(
        func=mdp.object_dropped_mid_carry,
        params={
            "object_cfgs":        [SceneEntityCfg(name) for name in _OBJECT_NAMES],
            "pose_cmd_name":      "ee_pose",
            "drop_height_world":  0.10,   # object below 10 cm during transport = dropped
            "enable_log":         True,
        },
    )

    # Failure: container knocked off the table or tipped over.
    container_fell: DoneTerm = DoneTerm(
        func=mdp.container_fell,
        params={
            "container_cfg":   SceneEntityCfg("container"),
            "minimum_height":  -0.05,   # world Z below which container is off the table
            "tilt_threshold":  0.5,     # ~28° roll/pitch before considered tipped
            "enable_log":      True,
        },
    )

    # Failure: container displaced more than 2 cm from its episode-start position.
    container_displaced: DoneTerm = DoneTerm(
        func=mdp.container_displaced,
        params={
            "container_cfg":    SceneEntityCfg("container"),
            "max_displacement": _C.max_displacement,
            "enable_log":       True,
        },
    )


##
# Environment configuration
##


@configclass
class HLEnvCfg(LLEnvCfg):
    """Evaluation configuration for the HL container pick-and-place task.

    Inherits all LL components (actions, observations, rewards, curriculum)
    and overrides scene, commands, events, terminations, and episode length.
    """

    scene:        HLSceneCfg        = HLSceneCfg(num_envs=4, env_spacing=2.5)
    commands:     HLCommandsCfg     = HLCommandsCfg()
    events:       HLEventCfg        = HLEventCfg()
    terminations: HLTerminationsCfg = HLTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 5 objects × ~7 s/object budget → ~35 s episode.
        self.episode_length_s = 35.0
        self.observations.policy.enable_corruption = False

        # LLEnvCfg lowers bounce_threshold_velocity to 0.01 m/s, which forces
        # PhysX to apply restitution on nearly every contact and makes objects
        # bounce when dropped into the bin (most visible when a second object
        # lands on top of one already inside).  Restore the Isaac Lab default so
        # sub-0.5 m/s impacts are treated as inelastic resting contacts.
        self.sim.physx.bounce_threshold_velocity = 0.5
