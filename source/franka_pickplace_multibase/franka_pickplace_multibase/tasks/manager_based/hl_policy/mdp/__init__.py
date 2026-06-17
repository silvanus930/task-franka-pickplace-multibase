# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for the HL classical pick-and-place / container environment.

Pulls in the full standard Isaac Lab MDP library (via the LL mdp namespace)
plus the reach-task reward helpers, then re-exports the HL-specific additions.
"""

# Full standard Isaac Lab MDP + reach rewards + LL custom MDP
from franka_pickplace_multibase.tasks.manager_based.ll_policy.mdp import *  # noqa: F401, F403

# HL-specific command terms
from .commands import (  # noqa: F401
    HLGripCommand,
    HLGripCommandCfg,
    HLPoseCommand,
    HLPoseCommandCfg,
    make_goal_marker_cfg,
    make_container_marker_cfg,
)

# Object catalog + container geometry (used by hl_env_cfg and events)
from .object_assets import (  # noqa: F401
    CONTAINER_CFG,
    OBJECT_CATALOG,
    ContainerGeomCfg,
    GraspObjectCfg,
    container_to_table_interior_half_extents,
    container_drop_slot_offsets_table,
    make_container_asset_cfg,
    make_container_rigid_cfg,
    make_object_rigid_cfg,
)

# HL-specific reset events
from .events import (  # noqa: F401
    reset_cube_and_goal_poses,
    reset_objects_and_goals,
    reset_scattered_objects_into_container,
    reset_typed_objects_from_scenario,
)

# HL-specific terminations
from .terminations import (  # noqa: F401
    all_objects_reached_goals,
    any_object_fell,
    container_displaced,
    container_fell,
    cube_reached_goal,
    object_dropped_mid_carry,
    objects_in_container,
)
