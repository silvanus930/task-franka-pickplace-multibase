# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for the LL goal-conditioned EE tracking environment.

Re-exports the full Isaac Lab standard MDP library so that ll_env_cfg.py can
import everything through this single ``mdp`` namespace.

Custom additions
----------------
commands.py
    GripperCommand / GripperCommandCfg

observations.py
    ee_pose_in_robot_base      (7D)  current EE pose in robot base frame
    gripper_pos_normalized     (1D)  normalised gripper opening [0, 1]
    grip_command_obs           (1D)  binary grip target (0=open, 1=close)

rewards.py
    orientation_command_error_tanh   tanh-kernel orientation tracking reward
    gripper_command_tracking         soft reward for matching grip target
"""

# Pull in the complete standard Isaac Lab MDP library (observations, rewards,
# terminations, events, commands, curriculum utilities …).
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Also pull in the reach-task reward functions that are not in the core MDP lib.
from isaaclab_tasks.manager_based.manipulation.reach.mdp import (  # noqa: F401
    orientation_command_error,
    position_command_error,
    position_command_error_tanh,
)

# Custom command terms.
from .commands import GripperCommand, GripperCommandCfg  # noqa: F401

# Custom observation terms.
from .observations import (  # noqa: F401
    ee_pose_in_robot_base,
    grip_command_obs,
    gripper_pos_normalized,
)

# Custom reward terms.
from .rewards import (  # noqa: F401
    ee_speed_while_closed,
    gripper_command_tracking,
    gripper_command_tracking_grasp_gated,
    gripper_grasp_contact_shaping,
    gripper_grasp_contact_shaping_grasp_gated,
    no_close_while_high,
    orientation_command_error_tanh,
)

# Custom curriculum terms.
from .curriculums import override_pose_z_range  # noqa: F401
