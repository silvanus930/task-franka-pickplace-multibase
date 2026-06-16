# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-Level goal-conditioned EE tracking policy for Franka.

Trains a policy that moves the end-effector to any commanded target pose
(x, y, z, rotx, roty, rotz, grip) from an arbitrary arm configuration.

Registered environments:
  Nepher-Franka-PickPlace-LL-v0        — training  (4096 envs, noise on)
  Nepher-Franka-PickPlace-LL-Play-v0   — evaluation (32 envs, noise off)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-Franka-PickPlace-LL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Franka-PickPlace-LL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)
