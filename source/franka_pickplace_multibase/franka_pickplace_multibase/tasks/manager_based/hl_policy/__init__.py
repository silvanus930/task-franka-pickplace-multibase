# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HL classical pick-and-place policy for Franka.

Registered environments:

``Nepher-Franka-PickPlace-HL-Multibase-Play-v0``
    Standard evaluation configuration (4 envs, no obs noise) with the
    hardcoded SeattleLabTable + DexCube scene.  Run via ``play.py``::

        python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-Play-v0

``Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0``
    Multi-base evaluation driven by a Nepher EnvHub preset: the scene
    (table, objects, lighting) is loaded from a preset identified by
    ``env_id`` / ``scene_id``.  Supports heterogeneous object types
    (rigid, deformable, articulated) and arbitrary goal configurations
    defined in the preset.  Run via ``play.py`` or evaluated with
    ``eval-nav``::

        python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0

Both environments use the same frozen LL RSL-RL checkpoint and the same
classical ``PickPlacePlanner`` state machine.
"""

import gymnasium as gym

from franka_pickplace_multibase.tasks.manager_based.ll_policy import agents

# ---------------------------------------------------------------------------
# Standard HL (hardcoded scene)
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg:HLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Multi-base HL (preset scene loaded from Nepher EnvHub)
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)
