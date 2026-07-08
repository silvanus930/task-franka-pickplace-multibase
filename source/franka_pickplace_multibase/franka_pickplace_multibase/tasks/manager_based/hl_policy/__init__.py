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
    ``env_id`` / ``scene_id``.  Default preset is
    ``franka-pickplace-multibase-sample`` — a 30-scenario typed benchmark
    with an 8-type YCB catalog (5 objects active per episode), the
    :class:`TypedPrebakedScenarioStrategy`, and per-episode object-type
    variation.  Override ``env_id`` to use any other compatible preset
    (e.g. ``franka-pickplace-base-sample``).  Run via ``play.py``::

        python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0

``Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0``
    One-env diagnostic variant with strict drop/fall checks but relaxed
    incidental container displacement. Use this before official scoring.

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

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-OpportunisticPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY_OpportunisticPlace",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-Safe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY_SAFE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-OpportunisticPlace-Safe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY_OpportunisticPlace_SAFE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-OpportunisticPlace-Video-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY_OpportunisticPlace_VIDEO",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_SAFE_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)
