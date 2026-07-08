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

``Nepher-Franka-PickPlace-HL-LL-Finetune-v0``
    Phase 1 training task: finetune the LL EE-tracking policy *inside* the full
    HL container task (real objects + bin, planner-driven commands, many envs,
    obs noise on, SafePlay terminations).  Resume from the best LL checkpoint to
    close the empty-table train/eval gap::

        python scripts/rsl_rl/train.py \
            --task=Nepher-Franka-PickPlace-HL-LL-Finetune-v0 --headless \
            --resume --checkpoint <path>/model_5400.pt

The PLAY / SAFE_PLAY environments use the same frozen LL RSL-RL checkpoint and
the same classical ``PickPlacePlanner`` state machine; the Finetune task uses
the same planner but trains the LL policy against it.
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
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_SAFE_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Phase 1: HL-in-the-loop LL finetune (training task on the eval distribution)
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-Franka-PickPlace-HL-LL-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_Finetune",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)
