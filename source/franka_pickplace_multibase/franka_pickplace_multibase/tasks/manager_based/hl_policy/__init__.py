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

``Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-TimingSafe-v0``
    Same 30-env strict benchmark as EnvhubPlay, with conservative HL
    planner timing for mustard ``finger_miss`` (longer ``grasp_hold_s``,
    slightly deeper mustard ``grasp_z``).  Use with frozen LL checkpoint for
    gated A/B eval — no LL retraining.

``Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-RetryGrasp-v0``
    Same 30-env strict benchmark with ``max_retries=5`` and
    ``grasp_hold_s=1.0``.  Gated A/B eval — no LL retraining.

``Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0``
    One-env diagnostic variant with strict drop/fall checks but relaxed
    incidental container displacement. Use this before official scoring.

``Nepher-Franka-PickPlace-HL-LL-Finetune-v0``
    HL-in-loop v2 (80 iters, S2 smoothness only).

``Nepher-Franka-PickPlace-HL-LL-FinetuneV3-v0``
    HL-in-loop v3 (recommended): mustard / finger_miss focus, 40 iters,
    grip + orientation + mild shallow shaping. Gate on official 30-env eval.

``Nepher-Franka-PickPlace-HL-LL-FinetuneV4-v0``
    HL-in-loop v4 (Option B): grasp-gated grip rewards, eval-strict
    terminations, 20-iter micro-finetune from ``5510``. Gate >= 15/90.

PLAY / SAFE_PLAY use a frozen LL checkpoint; Finetune tasks train the LL
executor against the same classical ``PickPlacePlanner`` state machine.
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
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-TimingSafe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY_TimingSafe",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-RetryGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY_RetryGrasp",
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
# HL-in-loop LL finetune (conservative v2 — Wave 3)
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-Franka-PickPlace-HL-LL-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_Finetune",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg_HLFinetune",
    },
)

# ---------------------------------------------------------------------------
# HL-in-loop LL finetune v3 (mustard / finger_miss — recommended)
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-Franka-PickPlace-HL-LL-FinetuneV3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_FinetuneV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg_HLFinetuneV3",
    },
)

# ---------------------------------------------------------------------------
# HL-in-loop LL finetune v4 (grasp-gated + strict terminations — Option B)
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-Franka-PickPlace-HL-LL-FinetuneV4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_FinetuneV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg_HLFinetuneV4",
    },
)
