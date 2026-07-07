# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the LL goal-conditioned EE-tracking policy.

Network sizing rationale
------------------------
Input: 41-dim observation (9 joint_pos + 9 joint_vel + 7 ee_pose_b +
       7 pose_cmd + 1 grip_cmd + 1 gripper_pos + 7 actions).
MLP [256, 128, 64] is sufficient for a purely reactive controller;
the task has no memory requirement (full Markov state provided).

Training speed guidance (A100 / 4090):
  4096 envs × 24 steps × 5000 iters  ≈  30–60 min to convergence.
  Reduce num_envs if VRAM is limited (2048 works fine, slower).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the LL goal-conditioned EE-tracking policy."""

    num_steps_per_env = 24
    max_iterations = 5_000
    save_interval = 200
    experiment_name = "franka_ll_ee_tracking"
    run_name = ""
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Compact MLP: task is reactive, not memory-intensive.
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LLPPORunnerCfg_SafeFinetune(LLPPORunnerCfg):
    """Conservative PPO settings for short LL finetunes from model_5400.

    * Lower learning rate and exploration noise
    * Shorter default horizon (override with ``--max_iterations`` on CLI)
    * More frequent checkpoint saves for gated eval
    """

    max_iterations = 200
    save_interval = 50
    resume = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class LLPPORunnerCfg_HLFinetune(LLPPORunnerCfg_SafeFinetune):
    """Extra-conservative PPO for short HL-in-loop LL finetunes.

    Lower LR and exploration than empty-table safe finetune; shorter horizon
    with frequent checkpoint saves for gated official eval.
    """

    max_iterations = 80
    save_interval = 25
    resume = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=1.0,
    )


@configclass
class LLPPORunnerCfg_HLFinetuneV3(LLPPORunnerCfg_SafeFinetune):
    """Ultra-conservative PPO for HL-in-loop v3 (mustard / finger_miss focus).

    Shorter than v2, lower LR/noise, frequent saves — pick the best *mid-run*
    checkpoint; later iters consistently regressed in v2 and safe finetunes.
    """

    max_iterations = 40
    save_interval = 10
    resume = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0015,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.8,
    )


@configclass
class LLPPORunnerCfg_HLFinetuneV4(LLPPORunnerCfg_SafeFinetune):
    """Micro PPO for HL-in-loop v4 (grasp-gated, strict terminations).

    Even shorter than v3 — 20 iters, checkpoint every 5. Stop at first peak;
    later checkpoints consistently regressed in v3 and safe finetunes.
    """

    max_iterations = 20
    save_interval = 5
    resume = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.22,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0012,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.8,
    )
