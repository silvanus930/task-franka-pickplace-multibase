# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained Franka LL EE-tracking policy checkpoint.

Syncs the latest (or requested) RSL-RL checkpoint from ``logs/`` into
``best_policy/best_policy.pt``, loads from there, runs the environment, and exports
JIT (.pt) and ONNX (.onnx) under ``best_policy/exported/``.

Usage
-----
    # Latest checkpoint from logs (synced to best_policy first):
    python play.py --task=Nepher-Franka-PickPlace-LL-Play-v0

    # Specific checkpoint (also copied to best_policy):
    python play.py --task=Nepher-Franka-PickPlace-LL-Play-v0 \\
        --checkpoint logs/rsl_rl/franka_ll_ee_tracking/<run>/model_5000.pt

    # Record a video:
    python play.py --task=Nepher-Franka-PickPlace-LL-Play-v0 --video --video_length 300

    # Run real-time (throttle to sim dt):
    python play.py --task=Nepher-Franka-PickPlace-LL-Play-v0 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a trained Franka LL EE-tracking policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during evaluation.")
parser.add_argument("--video_length", type=int, default=300, help="Number of steps to record.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
parser.add_argument("--task", type=str, default="Nepher-Franka-PickPlace-LL-Play-v0", help="Registered gym task ID.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle to real-time speed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import franka_pickplace_multibase.tasks  # noqa: F401

import policy_paths  # isort: skip


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load checkpoint and run the LL policy in the simulator."""
    # Resolve task name (strip trailing ":override" if present).
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root = policy_paths.log_root_path(agent_cfg.experiment_name)
    print(f"[INFO] LL policy logs: {log_root}")
    print(f"[INFO] LL policy deploy dir: {policy_paths.BEST_POLICY_DIR}")

    resume_path = policy_paths.sync_best_policy(
        agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        explicit_checkpoint=args_cli.checkpoint,
    )

    env_cfg.log_dir = policy_paths.BEST_POLICY_DIR

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(policy_paths.BEST_POLICY_DIR, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy for downstream use (HL policy, real-robot deployment, etc.).
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = policy_paths.BEST_POLICY_EXPORT_DIR
    os.makedirs(export_dir, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="ll_policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="ll_policy.onnx")
    print(f"[INFO] Exported policy to: {export_dir}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            if dones.any():
                unwrapped = env.unwrapped
                success = unwrapped.termination_manager.get_term("cube_at_goal")
                for i in torch.where(dones)[0].tolist():
                    if success[i]:
                        print(f"[INFO] env {i}: SUCCESS — pick-and-place complete, resetting")
                    elif unwrapped.reset_time_outs[i]:
                        print(f"[INFO] env {i}: episode timeout — resetting")

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
