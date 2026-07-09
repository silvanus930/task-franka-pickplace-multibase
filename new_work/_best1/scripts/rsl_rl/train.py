# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Franka LL goal-conditioned EE-tracking policy with RSL-RL.

Usage
-----
# Train from scratch (headless recommended for speed):
    python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --num_envs 4096

# Train with fewer envs (less VRAM):
    python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --num_envs 2048

# Resume from the latest checkpoint:
    python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --resume

# Resume from a specific checkpoint:
    python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --resume \\
        --checkpoint logs/rsl_rl/franka_ll_ee_tracking/<run>/model_2000.pt

# Cap training iterations:
    python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --max_iterations 3000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train the Franka LL EE-tracking policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Nepher-Franka-PickPlace-LL-v0", help="Registered gym task ID.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import franka_pickplace_multibase.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Entry point: configure, build env, run PPO training."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment to: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Resolve resume checkpoint *before* creating log_dir to avoid an empty
    # timestamped folder being picked up by get_checkpoint_path.
    resume_path = None
    if agent_cfg.resume:
        ckpt_arg = getattr(args_cli, "checkpoint", None)
        if ckpt_arg and os.path.isfile(os.path.expanduser(ckpt_arg)):
            resume_path = os.path.abspath(os.path.expanduser(ckpt_arg))
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO] Resuming from latest checkpoint: {resume_path}")

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume and resume_path is not None:
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
