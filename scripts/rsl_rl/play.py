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
parser.add_argument(
    "--video_warmup_steps",
    type=int,
    default=50,
    help="Sim steps to run before capturing frames (primes headless renderer).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
parser.add_argument("--task", type=str, default="Nepher-Franka-PickPlace-LL-Play-v0", help="Registered gym task ID.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--preset", type=str, default=None, help="Override EnvHub preset/env_id for HL EnvHub play tasks.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop play/eval after this many vectorized env steps.")
parser.add_argument("--max_episodes", type=int, default=None, help="Stop play/eval after this many completed episodes across envs.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle to real-time speed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

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


class _ManualVideoRecorder:
    """Record MP4 frames without gymnasium RecordVideo (Isaac Sim 5.1 SD-graph safe)."""

    def __init__(self, folder: str, video_length: int, warmup_steps: int, fps: float) -> None:
        self.folder = folder
        self.video_length = video_length
        self.warmup_steps = max(0, warmup_steps)
        self.fps = max(1.0, fps)
        self.frames: list = []
        self._primed = False

    def _prime_renderer(self, base_env) -> None:
        if self._primed:
            return
        for _ in range(5):
            base_env.sim.render()
        for _ in range(10):
            frame = base_env.render()
            if frame is not None and frame.size > 0 and frame.any():
                break
        self._primed = True

    def maybe_capture(self, base_env, step_idx: int) -> None:
        if step_idx < self.warmup_steps:
            return
        if len(self.frames) >= self.video_length:
            return
        if not self._primed:
            self._prime_renderer(base_env)
        frame = base_env.render()
        if frame is not None and frame.size > 0:
            self.frames.append(frame)

    def save(self) -> str | None:
        if not self.frames:
            print("[WARN] No video frames captured.")
            return None
        os.makedirs(self.folder, exist_ok=True)
        out_path = os.path.join(self.folder, "rl-video-step-0.mp4")
        import imageio

        imageio.mimsave(out_path, self.frames, fps=self.fps)
        print(f"[INFO] Saved video ({len(self.frames)} frames) to: {out_path}")
        return out_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load checkpoint and run the LL policy in the simulator."""
    # Resolve task name (strip trailing ":override" if present).
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.preset is not None:
        if not hasattr(env_cfg, "env_id"):
            raise ValueError("--preset is only supported by EnvHub-backed tasks with an env_id field.")
        env_cfg.env_id = args_cli.preset
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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    video_recorder = None
    if args_cli.video:
        video_folder = os.path.join(policy_paths.BEST_POLICY_DIR, "videos", "play")
        video_recorder = _ManualVideoRecorder(
            folder=video_folder,
            video_length=args_cli.video_length,
            warmup_steps=args_cli.video_warmup_steps,
            fps=1.0 / env.unwrapped.step_dt,
        )
        print("[INFO] Manual video recording enabled (avoids gym RecordVideo SD-graph crash).")
        print_dict(
            {
                "video_folder": video_folder,
                "video_length": args_cli.video_length,
                "video_warmup_steps": args_cli.video_warmup_steps,
                "fps": video_recorder.fps,
            },
            nesting=4,
        )

    print(f"[INFO] Loading model checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy for downstream use (HL policy, real-robot deployment, etc.).
    policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "actor_critic", None)
    if policy_nn is None:
        raise AttributeError("Could not find policy module on runner.alg")
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = policy_paths.BEST_POLICY_EXPORT_DIR
    os.makedirs(export_dir, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="ll_policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="ll_policy.onnx")
    print(f"[INFO] Exported policy to: {export_dir}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    episode_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    eval_metrics = {
        "episodes": 0,
        "success": 0,
        "failure": 0,
        "timeout": 0,
        "drop": 0,
        "episode_length_sum": 0.0,
    }

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            episode_steps += 1
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

            if dones.any():
                unwrapped = env.unwrapped
                termination_manager = unwrapped.termination_manager
                active_terms = set(getattr(termination_manager, "active_terms", []))
                success = torch.zeros_like(dones, dtype=torch.bool)
                failure = torch.zeros_like(dones, dtype=torch.bool)
                drop = torch.zeros_like(dones, dtype=torch.bool)
                for term_name in active_terms:
                    lower = term_name.lower()
                    term_done = termination_manager.get_term(term_name).bool()
                    if "goal" in lower or "success" in lower or "complete" in lower:
                        success |= term_done
                    if "fell" in lower or "drop" in lower or "fail" in lower or "displace" in lower or "unsafe" in lower:
                        failure |= term_done
                    if "drop" in lower:
                        drop |= term_done
                for i in torch.where(dones)[0].tolist():
                    eval_metrics["episodes"] += 1
                    eval_metrics["episode_length_sum"] += float(episode_steps[i].item())
                    eval_metrics["success"] += int(success[i].item())
                    eval_metrics["drop"] += int(drop[i].item())
                    if success[i]:
                        print(f"[INFO] env {i}: SUCCESS — pick-and-place complete, resetting")
                    elif unwrapped.reset_time_outs[i]:
                        eval_metrics["timeout"] += 1
                        print(f"[INFO] env {i}: episode timeout — resetting")
                    elif failure[i]:
                        eval_metrics["failure"] += 1
                        print(f"[INFO] env {i}: task failure — resetting")
                    episode_steps[i] = 0

        if video_recorder is not None:
            video_recorder.maybe_capture(env.unwrapped, timestep)
            timestep += 1
            if timestep >= video_recorder.warmup_steps + video_recorder.video_length:
                break
        elif args_cli.max_steps is not None:
            timestep += 1
            if timestep >= args_cli.max_steps:
                break

        if args_cli.max_episodes is not None and eval_metrics["episodes"] >= args_cli.max_episodes:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    episodes = eval_metrics["episodes"]
    avg_episode_length = eval_metrics["episode_length_sum"] / episodes if episodes else 0.0
    success_rate = eval_metrics["success"] / episodes if episodes else 0.0
    print(
        "[INFO] Evaluation metrics: "
        f"episodes={episodes}, success={eval_metrics['success']}, "
        f"failure={eval_metrics['failure']}, timeout={eval_metrics['timeout']}, "
        f"drop={eval_metrics['drop']}, avg_episode_length={avg_episode_length:.2f}, "
        f"success_rate={success_rate:.3f}"
    )

    if video_recorder is not None:
        video_recorder.save()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
