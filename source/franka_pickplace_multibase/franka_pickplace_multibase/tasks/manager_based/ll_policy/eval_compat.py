# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation compatibility wrapper for Franka pick-and-place environments.

Exposes a uniform ``task_completed`` / ``task_failed`` interface consumed by
the eval-nav framework, mirroring the contract implemented in task-spot-nav's
``eval_compat.py``.

Design:
- ``task_completed``: a success-named termination term fired (term name
  contains ``'goal'``, ``'success'``, or ``'complete'``).  Covers
  ``cube_at_goal`` (HL container task) and ``all_objects_reached_goals``
  (multi-object EnvHub path).  Intentionally excludes failure terms
  (``cube_fell``, ``object_dropped``, ``container_fell``,
  ``container_displaced``) even though they are also non-timeout terms.
- ``task_failed``: any object-fell / drop / displaced / explicit failure
  termination OR a hard timeout.  Kept separate from ``task_completed`` so
  the scorer can distinguish successes, clean failures, and timeouts.

``franka_pickplace_multibase/__init__.py`` exports ``EvalCompatEnv`` and
``wrap_for_eval`` from this module so eval-nav can discover them via the
``task_module`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EvalCompatEnv:
    """Thin wrapper that exposes manipulation environment state to eval-nav."""

    def __init__(self, env: "ManagerBasedRLEnv") -> None:
        self._env = env

    # ------------------------------------------------------------------
    # Attribute pass-through
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    # ------------------------------------------------------------------
    # Eval-nav interface: task_completed / task_failed
    # ------------------------------------------------------------------

    @property
    def task_completed(self) -> torch.Tensor:
        """``True`` for envs where a success termination fired.

        Iterates all active termination terms and ORs together any whose name
        contains ``'goal'``, ``'success'``, or ``'complete'``.  This covers
        ``cube_at_goal`` (container task) and ``all_objects_reached_goals``
        (multi-object EnvHub path) while correctly excluding non-timeout
        failure terms such as ``cube_fell``, ``object_dropped``,
        ``container_fell``, and ``container_displaced``.
        """
        try:
            base_env = self._env.unwrapped
            num_envs = base_env.num_envs
            device = base_env.device
            success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            tm = base_env.termination_manager
            for term_name in tm.active_terms:
                lower = term_name.lower()
                if "goal" in lower or "success" in lower or "complete" in lower:
                    success = success | tm.get_term(term_name).bool()
            return success
        except (AttributeError, KeyError):
            return torch.zeros(1, dtype=torch.bool)

    @property
    def task_failed(self) -> torch.Tensor:
        """``True`` for envs where a failure termination or timeout occurred.

        Matches any termination term whose name contains ``'fell'``,
        ``'drop'``, ``'fail'``, ``'displace'``, or ``'unsafe'``, plus hard
        timeouts. It intentionally does not classify every ``container`` term
        as failure because container-named success terms are valid in this
        task family.
        (``time_out=True`` terms that fired at the episode step limit).
        This covers ``cube_fell``, ``object_dropped``, ``container_fell``,
        ``container_displaced``, and ``time_out``.
        """
        try:
            base_env = self._env.unwrapped
            num_envs = base_env.num_envs
            device = base_env.device
            failure = torch.zeros(num_envs, dtype=torch.bool, device=device)
            tm = base_env.termination_manager
            for term_name in tm.active_terms:
                cfg = tm.get_term_cfg(term_name)
                lower = term_name.lower()
                if (
                    cfg.time_out
                    or "fell" in lower
                    or "drop" in lower
                    or "fail" in lower
                    or "displace" in lower
                    or "unsafe" in lower
                ):
                    failure = failure | tm.get_term(term_name).bool()
            return failure
        except (AttributeError, KeyError):
            return torch.zeros(1, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Gym interface pass-through
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)

    # ------------------------------------------------------------------
    # State logging (consumed by eval-nav StateLogger)
    # ------------------------------------------------------------------

    def _log_state(self, env_idx: int | None = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {}
        idx = env_idx if env_idx is not None else 0
        try:
            robot = self._env.unwrapped.scene["robot"]
            if robot is not None:
                pos_w = robot.data.root_pos_w
                state["position"] = pos_w[idx, :3].cpu().numpy() if torch.is_tensor(pos_w) else pos_w[idx, :3]
                quat_w = robot.data.root_quat_w[idx]
                state["quat_w"] = float(quat_w[0].cpu().item())
                state["joint_pos"] = robot.data.joint_pos[idx].cpu().numpy()
        except (AttributeError, KeyError):
            pass

        for prop_name in ("task_completed", "task_failed"):
            try:
                val = getattr(self, prop_name)
            except Exception:
                continue
            try:
                if torch.is_tensor(val):
                    state[prop_name] = bool(val[idx].cpu().item()) if val.numel() > 1 else bool(val.cpu().item())
                else:
                    state[prop_name] = bool(val)
            except Exception:
                pass

        return state


def wrap_for_eval(env: "ManagerBasedRLEnv") -> EvalCompatEnv:
    """Wrap a Franka pick-and-place HL environment for evaluation with eval-nav."""
    return EvalCompatEnv(env)
