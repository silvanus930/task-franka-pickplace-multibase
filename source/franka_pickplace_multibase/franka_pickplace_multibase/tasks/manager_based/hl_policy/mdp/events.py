# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the HL pick-and-place environment."""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from .commands import HLPoseCommand
from .spawn_utils import sample_xyz_offsets
from .object_assets import container_drop_slot_offsets_table, container_to_table_interior_half_extents


# ---------------------------------------------------------------------------
# Container task: non-overlap scatter + container drop goals
# ---------------------------------------------------------------------------


def reset_scattered_objects_into_container(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfgs: list[SceneEntityCfg],
    footprint_radii: list[float],
    spawn_region: dict[str, tuple[float, float]],
    container_pos_local: tuple[float, float, float],
    container_interior_half_x: float,
    container_interior_half_y: float,
    container_drop_z_local: float,
    pose_cmd_name: str = "ee_pose",
    max_attempts: int = 100,
    container_cfg: SceneEntityCfg | None = None,
    container_pos_range: dict[str, tuple[float, float]] | None = None,
    container_clearance: float = 0.10,
    object_spacing: float = 0.03,
) -> None:
    """Scatter M objects on the desk, randomise the container position, and set goals.

    The container is repositioned each episode by sampling a random robot-local
    XY position from ``container_pos_range``.  Its episode-start world pose is
    stored on ``env._hl_container_home_w`` for use by the
    ``container_displaced`` termination term.

    Objects are placed at random, non-overlapping positions in ``spawn_region``
    that keep at least ``container_clearance`` metres away from the container
    exterior and at least ``object_spacing`` metres away from each other.
    Each object's goal is set to the (sampled) container centre at the drop Z.

    Args:
        env:                        The RL environment.
        env_ids:                    Indices of envs being reset.
        object_cfgs:                Scene entity configs for each object (pick order).
        footprint_radii:            Per-object XY footprint radius (m) for overlap rejection.
        spawn_region:               Dict with ``"x"``, ``"y"`` ranges for scatter (local frame).
        container_pos_local:        Default container centre (x, y, z) in robot-local frame
                                    (used as fallback when ``container_pos_range`` is None).
        container_interior_half_x:  Container-local interior half-extent in x (m).
        container_interior_half_y:  Container-local interior half-extent in y (m).
        container_drop_z_local:     Z height for the drop target (local frame, e.g. rim Z).
        pose_cmd_name:              Name of the :class:`HLPoseCommand` term.
        max_attempts:               Rejection-sampling attempts per object per env.
        container_cfg:              Scene entity config for the container asset.  When
                                    provided the container is repositioned and its home
                                    pose is recorded.
        container_pos_range:        Dict with ``"x"`` and ``"y"`` ranges for sampling the
                                    container centre each episode (robot-local frame).
                                    When ``None``, ``container_pos_local`` is used unchanged.
        container_clearance:        Extra XY clearance (m) between object spawn and the
                                    container exterior wall (on top of the object radius).
        object_spacing:             Minimum extra gap (m) between object footprint circles.
    """
    if env_ids.numel() == 0:
        return

    N = env_ids.numel()
    M = len(object_cfgs)
    dev = env.device

    cix = container_interior_half_x
    ciy = container_interior_half_y
    half_table_x, half_table_y = container_to_table_interior_half_extents(cix, ciy)

    # ------------------------------------------------------------------
    # 1. Sample and write container position
    # ------------------------------------------------------------------
    if container_pos_range is not None:
        cx_range = container_pos_range.get("x", (container_pos_local[0], container_pos_local[0]))
        cy_range = container_pos_range.get("y", (container_pos_local[1], container_pos_local[1]))
        cx_l = torch.empty(N, device=dev).uniform_(*cx_range)  # (N,)
        cy_l = torch.empty(N, device=dev).uniform_(*cy_range)  # (N,)
    else:
        cx_l = torch.full((N,), container_pos_local[0], device=dev)
        cy_l = torch.full((N,), container_pos_local[1], device=dev)

    cz_l = container_pos_local[2]

    if container_cfg is not None:
        container_local_pos = torch.stack(
            [cx_l, cy_l, torch.full((N,), cz_l, device=dev)], dim=-1
        )  # (N, 3)
        identity_rot = torch.zeros(N, 4, device=dev)
        identity_rot[:, 0] = 1.0
        _reset_object_to_pose(env, env_ids, container_cfg, container_local_pos, identity_rot)

        # Store episode-start world position for displacement checking.
        if not hasattr(env, "_hl_container_home_w"):
            env._hl_container_home_w = torch.zeros(env.num_envs, 3, device=dev)
        origins = env.scene.env_origins[env_ids]
        env._hl_container_home_w[env_ids] = origins + container_local_pos

    # ------------------------------------------------------------------
    # 2. Scatter objects with clearance and spacing constraints
    # ------------------------------------------------------------------
    x_range = spawn_region.get("x", (-0.10, 0.10))
    y_range = spawn_region.get("y", (-0.20, 0.20))

    placed_xy: list[torch.Tensor] = []   # list of (N, 2) tensors
    placed_radii: list[float] = []

    for m in range(M):
        r_m = footprint_radii[m] if m < len(footprint_radii) else 0.05
        best = None

        # Clearance from container exterior (interior half + object radius + extra clearance).
        excl_x = half_table_x + r_m + container_clearance
        excl_y = half_table_y + r_m + container_clearance

        for _ in range(max_attempts):
            px = torch.empty(N, device=dev).uniform_(*x_range)
            py = torch.empty(N, device=dev).uniform_(*y_range)

            # Reject positions inside or too close to the container footprint.
            in_container = (
                (px - cx_l).abs() < excl_x
            ) & (
                (py - cy_l).abs() < excl_y
            )

            # Reject positions overlapping already-placed objects.
            too_close = torch.zeros(N, dtype=torch.bool, device=dev)
            for prev_i, prev_xy in enumerate(placed_xy):
                dist = torch.sqrt((px - prev_xy[:, 0]) ** 2 + (py - prev_xy[:, 1]) ** 2)
                min_sep = r_m + placed_radii[prev_i] + object_spacing
                too_close = too_close | (dist < min_sep)

            invalid = in_container | too_close
            if best is None:
                best = torch.stack([px, py], dim=-1)
            accepted = ~invalid
            best = torch.where(
                accepted.unsqueeze(-1),
                torch.stack([px, py], dim=-1),
                best,
            )
            if accepted.all():
                break

        placed_xy.append(best)   # (N, 2)
        placed_radii.append(r_m)

        # Build (N, 3) local-frame spawn position.
        default_z = object_cfgs[m].init_state.pos[2] if hasattr(object_cfgs[m], "init_state") else 0.055
        pz = torch.full((N,), default_z, device=dev)
        local_pos = torch.stack([best[:, 0], best[:, 1], pz], dim=-1)  # (N, 3)

        # Random yaw for spawn.
        yaw = torch.empty(N, device=dev).uniform_(-math.pi, math.pi)
        zeros = torch.zeros_like(yaw)
        local_rot = quat_from_euler_xyz(zeros, zeros, yaw)  # (N, 4)

        _reset_object_to_pose(env, env_ids, object_cfgs[m], local_pos, local_rot)

    # ------------------------------------------------------------------
    # 3. Set drop goals at fixed per-object slots inside the bin
    # ------------------------------------------------------------------
    goal_pos_local = torch.zeros(N, M, 3, device=dev)
    goal_rot_local = torch.zeros(N, M, 4, device=dev)
    goal_rot_local[:, :, 0] = 1.0  # identity quaternion

    slot_offsets = container_drop_slot_offsets_table(M, half_table_x, half_table_y)
    for m, (jx, jy) in enumerate(slot_offsets):
        goal_pos_local[:, m, 0] = cx_l + jx
        goal_pos_local[:, m, 1] = cy_l + jy
        goal_pos_local[:, m, 2] = container_drop_z_local

    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    pose_term.set_goals_from_strategy(env_ids, goal_pos_local, goal_rot_local)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _reset_object_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obj_cfg: SceneEntityCfg,
    local_pos: torch.Tensor,   # (N, 3) robot-local frame
    local_rot: torch.Tensor,   # (N, 4) wxyz
) -> None:
    """Set a rigid object's root state to an exact world-frame pose."""
    from isaaclab.assets import RigidObject

    obj: RigidObject = env.scene[obj_cfg.name]

    origins   = env.scene.env_origins[env_ids]
    world_pos = origins + local_pos

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3]  = world_pos
    root_state[:, 3:7] = local_rot
    root_state[:, 7:]  = 0.0

    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


# ---------------------------------------------------------------------------
# EnvHub multi-object path (unchanged)
# ---------------------------------------------------------------------------


def reset_objects_and_goals(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfgs: list[SceneEntityCfg],
    scenario_strategy,
    pose_cmd_name: str = "ee_pose",
) -> None:
    """Reset all object spawns and goal poses using a :class:`PrebakedScenarioStrategy`.

    Retained for the EnvHub path (``HLEnvCfg_Envhub``).
    """
    if env_ids.numel() == 0:
        return

    spawn_pos, spawn_rot = scenario_strategy.get_spawns(env_ids, env.device)
    goal_pos,  goal_rot  = scenario_strategy.get_goals(env_ids,  env.device)

    for m, obj_cfg in enumerate(object_cfgs):
        _reset_object_to_pose(env, env_ids, obj_cfg, spawn_pos[:, m], spawn_rot[:, m])

    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    pose_term.set_goals_from_strategy(env_ids, goal_pos, goal_rot)


# ---------------------------------------------------------------------------
# Backward-compatible single-object wrapper (used by non-envhub HLEnvCfg)
# ---------------------------------------------------------------------------

def reset_cube_and_goal_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    goal_pos_default: tuple[float, float, float],
    pose_cmd_name: str = "ee_pose",
    yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Randomise cube spawn and goal pose with identical XYZ offset sampling.

    Retained for backward compatibility with single-object / EnvHub fallback paths.
    """
    if env_ids.numel() == 0:
        return

    reset_root_state_uniform(env, env_ids, pose_range, {}, cube_cfg)

    offsets = sample_xyz_offsets(env_ids.numel(), pose_range, env.device)
    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    _set_single_goal(pose_term, env_ids, offsets, goal_pos_default, yaw_range, env)


def _set_single_goal(
    pose_term: HLPoseCommand,
    env_ids: torch.Tensor,
    offsets: torch.Tensor,
    goal_pos_default: tuple[float, float, float],
    yaw_range: tuple[float, float],
    env: ManagerBasedEnv,
) -> None:
    """Set a single-object goal via the legacy random-offset path."""
    base = torch.tensor(goal_pos_default, dtype=torch.float32, device=env.device)
    goal_pos_local = base.unsqueeze(0) + offsets

    goal_yaw = torch.empty(env_ids.numel(), device=env.device).uniform_(*yaw_range)
    zeros = torch.zeros_like(goal_yaw)
    goal_rot = quat_from_euler_xyz(zeros, zeros, goal_yaw)

    pose_term.set_goals_from_strategy(
        env_ids,
        goal_pos_local.unsqueeze(1),
        goal_rot.unsqueeze(1),
    )
