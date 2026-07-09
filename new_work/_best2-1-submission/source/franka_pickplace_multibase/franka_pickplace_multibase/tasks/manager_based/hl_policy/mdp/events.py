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
from .object_assets import (
    container_drop_slot_offsets_table,
    container_to_table_interior_half_extents,
)


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
# EnvHub typed-scenario path (franka-pickplace-multibase-sample)
# ---------------------------------------------------------------------------


def reset_typed_objects_from_scenario(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    all_object_cfgs: list[SceneEntityCfg],
    scenario_strategy,
    pose_cmd_name: str = "ee_pose",
    parking_pos: tuple[float, float, float] = (20.0, 0.0, 0.5),
    # Container params (optional).  When provided the container pose is read
    # from the scenario and goals are set to the drop zone.
    container_cfg: SceneEntityCfg | None = None,
    container_interior_half_x: float = 0.14,
    container_interior_half_y: float = 0.10,
    container_drop_z_local: float = 0.13,
) -> None:
    """Reset catalog objects for one episode using :class:`TypedPrebakedScenarioStrategy`.

    Parks the inactive catalog objects at ``parking_pos`` (robot-local frame,
    far off-table) and teleports the active objects to their scenario spawn
    positions.

    When ``container_cfg`` is supplied the function also:

    * Places the container at the pose defined in the scenario JSON
      (``scenario["container"]["pos"]`` and ``scenario["container"]["rot"]``).
    * Stores the container's episode-start world position and orientation on
      ``env._hl_container_home_w`` / ``env._hl_container_home_quat`` for the
      ``container_displaced`` termination.
    * Overrides the episode goals with per-slot container drop positions.

    When ``container_cfg`` is ``None`` (point-goal benchmarks), goals are read
    directly from the scenario file.

    Args:
        env:                        The RL environment.
        env_ids:                    Indices of environments being reset.
        all_object_cfgs:            Scene entity configs for all C catalog objects
                                    (``catalog_0`` … ``catalog_7``), in catalog order.
        scenario_strategy:          A :class:`TypedPrebakedScenarioStrategy` instance.
        pose_cmd_name:              Name of the :class:`HLPoseCommand` command term.
        parking_pos:                Robot-local position for inactive objects (m).
        container_cfg:              Scene entity config for the KLT bin.
        container_interior_half_x:  Container interior half-extent x (m).
        container_interior_half_y:  Container interior half-extent y (m).
        container_drop_z_local:     Drop target Z in robot-local frame (rim Z + offset).
    """
    if env_ids.numel() == 0:
        return

    N   = env_ids.numel()
    dev = env.device

    # ------------------------------------------------------------------
    # 1. Get per-catalog spawn positions (inactive slots → parking_pos).
    # ------------------------------------------------------------------
    spawn_pos_cat, spawn_rot_cat = scenario_strategy.get_catalog_spawns(env_ids, dev)
    # spawn_pos_cat: (N, C, 3),  spawn_rot_cat: (N, C, 4)

    # ------------------------------------------------------------------
    # 2. Teleport all catalog objects to their scenario-assigned positions.
    # ------------------------------------------------------------------
    for c, obj_cfg in enumerate(all_object_cfgs):
        _reset_object_to_pose(env, env_ids, obj_cfg, spawn_pos_cat[:, c], spawn_rot_cat[:, c])

    pose_term = env.command_manager.get_term(pose_cmd_name)

    # ------------------------------------------------------------------
    # 3a. Container mode: place container at its scenario-defined pose,
    #     then set drop-zone goals relative to that position.
    # ------------------------------------------------------------------
    if container_cfg is not None:
        # Container pose is explicitly defined per scenario — no runtime
        # randomisation.  This guarantees the pre-validated spawn clearance
        # from scenarios.json always holds.
        container_local_pos, container_rot = scenario_strategy.get_container_pose(env_ids, dev)
        cx_l = container_local_pos[:, 0]   # (N,) robot-local X
        cy_l = container_local_pos[:, 1]   # (N,) robot-local Y

        _reset_object_to_pose(env, env_ids, container_cfg, container_local_pos, container_rot)

        # Store episode-start world pose as reference for container_displaced.
        if not hasattr(env, "_hl_container_home_w"):
            env._hl_container_home_w    = torch.zeros(env.num_envs, 3, device=dev)
        if not hasattr(env, "_hl_container_home_quat"):
            env._hl_container_home_quat = torch.zeros(env.num_envs, 4, device=dev)
            env._hl_container_home_quat[:, 0] = 1.0  # identity quaternion
        origins = env.scene.env_origins[env_ids]
        env._hl_container_home_w[env_ids]    = origins + container_local_pos
        env._hl_container_home_quat[env_ids] = container_rot

        # Build per-slot drop goals inside the bin.
        M_active = scenario_strategy.num_active
        half_table_x, half_table_y = container_to_table_interior_half_extents(
            container_interior_half_x, container_interior_half_y
        )
        slot_offsets = container_drop_slot_offsets_table(M_active, half_table_x, half_table_y)

        goal_pos_local = torch.zeros(N, M_active, 3, device=dev)
        goal_rot_local = torch.zeros(N, M_active, 4, device=dev)
        goal_rot_local[:, :, 0] = 1.0  # identity quaternion

        # Rotate slot offsets by the container's yaw so goals align with the
        # rotated interior.  The slot pattern is defined for a zero-yaw
        # container; without this rotation, goals shift toward the container
        # walls when the container is rotated (up to ±22.5° in scenarios),
        # leaving only ~1 cm clearance and causing dropped objects to bounce.
        # Play-v0 always places the container at yaw=0, so it is unaffected.
        container_yaw = 2.0 * torch.atan2(container_rot[:, 3], container_rot[:, 0])  # (N,)
        cos_yaw = torch.cos(container_yaw)  # (N,)
        sin_yaw = torch.sin(container_yaw)  # (N,)

        for m, (jx, jy) in enumerate(slot_offsets):
            jx_rot = jx * cos_yaw - jy * sin_yaw  # (N,)
            jy_rot = jx * sin_yaw + jy * cos_yaw  # (N,)
            goal_pos_local[:, m, 0] = cx_l + jx_rot
            goal_pos_local[:, m, 1] = cy_l + jy_rot
            goal_pos_local[:, m, 2] = container_drop_z_local

        pose_term.set_goals_from_strategy(env_ids, goal_pos_local, goal_rot_local)

    # ------------------------------------------------------------------
    # 3b. Point-goal mode: use goal positions from the scenario file.
    # ------------------------------------------------------------------
    else:
        goal_pos, goal_rot = scenario_strategy.get_goals(env_ids, dev)
        pose_term.set_goals_from_strategy(env_ids, goal_pos, goal_rot)

    # ------------------------------------------------------------------
    # 4. Update per-env active catalog indices (grasp-metadata selection).
    # ------------------------------------------------------------------
    active_indices = scenario_strategy.get_active_indices(env_ids, dev)  # (N, M_active)
    # Optional pick-order reorder: secure tall/tippy objects (mustard=cat1,
    # cracker=cat3) FIRST so the arm cannot knock them off the table while
    # working other objects. Goals are arbitrary bin slots, so reordering the
    # pick sequence is safe. Gated on NEPHER_PICK_ORDER for measured comparison.
    import os as _os
    # Default pick order = "far" (farthest-from-bin object grasped first). This is
    # the measured-best order: it cuts early table knock-offs (39->8 events) and
    # bin-shove (25->17) vs the natural slot order, for +5 successes / 90. Set
    # NEPHER_PICK_ORDER=none to disable (natural order) for comparison.
    _po = _os.environ.get("NEPHER_PICK_ORDER", "far")
    if _po == "none":
        _po = None
    if _po in ("low", "tall"):
        # Order active objects by current height. "low" = flattest first (only the
        # first grasp descends from home at full depth); "tall" = tallest first
        # (remove tall obstacles early so the arm doesn't knock them while working
        # low objects -> fewer CUBE_FELL).
        all_z = torch.stack(
            [env.scene[cfg.name].data.root_pos_w[env_ids, 2] for cfg in all_object_cfgs], dim=1
        )  # (N, C)
        zkey = torch.gather(all_z, 1, active_indices.long())  # (N, M_active)
        if _po == "tall":
            zkey = -zkey
        order = torch.argsort(zkey, dim=1)
        active_indices = torch.gather(active_indices, 1, order)
    elif _po in ("near", "far", "farcube"):
        # Order by horizontal distance from the bin: "near" = nearest-to-bin first
        # (short transit, less sweep over the table); "far"/"farcube" = farthest first.
        home = getattr(env, "_hl_container_home_w", None)
        all_xy = torch.stack(
            [env.scene[cfg.name].data.root_pos_w[env_ids, :2] for cfg in all_object_cfgs], dim=1
        )  # (N, C, 2)
        bin_xy = home[env_ids, :2].unsqueeze(1) if home is not None else all_xy.mean(dim=1, keepdim=True)
        dist = torch.norm(all_xy - bin_xy, dim=2)  # (N, C)
        dkey = torch.gather(dist, 1, active_indices.long())  # (N, M_active)
        if _po in ("far", "farcube"):
            dkey = -dkey
        order = torch.argsort(dkey, dim=1)
        active_indices = torch.gather(active_indices, 1, order)
        if _po == "farcube":
            # Promote the depth-sensitive tiny DexCube (catalog idx 4) to FIRST so
            # it is grasped while the arm still has full reach from the home config
            # (the 2nd+ grasps under-descend, which the tiny cube can't tolerate).
            is_cube = (active_indices == 4)
            if is_cube.any():
                # stable: cube to front, others keep relative (far) order.
                key2 = torch.where(is_cube, torch.full_like(active_indices, -1),
                                   torch.arange(active_indices.shape[1], device=dev).unsqueeze(0).expand_as(active_indices))
                order2 = torch.argsort(key2, dim=1)
                active_indices = torch.gather(active_indices, 1, order2)
    elif _po:
        M_a = active_indices.shape[1]
        base = torch.arange(M_a, device=dev).unsqueeze(0).expand_as(active_indices).float() + 2.0
        key = torch.where(active_indices == 1, torch.zeros_like(base),
              torch.where(active_indices == 3, torch.ones_like(base), base))
        order = torch.argsort(key, dim=1)
        active_indices = torch.gather(active_indices, 1, order)
    pose_term.set_active_objects_from_typed_scenario(env_ids, active_indices)


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
