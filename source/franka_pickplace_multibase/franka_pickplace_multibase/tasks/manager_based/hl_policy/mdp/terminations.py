# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for the HL pick-and-place environment.

objects_in_container
--------------------
Multi-object container success check.  Returns ``True`` when:
  * The planner is fully done (``task_idx == num_objects - 1`` AND
    ``stage == DONE``).
  * Every object is inside the container XY footprint.
  * The gripper is open.
  * The success condition holds for ``success_dwell_s`` seconds.

No yaw or upright requirement — objects are simply dropped into the bin.

any_object_fell
---------------
Returns ``True`` for envs where ANY tracked object has fallen below
``minimum_height`` (e.g. fell off the table).

object_dropped_mid_carry
------------------------
Returns ``True`` when the planner is in the LIFT or CARRY stage (the
gripper should be holding an object aloft) but the current task object's
world Z has dropped below ``drop_height_world``.  This indicates the
gripper lost the object during transport.

container_fell
--------------
Returns ``True`` when the container asset has either fallen below
``minimum_height`` or has tilted beyond ``tilt_threshold`` radians.
Requires the container to be a dynamic ``RigidObjectCfg`` asset; for
static ``AssetBaseCfg`` containers (the default) the pose never changes
and this function always returns ``False`` without error.

cube_reached_goal / all_objects_reached_goals
---------------------------------------------
Retained for backward-compatible ``HLEnvCfg`` / EnvHub paths.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .object_assets import container_to_table_interior_half_extents

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LOG = logging.getLogger(__name__)


def _ensure_log_configured() -> None:
    if not _LOG.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


def _ensure_dwell_state(env: ManagerBasedRLEnv, key: str) -> dict[str, torch.Tensor]:
    attr = f"_hl_success_dwell_{key}"
    if not hasattr(env, attr):
        setattr(env, attr, {
            "ready_step": torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            ),
        })
    return getattr(env, attr)


def _ensure_diag_state(env: ManagerBasedRLEnv, key: str) -> dict[str, torch.Tensor]:
    attr = f"_hl_success_diag_{key}"
    if not hasattr(env, attr):
        setattr(env, attr, {
            "blocked_logged": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        })
    return getattr(env, attr)


# ---------------------------------------------------------------------------
# Container task: objects_in_container
# ---------------------------------------------------------------------------


def objects_in_container(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    pose_cmd_name: str = "ee_pose",
    container_cfg: SceneEntityCfg | None = None,
    container_pos_world_offset: tuple[float, float, float] = (0.55, -0.30, 0.03),
    container_interior_half_x: float = 0.14,
    container_interior_half_y: float = 0.10,
    grip_open_threshold: float = 0.8,
    gripper_open_val: float = 0.04,
    success_dwell_s: float = 1.0,
    enable_log: bool = True,
    typed_mode: bool = False,
    num_active: int = 0,
) -> torch.Tensor:
    """Return ``True`` for each env where all objects have been placed in the container.

    Success is **purely physical** — no planner stage requirement:

    * Every object centre is within the container interior footprint on the
      table XY plane (table +X = container +Y).
    * Gripper is open (robot is not holding an object over the bin).
    * Condition held for ``success_dwell_s`` seconds continuously.

    When ``typed_mode=True`` the function reads per-env active catalog indices
    from the :class:`HLPoseCommand` term and checks only those ``num_active``
    objects.  The remaining (parked) catalog objects are ignored, preventing
    false negatives from objects parked far off-table.

    Args:
        env:                         The RL environment.
        object_cfgs:                 Scene entity configs for all objects (or all
                                     catalog objects in typed mode).
        robot_cfg:                   Robot asset with finger joint IDs.
        pose_cmd_name:               Name of the :class:`HLPoseCommand` term
                                     (used in typed mode to fetch active indices).
        container_cfg:               Optional scene entity config for the container asset.
                                     When provided and the asset exposes ``data.root_pos_w``,
                                     the live container position is used for the XY check.
                                     Falls back to ``container_pos_world_offset`` otherwise.
        container_pos_world_offset:  Fallback container centre in robot-local frame (x, y, z).
        container_interior_half_x:   Container-local interior half-extent in x (m).
        container_interior_half_y:   Container-local interior half-extent in y (m).
        grip_open_threshold:         Min normalised finger opening.
        gripper_open_val:            Fully-open finger joint value (m).
        success_dwell_s:             Seconds the condition must hold before reset.
        enable_log:                  Print ``[HL] env N CONTAINER_SUCCESS`` on success.
        typed_mode:                  When ``True``, check only the per-env active objects
                                     (typed-scenario catalog path).
        num_active:                  Number of active objects per episode (used when
                                     ``typed_mode=True``).

    Returns:
        Boolean tensor ``(N,)``.
    """
    from .commands import HLPoseCommand

    M = len(object_cfgs)
    robot: Articulation = env.scene[robot_cfg.name]
    arange = torch.arange(env.num_envs, device=env.device)

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    grip_open  = (finger_pos.mean(dim=-1) / gripper_open_val).clamp(0.0, 1.0)
    grip_ok    = grip_open >= grip_open_threshold

    # Derive container XY centre in world frame.
    # Prefer live container pose when a dynamic container asset is provided.
    origins = env.scene.env_origins  # (N, 3)
    use_live = (
        container_cfg is not None
        and hasattr(env.scene[container_cfg.name], "data")
        and hasattr(env.scene[container_cfg.name].data, "root_pos_w")
    )
    if use_live:
        container_pos_w = env.scene[container_cfg.name].data.root_pos_w  # (N, 3)
        cx_w = container_pos_w[:, 0]
        cy_w = container_pos_w[:, 1]
    else:
        cx_w = origins[:, 0] + container_pos_world_offset[0]
        cy_w = origins[:, 1] + container_pos_world_offset[1]

    half_table_x, half_table_y = container_to_table_interior_half_extents(
        container_interior_half_x, container_interior_half_y
    )

    # In typed mode gather all catalog positions once, then index by active slot.
    if typed_mode and num_active > 0:
        pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
        active_idx = pose_term._active_catalog_indices  # (N, num_active)
        all_pos_w = torch.stack(
            [env.scene[cfg.name].data.root_pos_w for cfg in object_cfgs], dim=1
        )  # (N, C, 3)
        M_check = num_active
    else:
        active_idx = None
        all_pos_w  = None
        M_check    = M

    # Success: gripper open AND every (active) object inside the bin footprint.
    all_in = grip_ok
    for m in range(M_check):
        if active_idx is not None:
            cat_m = active_idx[:, m]          # (N,) catalog index per env
            pos   = all_pos_w[arange, cat_m]  # (N, 3)
        else:
            pos = env.scene[object_cfgs[m].name].data.root_pos_w   # (N, 3)
        dx = (pos[:, 0] - cx_w).abs()
        dy = (pos[:, 1] - cy_w).abs()
        in_xy = (dx < half_table_x) & (dy < half_table_y)
        all_in = all_in & in_xy

    dwell = _ensure_dwell_state(env, "container")
    step = env.common_step_counter
    ready_step = dwell["ready_step"]

    newly_ready = all_in & (ready_step < 0)
    ready_step  = torch.where(newly_ready, step, ready_step)
    ready_step  = torch.where(all_in, ready_step, torch.full_like(ready_step, -1))
    dwell["ready_step"] = ready_step

    dwell_elapsed_s = (step - ready_step).clamp(min=0).float() * env.step_dt
    success = all_in & (ready_step >= 0) & (dwell_elapsed_s >= success_dwell_s)

    if enable_log and success.any():
        _ensure_log_configured()
        for i in torch.where(success)[0].tolist():
            if active_idx is not None:
                positions = ", ".join(
                    _fmt_pos(all_pos_w[i, int(active_idx[i, m].item())])
                    for m in range(M_check)
                )
            else:
                positions = ", ".join(
                    _fmt_pos(env.scene[object_cfgs[m].name].data.root_pos_w[i])
                    for m in range(M)
                )
            msg = (
                f"[HL] env {i} CONTAINER_SUCCESS  M={M_check}  "
                f"object_positions=[{positions}]  "
                f"grip={grip_open[i].item():.2f}  -> episode reset"
            )
            _LOG.info(msg)
            print(msg, flush=True)

    return success


# ---------------------------------------------------------------------------
# Container task: any_object_fell
# ---------------------------------------------------------------------------


def any_object_fell(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    minimum_height: float = 0.0,
) -> torch.Tensor:
    """Return ``True`` for envs where ANY tracked object has fallen off the table.

    The check is performed in **env-local Z** (world Z minus env-origin Z) so
    that the threshold is meaningful regardless of where the environments are
    positioned in world space.

    The table surface sits at local Z ≈ ``TABLE_SURFACE_Z`` (~0.03 m) and
    table-top objects rest at local Z ≈ 0.04 – 0.07 m.  The default threshold
    ``0.0`` terminates whenever an object drops to or below the env-origin
    height — below the table top, the lower shelf of the lab table, or all the
    way to the simulation floor (~−1.05 m local Z).

    Args:
        env:             The RL environment.
        object_cfgs:     Scene entity configs for all objects to monitor.
        minimum_height:  Env-local Z (m) below which the object is considered
                         fallen.  Default ``0.0`` (at or below env-origin level).

    Returns:
        Boolean tensor ``(N,)``.
    """
    env_origin_z = env.scene.env_origins[:, 2]  # (N,)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in object_cfgs:
        obj = env.scene[cfg.name]
        local_z = obj.data.root_pos_w[:, 2] - env_origin_z
        fell = fell | (local_z < minimum_height)
    return fell


# ---------------------------------------------------------------------------
# Container task: object_dropped_mid_carry
# ---------------------------------------------------------------------------


def object_dropped_mid_carry(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    pose_cmd_name: str = "ee_pose",
    drop_height_world: float = 0.10,
    enable_log: bool = True,
) -> torch.Tensor:
    """Return ``True`` when a carried object falls back to table level.

    Fires while the planner is in the **CARRY** stage — i.e. the object should
    already be at full carry height — but the current task object's world-frame
    Z has dropped below ``drop_height_world`` **and** the planner has exhausted
    all grasp retries.

    Two conditions are intentionally excluded from triggering termination:

    * **LIFT stage** – the object is still rising from the table, so its Z is
      legitimately low at the start of the stage.
    * **Retries remaining** – the planner's own carry-miss recovery resets to
      ``PRE_GRASP`` and retries the grasp; terminating at this point would cut
      short a recoverable failure.  Termination only fires once
      ``retry_count >= max_retries`` (the planner has given up).

    Args:
        env:               The RL environment.
        object_cfgs:       Scene entity configs for all objects, in pick order.
        pose_cmd_name:     Name of the :class:`HLPoseCommand` term.
        drop_height_world: World-Z threshold (m).  An object below this height
                           during LIFT/CARRY is considered dropped.  Should be
                           set above the table surface (``~0.10 m``) to avoid
                           false positives.
        enable_log:        Print a ``[HL] OBJECT_DROPPED`` warning on first
                           detection per env.

    Returns:
        Boolean tensor ``(N,)``.
    """
    from ..classical_planner import Stage
    from .commands import HLPoseCommand

    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    planner  = pose_term.planner
    stage    = planner.stage
    task_idx = planner._task_idx
    arange   = torch.arange(env.num_envs, device=env.device)

    # Stack Z values for all tracked objects (N, M) or (N, C) in typed mode.
    obj_z_all = torch.stack(
        [env.scene[cfg.name].data.root_pos_w[:, 2] for cfg in object_cfgs],
        dim=1,
    )  # (N, M)

    # In typed mode task_idx is a pick-slot index (0..num_active-1); the
    # physical object for that slot is given by the per-env active catalog index.
    if getattr(pose_term, "_typed_mode", False):
        active_cat = pose_term._active_catalog_indices[arange, task_idx]  # (N,)
        cur_obj_z  = obj_z_all[arange, active_cat]  # (N,)
    else:
        cur_obj_z = obj_z_all[arange, task_idx]  # (N,)

    # Only check CARRY — during LIFT the object is still rising from the table
    # so its Z is legitimately below the threshold at the start of the stage.
    in_carry = stage == int(Stage.CARRY)

    # Do NOT terminate while retries remain: the planner's own carry-miss
    # recovery will reset to PRE_GRASP and try again.  Only terminate when
    # the planner has exhausted all retry attempts and the object is still
    # on the table (a genuine, unrecoverable drop).
    retries_exhausted = planner._retry_count >= planner.max_retries
    dropped = in_carry & retries_exhausted & (cur_obj_z < drop_height_world)

    if enable_log and dropped.any():
        _ensure_log_configured()
        for i in torch.where(dropped)[0].tolist():
            msg = (
                f"[HL] env {i} OBJECT_DROPPED  obj_idx={int(task_idx[i].item())}  "
                f"stage=CARRY  "
                f"obj_z={cur_obj_z[i].item():.3f} < {drop_height_world:.3f}"
                f"  -> episode reset"
            )
            _LOG.warning(msg)
            print(msg, flush=True)

    return dropped


# ---------------------------------------------------------------------------
# Container task: planner_grasp_failed
# ---------------------------------------------------------------------------


def planner_grasp_failed(
    env: ManagerBasedRLEnv,
    pose_cmd_name: str = "ee_pose",
    enable_log: bool = True,
) -> torch.Tensor:
    """Return ``True`` when every object was attempted and at least one remains unplaced.

    With skip-then-revisit scheduling the episode no longer ends on the first
    per-object grasp failure.  This term fires only after the planner has
    exhausted the primary pass and one deferred revisit for each object.
    """
    from .commands import HLPoseCommand

    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    planner = pose_term.planner

    failed = planner._planning_exhausted & ~planner._object_placed.all(dim=1)

    if enable_log and failed.any():
        _ensure_log_configured()
        for i in torch.where(failed)[0].tolist():
            placed = int(planner._object_placed[i].sum().item())
            abandoned = int(planner._object_abandoned[i].sum().item())
            msg = (
                f"[HL] env {i} PLANNER_GRASP_FAILED  "
                f"placed={placed}/{planner.num_objects}  "
                f"abandoned={abandoned}  "
                "-> episode reset"
            )
            _LOG.warning(msg)
            print(msg, flush=True)

    return failed


# ---------------------------------------------------------------------------
# Container task: planner_reach_failed
# ---------------------------------------------------------------------------


def planner_reach_failed(
    env: ManagerBasedRLEnv,
    pose_cmd_name: str = "ee_pose",
    command_error_threshold: float = 0.12,
    enable_log: bool = True,
) -> torch.Tensor:
    """Return ``True`` when skip-then-revisit scheduling is fully exhausted.

    Reach stalls on a single object now defer to the next object instead of
    ending the episode immediately.  This term aligns with
    :func:`planner_grasp_failed` and only fires once no pending or revisitable
    objects remain and the container task is still incomplete.
    """
    from .commands import HLPoseCommand

    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    planner = pose_term.planner

    failed = planner._planning_exhausted & ~planner._object_placed.all(dim=1)

    if enable_log and failed.any():
        _ensure_log_configured()
        ee_pos_w = pose_term.robot.data.body_pos_w[:, pose_term._body_idx]
        cmd_err = torch.norm(ee_pos_w - pose_term._target_pos_w, dim=-1)
        for i in torch.where(failed)[0].tolist():
            msg = (
                f"[HL] env {i} PLANNER_REACH_FAILED  "
                f"stage={int(planner.stage[i].item())}  "
                f"cmd_err={cmd_err[i].item():.4f}>{command_error_threshold:.4f}  "
                f"endpoint_err={planner._pos_err[i].item():.4f}  "
                f"reach_retry={int(planner._reach_retries[i].item())}/{planner.max_reach_retries}  "
                "-> episode reset"
            )
            _LOG.warning(msg)
            print(msg, flush=True)

    return failed


# ---------------------------------------------------------------------------
# Container task: container_fell
# ---------------------------------------------------------------------------


def container_fell(
    env: ManagerBasedRLEnv,
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    minimum_height: float = -0.05,
    tilt_threshold: float = 0.5,
    enable_log: bool = True,
) -> torch.Tensor:
    """Return ``True`` when the container has fallen or tipped over.

    Two sub-conditions trigger termination:

    * **Fell** – the container's world-Z centre drops below ``minimum_height``
      (e.g. it was knocked off the table).
    * **Tipped** – the container's roll or pitch exceeds ``tilt_threshold``
      radians (e.g. it was knocked onto its side).

    .. note::
        This condition only fires when the container is a **dynamic**
        ``RigidObjectCfg`` asset.  For the default static ``AssetBaseCfg``
        container the pose never changes and this function always returns
        ``False`` without raising an error.

    Args:
        env:              The RL environment.
        container_cfg:    Scene entity config for the container.
        minimum_height:   World-Z below which the container is considered
                          fallen (m).  Default ``-0.05 m``.
        tilt_threshold:   Roll or pitch magnitude (rad) that indicates the
                          container has tipped over.  Default ``0.5 rad``
                          (~28°).
        enable_log:       Print a ``[HL] CONTAINER_FELL/TIPPED`` warning when
                          the condition fires.

    Returns:
        Boolean tensor ``(N,)``.
    """
    container = env.scene[container_cfg.name]

    # Static / xform-only assets (XformPrimView, AssetBase without rigid-body
    # physics) have no `.data` attribute or no `root_pos_w` on their data —
    # return False immediately so they never trigger this term.
    if not hasattr(container, "data") or not hasattr(container.data, "root_pos_w"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    pos_w  = container.data.root_pos_w   # (N, 3)
    quat_w = container.data.root_quat_w  # (N, 4)

    fell   = pos_w[:, 2] < minimum_height
    roll, pitch, _ = euler_xyz_from_quat(quat_w)
    tipped = (roll.abs() > tilt_threshold) | (pitch.abs() > tilt_threshold)

    terminated = fell | tipped

    if enable_log and terminated.any():
        _ensure_log_configured()
        for i in torch.where(terminated)[0].tolist():
            reason = "FELL" if fell[i].item() else "TIPPED"
            msg = (
                f"[HL] env {i} CONTAINER_{reason}  "
                f"z={pos_w[i, 2].item():.3f}  "
                f"roll={roll[i].item():.3f}  pitch={pitch[i].item():.3f}"
                f"  -> episode reset"
            )
            _LOG.warning(msg)
            print(msg, flush=True)

    return terminated


# ---------------------------------------------------------------------------
# Container task: container_displaced
# ---------------------------------------------------------------------------


def container_displaced(
    env: ManagerBasedRLEnv,
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    max_displacement: float = 0.02,
    max_yaw_displacement: float = 0.1,
    enable_log: bool = True,
) -> torch.Tensor:
    """Return ``True`` when the container has been pushed or rotated from its
    scenario-defined initial pose.

    The reference state is stored on ``env._hl_container_home_w`` (position)
    and ``env._hl_container_home_quat`` (orientation) by the reset event at
    the start of each episode.  Both are set from the explicit container pose
    recorded in ``scenarios.json``, so the check is always relative to the
    per-scenario home.

    Two conditions are tested independently; either alone triggers the
    termination:

    * **XY displacement** — horizontal distance between current and home
      position exceeds ``max_displacement`` (default 2 cm).
    * **Yaw rotation** — absolute yaw difference between current and home
      orientation exceeds ``max_yaw_displacement`` (default 0.1 rad ≈ 5.7°).

    Z settling of the rigid body on the table surface is ignored.

    .. note::
        This condition requires the container to be a **dynamic**
        ``RigidObjectCfg`` asset.  For a static ``AssetBaseCfg`` the pose
        never changes and the function always returns ``False``.  If the home
        buffers have not been initialised yet (first step before reset), the
        function also returns ``False``.

    Args:
        env:                  The RL environment.
        container_cfg:        Scene entity config for the container.
        max_displacement:     XY distance threshold (m).  Default 0.02 m.
        max_yaw_displacement: Yaw angle threshold (rad).  Default 0.1 rad.
        enable_log:           Print a warning when fired.

    Returns:
        Boolean tensor ``(N,)``.
    """
    import math as _math

    container = env.scene[container_cfg.name]

    if not hasattr(container, "data") or not hasattr(container.data, "root_pos_w"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if not hasattr(env, "_hl_container_home_w"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    pos_w  = container.data.root_pos_w   # (N, 3)
    home_w = env._hl_container_home_w    # (N, 3)

    dx = pos_w[:, 0] - home_w[:, 0]
    dy = pos_w[:, 1] - home_w[:, 1]
    dist_xy = torch.sqrt(dx * dx + dy * dy)
    pos_displaced = dist_xy > max_displacement

    # Yaw-displacement check (uses scenario-defined home orientation).
    d_yaw        = torch.zeros(env.num_envs, device=env.device)
    yaw_displaced = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "_hl_container_home_quat"):
        def _extract_yaw(q: torch.Tensor) -> torch.Tensor:
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        quat_w   = container.data.root_quat_w          # (N, 4) wxyz
        home_q   = env._hl_container_home_quat         # (N, 4) wxyz
        yaw_cur  = _extract_yaw(quat_w)
        yaw_home = _extract_yaw(home_q)
        # Wrap difference to (-π, π].
        d_yaw = torch.remainder(yaw_cur - yaw_home + _math.pi, 2.0 * _math.pi) - _math.pi
        yaw_displaced = d_yaw.abs() > max_yaw_displacement

    displaced = pos_displaced | yaw_displaced

    if enable_log and displaced.any():
        _ensure_log_configured()
        for i in torch.where(displaced)[0].tolist():
            parts = []
            if pos_displaced[i]:
                parts.append(f"dist_xy={dist_xy[i].item():.4f}>{max_displacement:.4f}")
            if yaw_displaced[i]:
                parts.append(f"|d_yaw|={d_yaw[i].abs().item():.4f}>{max_yaw_displacement:.4f}")
            msg = (
                f"[HL] env {i} CONTAINER_DISPLACED  "
                + "  ".join(parts)
                + "  -> episode reset"
            )
            _LOG.warning(msg)
            print(msg, flush=True)

    return displaced


# ---------------------------------------------------------------------------
# Single-object (backward-compat)
# ---------------------------------------------------------------------------


def cube_reached_goal(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    pose_cmd_name: str = "ee_pose",
    pos_threshold: float = 0.025,
    ang_threshold: float = 0.15,
    yaw_threshold: float = 0.15,
    grip_open_threshold: float = 0.8,
    gripper_open_val: float = 0.04,
    success_dwell_s: float = 1.0,
    enable_log: bool = True,
    log_blocked: bool = True,
) -> torch.Tensor:
    """Return ``True`` for each env where single-object pick-and-place succeeded.

    Used by the hardcoded (non-envhub) :class:`HLEnvCfg` path when running in
    the legacy flat-goal mode (not container mode).
    """
    from ..classical_planner import Stage
    from .commands import HLPoseCommand

    cube = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w:  torch.Tensor = cube.data.root_pos_w
    cube_quat_w: torch.Tensor = cube.data.root_quat_w

    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    goal_pos_w  = pose_term.goal_pos_w[:, 0]
    goal_quat_w = pose_term.goal_quat_w[:, 0]
    stage = pose_term.planner.stage

    xy_err = torch.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1)

    roll, pitch, cube_yaw = euler_xyz_from_quat(cube_quat_w)
    upright = (roll.abs() < ang_threshold) & (pitch.abs() < ang_threshold)

    _, _, goal_yaw = euler_xyz_from_quat(goal_quat_w)
    yaw_diff = wrap_to_pi(cube_yaw - goal_yaw)
    half_pi = 0.5 * math.pi
    yaw_err = torch.abs((yaw_diff + 0.25 * half_pi) % half_pi - 0.25 * half_pi)
    yaw_ok = yaw_err < yaw_threshold

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    grip_open = (finger_pos.mean(dim=-1) / gripper_open_val).clamp(0.0, 1.0)

    xy_ok   = xy_err < pos_threshold
    grip_ok = grip_open >= grip_open_threshold
    at_done = stage >= int(Stage.DONE)

    physically_ready = xy_ok & upright & yaw_ok & grip_ok & at_done

    dwell = _ensure_dwell_state(env, "single")
    step = env.common_step_counter
    ready_step = dwell["ready_step"]

    newly_ready = physically_ready & (ready_step < 0)
    ready_step = torch.where(newly_ready, step, ready_step)
    ready_step = torch.where(physically_ready, ready_step, torch.full_like(ready_step, -1))
    dwell["ready_step"] = ready_step

    dwell_elapsed_s = (step - ready_step).clamp(min=0).float() * env.step_dt
    placed = physically_ready & (ready_step >= 0) & (dwell_elapsed_s >= success_dwell_s)

    if enable_log and placed.any():
        _ensure_log_configured()
        for i in torch.where(placed)[0].tolist():
            p = cube_pos_w[i]
            msg = (
                f"[HL] env {i} SUCCESS  cube_w=({p[0].item():.3f}, {p[1].item():.3f}, {p[2].item():.3f})  "
                f"xy_err={xy_err[i].item():.4f}  grip={grip_open[i].item():.2f}  "
                f"stage={int(stage[i].item())}  -> episode reset"
            )
            _LOG.info(msg)
            print(msg, flush=True)

    if log_blocked and enable_log:
        diag = _ensure_diag_state(env, "single")
        diag["blocked_logged"] &= stage >= int(Stage.DONE)

        near_goal = xy_err < pos_threshold * 2.0
        in_done = stage >= int(Stage.DONE)
        blocked = in_done & near_goal & ~physically_ready & ~diag["blocked_logged"]
        if blocked.any():
            _ensure_log_configured()
            for i in torch.where(blocked)[0].tolist():
                reasons: list[str] = []
                if not xy_ok[i]:
                    reasons.append(f"xy={xy_err[i].item():.4f} (need<{pos_threshold})")
                if not upright[i]:
                    reasons.append(f"roll={roll[i].item():.3f} pitch={pitch[i].item():.3f} (need<{ang_threshold})")
                if not yaw_ok[i]:
                    reasons.append(f"yaw={yaw_err[i].item():.3f} (need<{yaw_threshold})")
                if not grip_ok[i]:
                    reasons.append(f"grip={grip_open[i].item():.2f} (need>={grip_open_threshold})")
                msg = (
                    f"[HL] env {i} success blocked at stage {int(stage[i].item())}: "
                    + ", ".join(reasons)
                )
                _LOG.warning(msg)
                print(msg, flush=True)
                diag["blocked_logged"][i] = True

    return placed


# ---------------------------------------------------------------------------
# Multi-object (envhub sequential pick-and-place path)
# ---------------------------------------------------------------------------


def all_objects_reached_goals(
    env: ManagerBasedRLEnv,
    cube_cfgs: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    pose_cmd_name: str = "ee_pose",
    pos_threshold: float = 0.025,
    ang_threshold: float = 0.15,
    yaw_threshold: float = 0.15,
    grip_open_threshold: float = 0.8,
    gripper_open_val: float = 0.04,
    success_dwell_s: float = 1.0,
    enable_log: bool = True,
    typed_mode: bool = False,
    num_active: int = 0,
) -> torch.Tensor:
    """Return ``True`` when all active objects have been sequentially placed (EnvHub path).

    Retained for the EnvHub ``HLEnvCfg_Envhub`` path which uses point goals
    with yaw/upright requirements.

    In **typed-scenario mode** (``typed_mode=True``, ``num_active > 0``), each
    episode only ``num_active`` of the ``len(cube_cfgs)`` catalog objects are
    tracked.  The active set is read from
    ``pose_term._active_catalog_indices (N, num_active)``, and positions are
    gathered per-env from the full catalog pose stack.
    """
    from ..classical_planner import Stage
    from .commands import HLPoseCommand

    robot: Articulation = env.scene[robot_cfg.name]
    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    planner  = pose_term.planner
    stage    = planner.stage
    task_idx = planner._task_idx
    arange   = torch.arange(env.num_envs, device=env.device)

    if typed_mode and num_active > 0:
        # Typed path: per-env active catalog subset.
        M = num_active
        active_idx = pose_term._active_catalog_indices  # (N, M)
        # Stack all catalog object positions / quaternions.
        all_pos  = torch.stack([env.scene[c.name].data.root_pos_w  for c in cube_cfgs], dim=1)  # (N, C, 3)
        all_quat = torch.stack([env.scene[c.name].data.root_quat_w for c in cube_cfgs], dim=1)  # (N, C, 4)
    else:
        M = len(cube_cfgs)
        active_idx = None
        all_pos = all_quat = None

    fully_done = (task_idx >= M - 1) & (stage >= int(Stage.DONE))

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    grip_open  = (finger_pos.mean(dim=-1) / gripper_open_val).clamp(0.0, 1.0)
    grip_ok    = grip_open >= grip_open_threshold

    all_placed = fully_done & grip_ok
    for m in range(M):
        if typed_mode and active_idx is not None:
            cat_m = active_idx[:, m]                   # (N,) catalog index for slot m
            cube_pos_w  = all_pos[arange,  cat_m]      # (N, 3)
            cube_quat_w = all_quat[arange, cat_m]      # (N, 4)
        else:
            cube_m = env.scene[cube_cfgs[m].name]
            cube_pos_w  = cube_m.data.root_pos_w
            cube_quat_w = cube_m.data.root_quat_w

        goal_pos_w  = pose_term.goal_pos_w[:, m]
        goal_quat_w = pose_term.goal_quat_w[:, m]

        xy_err = torch.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1)
        roll, pitch, cube_yaw = euler_xyz_from_quat(cube_quat_w)
        upright = (roll.abs() < ang_threshold) & (pitch.abs() < ang_threshold)

        _, _, goal_yaw = euler_xyz_from_quat(goal_quat_w)
        yaw_diff = wrap_to_pi(cube_yaw - goal_yaw)
        half_pi  = 0.5 * math.pi
        yaw_err  = torch.abs((yaw_diff + 0.25 * half_pi) % half_pi - 0.25 * half_pi)
        yaw_ok   = yaw_err < yaw_threshold

        all_placed = all_placed & (xy_err < pos_threshold) & upright & yaw_ok

    dwell = _ensure_dwell_state(env, "multi")
    step = env.common_step_counter
    ready_step = dwell["ready_step"]

    newly_ready = all_placed & (ready_step < 0)
    ready_step  = torch.where(newly_ready, step, ready_step)
    ready_step  = torch.where(all_placed, ready_step, torch.full_like(ready_step, -1))
    dwell["ready_step"] = ready_step

    dwell_elapsed_s = (step - ready_step).clamp(min=0).float() * env.step_dt
    success = all_placed & (ready_step >= 0) & (dwell_elapsed_s >= success_dwell_s)

    if enable_log and success.any():
        _ensure_log_configured()
        for i in torch.where(success)[0].tolist():
            if typed_mode and active_idx is not None:
                positions = ", ".join(
                    _fmt_pos(all_pos[i, int(active_idx[i, m].item())])
                    for m in range(M)
                )
            else:
                positions = ", ".join(
                    _fmt_pos(env.scene[cube_cfgs[m].name].data.root_pos_w[i])
                    for m in range(M)
                )
            msg = (
                f"[HL] env {i} ALL_DONE  M={M}  "
                f"cube_positions=[{positions}]  "
                f"grip={grip_open[i].item():.2f}  -> episode reset"
            )
            _LOG.info(msg)
            print(msg, flush=True)

    return success


def _fmt_pos(v: torch.Tensor) -> str:
    return f"({v[0].item():.3f},{v[1].item():.3f},{v[2].item():.3f})"
