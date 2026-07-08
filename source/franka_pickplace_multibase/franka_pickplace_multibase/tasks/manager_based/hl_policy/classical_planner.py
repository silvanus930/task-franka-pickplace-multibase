# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal pick-and-place trajectory planner for the Franka HL policy.

Nine stages (plus absorbing DONE):

    PRE_GRASP → DESCEND → GRASP → LIFT → CARRY → LOWER → RELEASE → RETRACT → DONE

Gripper state per stage:

    open  open  closed  closed  closed  closed  open  open  open

Motion sequence:
  1. PRE_GRASP  – hover above object at standoff height, gripper open.
  2. DESCEND    – lower to grasp Z with gripper still open.
  3. GRASP      – close gripper and hold until it settles.
  4. LIFT       – raise to carry height at object XY, gripper closed.
  5. CARRY      – move to goal XY at carry height, gripper closed.
  6. LOWER      – descend to drop height above container rim.
  7. RELEASE    – open gripper (drop into container).
  8. RETRACT    – lift up at goal XY with gripper open (clear the container).
  9. DONE       – hold at retract height.

Per-object grasp metadata
-------------------------
``step()`` accepts per-env tensors ``grasp_z_off``, ``grasp_sym``, and
``grasp_yaw_off`` that override the scalar constructor defaults for the
currently active object.  These are supplied by ``HLPoseCommand`` which
selects the correct row from per-object metadata arrays using ``_task_idx``.

* ``grasp_sym``     – yaw symmetry in radians.  ``0.0`` = rotationally
  symmetric (round objects): the wrist stays at neutral yaw (0 rad).
  ``math.pi`` = 180° symmetry (long-axis objects).
  ``math.pi/2`` = 90° symmetry (square cross-section boxes).
* ``grasp_z_off``   – TCP below object *centre* at grasp depth (m, negative).
* ``grasp_yaw_off`` – constant yaw added after symmetry-folding (rad).

Drop-into-container placement
------------------------------
``place_z`` for LOWER/RELEASE is derived from ``goal_pos_w[:, 2]`` (the
container rim target height set by the command term) instead of
``gz + grasp_z_offset``.  Placement is orientation-agnostic: the place-yaw
gate is disabled and in-gripper yaw compensation is skipped for the goal yaw
(the object just needs to land inside the bin walls).

If the cube is not detected as lifted during CARRY the planner recycles to
PRE_GRASP (up to ``max_retries`` times).

Multi-object support
--------------------
Pass ``num_objects > 1`` to handle sequential pick-and-place of N objects.
Each env tracks ``_task_idx`` (which object is currently being worked on).
When DONE is reached for object ``k``, that slot is marked placed and the
planner advances to the next pending object.

**Skip-then-revisit:** If the current object exhausts grasp or reach retries,
it is deferred and the planner moves on to the next object.  After all
immediate objects are attempted, deferred objects are revisited once with a
fresh retry budget.  Objects that fail again are abandoned.  The episode only
terminates as a planner failure when every object is placed or abandoned.

``HLPoseCommand`` is responsible for feeding the correct current-object poses
and metadata (indexed by ``_task_idx``) into each ``step()`` call.

**Option A – static-endpoint command design.**
Each step the planner emits the *endpoint* of the current stage as the
commanded EE pose, not an interpolated mid-segment value.  This matches the
LL training distribution exactly: ``UniformPoseCommand`` also holds a fixed
target until resampling.  Stage advancement is gated purely on the LL EE
arriving within (``pos_tol``, ``ang_tol``) of that endpoint, plus a minimum
dwell time.  No trajectory interpolation is performed.

All positions are world-frame.  Every Z target is expressed for ``panda_hand``
(= TCP_Z + hand_tcp_offset_z).
"""

from __future__ import annotations

import math
from enum import IntEnum

import torch

from isaaclab.utils.math import (
    euler_xyz_from_quat,
    normalize,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
)


class Stage(IntEnum):
    PRE_GRASP = 0  # hover above object, gripper open
    DESCEND   = 1  # lower to grasp Z, gripper open
    GRASP     = 2  # close gripper, hold
    LIFT      = 3  # raise to carry height at object XY
    CARRY     = 4  # move to goal XY at carry height
    LOWER     = 5  # descend to drop height (above container rim)
    RELEASE   = 6  # open gripper (drop into container)
    RETRACT   = 7  # lift up at goal XY, gripper open
    DONE      = 8  # hold at retract height


STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in Stage)


def object_settled_in_container(
    cube_pos_w: torch.Tensor,
    goal_pos_w: torch.Tensor,
    place_verify_xy: float,
    z_above_goal_max: float = 0.10,
) -> torch.Tensor:
    """Return per-env mask: object centre is near the bin drop target (loose check).

    Used for suppressing false carry-drop failures. Prefer
    :func:`object_inside_container_interior` for opportunistic placement.
    """
    xy_ok = torch.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1) < place_verify_xy
    z_ok = cube_pos_w[:, 2] <= (goal_pos_w[:, 2] + z_above_goal_max)
    return xy_ok & z_ok


def object_inside_container_interior(
    cube_pos_w: torch.Tensor,
    goal_pos_w: torch.Tensor,
    half_table_x: float,
    half_table_y: float,
    floor_z: float,
    rim_z: float,
    xy_margin: float = 0.80,
) -> torch.Tensor:
    """Return per-env mask: object centre is inside the bin interior (not on rim).

    Uses axis-aligned bounds in table XY (same frame as ``objects_in_container``).
    """
    hx = half_table_x * xy_margin
    hy = half_table_y * xy_margin
    dx = (cube_pos_w[:, 0] - goal_pos_w[:, 0]).abs()
    dy = (cube_pos_w[:, 1] - goal_pos_w[:, 1]).abs()
    xy_ok = (dx < hx) & (dy < hy)
    z_ok = (cube_pos_w[:, 2] >= floor_z) & (cube_pos_w[:, 2] <= rim_z + 0.02)
    return xy_ok & z_ok


# 0 = open, 1 = closed
_STAGE_GRIP: list[float] = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


from .grasp_yaw import (
    ClosingAxis,
    GripperFrameConfig,
    GraspYawDebug,
    closing_direction_world_from_target,
    compute_grasp_yaw,
    compute_grasp_yaw_symmetry,
    normalize_yaw,
)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class PickPlacePlanner:
    """Vectorised pick-and-place planner (9 stages, static-endpoint commands).

    Supports sequential multi-object pick-and-place via ``num_objects`` and
    per-object grasp metadata tensors supplied on each ``step()`` call.

    In **container mode** (``container_drop=True``) LOWER/RELEASE target a
    fixed drop height above the container rim rather than pressing the object
    onto the table surface.  Placement is orientation-agnostic (no yaw gate,
    no in-gripper yaw compensation on the goal side) since the object just
    needs to land inside the bin walls.

    Each ``step()`` call returns the *endpoint* of the current stage so that
    the frozen LL policy sees the same piecewise-constant command distribution
    it was trained on.  Stage transitions are gated on LL EE arrival within
    tolerance, not on elapsed time alone.
    """

    def __init__(
        self,
        num_envs: int,
        device:   str | torch.device,
        *,
        num_objects:       int   = 1,
        hand_tcp_offset_z: float = 0.107,
        pre_approach_z:    float = 0.12,
        carry_z:           float = 0.22,   # raised to clear container rim
        grasp_z_offset:    float = -0.01,  # default, overridden per-object
        release_z_offset:  float = 0.0,
        retract_approach_z: float = 0.12,
        pos_tol:           float = 0.015,
        ang_tol:           float = 0.08,
        pos_tol_approach:  float = 0.025,
        ang_tol_approach:  float = 0.20,
        pos_tol_grasp:     float = 0.045,
        ang_tol_grasp:     float = 0.45,
        pos_tol_transport: float = 0.020,
        ang_tol_transport: float = 0.22,
        pos_tol_place:     float = 0.025,  # relaxed: drop not press
        pos_tol_retract:   float = 0.06,
        ang_tol_retract:   float = 1.5,
        min_stage_dur:     float = 0.25,
        grasp_hold_s:      float = 0.45,
        release_hold_s:    float = 0.30,
        max_retries:       int   = 3,
        min_carry_cube_z:  float = 0.08,
        grasp_secure_xy_tol: float = 0.06,
        wrist_soft_limit:  float = 2.5,
        yaw_switch_cooldown: int = 20,
        yaw_k_max:         int   = 3,
        pitch_cmd:         float = math.pi,
        pitch_transport:   float = math.pi,
        # Container-drop mode: disable yaw gate, use goal_z directly for LOWER.
        container_drop:    bool  = True,
        place_yaw_gate:    float = 10.0,   # effectively disabled in container mode
        max_step:          float = 0.05,
        pre_grasp_settle_s:   float = 1.5,
        pre_grasp_settle_ang: float = 0.6,
        pre_grasp_yaw_tol:    float = 0.12,
        lift_anchor_radius:   float = 0.05,
        place_settle_s:       float = 0.3,
        place_settle_max:     float = 0.8,
        stall_pos_tol:        float = 0.06,
        stall_time_s:         float = 4.0,
        retreat_steps:        int   = 25,
        retreat_z:            float = 0.40,
        max_reach_retries:    int   = 3,
        carry_drop_gap:       float = 0.04,
        stage_escape_s:       float = 2.0,
        stage_escape_pos_mult: float = 2.5,
        # Object-aware grasp (orientation + finger contact).
        upright_up_z_min:       float = 0.75,
        lying_up_z_max:         float = 0.45,
        lying_grasp_z_scale:    float = 0.55,
        grasp_finger_empty_max: float = 0.004,
        grasp_finger_contact_min: float = 0.010,
        # Placement verification relaxed for container drop.
        place_verify_xy:      float = 0.18,  # bin interior half-extent; re-pick only if missed bin
        place_verify_yaw:     float = 10.0,  # disabled: no yaw requirement
        max_place_retries:    int   = 1,
        max_lower_retries:    int   = 1,
        hurry_after_s:        float = 25.0,  # longer budget per object
        hurry_scale:          float = 0.6,
        container_retract_xy_offset: tuple[float, float] = (-0.05, 0.12),
        # Opportunistic placement: count early bin landings during transport.
        opportunistic_container_place: bool = False,
        opportunistic_z_above_goal_max: float = 0.10,
        container_interior_half_table_x: float = 0.10,
        container_interior_half_table_y: float = 0.14,
        container_floor_z: float = 0.035,
        container_rim_z: float = 0.11,
        # Grasp-yaw alignment (see grasp_yaw.py).
        ang_tol_pre_grasp: float = 0.45,
        grasp_closing_axis: str = "ee_x",
        grasp_yaw_frame_offset: float = 0.0,
        # After N grasp failures, add +90° yaw flip for elongated objects (safe fallback).
        grasp_yaw_flip_enabled: bool = True,
        grasp_yaw_flip_after_retries: int = 2,
        grasp_yaw_flip_rad: float = math.pi / 2,
    ) -> None:
        self.num_envs = num_envs
        self.device   = device

        self.num_objects      = num_objects
        self.H                = hand_tcp_offset_z
        self.pre_approach_z   = pre_approach_z
        self.carry_z          = carry_z
        self.grasp_z_offset   = grasp_z_offset
        self.release_z_offset  = release_z_offset
        self.retract_approach_z = retract_approach_z
        self.pos_tol           = pos_tol
        self.ang_tol           = ang_tol
        self.pos_tol_approach  = pos_tol_approach
        self.ang_tol_approach  = ang_tol_approach
        self.pos_tol_grasp     = pos_tol_grasp
        self.ang_tol_grasp     = ang_tol_grasp
        self.pos_tol_transport = pos_tol_transport
        self.ang_tol_transport = ang_tol_transport
        self.pos_tol_place     = pos_tol_place
        self.min_stage_dur    = min_stage_dur
        self.grasp_hold_s     = grasp_hold_s
        self.release_hold_s   = release_hold_s
        self.max_retries      = max_retries
        self.min_carry_cube_z = min_carry_cube_z
        self.grasp_secure_xy_tol = grasp_secure_xy_tol
        self.wrist_soft_limit    = wrist_soft_limit
        self.yaw_switch_cooldown = yaw_switch_cooldown
        self.yaw_k_max           = yaw_k_max
        self.pitch_cmd           = pitch_cmd
        self.pitch_transport     = pitch_transport
        self.container_drop      = container_drop
        self.place_yaw_gate      = place_yaw_gate
        self.max_step            = max_step
        self.pre_grasp_settle_s   = pre_grasp_settle_s
        self.pre_grasp_settle_ang = pre_grasp_settle_ang
        self.pre_grasp_yaw_tol    = pre_grasp_yaw_tol
        self.lift_anchor_radius   = lift_anchor_radius
        self.place_settle_s       = place_settle_s
        self.place_settle_max     = place_settle_max
        self.stall_pos_tol        = stall_pos_tol
        self.stall_time_s         = stall_time_s
        self.retreat_steps        = retreat_steps
        self.retreat_z            = retreat_z
        self.max_reach_retries    = max_reach_retries
        self.carry_drop_gap        = carry_drop_gap
        self.stage_escape_s        = stage_escape_s
        self.stage_escape_pos_mult = stage_escape_pos_mult
        self.upright_up_z_min       = upright_up_z_min
        self.lying_up_z_max         = lying_up_z_max
        self.lying_grasp_z_scale    = lying_grasp_z_scale
        self.grasp_finger_empty_max = grasp_finger_empty_max
        self.grasp_finger_contact_min = grasp_finger_contact_min
        self.place_verify_xy       = place_verify_xy
        self.place_verify_yaw      = place_verify_yaw
        self.max_place_retries     = max_place_retries
        self.max_lower_retries     = max_lower_retries
        self.hurry_after_s         = hurry_after_s
        self.hurry_scale           = hurry_scale
        self.container_retract_xy_offset = container_retract_xy_offset
        self.opportunistic_container_place = opportunistic_container_place
        self.opportunistic_z_above_goal_max = opportunistic_z_above_goal_max
        self.container_interior_half_table_x = container_interior_half_table_x
        self.container_interior_half_table_y = container_interior_half_table_y
        self.container_floor_z = container_floor_z
        self.container_rim_z = container_rim_z
        self.ang_tol_pre_grasp = ang_tol_pre_grasp
        closing = ClosingAxis.EE_X if grasp_closing_axis == "ee_x" else ClosingAxis.EE_Y
        self._gripper_frame = GripperFrameConfig(
            closing_axis=closing,
            yaw_offset=grasp_yaw_frame_offset,
        )
        self.grasp_yaw_flip_enabled = grasp_yaw_flip_enabled
        self.grasp_yaw_flip_after_retries = grasp_yaw_flip_after_retries
        self.grasp_yaw_flip_rad = grasp_yaw_flip_rad

        N, dev = num_envs, device
        self._stage       = torch.full((N,), int(Stage.PRE_GRASP), dtype=torch.long, device=dev)
        self._elapsed     = torch.zeros(N, device=dev)
        self._yaw         = torch.zeros(N, device=dev)
        self._retry_count = torch.zeros(N, dtype=torch.long, device=dev)
        self._place_retries = torch.zeros(N, dtype=torch.long, device=dev)
        self._place_bias_xy  = torch.zeros(N, 2, device=dev)
        self._place_bias_yaw = torch.zeros(N, device=dev)
        self._lower_retries  = torch.zeros(N, dtype=torch.long, device=dev)
        self._episode_t      = torch.zeros(N, device=dev)

        self._yaw_k         = torch.zeros(N, dtype=torch.long, device=dev)
        self._yaw_switch_cd = torch.zeros(N, dtype=torch.long, device=dev)
        self._pre_settle = torch.zeros(N, device=dev)
        self._place_offset  = torch.zeros(N, 2, device=dev)
        self._yaw_offset    = torch.zeros(N, device=dev)
        self._x_axis        = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3)
        self._lift_xy       = torch.zeros(N, 2, device=dev)
        self._retreat_ctr   = torch.zeros(N, dtype=torch.long, device=dev)
        self._reach_retries = torch.zeros(N, dtype=torch.long, device=dev)
        self._retract_xy_off = torch.tensor(
            container_retract_xy_offset, dtype=torch.float32, device=dev
        )
        self._target_pos  = torch.zeros(N, 3, device=dev)
        self._target_quat = torch.zeros(N, 4, device=dev)
        self._target_quat[:, 0] = 1.0
        self._task_idx = torch.zeros(N, dtype=torch.long, device=dev)

        # Per-object progress for skip-then-revisit multi-object scheduling.
        self._object_placed = torch.zeros(N, num_objects, dtype=torch.bool, device=dev)
        self._object_deferred = torch.zeros(N, num_objects, dtype=torch.bool, device=dev)
        self._object_abandoned = torch.zeros(N, num_objects, dtype=torch.bool, device=dev)
        self._planning_exhausted = torch.zeros(N, dtype=torch.bool, device=dev)
        self._skip_event = torch.zeros(N, dtype=torch.bool, device=dev)

        # Per-object grasp metadata cached tensors (set each step from command).
        self._cur_grasp_z_off  = torch.full((N,), grasp_z_offset, device=dev)
        self._cur_grasp_sym    = torch.full((N,), math.pi / 2, device=dev)
        self._cur_grasp_yaw_off = torch.zeros(N, device=dev)
        self._cur_upright_height = torch.zeros(N, device=dev)
        self._cur_grasp_offset_local = torch.zeros(N, 3, device=dev)
        self._cur_grasp_long_axis_local = torch.zeros(N, 3, device=dev)
        self._cur_footprint_xy = torch.zeros(N, 2, device=dev)
        self._cur_grasp_frame_yaw_off = torch.zeros(N, device=dev)
        self._grasp_yaw_flip = torch.zeros(N, device=dev)
        self._grasp_yaw_debug: GraspYawDebug | None = None
        self._cur_grasp_z_eff = torch.full((N,), grasp_z_offset, device=dev)
        self._grasp_orient_state = torch.zeros(N, dtype=torch.long, device=dev)
        self._grasp_verify_xy_ok = torch.ones(N, dtype=torch.bool, device=dev)
        self._grasp_verify_finger_ok = torch.ones(N, dtype=torch.bool, device=dev)
        self._object_center_xy = torch.zeros(N, 2, device=dev)
        self._grasp_aim_xy = torch.zeros(N, 2, device=dev)
        self._aim_offset_norm = torch.zeros(N, device=dev)
        self._aim_error_xy = torch.zeros(N, device=dev)
        self._secure_error_xy = torch.zeros(N, device=dev)
        self._used_corrected_aim_for_verify = torch.zeros(N, dtype=torch.bool, device=dev)
        self._local_up_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)

        self._grip_table  = torch.tensor(_STAGE_GRIP, device=dev)
        self._next_stage  = torch.arange(1, len(Stage) + 1, dtype=torch.long, device=dev).clamp_(max=int(Stage.DONE))

        pos_tol_stages = [
            pos_tol, pos_tol_approach, pos_tol_grasp,
            pos_tol_transport, pos_tol_transport, pos_tol_place,
            pos_tol_grasp, pos_tol_retract, pos_tol,
        ]
        ang_tol_stages = [
            ang_tol_pre_grasp, ang_tol_approach, ang_tol_grasp,
            ang_tol_transport, ang_tol_transport, ang_tol_transport,
            ang_tol_grasp, ang_tol_retract, ang_tol,
        ]
        self._pos_tol_table = torch.tensor(pos_tol_stages, dtype=torch.float32, device=dev)
        self._ang_tol_table = torch.tensor(ang_tol_stages, dtype=torch.float32, device=dev)

        esc = stage_escape_pos_mult
        self._esc_mult_table = torch.tensor(
            [0.0, 1.2, 0.0, esc, esc, 2.0, 0.0, esc, 0.0],
            dtype=torch.float32, device=dev,
        )

        self._pos_err    = torch.zeros(N, device=dev)
        self._ang_err    = torch.zeros(N, device=dev)
        self._yaw_err    = torch.zeros(N, device=dev)
        self._track_ok   = torch.zeros(N, dtype=torch.bool, device=dev)
        self._grasp_miss      = torch.zeros(N, dtype=torch.bool, device=dev)
        self._place_miss      = torch.zeros(N, dtype=torch.bool, device=dev)
        self._opportunistic_place     = torch.zeros(N, dtype=torch.bool, device=dev)
        self._opportunistic_place_obj = torch.full((N,), -1, dtype=torch.long, device=dev)
        self._lift_confirmed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._stage_changed  = torch.zeros(N, dtype=torch.bool, device=dev)
        self._pos_tol_eff    = torch.full((N,), pos_tol, device=dev)
        self._ang_tol_eff    = torch.full((N,), ang_tol, device=dev)

    @property
    def stage(self) -> torch.Tensor:
        return self._stage

    def is_fully_done(self) -> torch.Tensor:
        """Return bool tensor: True for envs where every object has been placed."""
        return self._object_placed.all(dim=1)

    def _reset_stage_state(
        self,
        env_mask: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> None:
        """Reset planner stage counters when switching to a new object."""
        if not env_mask.any():
            return
        self._stage[env_mask] = int(Stage.PRE_GRASP)
        self._elapsed[env_mask] = 0.0
        self._pre_settle[env_mask] = 0.0
        self._retry_count[env_mask] = 0
        self._reach_retries[env_mask] = 0
        self._yaw_k[env_mask] = 0
        self._yaw_switch_cd[env_mask] = 0
        self._place_offset[env_mask] = 0.0
        self._yaw_offset[env_mask] = 0.0
        self._lift_xy[env_mask] = 0.0
        self._retreat_ctr[env_mask] = 0
        self._place_retries[env_mask] = 0
        self._place_bias_xy[env_mask] = 0.0
        self._place_bias_yaw[env_mask] = 0.0
        self._lower_retries[env_mask] = 0
        self._lift_confirmed[env_mask] = False
        self._grasp_yaw_flip[env_mask] = 0.0

        env_ids = torch.where(env_mask)[0]
        self._yaw[env_ids] = self._resolve_grasp_yaw(cube_quat_w[env_ids], env_ids)

    def _advance_to_next_object(
        self,
        env_mask: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> None:
        """Pick the next pending object: fresh slots first, then deferred revisits."""
        if not env_mask.any():
            return

        M = self.num_objects
        device = self.device
        idx_range = torch.arange(M, device=device).unsqueeze(0).expand(self.num_envs, -1)
        sentinel = M + 1000

        pending_first = (~self._object_placed) & (~self._object_deferred) & (~self._object_abandoned)
        pending_revisit = (~self._object_placed) & self._object_deferred & (~self._object_abandoned)

        next_first = torch.where(pending_first, idx_range, sentinel).min(dim=1).values
        next_revisit = torch.where(pending_revisit, idx_range, sentinel).min(dim=1).values

        has_first = next_first < sentinel
        has_revisit = next_revisit < sentinel
        has_next = has_first | has_revisit
        new_idx = torch.where(has_first, next_first, next_revisit)

        moved = env_mask & has_next
        all_placed = env_mask & self._object_placed.all(dim=1)
        exhausted = env_mask & ~has_next & ~all_placed

        self._task_idx = torch.where(moved, new_idx, self._task_idx)
        self._reset_stage_state(moved, cube_quat_w)

        self._planning_exhausted = torch.where(
            exhausted,
            torch.ones_like(self._planning_exhausted),
            self._planning_exhausted,
        )
        self._planning_exhausted = torch.where(
            moved | all_placed,
            torch.zeros_like(self._planning_exhausted),
            self._planning_exhausted,
        )

        self._stage = torch.where(
            all_placed | exhausted,
            torch.full_like(self._stage, int(Stage.DONE)),
            self._stage,
        )

    def _defer_current_object(
        self,
        env_mask: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> None:
        """Skip the current object and schedule a later revisit when possible."""
        if not env_mask.any():
            return

        env_ids = torch.where(env_mask)[0]
        obj_idx = self._task_idx[env_ids]
        is_revisit = self._object_deferred[env_ids, obj_idx]

        first_skip = env_ids[~is_revisit]
        if first_skip.numel() > 0:
            first_obj = self._task_idx[first_skip]
            self._object_deferred[first_skip, first_obj] = True

        revisit_fail = env_ids[is_revisit]
        if revisit_fail.numel() > 0:
            fail_obj = self._task_idx[revisit_fail]
            self._object_abandoned[revisit_fail, fail_obj] = True

        self._skip_event[env_mask] = True
        self._advance_to_next_object(env_mask, cube_quat_w)

    def _update_grasp_orientation(
        self,
        cube_quat_w: torch.Tensor,
        cube_pos_w: torch.Tensor,
    ) -> None:
        """Classify upright vs lying and compute effective grasp-Z for this step."""
        has_meta = self._cur_upright_height > 0.0
        world_up = quat_apply(cube_quat_w, self._local_up_axis)
        up_z = world_up[:, 2].abs()

        upright = up_z >= self.upright_up_z_min
        lying_quat = up_z <= self.lying_up_z_max
        lying_height = has_meta & (~upright) & (up_z < self.upright_up_z_min)
        lying = has_meta & (lying_quat | lying_height)

        self._grasp_orient_state = torch.where(
            lying,
            torch.full_like(self._grasp_orient_state, 2),
            torch.where(
                upright | ~has_meta,
                torch.ones_like(self._grasp_orient_state),
                torch.zeros_like(self._grasp_orient_state),
            ),
        )

        base_z = self._cur_grasp_z_off
        lying_z_raw = base_z * self.lying_grasp_z_scale
        lying_z_max = torch.full_like(base_z, -0.005)
        lying_z = torch.clamp(lying_z_raw, min=base_z, max=lying_z_max)
        self._cur_grasp_z_eff = torch.where(lying, lying_z, base_z)

    def _update_grasp_aim_point(
        self,
        cube_pos_w: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute corrected grasp XY from optional object-frame offset."""
        self._object_center_xy = cube_pos_w[:, :2]

        offset_w = quat_apply(cube_quat_w, self._cur_grasp_offset_local)
        offset_valid = torch.isfinite(offset_w).all(dim=-1)
        local_norm = torch.norm(self._cur_grasp_offset_local, dim=-1)
        orient_known = self._grasp_orient_state != 0
        use_offset = (local_norm > 1e-6) & offset_valid & orient_known
        offset_w = torch.where(use_offset.unsqueeze(-1), offset_w, torch.zeros_like(offset_w))

        grasp_point_w = cube_pos_w + offset_w
        self._grasp_aim_xy = grasp_point_w[:, :2]
        self._aim_offset_norm = torch.norm(offset_w, dim=-1)
        self._used_corrected_aim_for_verify = use_offset

        grasp_x = torch.where(use_offset, grasp_point_w[:, 0], cube_pos_w[:, 0])
        grasp_y = torch.where(use_offset, grasp_point_w[:, 1], cube_pos_w[:, 1])
        return grasp_x, grasp_y

    def _elongated_object_mask(self) -> torch.Tensor:
        """True for objects where a ±90° yaw flip can fix a mis-aligned grasp."""
        return self._uses_shape_grasp_yaw() & (self._cur_grasp_sym > 1e-6)

    def _uses_shape_grasp_yaw(self) -> torch.Tensor:
        """True when grasp yaw is tied to object shape (boxes / catalog long axis)."""
        has_catalog = torch.norm(self._cur_grasp_long_axis_local, dim=-1) > 1e-6
        fp = self._cur_footprint_xy
        is_box = (
            (fp[:, 0] > 1e-6)
            & (fp[:, 1] > 1e-6)
            & ((fp[:, 0] - fp[:, 1]).abs() > 0.003)
        )
        return has_catalog | is_box

    def _merge_grasp_yaw_debug(self, env_ids: torch.Tensor, debug: GraspYawDebug) -> None:
        """Merge per-env grasp-yaw debug; supports partial env_id batches."""
        if env_ids.numel() == self.num_envs:
            self._grasp_yaw_debug = debug
            return
        if self._grasp_yaw_debug is None:
            self._grasp_yaw_debug = debug
            return
        prev = self._grasp_yaw_debug
        prev.object_yaw[env_ids] = debug.object_yaw
        prev.target_yaw[env_ids] = debug.target_yaw
        prev.long_axis_w[env_ids] = debug.long_axis_w
        prev.closing_axis_w[env_ids] = debug.closing_axis_w
        prev.width_alignment[env_ids] = debug.width_alignment
        for i, eid in enumerate(env_ids.tolist()):
            prev.source[eid] = debug.source[i]

    def _resolve_grasp_yaw(
        self,
        obj_quat: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Object-aligned wrist yaw for PRE_GRASP; stores debug vectors for logging."""
        batch = obj_quat.shape[0]
        if env_ids is None:
            env_ids = torch.arange(batch, device=self.device)
        else:
            env_ids = env_ids.reshape(-1)
        if env_ids.numel() != batch:
            raise ValueError(
                f"grasp yaw batch mismatch: obj_quat={batch}, env_ids={env_ids.numel()}"
            )

        yaw_off = (
            self._cur_grasp_yaw_off[env_ids]
            + self._cur_grasp_frame_yaw_off[env_ids]
        )
        yaw, debug = compute_grasp_yaw(
            obj_quat,
            self._cur_grasp_sym[env_ids],
            yaw_off,
            self._cur_grasp_long_axis_local[env_ids],
            self._cur_footprint_xy[env_ids],
            self._cur_upright_height[env_ids],
            self._gripper_frame,
        )
        yaw = normalize_yaw(yaw + self._grasp_yaw_flip[env_ids])
        debug.target_yaw = yaw
        debug.closing_axis_w = closing_direction_world_from_target(yaw, self._gripper_frame)
        long_xy = torch.cat(
            [debug.long_axis_w[:, :2], torch.zeros_like(debug.long_axis_w[:, 2:3])],
            dim=-1,
        )
        long_xy = normalize(long_xy)
        debug.width_alignment = (debug.closing_axis_w * long_xy).sum(dim=-1).abs()

        self._merge_grasp_yaw_debug(env_ids, debug)
        return yaw

    def _apply_yaw_flip_on_grasp_miss(self, env_mask: torch.Tensor) -> None:
        """On grasp miss, alternate +90° wrist yaw for elongated/box objects."""
        if not self.grasp_yaw_flip_enabled or not env_mask.any():
            return
        eligible = env_mask & self._elongated_object_mask()
        should_flip = eligible & (
            self._retry_count >= self.grasp_yaw_flip_after_retries
        )
        if not should_flip.any():
            return
        flip_val = torch.tensor(self.grasp_yaw_flip_rad, device=self.device, dtype=torch.float32)
        half_flip = 0.5 * self.grasp_yaw_flip_rad
        currently_flipped = self._grasp_yaw_flip > half_flip
        self._grasp_yaw_flip = torch.where(
            should_flip,
            torch.where(currently_flipped, torch.zeros_like(self._grasp_yaw_flip), flip_val),
            self._grasp_yaw_flip,
        )

    def _check_grasp_secure(
        self,
        cube_pos_w: torch.Tensor,
        ee_pos_w: torch.Tensor,
        finger_open: torch.Tensor | None,
        in_grasp_phase: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-env grasp security using XY proximity and finger opening."""
        use_corrected = self._used_corrected_aim_for_verify
        verify_xy = torch.where(
            use_corrected.unsqueeze(-1),
            self._grasp_aim_xy,
            cube_pos_w[:, :2],
        )
        self._secure_error_xy = torch.norm(verify_xy - ee_pos_w[:, :2], dim=-1)
        xy_ok = self._secure_error_xy < self.grasp_secure_xy_tol
        self._grasp_verify_xy_ok = xy_ok

        if finger_open is None:
            self._grasp_verify_finger_ok = torch.ones_like(xy_ok)
            return xy_ok

        finger_empty = finger_open < self.grasp_finger_empty_max
        finger_contact = finger_open >= self.grasp_finger_contact_min
        finger_ok = finger_contact | (~finger_empty & xy_ok)
        finger_ok = torch.where(in_grasp_phase, finger_ok, torch.ones_like(finger_ok))
        self._grasp_verify_finger_ok = finger_ok
        return xy_ok & finger_ok

    def reset(
        self,
        env_ids:     torch.Tensor,
        ee_pos_w:    torch.Tensor | None = None,
        ee_quat_w:   torch.Tensor | None = None,
        cube_quat_w: torch.Tensor | None = None,
        grasp_sym:   torch.Tensor | None = None,
        grasp_yaw_off: torch.Tensor | None = None,
        grasp_long_axis_local: torch.Tensor | None = None,
        footprint_xy: torch.Tensor | None = None,
        grasp_frame_yaw_off: torch.Tensor | None = None,
    ) -> None:
        """Reset selected envs to PRE_GRASP, task_idx = 0."""
        if env_ids.numel() == 0:
            return
        ids = env_ids
        if grasp_sym is not None:
            self._cur_grasp_sym[ids] = grasp_sym[ids]
        if grasp_yaw_off is not None:
            self._cur_grasp_yaw_off[ids] = grasp_yaw_off[ids]
        if grasp_long_axis_local is not None:
            self._cur_grasp_long_axis_local[ids] = grasp_long_axis_local[ids]
        if footprint_xy is not None:
            self._cur_footprint_xy[ids] = footprint_xy[ids]
        if grasp_frame_yaw_off is not None:
            self._cur_grasp_frame_yaw_off[ids] = grasp_frame_yaw_off[ids]

        self._stage[ids]          = int(Stage.PRE_GRASP)
        self._elapsed[ids]        = 0.0
        self._retry_count[ids]    = 0
        self._task_idx[ids]       = 0
        self._object_placed[ids] = False
        self._object_deferred[ids] = False
        self._object_abandoned[ids] = False
        self._planning_exhausted[ids] = False
        self._skip_event[ids] = False
        self._target_pos[ids]     = 0.0
        self._target_quat[ids]    = 0.0
        self._target_quat[ids, 0] = 1.0
        self._yaw[ids]            = (
            self._resolve_grasp_yaw(cube_quat_w[ids], ids)
            if cube_quat_w is not None else torch.zeros(ids.numel(), device=self.device)
        )
        self._yaw_k[ids]          = 0
        self._yaw_switch_cd[ids]  = 0
        self._place_offset[ids]   = 0.0
        self._yaw_offset[ids]     = 0.0
        self._lift_xy[ids]        = 0.0
        self._retreat_ctr[ids]    = 0
        self._reach_retries[ids]  = 0
        self._pre_settle[ids]     = 0.0
        self._place_retries[ids]  = 0
        self._place_bias_xy[ids]  = 0.0
        self._place_bias_yaw[ids] = 0.0
        self._lower_retries[ids]  = 0
        self._grasp_yaw_flip[ids] = 0.0
        self._episode_t[ids]      = 0.0
        self._cur_grasp_z_eff[ids] = self._cur_grasp_z_off[ids]
        self._grasp_orient_state[ids] = 1
        self._grasp_verify_xy_ok[ids] = True
        self._grasp_verify_finger_ok[ids] = True
        self._aim_offset_norm[ids] = 0.0
        self._aim_error_xy[ids] = 0.0
        self._secure_error_xy[ids] = 0.0
        self._used_corrected_aim_for_verify[ids] = False

    def step(
        self,
        cube_pos_w:  torch.Tensor,   # (N, 3) current object
        cube_quat_w: torch.Tensor,   # (N, 4) wxyz
        goal_pos_w:  torch.Tensor,   # (N, 3) container drop target (XY = bin centre, Z = drop height)
        goal_quat_w: torch.Tensor,   # (N, 4) wxyz (used for orientation-agnostic drop; neutral)
        ee_pos_w:    torch.Tensor,   # (N, 3) panda_hand
        ee_quat_w:   torch.Tensor,   # (N, 4)
        dt:          float,
        wrist_angle: torch.Tensor | None = None,
        # Per-object grasp metadata (selected by HLPoseCommand using task_idx).
        grasp_z_off:   torch.Tensor | None = None,  # (N,)
        grasp_sym:     torch.Tensor | None = None,  # (N,)
        grasp_yaw_off: torch.Tensor | None = None,  # (N,)
        upright_height: torch.Tensor | None = None,  # (N,)
        grasp_offset_local: torch.Tensor | None = None,  # (N, 3)
        grasp_long_axis_local: torch.Tensor | None = None,  # (N, 3)
        footprint_xy: torch.Tensor | None = None,  # (N, 2)
        grasp_frame_yaw_off: torch.Tensor | None = None,  # (N,) per-object 0 or π/2
        finger_open:   torch.Tensor | None = None,  # (N,) mean finger joint opening (m)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One planner step.  Returns ``(end_pos_w, end_quat_w, grip)``."""

        # Update current object grasp metadata.
        if grasp_z_off is not None:
            self._cur_grasp_z_off.copy_(grasp_z_off)
        if grasp_sym is not None:
            self._cur_grasp_sym.copy_(grasp_sym)
        if grasp_yaw_off is not None:
            self._cur_grasp_yaw_off.copy_(grasp_yaw_off)
        if upright_height is not None:
            self._cur_upright_height.copy_(upright_height)
        if grasp_offset_local is not None:
            self._cur_grasp_offset_local.copy_(grasp_offset_local)
        if grasp_long_axis_local is not None:
            self._cur_grasp_long_axis_local.copy_(grasp_long_axis_local)
        if footprint_xy is not None:
            self._cur_footprint_xy.copy_(footprint_xy)
        if grasp_frame_yaw_off is not None:
            self._cur_grasp_frame_yaw_off.copy_(grasp_frame_yaw_off)

        # Yaw is set and settled in PRE_GRASP only; frozen before DESCEND/GRASP.
        in_pre_grasp = self._stage == int(Stage.PRE_GRASP)
        in_descend = self._stage == int(Stage.DESCEND)
        past_descend = self._stage > int(Stage.DESCEND)
        if in_pre_grasp.any():
            self._update_grasp_orientation(cube_quat_w, cube_pos_w)
            new_yaw = self._resolve_grasp_yaw(cube_quat_w)
            self._yaw = torch.where(in_pre_grasp, new_yaw, self._yaw)
        if in_descend.any():
            self._update_grasp_orientation(cube_quat_w, cube_pos_w)
        if past_descend.any():
            self._cur_grasp_z_eff = torch.where(
                past_descend, self._cur_grasp_z_off, self._cur_grasp_z_eff
            )

        grasp_x, grasp_y = self._update_grasp_aim_point(cube_pos_w, cube_quat_w)

        # Wrist-unwind: skip when shape-sensitive grasp yaw is active.
        shape_yaw = self._uses_shape_grasp_yaw()
        if wrist_angle is not None:
            self._yaw_switch_cd = (self._yaw_switch_cd - 1).clamp_(min=0)
            unwind_ok = (self._stage <= int(Stage.DESCEND)) & (~shape_yaw)
            ready = (self._yaw_switch_cd == 0) & unwind_ok
            over_pos = ready & (wrist_angle >  self.wrist_soft_limit) & (self._yaw_k > -self.yaw_k_max)
            over_neg = ready & (wrist_angle < -self.wrist_soft_limit) & (self._yaw_k <  self.yaw_k_max)
            self._yaw_k = torch.where(over_pos, self._yaw_k - 1, self._yaw_k)
            self._yaw_k = torch.where(over_neg, self._yaw_k + 1, self._yaw_k)
            switched = over_pos | over_neg
            if switched.any():
                self._yaw_switch_cd = torch.where(
                    switched,
                    torch.full_like(self._yaw_switch_cd, self.yaw_switch_cooldown),
                    self._yaw_switch_cd,
                )

        # Track in-gripper XY offset while holding (table placement only).
        holding = (self._stage >= int(Stage.GRASP)) & (self._stage <= int(Stage.LOWER))
        if holding.any() and not self.container_drop:
            off = cube_pos_w[:, :2] - ee_pos_w[:, :2]
            self._place_offset = torch.where(holding.unsqueeze(-1), off, self._place_offset)

        # Lift anchor: EE XY at GRASP.
        cap = self._stage == int(Stage.GRASP)
        if cap.any():
            self._lift_xy = torch.where(cap.unsqueeze(-1), ee_pos_w[:, :2], self._lift_xy)

        # In-gripper yaw offset (only used in non-container mode via place_bias).
        grasp_hold = (self._stage == int(Stage.GRASP)) | (self._stage == int(Stage.LIFT))
        if grasp_hold.any() and not self.container_drop:
            cv = quat_apply(cube_quat_w, self._x_axis)
            ev = quat_apply(ee_quat_w, self._x_axis)
            cube_h = torch.atan2(cv[:, 1], cv[:, 0])
            grip_h = torch.atan2(ev[:, 1], ev[:, 0])
            hp = 0.5 * math.pi
            yoff = (cube_h - grip_h + 0.25 * hp) % hp - 0.25 * hp
            self._yaw_offset = torch.where(grasp_hold, yoff, self._yaw_offset)

        end_pos, end_quat = self._end_pose(cube_pos_w, goal_pos_w, goal_quat_w, grasp_x, grasp_y)
        old_stage = self._stage.clone()

        self._elapsed += dt
        self._episode_t += dt
        hurry_mul = torch.where(
            self._episode_t > self.hurry_after_s,
            torch.full_like(self._episode_t, self.hurry_scale),
            torch.ones_like(self._episode_t),
        )

        pos_err  = torch.norm(ee_pos_w - end_pos, dim=-1)
        ang_err  = quat_error_magnitude(ee_quat_w, end_quat)
        _, _, ee_yaw = euler_xyz_from_quat(ee_quat_w)
        self._yaw_err = torch.abs(
            torch.atan2(torch.sin(ee_yaw - self._yaw), torch.cos(ee_yaw - self._yaw))
        )
        in_aim_phase = self._stage <= int(Stage.GRASP)
        self._aim_error_xy = torch.where(
            in_aim_phase,
            torch.norm(ee_pos_w[:, :2] - self._grasp_aim_xy, dim=-1),
            self._aim_error_xy,
        )
        pos_tol_eff = self._pos_tol_table[self._stage]
        ang_tol_eff = self._ang_tol_table[self._stage]
        track_ok = (pos_err < pos_tol_eff) & (ang_err < ang_tol_eff)
        self._pos_tol_eff.copy_(pos_tol_eff)
        self._ang_tol_eff.copy_(ang_tol_eff)
        self._pos_err.copy_(pos_err)
        self._ang_err.copy_(ang_err)
        self._track_ok.copy_(track_ok)
        self._grasp_miss.fill_(False)
        self._place_miss.fill_(False)
        self._skip_event.fill_(False)
        self._opportunistic_place.fill_(False)
        self._opportunistic_place_obj.fill_(-1)

        arange = torch.arange(self.num_envs, device=self.device)
        cur_obj = self._task_idx
        object_active = ~(self._object_placed[arange, cur_obj] | self._object_abandoned[arange, cur_obj])

        # Opportunistic in-bin placement: only after a confirmed lift, during drop.
        if self.opportunistic_container_place and self.container_drop:
            in_drop_phase = (
                (self._stage == int(Stage.LOWER))
                | (self._stage == int(Stage.RELEASE))
            )
            inside = object_inside_container_interior(
                cube_pos_w,
                goal_pos_w,
                self.container_interior_half_table_x,
                self.container_interior_half_table_y,
                self.container_floor_z,
                self.container_rim_z,
            )
            opp_mask = in_drop_phase & inside & self._lift_confirmed & object_active
            if opp_mask.any():
                self._opportunistic_place.copy_(opp_mask)
                self._opportunistic_place_obj[opp_mask] = self._task_idx[opp_mask]
                self._object_placed[opp_mask, self._task_idx[opp_mask]] = True
                self._retry_count[opp_mask] = 0
                self._advance_to_next_object(opp_mask, cube_quat_w)
                cur_obj = self._task_idx
                object_active = ~(
                    self._object_placed[arange, cur_obj]
                    | self._object_abandoned[arange, cur_obj]
                )

        in_grasp   = self._stage == int(Stage.GRASP)
        in_release = self._stage == int(Stage.RELEASE)
        in_lower   = self._stage == int(Stage.LOWER)
        in_done    = self._stage == int(Stage.DONE)

        in_grasp_phase = in_grasp | (self._stage == int(Stage.LIFT))
        grasp_secure = self._check_grasp_secure(
            cube_pos_w,
            ee_pos_w,
            finger_open,
            in_grasp_phase,
        )

        at_end = (self._elapsed >= self.min_stage_dur) & track_ok

        # PRE_GRASP orientation-stall escape.
        in_pre_grasp = self._stage == int(Stage.PRE_GRASP)
        in_pos       = pos_err < pos_tol_eff
        self._pre_settle = torch.where(
            in_pre_grasp & in_pos,
            self._pre_settle + dt,
            torch.zeros_like(self._pre_settle),
        )
        pre_settle_req = self.pre_grasp_settle_s * hurry_mul
        pre_grasp_settled = in_pre_grasp & (
            ((self._pre_settle >= pre_settle_req) & (ang_err < self.pre_grasp_settle_ang))
            | (self._pre_settle >= 3.0 * pre_settle_req)
        )

        grasp_hold_done = self._elapsed >= self.min_stage_dur + self.grasp_hold_s
        grasp_ok = in_grasp & grasp_hold_done & track_ok & grasp_secure
        grasp_pos_escape = (
            in_grasp
            & grasp_hold_done
            & (self._elapsed >= self.min_stage_dur + self.grasp_hold_s + self.stage_escape_s * hurry_mul)
            & (pos_err < pos_tol_eff)
            & grasp_secure
        )
        release_ok = in_release & (self._elapsed >= self.min_stage_dur + self.release_hold_s) & track_ok

        # LOWER: in container mode just use a time-based settle (no yaw gate).
        if self.container_drop:
            lower_min = self._elapsed >= self.min_stage_dur + self.place_settle_s * hurry_mul
            lower_cap = self._elapsed >= self.min_stage_dur + self.place_settle_max * hurry_mul
            lower_ok  = in_lower & track_ok & (lower_min | lower_cap)
        else:
            ev = quat_apply(ee_quat_w, self._x_axis)
            gv = quat_apply(goal_quat_w, self._x_axis)
            grip_h = torch.atan2(ev[:, 1], ev[:, 0])
            goal_h = torch.atan2(gv[:, 1], gv[:, 0])
            hp = 0.5 * math.pi
            pred_yaw_err = torch.abs((grip_h + self._yaw_offset - goal_h + 0.25 * hp) % hp - 0.25 * hp)
            lower_min = self._elapsed >= self.min_stage_dur + self.place_settle_s * hurry_mul
            lower_cap = self._elapsed >= self.min_stage_dur + self.place_settle_max * hurry_mul
            lower_ok  = in_lower & track_ok & ((lower_min & (pred_yaw_err < self.place_yaw_gate)) | lower_cap)

        can_advance = at_end.clone()
        can_advance = torch.where(in_grasp,   grasp_ok | grasp_pos_escape,   can_advance)
        can_advance = torch.where(in_release, release_ok, can_advance)
        can_advance = torch.where(in_lower,   lower_ok,   can_advance)
        can_advance = can_advance | pre_grasp_settled

        # Tolerance-floor escape.
        esc_bound = self._esc_mult_table[self._stage] * pos_tol_eff
        stage_escaped = (
            (self._elapsed >= self.min_stage_dur + self.stage_escape_s * hurry_mul)
            & (pos_err < esc_bound)
        )
        lower_redo = stage_escaped & in_lower & (self._lower_retries < self.max_lower_retries)
        if lower_redo.any():
            self._lower_retries = torch.where(lower_redo, self._lower_retries + 1, self._lower_retries)
            self._stage   = torch.where(lower_redo, torch.full_like(self._stage, int(Stage.CARRY)), self._stage)
            self._elapsed = torch.where(lower_redo, torch.zeros_like(self._elapsed), self._elapsed)
        can_advance = can_advance | (stage_escaped & ~lower_redo)
        can_advance &= ~in_done

        # Config-break retreat.
        in_approach = self._stage <= int(Stage.DESCEND)
        stalled = (in_approach & (pos_err > self.stall_pos_tol)
                   & (self._elapsed > self.stall_time_s)
                   & (self._retreat_ctr == 0)
                   & (self._reach_retries < self.max_reach_retries))
        if stalled.any():
            self._retreat_ctr   = torch.where(stalled, torch.full_like(self._retreat_ctr, self.retreat_steps), self._retreat_ctr)
            self._reach_retries = torch.where(stalled, self._reach_retries + 1, self._reach_retries)
            self._elapsed       = torch.where(stalled, torch.zeros_like(self._elapsed), self._elapsed)
        retreating = self._retreat_ctr > 0
        self._retreat_ctr = (self._retreat_ctr - 1).clamp_(min=0)
        can_advance = can_advance & ~retreating

        reach_exhausted = (
            in_approach
            & (pos_err > self.stall_pos_tol)
            & (self._elapsed > self.stall_time_s)
            & (self._reach_retries >= self.max_reach_retries)
            & (self._retreat_ctr == 0)
            & object_active
        )
        if reach_exhausted.any():
            self._defer_current_object(reach_exhausted, cube_quat_w)
            object_active = ~(self._object_placed[arange, cur_obj] | self._object_abandoned[arange, cur_obj])

        # Grasp miss / carry drop recovery.
        in_carry     = self._stage == int(Stage.CARRY)
        grasp_failed = in_grasp & (self._elapsed >= self.min_stage_dur + self.grasp_hold_s) & ~grasp_secure
        drop_gap = (ee_pos_w[:, 2] - self.H) - cube_pos_w[:, 2]
        carry_missed = (
            in_carry
            & (cube_pos_w[:, 2] < self.min_carry_cube_z)
            & (drop_gap > self.carry_drop_gap)
        )
        if self.opportunistic_container_place and self.container_drop:
            settled = object_settled_in_container(
                cube_pos_w,
                goal_pos_w,
                self.place_verify_xy,
                self.opportunistic_z_above_goal_max,
            )
            carry_missed = carry_missed & ~settled
        grasp_miss   = (grasp_failed | carry_missed) & (self._retry_count < self.max_retries)
        if grasp_miss.any():
            self._grasp_miss.copy_(grasp_miss)
            self._retry_count[grasp_miss] += 1
            self._apply_yaw_flip_on_grasp_miss(grasp_miss)
            self._lift_confirmed[grasp_miss] = False
            self._stage[grasp_miss]       = int(Stage.PRE_GRASP)
            self._elapsed[grasp_miss]     = 0.0
            self._pre_settle[grasp_miss]  = 0.0
            miss_ids = torch.where(grasp_miss)[0]
            self._yaw[miss_ids] = self._resolve_grasp_yaw(cube_quat_w[miss_ids], miss_ids)

        grasp_still_failing = grasp_failed | carry_missed
        grasp_exhausted = (
            grasp_still_failing
            & (self._retry_count >= self.max_retries)
            & object_active
        )
        if grasp_exhausted.any():
            self._grasp_miss.copy_(grasp_exhausted)
            self._defer_current_object(grasp_exhausted, cube_quat_w)

        if can_advance.any():
            adv = can_advance & ~grasp_miss & ~grasp_exhausted
            lifted = adv & (old_stage == int(Stage.LIFT))
            self._lift_confirmed = torch.where(
                lifted,
                torch.ones_like(self._lift_confirmed),
                self._lift_confirmed,
            )
            self._elapsed = torch.where(adv, torch.zeros_like(self._elapsed), self._elapsed)
            self._stage   = torch.where(adv, self._next_stage[self._stage], self._stage)

        # Closed-loop placement verification (container mode: only check XY vs bin).
        newly_done = (self._stage == int(Stage.DONE)) & (old_stage != int(Stage.DONE))
        hp = 0.5 * math.pi
        if newly_done.any():
            place_xy_err = torch.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1)
            if self.container_drop:
                place_missed = (
                    newly_done
                    & (place_xy_err > self.place_verify_xy)
                    & (self._place_retries < self.max_place_retries)
                )
            else:
                cvx = quat_apply(cube_quat_w, self._x_axis)
                gvx = quat_apply(goal_quat_w, self._x_axis)
                cube_head = torch.atan2(cvx[:, 1], cvx[:, 0])
                goal_head = torch.atan2(gvx[:, 1], gvx[:, 0])
                place_yaw_err = torch.abs((cube_head - goal_head + 0.25 * hp) % hp - 0.25 * hp)
                place_missed = (
                    newly_done
                    & ((place_xy_err > self.place_verify_xy) | (place_yaw_err > self.place_verify_yaw))
                    & (self._place_retries < self.max_place_retries)
                )
            if place_missed.any():
                self._place_miss.copy_(place_missed)
                self._place_retries[place_missed] += 1
                self._stage[place_missed]        = int(Stage.PRE_GRASP)
                self._elapsed[place_missed]      = 0.0
                self._pre_settle[place_missed]   = 0.0
                place_ids = torch.where(place_missed)[0]
                self._yaw[place_ids] = self._resolve_grasp_yaw(
                    cube_quat_w[place_ids], place_ids
                )
                self._place_offset[place_missed] = 0.0
                self._yaw_offset[place_missed]   = 0.0
                self._retreat_ctr[place_missed]  = 0
                self._lower_retries[place_missed] = 0
                if not self.container_drop:
                    err_xy = (cube_pos_w[:, :2] - goal_pos_w[:, :2]).clamp(-0.05, 0.05)
                    self._place_bias_xy = torch.where(
                        place_missed.unsqueeze(-1),
                        (self._place_bias_xy + err_xy).clamp(-0.05, 0.05),
                        self._place_bias_xy,
                    )
            newly_done = newly_done & ~place_missed

            place_exhausted = (
                place_missed
                & (self._place_retries >= self.max_place_retries)
                & object_active
            )
            if place_exhausted.any():
                self._defer_current_object(place_exhausted, cube_quat_w)

        # Multi-object: mark placed objects and advance the pick queue.
        if newly_done.any():
            env_done = torch.where(newly_done)[0]
            self._object_placed[env_done, self._task_idx[env_done]] = True
            self._advance_to_next_object(newly_done, cube_quat_w)

        self._stage_changed = old_stage != self._stage

        self._target_pos.copy_(end_pos)
        self._target_quat.copy_(end_quat)

        # Config-break retreat: command a high pose above current object.
        retreat_pos = torch.stack(
            [cube_pos_w[:, 0], cube_pos_w[:, 1],
             torch.full_like(cube_pos_w[:, 2], self.retreat_z + self.H)],
            dim=-1,
        )
        cmd_target = torch.where(retreating.unsqueeze(-1), retreat_pos, end_pos)

        # Smooth pursuit carrot.
        delta = cmd_target - ee_pos_w
        dist  = torch.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_step / (dist + 1e-6), max=1.0)
        cmd_pos = ee_pos_w + delta * scale
        return cmd_pos, end_quat, self._grip_table[self._stage]

    # ------------------------------------------------------------------

    def _end_pose(
        self,
        cube_pos_w: torch.Tensor,
        goal_pos_w: torch.Tensor,
        goal_quat_w: torch.Tensor,
        grasp_x: torch.Tensor,
        grasp_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Target (position, quaternion) in world frame for the current stage."""
        H  = self.H
        cx, cy, cz = cube_pos_w[:, 0], cube_pos_w[:, 1], cube_pos_w[:, 2]
        gx, gy, gz = goal_pos_w[:, 0], goal_pos_w[:, 1], goal_pos_w[:, 2]

        # Per-object grasp Z offset (replaces scalar grasp_z_offset for DESCEND/GRASP).
        per_z = self._cur_grasp_z_eff

        if self.container_drop:
            # Keep grasp yaw through CARRY/LOWER/RELEASE so TCP stays over object centre.
            yaw = self._yaw
        else:
            goal_yaw_sym = (
                compute_grasp_yaw_symmetry(goal_quat_w, self._cur_grasp_sym, self._cur_grasp_yaw_off)
                - self._yaw_offset
                - self._place_bias_yaw
            )
            use_goal_ori = self._stage >= int(Stage.CARRY)
            yaw = torch.where(use_goal_ori, goal_yaw_sym, self._yaw)

        shape_yaw = self._uses_shape_grasp_yaw()
        in_grasp_approach = self._stage <= int(Stage.GRASP)
        skip_unwind = shape_yaw & in_grasp_approach
        yaw = yaw + torch.where(
            skip_unwind,
            torch.zeros_like(yaw),
            self._yaw_k.to(yaw.dtype) * (0.5 * math.pi),
        )

        # Z targets:
        z_pre     = cz + self.pre_approach_z + H
        z_grasp   = cz + per_z + H           # per-object grasp depth
        z_carry   = torch.full_like(cz, self.carry_z + H)
        if self.container_drop:
            # LOWER = goal_z directly (set by command to rim + drop_offset).
            # RELEASE = slightly below LOWER (open fingers inside the bin).
            z_place   = gz + H
            z_release = gz + self.release_z_offset + H
        else:
            z_place   = gz + per_z + H
            z_release = gz + self.release_z_offset + H
        z_retract = gz + self.retract_approach_z + H

        z_table = torch.stack(
            [
                z_pre, z_grasp, z_grasp, z_carry, z_carry,
                z_place, z_release, z_retract, z_retract,
            ],
            dim=1,
        )
        ez = z_table.gather(1, self._stage.unsqueeze(-1)).squeeze(-1)

        # XY targets.
        use_goal = self._stage >= int(Stage.CARRY)
        placing  = use_goal & (self._stage <= int(Stage.RELEASE))
        if self.container_drop:
            # Command TCP directly over bin centre; object drops straight down on release.
            ex_place = gx
            ey_place = gy
        else:
            ex_place = gx - self._place_offset[:, 0] - self._place_bias_xy[:, 0]
            ey_place = gy - self._place_offset[:, 1] - self._place_bias_xy[:, 1]

        is_lift = self._stage == int(Stage.LIFT)
        d = torch.stack([grasp_x, grasp_y], dim=-1) - self._lift_xy
        n = torch.norm(d, dim=-1, keepdim=True)
        lift_xy = self._lift_xy + d * torch.clamp(self.lift_anchor_radius / (n + 1e-6), max=1.0)
        ex = torch.where(placing, ex_place, torch.where(use_goal, gx, torch.where(is_lift, lift_xy[:, 0], grasp_x)))
        ey = torch.where(placing, ey_place, torch.where(use_goal, gy, torch.where(is_lift, lift_xy[:, 1], grasp_y)))

        # Container RETRACT: shift XY toward the robot before lifting out of the bin.
        is_retract = self._stage == int(Stage.RETRACT)
        if self.container_drop:
            ex = torch.where(
                is_retract,
                gx + self._retract_xy_off[0],
                ex,
            )
            ey = torch.where(
                is_retract,
                gy + self._retract_xy_off[1],
                ey,
            )

        end_pos  = torch.stack([ex, ey, ez], dim=-1)

        transport = (self._stage == int(Stage.LIFT)) | (self._stage == int(Stage.CARRY))
        pitch = torch.where(transport,
                            torch.full_like(yaw, self.pitch_transport),
                            torch.full_like(yaw, self.pitch_cmd))
        end_quat = quat_from_euler_xyz(
            torch.zeros_like(yaw),
            pitch,
            yaw,
        )
        return end_pos, end_quat
