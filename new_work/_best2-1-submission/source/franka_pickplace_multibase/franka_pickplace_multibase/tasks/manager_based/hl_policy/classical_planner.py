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
When DONE is reached for object ``k < num_objects - 1`` the planner
automatically increments ``_task_idx`` and resets to PRE_GRASP for the next
object.  When DONE is reached for the final object (``k == num_objects - 1``)
the planner stays in DONE.

**Skip-then-revisit:** If the current object exhausts grasp or reach retries,
it is deferred and the planner moves to the next pending object.  After all
immediate objects are attempted, deferred objects are revisited once with a
fresh retry budget.  Objects that fail again are abandoned.

**Grasp approach hardening:** Yaw is settled in ``PRE_GRASP`` only (frozen
before ``DESCEND``).  Grasp misses trigger a high retreat and optional ±90°
yaw flip on elongated objects.  During ``DESCEND``, if the EE is near object
height but XY-misaligned, Z is held at hover until centred.

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
    quat_mul,
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


# 0 = open, 1 = closed
_STAGE_GRIP: list[float] = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

# Hard catalogs dominate the remaining grasp_miss logs. Give only these object
# classes one extra retry plus a deliberately different retry pose schedule.
HARD_GRASP_CATALOGS: tuple[int, ...] = (0, 2, 3)
_HARD_GRASP_CATALOG_SET = set(HARD_GRASP_CATALOGS)
_HARD_RETRY_PATTERNS: tuple[tuple[float, float, float, float], ...] = (
    (0.000, 0.000, 0.000, 0.0000),   # baseline
    (0.000, 0.000, -0.002, 0.0000),  # slightly deeper
    (0.006, 0.000, -0.002, 0.0000),  # gentle local +x probe
    (-0.006, 0.000, -0.002, 0.0000), # gentle local -x probe
)
_HARD_CATALOG_EXTRA_DEPTH: dict[int, float] = {0: 0.002, 2: 0.002, 3: 0.002}


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to ``[-pi, pi]`` for stable yaw comparisons."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grasp_yaw(
    obj_quat: torch.Tensor,
    sym: torch.Tensor,
    yaw_off: torch.Tensor,
) -> torch.Tensor:
    """Compute grasp yaw snapped to the object's symmetry period.

    Args:
        obj_quat: ``(N, 4)`` object quaternion (w, x, y, z).
        sym:      ``(N,)`` symmetry period in radians.
                  ``0`` → rotationally symmetric; return neutral wrist (0 rad).
        yaw_off:  ``(N,)`` constant yaw offset added after folding (rad).

    Returns:
        ``(N,)`` grasp yaw in world frame.
    """
    _, _, yaw = euler_xyz_from_quat(obj_quat)
    yaw = _wrap_to_pi(yaw + yaw_off)

    # Rotationally symmetric objects: neutral wrist.
    neutral_mask = sym <= 0.0
    # Avoid division by zero for neutral case.
    safe_sym = torch.where(neutral_mask, torch.ones_like(sym), sym)
    folded = (yaw + 0.5 * safe_sym) % safe_sym - 0.5 * safe_sym
    return torch.where(neutral_mask, torch.zeros_like(yaw), _wrap_to_pi(folded))


def _symmetry_aware_yaw_error(
    yaw_a: torch.Tensor,
    yaw_b: torch.Tensor,
    sym: torch.Tensor,
) -> torch.Tensor:
    """Return the minimum yaw difference modulo the object's grasp symmetry."""
    base = _wrap_to_pi(yaw_a - yaw_b)
    neutral_mask = sym <= 0.0
    safe_sym = torch.where(neutral_mask, torch.full_like(sym, 2.0 * math.pi), sym)
    folded = (base + 0.5 * safe_sym) % safe_sym - 0.5 * safe_sym
    return torch.abs(torch.where(neutral_mask, base, _wrap_to_pi(folded)))


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
        release_z_offset:  float = -0.020,
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
        pre_grasp_settle_s:   float = 5.0,
        pre_grasp_settle_ang: float = 0.75,
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
        # Placement verification relaxed for container drop.
        place_verify_xy:      float = 0.18,  # bin interior half-extent; re-pick only if missed bin
        place_verify_yaw:     float = 10.0,  # disabled: no yaw requirement
        max_place_retries:    int   = 1,
        max_lower_retries:    int   = 1,
        hurry_after_s:        float = 25.0,  # longer budget per object
        hurry_scale:          float = 0.6,
        container_retract_xy_offset: tuple[float, float] = (-0.05, 0.12),
        skip_then_revisit: bool = True,
        pre_grasp_yaw_tol: float = 0.35,
        freeze_yaw_before_descend: bool = True,
        grasp_yaw_flip_enabled: bool = True,
        grasp_yaw_flip_after_retries: int = 1,
        grasp_yaw_flip_rad: float = math.pi / 2,
        grasp_miss_retreat: bool = True,
        descend_xy_gate_z_margin: float = 0.08,
        keep_grasp_yaw_container: bool = True,
        release_at_container_center: bool = False,
        safe_release_above_rim: bool = True,
        table_z: float = 0.03,
        pre_grasp_pos_relax_thresh: float = 0.02,
        pre_grasp_relaxed_ang_bonus: float = 0.10,
        pre_grasp_relaxed_yaw_bonus: float = 0.08,
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
        # Grasp pitch (pi = straight down; pi/2 = horizontal). Tunable to test a
        # tilted/"horizontal" grip. NOTE the LL policy was trained only on
        # pitch in (2.8, pi) (~20 deg of vertical), so large tilts are untracked.
        import os as _os2
        self.pitch_cmd = float(_os2.environ.get("NEPHER_GRASP_PITCH", str(self.pitch_cmd)))
        self.container_drop      = container_drop
        self.place_yaw_gate      = place_yaw_gate
        self.max_step            = max_step
        self.pre_grasp_settle_s   = pre_grasp_settle_s
        self.pre_grasp_settle_ang = pre_grasp_settle_ang
        self.lift_anchor_radius   = lift_anchor_radius
        self.place_settle_s       = place_settle_s
        self.place_settle_max     = place_settle_max
        # stall_pos_tol gates the config-break retreat (the top miner's bent-elbow
        # fix). Default 0.06 is just ABOVE the multibase under-descend pos_err
        # (~0.058), so the retreat never fires and the grasp stalls to timeout.
        # Lower it so the under-descend triggers the retreat -> re-extend -> reach
        # full depth (converts grasp-stall timeouts, the whole gap to the top agent).
        import os as _osS
        self.stall_pos_tol        = float(_osS.environ.get("NEPHER_STALL_POS_TOL", str(stall_pos_tol)))
        self.stall_time_s         = float(_osS.environ.get("NEPHER_STALL_TIME", str(stall_time_s)))
        # Lateral offset added to the config-break retreat so the re-descent is
        # diagonal (breaks the bent-elbow trap on hard multibase objects). 0 = the
        # base straight-up retreat (which re-traps).
        self.retreat_xy_break     = float(_osS.environ.get("NEPHER_RETREAT_XY_BREAK", "0.15"))
        self.retreat_steps        = int(float(_osS.environ.get("NEPHER_RETREAT_STEPS", "45")))
        self.retreat_z            = retreat_z
        self.max_reach_retries    = int(float(_osS.environ.get("NEPHER_MAX_REACH_RETRIES", "12")))
        self.carry_drop_gap        = carry_drop_gap
        self.stage_escape_s        = stage_escape_s
        self.stage_escape_pos_mult = stage_escape_pos_mult
        self.place_verify_xy       = place_verify_xy
        self.place_verify_yaw      = place_verify_yaw
        self.max_place_retries     = max_place_retries
        self.max_lower_retries     = max_lower_retries
        self.hurry_after_s         = hurry_after_s
        self.hurry_scale           = hurry_scale
        self.container_retract_xy_offset = container_retract_xy_offset
        self.skip_then_revisit = skip_then_revisit
        self.pre_grasp_yaw_tol = pre_grasp_yaw_tol
        self.freeze_yaw_before_descend = freeze_yaw_before_descend
        self.grasp_yaw_flip_enabled = grasp_yaw_flip_enabled
        self.grasp_yaw_flip_after_retries = grasp_yaw_flip_after_retries
        self.grasp_yaw_flip_rad = grasp_yaw_flip_rad
        self.grasp_miss_retreat = grasp_miss_retreat
        self.descend_xy_gate_z_margin = descend_xy_gate_z_margin
        self.keep_grasp_yaw_container = keep_grasp_yaw_container
        self.release_at_container_center = release_at_container_center
        self.safe_release_above_rim = safe_release_above_rim
        self.table_z = table_z
        self.pre_grasp_pos_relax_thresh = pre_grasp_pos_relax_thresh
        self.pre_grasp_relaxed_ang_bonus = pre_grasp_relaxed_ang_bonus
        self.pre_grasp_relaxed_yaw_bonus = pre_grasp_relaxed_yaw_bonus

        # Tunable overrides (env-var, for fast measured sweeps). 0.0 = no change.
        import os as _os
        self.grasp_z_bias = float(_os.environ.get("NEPHER_GRASP_Z_BIAS", "0.015"))
        # Extra grasp-depth bias per task_idx (compensates worsening under-descent
        # on later grasps). z_grasp -= task_idx * grasp_bias_ramp. Default off.
        self.grasp_bias_ramp = float(_os.environ.get("NEPHER_GRASP_BIAS_RAMP", "0.0"))
        # RETRY DEPTH BOOST: when the descend stalls on a hard low target and the
        # config-break fires (reach_retries>0), a gentle/precise LL under-reaches
        # (stops ~6cm short). Command DEEPER by reach_retries*boost on each retry
        # to force it down — recovers the hard-reach scenarios a gentle policy
        # can't reach alone, without over-deepening normal grasps (boost=0 when
        # not stalled). Default off.
        # BAKED 0.012: see retry_jitter note — the jitter(0.015)+depth(0.012) PAIR is
        # the only measured gain over baseline (seed-42 26 vs 23/90). Depth alone was
        # neutral (22/90), jitter alone slightly worse (21/90); together they synergize.
        self.retry_depth_boost = float(_os.environ.get("NEPHER_RETRY_DEPTH", "0.006"))
        self.enable_grasp_z_clamp = _os.environ.get("NEPHER_ENABLE_GRASP_Z_CLAMP", "1") != "0"
        self.min_grasp_z = float(_os.environ.get("NEPHER_MIN_GRASP_Z", "0.060"))
        # Per-stage motion cap: slower (gentler) descent/placement reduces tipping
        # of tall objects and bin-shoving on contact. Defaults to global max_step.
        self.descend_max_step = float(_os.environ.get("NEPHER_DESCEND_STEP", str(max_step)))
        self.place_max_step = float(_os.environ.get("NEPHER_PLACE_STEP", str(max_step)))
        # Time-budget tuning: the 35s episode cap forces 5 objects ~7s each.
        # pre_grasp settle is the biggest sink (up to 5s/object). Tunable.
        self.pre_grasp_settle_s = float(_os.environ.get("NEPHER_PRE_SETTLE", str(self.pre_grasp_settle_s)))
        self.grasp_hold_s = float(_os.environ.get("NEPHER_GRASP_HOLD", str(self.grasp_hold_s)))
        self.release_hold_s = float(_os.environ.get("NEPHER_RELEASE_HOLD", str(self.release_hold_s)))
        self.post_release_wait_s = float(_os.environ.get("NEPHER_POST_RELEASE_WAIT", "0.15"))
        self.release_z_bias = float(_os.environ.get("NEPHER_RELEASE_Z_BIAS", "0.020"))
        # Pre-grasp hover height: raise it so open fingers clear tall objects'
        # tops during XY centering (else they dangle alongside & tip the bottle).
        self.pre_approach_z = float(_os.environ.get("NEPHER_PRE_APPROACH_Z", "0.18"))
        # MUSTARD-only extra hover (added to pre_approach_z for the tall bottle only):
        # straighter descent down the oval channel -> fewer descent-clips, WITHOUT the
        # global travel-time cost of raising every object's hover. Default 0 = off.
        self.must_hover = float(_os.environ.get("NEPHER_MUST_HOVER", "0.12"))
        # Drop objects from above the bin rim (don't enter the bin & shove it).
        self.drop_z_bonus = float(_os.environ.get("NEPHER_DROP_Z_BONUS", "0.04"))
        # Per-task_idx LATE drop: by the 4th/5th object the bin holds a growing
        # pile; dropping the last objects from a bit higher clears the pile so they
        # don't land on it and shove the bin. A GLOBAL higher drop hurts (early
        # objects bounce) — this only raises the last `late_count` placements.
        self.late_drop_bonus = float(_os.environ.get("NEPHER_LATE_DROP", "0.0"))
        self.late_drop_count = int(float(_os.environ.get("NEPHER_LATE_DROP_COUNT", "2")))
        # Placement spread: the bin is small (~0.20x0.28m interior). When all 5
        # objects target the bin CENTRE they pile up; the 3rd+ lands on the pile,
        # bounces against a wall, and shoves the bin past the 2cm limit
        # (CONTAINER_DISPLACED = 60% of failures). Spread the drop points across
        # the bin floor (table frame, indexed by pick slot) so objects don't pile.
        # x = robot-radial (interior half ~0.10), y = lateral (interior half ~0.14).
        self.place_spread = float(_os.environ.get("NEPHER_PLACE_SPREAD", "0.0"))
        # Centre-first quincunx, all well inside the rim (table half-extents
        # x~0.10, y~0.14; object radius ~0.04). Each new object targets an empty
        # spot around the centre one, so placements don't land on / knock the pile.
        # Row along the bin's long (y) axis: max separation, no piling. Scaled
        # by NEPHER_PLACE_SPREAD (0.75 pulls objects ~25% off the rim so they
        # don't tip out). This row pattern is the only one that killed bin shove.
        # Clean sequential row along the bin LONG (y) axis only (x=0): each object
        # placed 0.04m from the previous into empty space -> no piling (no shove)
        # and max |y|=0.08 < 0.14 rim (no tip-out). The one geometry that may be
        # both non-piling AND off-the-rim.
        _spread_xy = [
            (0.000, -0.080),   # slot 0
            (0.000, -0.040),   # slot 1
            (0.000,  0.000),   # slot 2
            (0.000,  0.040),   # slot 3
            (0.000,  0.080),   # slot 4
        ]
        self._place_spread_xy = torch.tensor(_spread_xy, dtype=torch.float32, device=device)
        # Retract lift-out fix: the #1 bin-shove cause is the gripper retracting
        # DIAGONALLY (shifting toward the robot while still low) right after
        # placing -> it drags the rim. Lift STRAIGHT UP over the drop point until
        # the fingertips clear the rim by this margin (world Z above goal/bin z),
        # THEN shift XY toward the robot. Set 0 to restore the old diagonal retract.
        # DEFAULT OFF (0.0): at clear=0.19 this halved RETRACT-stage shoves (20->10,
        # total DISP 54->38) BUT the straight-up retract left the gripper sweeping
        # the tall mustard during the next pick's approach (CUBE_FELL 25->40, mostly
        # catalog_1 mustard) -> net 11/90 < baseline 13/90. Failures are conserved;
        # baseline diagonal retract kept. Set NEPHER_RETRACT_CLEAR=0.19 to re-enable.
        self.retract_clear_z = float(_os.environ.get("NEPHER_RETRACT_CLEAR", "0.0"))
        # SAFE TRANSIT CEILING (user strategy: up -> over -> down for EVERY lateral
        # move). The arm was cutting corners diagonally: carry_z=0.22 holds the
        # carried object at ~0.08-0.11 (bin-rim height) and the empty transit dips
        # to ~0.22 (mustard-top height) -> drags the bin / knocks the mustard. This
        # forces the EE up to transit_ceiling_z before any lateral motion in the
        # transit/approach stages, descending only once XY-aligned with the target.
        # Fingertip = ee_z - H(0.107); carried obj hangs ~0.06 below that. 0.40 ->
        # fingertip 0.29 and carried-obj-bottom ~0.23, clearing the mustard (~0.23)
        # and bin rim (0.11). Set 0 to disable.
        # Bottomed-out DESCEND advance: number of consecutive steps the EE z must
        # stop dropping before the planner treats the grasp as bottomed-out and
        # advances DESCEND->GRASP (breaks the unreachable-target timeout-stall loop;
        # see _end_pose/step). ~15 steps = 0.5s at 30Hz. 0 disables.
        # TESTED: breaking the DESCEND stall (30->3) did NOT fix the timeouts (the
        # stuck objects are ungraspable in their pose - gripper bottoms out but can't
        # close on them) and premature grasps knocked other objects (6/90 < 17/90).
        # The stall is a SYMPTOM of an ungraspable object, not the cause. Default OFF.
        self.descend_bottom_steps = int(float(_os.environ.get("NEPHER_DESCEND_BOTTOM", "0")))
        self.transit_ceiling_z = float(_os.environ.get("NEPHER_TRANSIT_CEILING", "0.40"))
        self.transit_align_tol = float(_os.environ.get("NEPHER_TRANSIT_ALIGN", "0.05"))
        # Tighter alignment specifically for the narrow mustard bottle (see the
        # transit-ceiling block). 0.03 vs the global 0.05: tight enough to center
        # the gripper on the 3.5cm bottle, loose enough to stay achievable by the
        # LL (~2cm XY residual) so the mustard still descends (no stall).
        # NOTE: tightening this (0.03) made the mustard WORSE (CUBE_FELL 15->20):
        # the arm does XY corrections near the bottle and clips it. Default = global
        # (no-op); kept env-gated for the record.
        self.mustard_align_tol = float(_os.environ.get("NEPHER_MUSTARD_ALIGN", "0.05"))
        # Symmetry-aware angular error on grasp-approach stages — TESTED, REJECTED:
        # the PRE_GRASP stalls are genuine kinematic unreachability (high pos_err
        # too), not symmetry-flip artifacts; relaxing the check let the gripper
        # descend mid-rotation -> 7/90 (worse). Default OFF.
        self.sym_ang_err = _os.environ.get("NEPHER_SYM_ANGERR", "0") != "0"
        # Transit heights: keep the arm/elbow high during LIFT/CARRY and retract
        # so it doesn't sweep through tall objects (like the pre-grasp hover fix).
        self.carry_z = float(_os.environ.get("NEPHER_CARRY_Z", str(self.carry_z)))
        self.retract_approach_z = float(_os.environ.get("NEPHER_RETRACT_Z", str(self.retract_approach_z)))
        # Per-stage step table (indexed by Stage). LIFT/CARRY HOLD the object ->
        # keep gentle (transport_max_step) to avoid dropping/swinging it (esp. the
        # mustard, which slips mid-carry). PRE_GRASP (empty approach) and RETRACT
        # (empty, AFTER release) carry NOTHING -> move FAST to save time for free;
        # transit_ceiling forces them up-and-over first so fast RETRACT clears the bin.
        self.transport_max_step = float(_os.environ.get("NEPHER_TRANSPORT_STEP", str(max_step)))
        self.empty_transit_step = float(_os.environ.get("NEPHER_EMPTY_STEP", str(max_step)))
        _steps = [max_step] * len(Stage)
        _steps[int(Stage.PRE_GRASP)] = self.empty_transit_step
        _steps[int(Stage.DESCEND)] = self.descend_max_step
        _steps[int(Stage.LOWER)]   = self.place_max_step
        _steps[int(Stage.RELEASE)] = self.place_max_step
        _steps[int(Stage.LIFT)]    = self.transport_max_step
        _steps[int(Stage.CARRY)]   = self.transport_max_step
        _steps[int(Stage.RETRACT)] = self.empty_transit_step
        self._step_table = torch.tensor(_steps, dtype=torch.float32, device=device)

        N, dev = num_envs, device
        self._stage       = torch.full((N,), int(Stage.PRE_GRASP), dtype=torch.long, device=dev)
        self._elapsed     = torch.zeros(N, device=dev)
        self._yaw         = torch.zeros(N, device=dev)
        self._retry_count = torch.zeros(N, dtype=torch.long, device=dev)
        # Bottomed-out detection for DESCEND (timeout-stall fix): track the lowest
        # EE z reached this descent and how many steps since it last dropped.
        self._descend_min_z   = torch.full((N,), 1e3, device=dev)
        self._descend_stall_n = torch.zeros(N, dtype=torch.long, device=dev)
        # VIA-HOME config reset: after placing, the arm sits in a constrained config
        # and the next grasp under-descends; the FIRST grasp from the home config
        # descends fully. Capture the home EE pose at reset and route the arm back
        # through it between picks so every grasp starts from the home config.
        self._home_ee_pos = torch.zeros(N, 3, device=dev)
        self._home_ee_z   = torch.full((N,), 0.45, device=dev)
        self._via_home    = torch.zeros(N, dtype=torch.bool, device=dev)
        import os as _os3
        self.via_home = _os3.environ.get("NEPHER_VIA_HOME", "0") != "0"
        # Per-retry approach jitter: a grasp that under-descends will repeat the
        # IDENTICAL failing descent forever (config-limited) -> timeout. Offset the
        # grasp XY by a small cycling amount on each retry so the arm approaches from
        # a different config and can occasionally reach full depth (converts grasp-
        # stall timeouts, the entire gap to the top agent). 0 disables.
        # BAKED 0.015: confirmed only-WITH retry_depth (0.012); the pair is the sole
        # measured improvement over baseline — seed-42 (validator seed) reproducibly
        # 26 vs 23/90, +5/270 aggregate across seeds 42/43/44. Neither knob helps alone.
        self.retry_jitter = float(_os3.environ.get("NEPHER_RETRY_JITTER", "0.015"))
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
        # Retract XY offset: env-var override (e.g. "0,0" for a straight-up
        # retract that lifts out of the bin without dragging across the rim).
        _rxy = _os.environ.get("NEPHER_RETRACT_XY")
        if _rxy:
            container_retract_xy_offset = tuple(float(x) for x in _rxy.split(","))
        self._retract_xy_off = torch.tensor(
            container_retract_xy_offset, dtype=torch.float32, device=dev
        )
        self._target_pos  = torch.zeros(N, 3, device=dev)
        self._target_quat = torch.zeros(N, 4, device=dev)
        self._target_quat[:, 0] = 1.0
        self._task_idx = torch.zeros(N, dtype=torch.long, device=dev)

        # Per-object progress for skip-then-revisit scheduling.
        self._object_placed = torch.zeros(N, num_objects, dtype=torch.bool, device=dev)
        self._object_deferred = torch.zeros(N, num_objects, dtype=torch.bool, device=dev)
        self._object_abandoned = torch.zeros(N, num_objects, dtype=torch.bool, device=dev)
        self._planning_exhausted = torch.zeros(N, dtype=torch.bool, device=dev)
        self._skip_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self._grasp_yaw_flip = torch.zeros(N, device=dev)

        # Per-object grasp metadata cached tensors (set each step from command).
        self._cur_grasp_z_off  = torch.full((N,), grasp_z_offset, device=dev)
        self._cur_grasp_sym    = torch.full((N,), math.pi / 2, device=dev)
        self._cur_grasp_yaw_off = torch.zeros(N, device=dev)
        self._cur_grasp_offset_local = torch.zeros(N, 3, device=dev)
        self._cur_catalog_idx = torch.full((N,), -1, dtype=torch.long, device=dev)
        self._cur_max_grasp_retries = torch.full((N,), max_retries, dtype=torch.long, device=dev)
        self._retry_pattern_local = torch.zeros(N, 3, device=dev)
        self._retry_pattern_dyaw = torch.zeros(N, device=dev)
        self._retry_pattern_idx = torch.zeros(N, dtype=torch.long, device=dev)
        self._catalog_extra_depth = torch.zeros(N, device=dev)
        self._grasp_start_cube_z = torch.zeros(N, device=dev)
        self._required_lift_z = torch.zeros(N, device=dev)
        self._target_grasp_z = torch.zeros(N, device=dev)
        self._grasp_miss_is_carry = torch.zeros(N, dtype=torch.bool, device=dev)
        self._stuck_elapsed = torch.zeros(N, device=dev)
        self._retract_phase_xy_clear = torch.zeros(N, dtype=torch.bool, device=dev)
        self._retract_release_xy = torch.zeros(N, 2, device=dev)
        self._retract_clear_z = torch.zeros(N, device=dev)

        _stage_grip = list(_STAGE_GRIP)
        # Optionally CLOSE the gripper during RETRACT (default open): after RELEASE
        # the object is already dropped, so closing the fingers shrinks the EE
        # profile (~8cm open -> ~2cm closed) so it doesn't catch the bin rim / a
        # placed object while lifting out (targets the RETRACT bin-shoves).
        if float(_os.environ.get("NEPHER_RETRACT_GRIP", "0.0")) > 0.5:
            _stage_grip[int(Stage.RETRACT)] = 1.0
        self._grip_table  = torch.tensor(_stage_grip, device=dev)
        self._next_stage  = torch.arange(1, len(Stage) + 1, dtype=torch.long, device=dev).clamp_(max=int(Stage.DONE))

        # ---- MUSTARD TIP-OVER (env-gated NEPHER_MUSTARD_TIP, default OFF) ----
        # Tall UPRIGHT objects (mustard, ~13cm, centre z~0.065) tip during top-down
        # grasp = our #1 failure (~55%). Knock the bottle onto its SIDE first -> a
        # low, stable 3.5cm cylinder that grasps cleanly. Per-env tip phase:
        #   0 idle/not-needed, 1 approach beside (gripper closed), 2 push-through
        #   (topple it in the push dir), 3 done -> normal grasp picks up the lying
        #   cylinder (grasp_z lowered to tip_lying_z, yaw across the lying axis).
        # tip_height 0.055 selects ONLY the mustard (others rest at z<=0.05).
        self.mustard_tip   = _os.environ.get("NEPHER_MUSTARD_TIP", "0") != "0"
        self.tip_height    = float(_os.environ.get("NEPHER_TIP_HEIGHT", "0.055"))
        self.tip_push_z    = float(_os.environ.get("NEPHER_TIP_PUSH_Z", "0.085"))
        self.tip_standoff  = float(_os.environ.get("NEPHER_TIP_STANDOFF", "0.085"))
        self.tip_overshoot = float(_os.environ.get("NEPHER_TIP_OVERSHOOT", "0.07"))
        self.tip_lying_z   = float(_os.environ.get("NEPHER_TIP_LYING_Z", "0.035"))
        self.tip_yaw_off   = float(_os.environ.get("NEPHER_TIP_YAW_OFF", str(math.pi / 2)))
        # NECK-GRASP: after toppling, the cap/neck points in +tip_dir (the bottle fell
        # that way). Offset the grasp toward the NARROW neck (~1.5cm) for a form-closure
        # grip instead of pinching the smooth 3.5cm mid-body (which slips mid-carry).
        # 0.0 = grab centre (old behaviour). Stays within grasp_secure_xy_tol (0.06).
        self.tip_neck_off  = float(_os.environ.get("NEPHER_TIP_NECK_OFF", "0.0"))
        self._tip_phase    = torch.zeros(N, dtype=torch.long, device=dev)
        self._tip_dir      = torch.zeros(N, 2, device=dev)

        pos_tol_stages = [
            pos_tol, pos_tol_approach, pos_tol_grasp,
            pos_tol_transport, pos_tol_transport, pos_tol_place,
            pos_tol_grasp, pos_tol_retract, pos_tol,
        ]
        ang_tol_stages = [
            ang_tol, ang_tol_approach, ang_tol_grasp,
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
        self._endpoint_err = torch.zeros(N, device=dev)
        self._cmd_err = torch.zeros(N, device=dev)
        self._track_ok   = torch.zeros(N, dtype=torch.bool, device=dev)
        self._grasp_miss      = torch.zeros(N, dtype=torch.bool, device=dev)
        self._place_miss      = torch.zeros(N, dtype=torch.bool, device=dev)
        self._stage_changed  = torch.zeros(N, dtype=torch.bool, device=dev)
        self._pos_tol_eff    = torch.full((N,), pos_tol, device=dev)
        self._ang_tol_eff    = torch.full((N,), ang_tol, device=dev)

    @property
    def stage(self) -> torch.Tensor:
        return self._stage

    def is_fully_done(self) -> torch.Tensor:
        """Return bool tensor: True for envs where all objects have been placed."""
        if self.skip_then_revisit and self.num_objects > 1:
            return self._object_placed.all(dim=1)
        return (self._stage == int(Stage.DONE)) & (self._task_idx >= self.num_objects - 1)

    def _resolve_grasp_yaw(
        self,
        obj_quat: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Object-aligned grasp yaw plus optional flip offset."""
        if env_ids is None:
            sym = self._cur_grasp_sym
            yaw_off = self._cur_grasp_yaw_off
            flip = self._grasp_yaw_flip
        else:
            sym = self._cur_grasp_sym[env_ids]
            yaw_off = self._cur_grasp_yaw_off[env_ids]
            flip = self._grasp_yaw_flip[env_ids]
        return _wrap_to_pi(_grasp_yaw(obj_quat, sym, yaw_off) + flip)

    def _reset_stage_state(
        self,
        env_mask: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> None:
        """Reset stage counters when switching to a new object."""
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
        self._tip_phase[env_mask] = 0
        self._tip_dir[env_mask] = 0.0
        self._descend_min_z[env_mask] = 1e3
        self._descend_stall_n[env_mask] = 0
        self._grasp_yaw_flip[env_mask] = 0.0
        self._stuck_elapsed[env_mask] = 0.0

        env_ids = torch.where(env_mask)[0]
        self._yaw[env_ids] = self._resolve_grasp_yaw(cube_quat_w[env_ids], env_ids)

    def _advance_to_next_object(
        self,
        env_mask: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> None:
        """Pick the next pending object: fresh slots first, then deferred revisits."""
        if not env_mask.any() or not self.skip_then_revisit:
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
        if self.via_home:
            self._via_home = torch.where(moved, torch.ones_like(self._via_home), self._via_home)

    def _defer_current_object(
        self,
        env_mask: torch.Tensor,
        cube_quat_w: torch.Tensor,
    ) -> None:
        """Skip the current object; revisit later or abandon on second failure."""
        if not env_mask.any() or not self.skip_then_revisit:
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

    def _apply_yaw_flip_on_grasp_miss(self, env_mask: torch.Tensor) -> None:
        """Alternate ±90° wrist yaw on elongated objects after grasp misses."""
        if not self.grasp_yaw_flip_enabled or not env_mask.any():
            return
        elongated = env_mask & (self._cur_grasp_sym > 0.4)
        should_flip = elongated & (self._retry_count >= self.grasp_yaw_flip_after_retries)
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

    def _update_hard_grasp_retry_state(self) -> None:
        """Build per-env retry offsets for the hard catalog grasp schedule."""
        cat_idx = self._cur_catalog_idx
        hard_mask = torch.zeros_like(cat_idx, dtype=torch.bool)
        for hard_idx in HARD_GRASP_CATALOGS:
            hard_mask |= cat_idx == hard_idx

        self._cur_max_grasp_retries = torch.where(
            hard_mask,
            torch.full_like(self._cur_max_grasp_retries, 4),
            torch.full_like(self._cur_max_grasp_retries, self.max_retries),
        )

        pattern_idx = torch.clamp(self._retry_count, max=len(_HARD_RETRY_PATTERNS) - 1)
        self._retry_pattern_idx.copy_(torch.where(hard_mask, pattern_idx, torch.zeros_like(pattern_idx)))

        retry_local = torch.zeros_like(self._retry_pattern_local)
        retry_dyaw = torch.zeros_like(self._retry_pattern_dyaw)
        extra_depth = torch.zeros_like(self._catalog_extra_depth)

        for idx, (dx, dy, dz, dyaw) in enumerate(_HARD_RETRY_PATTERNS):
            sel = hard_mask & (pattern_idx == idx)
            if sel.any():
                retry_local[sel, 0] = dx
                retry_local[sel, 1] = dy
                retry_local[sel, 2] = dz
                retry_dyaw[sel] = dyaw

        for hard_idx, depth in _HARD_CATALOG_EXTRA_DEPTH.items():
            sel = cat_idx == hard_idx
            if sel.any():
                extra_depth[sel] = depth

        self._retry_pattern_local.copy_(retry_local)
        self._retry_pattern_dyaw.copy_(retry_dyaw)
        self._catalog_extra_depth.copy_(extra_depth)


    def reset(
        self,
        env_ids:     torch.Tensor,
        ee_pos_w:    torch.Tensor | None = None,
        ee_quat_w:   torch.Tensor | None = None,
        cube_quat_w: torch.Tensor | None = None,
        grasp_sym:   torch.Tensor | None = None,
        grasp_yaw_off: torch.Tensor | None = None,
    ) -> None:
        """Reset selected envs to PRE_GRASP, task_idx = 0."""
        if env_ids.numel() == 0:
            return
        ids = env_ids
        sym = grasp_sym[ids] if grasp_sym is not None else self._cur_grasp_sym[ids]
        yaw_off = grasp_yaw_off[ids] if grasp_yaw_off is not None else self._cur_grasp_yaw_off[ids]

        # Capture the home EE pose (arm at start config -> grasps descend fully).
        if ee_pos_w is not None:
            self._home_ee_pos[ids] = ee_pos_w[ids]
        self._via_home[ids] = False

        self._stage[ids]          = int(Stage.PRE_GRASP)
        self._elapsed[ids]        = 0.0
        self._retry_count[ids]    = 0
        self._task_idx[ids]       = 0
        self._tip_phase[ids]      = 0
        self._tip_dir[ids]        = 0.0
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
        self._episode_t[ids]      = 0.0
        self._object_placed[ids] = False
        self._object_deferred[ids] = False
        self._object_abandoned[ids] = False
        self._planning_exhausted[ids] = False
        self._skip_event[ids] = False
        self._grasp_yaw_flip[ids] = 0.0
        self._descend_min_z[ids] = 1e3
        self._descend_stall_n[ids] = 0
        self._stuck_elapsed[ids] = 0.0
        self._grasp_start_cube_z[ids] = 0.0

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
        grasp_offset_local: torch.Tensor | None = None,  # (N, 3)
        catalog_idx: torch.Tensor | None = None,  # (N,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One planner step.  Returns ``(end_pos_w, end_quat_w, grip)``."""

        # Update current object grasp metadata.
        if grasp_z_off is not None:
            self._cur_grasp_z_off.copy_(grasp_z_off)
        if grasp_sym is not None:
            self._cur_grasp_sym.copy_(grasp_sym)
        if grasp_yaw_off is not None:
            self._cur_grasp_yaw_off.copy_(grasp_yaw_off)
        if grasp_offset_local is not None:
            self._cur_grasp_offset_local.copy_(grasp_offset_local)
        if catalog_idx is not None:
            self._cur_catalog_idx.copy_(catalog_idx)
        else:
            self._cur_catalog_idx.fill_(-1)
        self._update_hard_grasp_retry_state()

        # Yaw: settle in PRE_GRASP only; freeze before DESCEND so the wrist does
        # not sweep the object while descending.
        in_pre_grasp_yaw = self._stage == int(Stage.PRE_GRASP)
        if in_pre_grasp_yaw.any():
            new_yaw = self._resolve_grasp_yaw(cube_quat_w)
            self._yaw = torch.where(in_pre_grasp_yaw, new_yaw, self._yaw)
        elif not self.freeze_yaw_before_descend:
            approaching = self._stage <= int(Stage.DESCEND)
            if approaching.any():
                new_yaw = self._resolve_grasp_yaw(cube_quat_w)
                self._yaw = torch.where(approaching, new_yaw, self._yaw)

        # Wrist-unwind: only during PRE_GRASP when yaw is being actively settled.
        if wrist_angle is not None:
            self._yaw_switch_cd = (self._yaw_switch_cd - 1).clamp_(min=0)
            if self.freeze_yaw_before_descend:
                unwind_ok = in_pre_grasp_yaw
            else:
                unwind_ok = self._stage <= int(Stage.DESCEND)
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

        self._cur_ee_pos = ee_pos_w  # stash for retract-clearance gating in _end_pose
        # VIA-HOME: clear the flag once the EE has returned near the home pose, so the
        # next step heads to the object hover (descending from the home config).
        if self.via_home and self._via_home.any():
            home_tgt = torch.stack([self._home_ee_pos[:, 0], self._home_ee_pos[:, 1], self._home_ee_z], dim=-1)
            reached = torch.norm(ee_pos_w - home_tgt, dim=-1) < 0.07
            self._via_home = self._via_home & ~reached

        # ---- MUSTARD TIP-OVER phase machine ----
        if self.mustard_tip:
            cz_now = cube_pos_w[:, 2]
            in_pg  = self._stage == int(Stage.PRE_GRASP)
            tall   = cz_now > self.tip_height
            ee_xy  = ee_pos_w[:, :2]
            # ACTIVATE: tall upright object at PRE_GRASP, not yet tipped, not mid
            # config-break. Push radially OUTWARD from the robot base (toward open
            # table space, away from the cluster) -> bottle topples forward.
            start = in_pg & tall & (self._tip_phase == 0) & (self._reach_retries == 0)
            if start.any():
                r = torch.norm(cube_pos_w[:, :2], dim=-1, keepdim=True) + 1e-6
                d = cube_pos_w[:, :2] / r
                self._tip_dir = torch.where(start.unsqueeze(-1), d, self._tip_dir)
                self._tip_phase = torch.where(start, torch.ones_like(self._tip_phase), self._tip_phase)
            # 1 -> 2: EE reached the near-side pre-push pose (high, beside bottle).
            pre_xy = cube_pos_w[:, :2] - self._tip_dir * self.tip_standoff
            to_2 = (self._tip_phase == 1) & (torch.norm(ee_xy - pre_xy, dim=-1) < 0.035)
            self._tip_phase = torch.where(to_2, torch.full_like(self._tip_phase, 2), self._tip_phase)
            # 2 -> 3: pushed past centre OR object already toppled (cz dropped low).
            push_xy = cube_pos_w[:, :2] + self._tip_dir * self.tip_overshoot
            to_3 = (self._tip_phase == 2) & (
                (torch.norm(ee_xy - push_xy, dim=-1) < 0.04) | (cz_now < self.tip_height))
            self._tip_phase = torch.where(to_3, torch.full_like(self._tip_phase, 3), self._tip_phase)
            import os as _ostip
            if _ostip.environ.get("NEPHER_DIAG") and (start.any() or to_3.any()):
                self._tip_act = getattr(self, "_tip_act", 0) + int(start.sum())
                self._tip_top = getattr(self, "_tip_top", 0) + int(to_3.sum())
                # min cz of currently-tipping envs (did it actually go down to lying ~0.018?)
                tipping_now = self._tip_phase >= 1
                minz = float(cz_now[tipping_now].min()) if tipping_now.any() else -1.0
                print(f"[TIP] activated_tot={self._tip_act} toppled_tot={self._tip_top} "
                      f"now_p1={int((self._tip_phase==1).sum())} p2={int((self._tip_phase==2).sum())} "
                      f"p3={int((self._tip_phase==3).sum())} min_tip_cz={minz:.3f}", flush=True)

        end_pos, end_quat = self._end_pose(cube_pos_w, cube_quat_w, goal_pos_w, goal_quat_w)
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
        yaw_err = _symmetry_aware_yaw_error(ee_yaw, self._yaw, self._cur_grasp_sym)
        # BUG FIX: ang_err ignored grasp symmetry, so a wrist sitting at an
        # EQUIVALENT rotated grasp (180° for boxes sym=pi, 90° for the cube
        # sym=pi/2) reported ang_err~pi and never satisfied the tight PRE_GRASP
        # tolerance -> 3s stall, then a stage-escape that descended COCKED and
        # could knock the object. Fold the target by the grasp symmetry about the
        # vertical axis on the grasp-approach stages and take the min error.
        if self.sym_ang_err:
            approach = self._stage <= int(Stage.GRASP)
            if approach.any():
                sym = self._cur_grasp_sym  # (N,)
                ang_sym = ang_err.clone()
                for k in (1, -1, 2, -2):
                    half = 0.5 * k * sym
                    qz = torch.zeros_like(end_quat)
                    qz[:, 0] = torch.cos(half)
                    qz[:, 3] = torch.sin(half)
                    tgt_k = quat_mul(qz, end_quat)
                    ang_sym = torch.minimum(ang_sym, quat_error_magnitude(ee_quat_w, tgt_k))
                ang_err = torch.where(approach, ang_sym, ang_err)
        pos_tol_eff = self._pos_tol_table[self._stage]
        ang_tol_eff = self._ang_tol_table[self._stage]
        in_pre_grasp = self._stage == int(Stage.PRE_GRASP)
        pre_grasp_base_tol = torch.full_like(ang_tol_eff, 0.70)
        pre_grasp_close_tol = torch.full_like(ang_tol_eff, 0.80)
        ang_tol_eff = torch.where(in_pre_grasp, torch.maximum(ang_tol_eff, pre_grasp_base_tol), ang_tol_eff)
        ang_tol_eff = torch.where(in_pre_grasp & (pos_err < 0.025), pre_grasp_close_tol, ang_tol_eff)
        track_ok = (pos_err < pos_tol_eff) & (ang_err < ang_tol_eff)
        self._pos_tol_eff.copy_(pos_tol_eff)
        self._ang_tol_eff.copy_(ang_tol_eff)
        self._pos_err.copy_(pos_err)
        self._endpoint_err.copy_(pos_err)
        self._ang_err.copy_(ang_err)
        self._track_ok.copy_(track_ok)
        self._grasp_miss.fill_(False)
        self._grasp_miss_is_carry.fill_(False)
        self._place_miss.fill_(False)
        self._skip_event.fill_(False)

        arange = torch.arange(self.num_envs, device=self.device)
        cur_obj = self._task_idx
        object_active = ~(self._object_placed[arange, cur_obj] | self._object_abandoned[arange, cur_obj])

        in_grasp   = self._stage == int(Stage.GRASP)
        in_release = self._stage == int(Stage.RELEASE)
        in_lower   = self._stage == int(Stage.LOWER)
        in_done    = self._stage == int(Stage.DONE)

        cube_to_ee_xy = torch.norm(cube_pos_w[:, :2] - ee_pos_w[:, :2], dim=-1)
        grasp_secure  = cube_to_ee_xy < self.grasp_secure_xy_tol

        at_end = (self._elapsed >= self.min_stage_dur) & track_ok

        # PRE_GRASP orientation-stall escape.
        in_pos       = pos_err < pos_tol_eff
        self._pre_settle = torch.where(
            in_pre_grasp & in_pos,
            self._pre_settle + dt,
            torch.zeros_like(self._pre_settle),
        )
        pre_settle_req = self.pre_grasp_settle_s * hurry_mul
        very_close = in_pre_grasp & (pos_err < self.pre_grasp_pos_relax_thresh)
        pre_grasp_ang_tol = torch.where(
            very_close,
            torch.full_like(ang_err, self.pre_grasp_settle_ang + self.pre_grasp_relaxed_ang_bonus),
            torch.full_like(ang_err, self.pre_grasp_settle_ang),
        )
        pre_grasp_yaw_tol = torch.where(
            very_close,
            torch.full_like(yaw_err, self.pre_grasp_yaw_tol + self.pre_grasp_relaxed_yaw_bonus),
            torch.full_like(yaw_err, self.pre_grasp_yaw_tol),
        )
        pre_grasp_settled = in_pre_grasp & (
            (
                (self._pre_settle >= pre_settle_req)
                & (ang_err < pre_grasp_ang_tol)
                & (yaw_err < pre_grasp_yaw_tol)
            )
            | (self._pre_settle >= 3.0 * pre_settle_req)
        )

        grasp_ok   = in_grasp   & (self._elapsed >= self.min_stage_dur + self.grasp_hold_s)   & track_ok & grasp_secure
        release_ok = in_release & (
            self._elapsed >= self.min_stage_dur + self.release_hold_s + self.post_release_wait_s
        ) & track_ok

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
        can_advance = torch.where(in_grasp,   grasp_ok,   can_advance)
        can_advance = torch.where(in_release, release_ok, can_advance)
        can_advance = torch.where(in_lower,   lower_ok,   can_advance)
        can_advance = can_advance | pre_grasp_settled

        # BOTTOMED-OUT DESCEND advance (validator-log timeout-stall fix): for some
        # objects the grasp target z is commanded ~6-7cm below the object centre
        # (per_z + bias) = below the table = UNREACHABLE; the gripper bottoms out on
        # contact ~6cm above target, pos_err never clears -> 3s stall -> escape ->
        # grasp at wrong height -> grasp_miss -> PRE_GRASP/DESCEND loop -> 1050-step
        # TIMEOUT (~23/90 validator fails). Detect that the EE has STOPPED descending
        # (z no longer dropping for descend_bottom_steps) and grasp there. This is
        # progress-based (robust to object height/position) and leaves normal grasps
        # untouched (they reach depth & advance via track_ok before bottoming counts).
        carrot_following = (self._cmd_err < 0.075) & (self._endpoint_err > pos_tol_eff)
        self._stuck_elapsed = torch.where(
            track_ok | carrot_following,
            torch.zeros_like(self._stuck_elapsed),
            self._stuck_elapsed + dt,
        )

        in_descend = self._stage == int(Stage.DESCEND)
        not_descend = ~in_descend
        self._descend_min_z = torch.where(not_descend, torch.full_like(self._descend_min_z, 1e3), self._descend_min_z)
        self._descend_stall_n = torch.where(not_descend, torch.zeros_like(self._descend_stall_n), self._descend_stall_n)
        ee_z = ee_pos_w[:, 2]
        improving = in_descend & (ee_z < self._descend_min_z - 0.003)
        self._descend_min_z = torch.where(improving, ee_z, self._descend_min_z)
        self._descend_stall_n = torch.where(
            improving, torch.zeros_like(self._descend_stall_n),
            torch.where(in_descend, self._descend_stall_n + 1, self._descend_stall_n),
        )
        if self.descend_bottom_steps > 0:
            bottomed = (
                in_descend
                & (self._descend_stall_n >= self.descend_bottom_steps)
                & (cube_to_ee_xy < self.grasp_secure_xy_tol)
            )
            can_advance = can_advance | bottomed

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
        import os as _osd
        if _osd.environ.get("NEPHER_DIAG"):
            in_desc = self._stage == int(Stage.DESCEND)
            longstuck = in_desc & (self._stuck_elapsed > self.stall_time_s)
            if longstuck.any():
                pe = pos_err[longstuck]
                _ls = longstuck
                print(f"[DESCEND-STALL] n={int(_ls.sum())} pos_err mean={pe.mean():.3f} | "
                      f"pos>tol={int((_ls&(pos_err>self.stall_pos_tol)).sum())} "
                      f"elapsed>time={int((_ls&(self._stuck_elapsed>self.stall_time_s)).sum())} "
                      f"retreat_ctr==0={int((_ls&(self._retreat_ctr==0)).sum())} "
                      f"reach<max={int((_ls&(self._reach_retries<self.max_reach_retries)).sum())} "
                      f"in_approach={int((_ls&(self._stage<=int(Stage.DESCEND))).sum())}", flush=True)
        stalled = (in_approach & (pos_err > self.stall_pos_tol)
                   & (self._stuck_elapsed > self.stall_time_s)
                   & (self._retreat_ctr == 0)
                   & (self._reach_retries < self.max_reach_retries))
        if stalled.any():
            import os as _osr
            if _osr.environ.get("NEPHER_DIAG"):
                print(f"[RETREAT-FIRE] n={int(stalled.sum())} reach_retries(max)={int(self._reach_retries.max())} "
                      f"xy_break={self.retreat_xy_break:.3f} ee_z={float(ee_pos_w[stalled][:,2].mean()):.3f}", flush=True)
            self._retreat_ctr   = torch.where(stalled, torch.full_like(self._retreat_ctr, self.retreat_steps), self._retreat_ctr)
            self._reach_retries = torch.where(stalled, self._reach_retries + 1, self._reach_retries)
            self._elapsed       = torch.where(stalled, torch.zeros_like(self._elapsed), self._elapsed)
            self._stuck_elapsed = torch.where(stalled, torch.zeros_like(self._stuck_elapsed), self._stuck_elapsed)
        retreating = self._retreat_ctr > 0
        self._retreat_ctr = (self._retreat_ctr - 1).clamp_(min=0)
        can_advance = can_advance & ~retreating

        reach_exhausted = (
            in_approach
            & (pos_err > self.stall_pos_tol)
            & (self._stuck_elapsed > self.stall_time_s)
            & (self._reach_retries >= self.max_reach_retries)
            & (self._retreat_ctr == 0)
            & object_active
        )
        if reach_exhausted.any() and self.skip_then_revisit:
            self._defer_current_object(reach_exhausted, cube_quat_w)
            object_active = ~(self._object_placed[arange, cur_obj] | self._object_abandoned[arange, cur_obj])

        entered_grasp = (old_stage != int(Stage.GRASP)) & (self._stage == int(Stage.GRASP))
        self._grasp_start_cube_z = torch.where(entered_grasp, cube_pos_w[:, 2], self._grasp_start_cube_z)

        # Grasp miss / carry drop recovery.
        in_carry     = self._stage == int(Stage.CARRY)
        drop_gap = (ee_pos_w[:, 2] - self.H) - cube_pos_w[:, 2]
        lift_verify_ready = in_carry & (self._elapsed >= self.min_stage_dur + 0.25)
        required_lift_z = torch.maximum(
            torch.full_like(cube_pos_w[:, 2], self.min_carry_cube_z),
            self._grasp_start_cube_z + 0.045,
        )
        # catalog_1 often rides low in the fingers even on successful carries.
        # Keep the actual grasp pose unchanged, but soften carry verification so
        # we do not reschedule a real pickup as a false miss.
        catalog1_mask = self._cur_catalog_idx == 1
        required_lift_z_eff = torch.where(
            catalog1_mask,
            torch.full_like(required_lift_z, 0.065),
            required_lift_z,
        )
        self._required_lift_z.copy_(required_lift_z_eff)
        gripper_closed = self._grip_table[self._stage] > 0.5
        carry_success = (
            (cube_pos_w[:, 2] > (self._grasp_start_cube_z + 0.045))
            | (cube_pos_w[:, 2] > self.min_carry_cube_z)
            | (catalog1_mask & gripper_closed & (cube_pos_w[:, 2] > 0.065))
        )
        carry_missed = (
            lift_verify_ready
            & ~carry_success
            & (drop_gap > self.carry_drop_gap)
        )
        grasp_failed = (
            in_carry
            & lift_verify_ready
            & ~grasp_secure
            & ~carry_success
            & (drop_gap > 0.0)
        )
        grasp_miss   = (grasp_failed | carry_missed) & (self._retry_count < self._cur_max_grasp_retries) & object_active
        if grasp_miss.any():
            self._grasp_miss.copy_(grasp_miss)
            self._grasp_miss_is_carry.copy_(carry_missed & grasp_miss)
            self._retry_count[grasp_miss] += 1
            self._apply_yaw_flip_on_grasp_miss(grasp_miss)
            if self.grasp_miss_retreat:
                self._retreat_ctr = torch.where(
                    grasp_miss,
                    torch.full_like(self._retreat_ctr, self.retreat_steps),
                    self._retreat_ctr,
                )
            self._stage[grasp_miss]       = int(Stage.PRE_GRASP)
            self._elapsed[grasp_miss]     = 0.0
            self._pre_settle[grasp_miss]  = 0.0
            miss_ids = torch.where(grasp_miss)[0]
            self._yaw[miss_ids] = self._resolve_grasp_yaw(cube_quat_w[miss_ids], miss_ids)

        grasp_still_failing = (grasp_failed | carry_missed) & object_active
        grasp_exhausted = grasp_still_failing & (self._retry_count >= self._cur_max_grasp_retries)
        if grasp_exhausted.any() and self.skip_then_revisit:
            self._grasp_miss.copy_(grasp_exhausted)
            self._defer_current_object(grasp_exhausted, cube_quat_w)
            object_active = ~(self._object_placed[arange, cur_obj] | self._object_abandoned[arange, cur_obj])

        if can_advance.any():
            adv = can_advance & ~grasp_miss & ~grasp_exhausted
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
                cur_sym_m = self._cur_grasp_sym[place_missed]
                cur_off_m = self._cur_grasp_yaw_off[place_missed]
                self._yaw[place_missed] = _grasp_yaw(cube_quat_w[place_missed], cur_sym_m, cur_off_m)
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
            if place_exhausted.any() and self.skip_then_revisit:
                self._defer_current_object(place_exhausted, cube_quat_w)

        # Multi-object: mark placed and advance pick queue.
        if newly_done.any():
            env_done = torch.where(newly_done)[0]
            self._object_placed[env_done, self._task_idx[env_done]] = True
            if self.skip_then_revisit and self.num_objects > 1:
                self._advance_to_next_object(newly_done, cube_quat_w)
            elif self.num_objects > 1:
                has_more = self._task_idx < (self.num_objects - 1)
                advance_task = newly_done & has_more
                if advance_task.any():
                    self._task_idx[advance_task] += 1
                    self._reset_stage_state(advance_task, cube_quat_w)
                    if self.via_home:
                        self._via_home[advance_task] = True

        self._stage_changed = old_stage != self._stage

        import os as _os
        # Grasp-lift-rate proxy: at each LIFT->CARRY transition, did the object
        # actually come up? (cube_z > 0.10 = good grasp, else failed grasp).
        if _os.environ.get("NEPHER_DIAG"):
            entered_carry = (old_stage == int(Stage.LIFT)) & (self._stage == int(Stage.CARRY))
            if entered_carry.any():
                lifted = entered_carry & (cube_pos_w[:, 2] > 0.10)
                self._diag_grasp_attempts = getattr(self, "_diag_grasp_attempts", 0) + int(entered_carry.sum().item())
                self._diag_grasp_lifts = getattr(self, "_diag_grasp_lifts", 0) + int(lifted.sum().item())
                a, l = self._diag_grasp_attempts, self._diag_grasp_lifts
                # Bucket lift success by object class (round sym~0 vs box sym~pi vs cube sym~pi/2).
                buckets = getattr(self, "_diag_sym_buckets", None)
                if buckets is None:
                    buckets = self._diag_sym_buckets = {}
                syms = self._cur_grasp_sym
                for idx in entered_carry.nonzero(as_tuple=True)[0].tolist():
                    s = float(syms[idx])
                    key = "round" if s < 0.4 else ("cube90" if s < 2.0 else "box180")
                    rec = buckets.setdefault(key, [0, 0])
                    rec[0] += 1
                    if bool(lifted[idx]):
                        rec[1] += 1
                bstr = " ".join(f"{k}:{v[1]}/{v[0]}" for k, v in sorted(buckets.items()))
                print(f"[GRASP] attempts={a} lifts={l} lift_rate={l/max(a,1):.3f} | by_class {bstr}", flush=True)
        # Progress proxy: count every object placement (newly_done) across all envs.
        if _os.environ.get("NEPHER_DIAG") and newly_done.any():
            n = int(newly_done.sum().item())
            tot = getattr(self, "_diag_place_total", 0) + n
            self._diag_place_total = tot
            print(f"[PLACE] +{n} objects placed (cumulative={tot})", flush=True)
        if _os.environ.get("NEPHER_DIAG_TRACE"):
            ntr = int(_os.environ.get("NEPHER_DIAG_TRACE"))
            chg = self._stage_changed
            for i in range(min(ntr, self.num_envs)):
                if chg[i]:
                    cz = cube_pos_w[i, 2].item()
                    ez = ee_pos_w[i, 2].item()
                    # EE-vs-cube offset (grasp targets cube XY); reveals systematic bias.
                    dx = (ee_pos_w[i, 0] - cube_pos_w[i, 0]).item()
                    dy = (ee_pos_w[i, 1] - cube_pos_w[i, 1]).item()
                    print(
                        f"[TRACE] env{i} t={self._episode_t[i].item():.1f} "
                        f"{STAGE_NAMES[int(old_stage[i].item())]}->{STAGE_NAMES[int(self._stage[i].item())]} "
                        f"obj={int(self._task_idx[i].item())} "
                        f"pos_err={pos_err[i].item():.3f} ang_err={ang_err[i].item():.3f} "
                        f"dx={dx:+.3f} dy={dy:+.3f} "
                        f"cube_z={cz:.3f} ee_z={ez:.3f} retry={int(self._retry_count[i].item())}",
                        flush=True,
                    )

        self._target_pos.copy_(end_pos)
        self._target_quat.copy_(end_quat)

        # Config-break retreat: command a high pose above current object, OFFSET in
        # XY (cycling per retry). The base retreat goes straight up & re-descends
        # straight down, which re-enters the same bent-elbow trap on the harder
        # multibase objects. Offsetting the retreat XY makes the RE-DESCENT diagonal
        # (lands the arm in a different config), which actually breaks the trap.
        _rk = self._reach_retries.clamp(min=1).to(cube_pos_w.dtype)
        _rang = _rk * 1.7
        _rmag = self.retreat_xy_break
        retreat_pos = torch.stack(
            [cube_pos_w[:, 0] + _rmag * torch.cos(_rang),
             cube_pos_w[:, 1] + _rmag * torch.sin(_rang),
             torch.full_like(cube_pos_w[:, 2], self.retreat_z + self.H)],
            dim=-1,
        )
        cmd_target = torch.where(retreating.unsqueeze(-1), retreat_pos, end_pos)

        # Smooth pursuit carrot (per-stage motion cap for gentler descent/place).
        delta = cmd_target - ee_pos_w
        dist  = torch.norm(delta, dim=-1, keepdim=True)
        eff_max_step = self._step_table[self._stage].unsqueeze(-1)
        scale = torch.clamp(eff_max_step / (dist + 1e-6), max=1.0)
        cmd_pos = ee_pos_w + delta * scale
        self._cmd_err.copy_(torch.norm(ee_pos_w - cmd_pos, dim=-1))
        grip = self._grip_table[self._stage]
        # MUSTARD TIP-OVER: close the gripper during the push (phases 1-2) so the
        # closed fingertip acts as a rigid pusher to topple the bottle.
        if self.mustard_tip:
            tipping = (self._tip_phase == 1) | (self._tip_phase == 2)
            grip = torch.where(tipping, torch.ones_like(grip), grip)
        return cmd_pos, end_quat, grip

    # ------------------------------------------------------------------

    def _end_pose(
        self,
        cube_pos_w: torch.Tensor,
        cube_quat_w: torch.Tensor,
        goal_pos_w: torch.Tensor,
        goal_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Target (position, quaternion) in world frame for the current stage."""
        H  = self.H
        cx, cy, cz = cube_pos_w[:, 0], cube_pos_w[:, 1], cube_pos_w[:, 2]
        gx, gy, gz = goal_pos_w[:, 0], goal_pos_w[:, 1], goal_pos_w[:, 2]

        # Per-retry approach jitter (config-cycling for grasp-stall recovery): shift
        # the grasp XY by a small cycling offset on each retry so a config-limited
        # under-descend isn't repeated identically.
        if self.retry_jitter > 0.0:
            k = self._retry_count.clamp(0, 4)
            active = (k > 0).to(cx.dtype)
            hard_mask = self._catalog_extra_depth > 0.0
            active = active * (~hard_mask).to(cx.dtype)
            ang = k.to(cx.dtype) * 1.7  # irrational-ish step -> spreads directions
            cx = cx + self.retry_jitter * torch.cos(ang) * active
            cy = cy + self.retry_jitter * torch.sin(ang) * active

        # Per-object grasp Z offset (replaces scalar grasp_z_offset for DESCEND/GRASP).
        per_z = self._cur_grasp_z_off
        # Conservative object-local grasp offsets let us bias difficult shapes
        # without changing the planner interface or overfitting defaults yet.
        total_offset_local = self._cur_grasp_offset_local + self._retry_pattern_local
        offset_w = quat_apply(cube_quat_w, total_offset_local)
        cx = cx + offset_w[:, 0]
        cy = cy + offset_w[:, 1]
        per_z = per_z + offset_w[:, 2]
        # MUSTARD TIP-OVER: once toppled (phase 3, object now low), grasp the LYING
        # 3.5cm cylinder near its centre (tip_lying_z) instead of the old upright
        # base offset (-0.04, which would aim below the table for a lying bottle).
        if self.mustard_tip:
            _lying = (self._tip_phase == 3) & (cube_pos_w[:, 2] < self.tip_height)
            per_z = torch.where(_lying, torch.full_like(per_z, self.tip_lying_z), per_z)

        if self.container_drop:
            if self.keep_grasp_yaw_container:
                yaw_cmd = self._yaw
            else:
                yaw_cmd = torch.zeros_like(gz)
        else:
            goal_yaw_sym = _grasp_yaw(goal_quat_w, self._cur_grasp_sym, self._cur_grasp_yaw_off) \
                           - self._yaw_offset - self._place_bias_yaw
            use_goal_ori = self._stage >= int(Stage.CARRY)
            yaw_cmd = torch.where(use_goal_ori, goal_yaw_sym, self._yaw)

        # Wrist-unwind (only meaningful when grasp_sym > 0).
        yaw = yaw_cmd + self._yaw_k.to(yaw_cmd.dtype) * (0.5 * math.pi) + self._retry_pattern_dyaw

        # MUSTARD TIP-OVER: grasp the LYING cylinder with fingers across its long
        # axis (= tip_dir). Override the tilted-quat-derived yaw to (push_angle +
        # offset); offset tunable since the gripper yaw->finger-axis convention is
        # empirical (default pi/2 = fingers perpendicular to the lying axis).
        if self.mustard_tip:
            _lyg = (self._tip_phase == 3) & (cube_pos_w[:, 2] < self.tip_height) \
                   & (self._stage <= int(Stage.DESCEND))
            if _lyg.any():
                _tip_ang = torch.atan2(self._tip_dir[:, 1], self._tip_dir[:, 0]) + self.tip_yaw_off
                yaw = torch.where(_lyg, _tip_ang, yaw)

        # Z targets:
        # MUSTARD-SPECIFIC higher hover: the tall (13cm) bottle clips a finger on the
        # descent (CUBE_FELL). A higher pre-grasp hover -> straighter vertical descent
        # down the narrow oval channel -> fewer clips. Applied ONLY to the mustard
        # (grasp_z_off<-0.035) so the other 4 objects keep the low hover (a GLOBAL
        # raise costs travel time on every object and nets worse). Default 0 = off.
        _must_h = (self._cur_grasp_z_off < -0.035).to(cz.dtype) * self.must_hover
        z_pre     = cz + self.pre_approach_z + _must_h + H
        # grasp_z_bias compensates the LL policy's systematic vertical under-descent:
        # the commanded grasp height is lowered so the EE actually reaches grasp depth.
        # The under-descent WORSENS for later grasps (the arm config degrades after
        # several picks) -> ramp extra bias by task_idx. A GLOBAL bias increase
        # over-deepens the 1st grasp (breaks placement); ramping only deepens the
        # late grasps that actually under-descend (targets the ~23 grasp-stalls).
        ramp = self._task_idx.to(cz.dtype) * self.grasp_bias_ramp
        # retry_boost: deepen the command by reach_retries*boost so a stalled gentle
        # LL gets forced down to a hard low target (0 until the first config-break).
        retry_boost = self._reach_retries.to(cz.dtype) * self.retry_depth_boost
        z_grasp_raw = cz + per_z + H - self.grasp_z_bias - ramp - retry_boost
        # Conservative absolute clamp only: keep pathological targets from going
        # too low without lifting every grasp by the hand offset.
        if self.enable_grasp_z_clamp:
            z_grasp = torch.maximum(
                z_grasp_raw,
                torch.full_like(z_grasp_raw, self.min_grasp_z),
            )
        else:
            z_grasp = z_grasp_raw
        z_grasp = z_grasp - self._catalog_extra_depth
        if self.enable_grasp_z_clamp:
            z_grasp = torch.maximum(
                z_grasp,
                torch.full_like(z_grasp, self.min_grasp_z),
            )
        self._target_grasp_z.copy_(z_grasp)
        z_carry   = torch.full_like(cz, self.carry_z + H)
        if self.container_drop:
            # LOWER = goal_z directly (set by command to rim + drop_offset).
            # RELEASE = slightly below LOWER (open fingers inside the bin).
            # drop_z_bonus raises the release so the gripper drops the object from
            # above the rim instead of entering the bin and shoving it.
            # late_drop_bonus adds extra height for the last `late_drop_count`
            # placements (the pile is highest then).
            late = (self._task_idx >= (self.num_objects - self.late_drop_count)).to(gz.dtype) \
                   * self.late_drop_bonus
            z_place   = gz + H + self.drop_z_bonus + late
            if self.safe_release_above_rim:
                z_release = z_place + self.release_z_bias
            else:
                z_release = gz + self.release_z_offset + H + self.drop_z_bonus + late + self.release_z_bias
        else:
            z_place   = gz + per_z + H
            z_release = gz + self.release_z_offset + H + self.release_z_bias
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
        gx_c = gx - self._place_offset[:, 0] - self._place_bias_xy[:, 0]
        gy_c = gy - self._place_offset[:, 1] - self._place_bias_xy[:, 1]
        # Spread drop points across the bin floor by pick slot so objects don't
        # pile at centre and bounce the bin (see __init__ place_spread note).
        if self.container_drop and self.place_spread > 0.0:
            slot = self._task_idx.clamp(0, self._place_spread_xy.shape[0] - 1)
            sx = self._place_spread_xy[slot, 0] * self.place_spread
            sy = self._place_spread_xy[slot, 1] * self.place_spread
            gx_c = gx_c + sx
            gy_c = gy_c + sy

        if self.container_drop and self.release_at_container_center:
            is_release = self._stage == int(Stage.RELEASE)
            centre_x = gx - self._place_offset[:, 0] - self._place_bias_xy[:, 0]
            centre_y = gy - self._place_offset[:, 1] - self._place_bias_xy[:, 1]
            gx_c = torch.where(is_release, centre_x, gx_c)
            gy_c = torch.where(is_release, centre_y, gy_c)

        is_lift = self._stage == int(Stage.LIFT)
        d = torch.stack([cx, cy], dim=-1) - self._lift_xy
        n = torch.norm(d, dim=-1, keepdim=True)
        lift_xy = self._lift_xy + d * torch.clamp(self.lift_anchor_radius / (n + 1e-6), max=1.0)
        ex = torch.where(placing, gx_c, torch.where(use_goal, gx, torch.where(is_lift, lift_xy[:, 0], cx)))
        ey = torch.where(placing, gy_c, torch.where(use_goal, gy, torch.where(is_lift, lift_xy[:, 1], cy)))

        # Container RETRACT: lift STRAIGHT UP over the drop point until the gripper
        # clears the rim, THEN shift XY toward the robot. The old code shifted XY
        # while still low -> the gripper dragged the rim and shoved the bin (the #1
        # displacement cause, 20/54 events at RETRACT). Gate the lateral shift on
        # the EE having risen retract_clear_z above the bin/goal z.
        is_retract = self._stage == int(Stage.RETRACT)
        if self.container_drop:
            self._retract_release_xy[:, 0] = gx_c
            self._retract_release_xy[:, 1] = gy_c
            self._retract_clear_z.copy_(z_release + self.retract_clear_z)
            if self.retract_clear_z > 0.0 and getattr(self, "_cur_ee_pos", None) is not None:
                cleared = self._cur_ee_pos[:, 2] >= self._retract_clear_z
            else:
                cleared = torch.ones_like(gx, dtype=torch.bool)
            self._retract_phase_xy_clear.copy_(is_retract & cleared)
            shift = is_retract & cleared
            # Strict two-phase retract:
            # 1) keep XY pinned to the release slot while lifting vertically.
            # 2) only after clearing the release height by retract_clear_z may any
            #    configured lateral retract offset be applied.
            ex = torch.where(is_retract, gx_c, ex)
            ey = torch.where(is_retract, gy_c, ey)
            ex = torch.where(shift, gx_c + self._retract_xy_off[0], ex)
            ey = torch.where(shift, gy_c + self._retract_xy_off[1], ey)

        # SAFE TRANSIT CEILING: up -> over -> down. In the transit/approach stages,
        # lift the EE to transit_ceiling_z while it is still moving laterally toward
        # the target, and only let it descend once XY-aligned. LIFT always lifts to
        # the ceiling (straight up after grasp); PRE_GRASP/CARRY/RETRACT raise to
        # the ceiling only while far in XY (so the final descent stays vertical).
        if self.transit_ceiling_z > 0.0 and getattr(self, "_cur_ee_pos", None) is not None:
            ee_xy = self._cur_ee_pos[:, :2]
            tgt_xy = torch.stack([ex, ey], dim=-1)
            # Per-object alignment: the narrow mustard bottle (grasp_z_off≈-0.04)
            # is the #1 CUBE_FELL cause (15/21) — the ~8cm gripper barely clears the
            # 3.5cm bottle, so the loose 5cm align lets a finger clip & tip it. Hold
            # the arm high until it is TIGHTLY centred over the mustard, so the
            # vertical descent doesn't graze the bottle.
            is_must = self._cur_grasp_z_off < -0.035
            align_tol = torch.where(is_must,
                                    torch.full_like(ee_xy[:, 0], self.mustard_align_tol),
                                    torch.full_like(ee_xy[:, 0], self.transit_align_tol))
            xy_far = torch.norm(ee_xy - tgt_xy, dim=-1) > align_tol
            is_pre   = self._stage == int(Stage.PRE_GRASP)
            is_lift  = self._stage == int(Stage.LIFT)
            is_carry = self._stage == int(Stage.CARRY)
            is_retr  = self._stage == int(Stage.RETRACT)
            raise_z = is_lift | ((is_pre | is_carry | is_retr) & xy_far)
            # CONFIG-BREAK FIX (from top miner's planner): after a config-break
            # retreat (reach_retries>0), the PRE_GRASP re-approach must be DIAGONAL
            # to break the bent-elbow IK trap. The ceiling's forced-vertical descent
            # re-enters the same trap -> loops to timeout. Disable the ceiling for
            # the grasp approach of envs that have done a config-break retreat.
            raise_z = raise_z & ~(is_pre & (self._reach_retries > 0))
            ez = torch.where(raise_z, torch.clamp(ez, min=self.transit_ceiling_z), ez)

        # DESCEND XY gate: near object height, hold hover Z until XY is centred so
        # vertical motion does not push/slide the object with misaligned finger pads.
        if self.descend_xy_gate_z_margin > 0.0 and getattr(self, "_cur_ee_pos", None) is not None:
            is_desc = self._stage == int(Stage.DESCEND)
            near_obj = self._cur_ee_pos[:, 2] < (
                cz + self.pre_approach_z + self.H + self.descend_xy_gate_z_margin
            )
            aim_xy = torch.stack([cx, cy], dim=-1)
            xy_err = torch.norm(self._cur_ee_pos[:, :2] - aim_xy, dim=-1)
            is_must = self._cur_grasp_z_off < -0.035
            align_tol = torch.where(
                is_must,
                torch.full_like(xy_err, self.mustard_align_tol),
                torch.full_like(xy_err, self.transit_align_tol),
            )
            gate = is_desc & near_obj & (xy_err > align_tol)
            z_hover = cz + self.pre_approach_z + _must_h + H
            ez = torch.where(gate, z_hover, ez)

        # MUSTARD TIP-OVER target override (phases 1-2): replace the grasp approach
        # with a side-push trajectory at tip_push_z. Phase 1 = near-side pre-push
        # pose (beside the bottle); phase 2 = drive through to the far side, toppling
        # the bottle in tip_dir. Gripper is forced closed for these phases (in step).
        if self.mustard_tip:
            _th = self.tip_push_z + H
            _p1 = self._tip_phase == 1
            _p2 = self._tip_phase == 2
            _pre_x = cx - self._tip_dir[:, 0] * self.tip_standoff
            _pre_y = cy - self._tip_dir[:, 1] * self.tip_standoff
            _psh_x = cx + self._tip_dir[:, 0] * self.tip_overshoot
            _psh_y = cy + self._tip_dir[:, 1] * self.tip_overshoot
            ex = torch.where(_p1, _pre_x, torch.where(_p2, _psh_x, ex))
            ey = torch.where(_p1, _pre_y, torch.where(_p2, _psh_y, ey))
            ez = torch.where(_p1 | _p2, torch.full_like(ez, _th), ez)
            # NECK-GRASP (phase 3, lying): shift the grasp XY toward the cap/neck end
            # (+tip_dir) so the gripper closes on the narrow neck (form closure) rather
            # than pinching the smooth mid-body. Within grasp_secure_xy_tol (0.06).
            if self.tip_neck_off != 0.0:
                _p3g = (self._tip_phase == 3) & (cube_pos_w[:, 2] < self.tip_height) \
                       & (self._stage <= int(Stage.GRASP))
                ex = torch.where(_p3g, ex + self._tip_dir[:, 0] * self.tip_neck_off, ex)
                ey = torch.where(_p3g, ey + self._tip_dir[:, 1] * self.tip_neck_off, ey)

        # VIA-HOME override: while routing back to the home config (set between picks),
        # command the captured home EE pose (XY + a high z) so the arm un-twists to the
        # start config before descending to the next object. Takes precedence over all.
        if self.via_home:
            ex = torch.where(self._via_home, self._home_ee_pos[:, 0], ex)
            ey = torch.where(self._via_home, self._home_ee_pos[:, 1], ey)
            ez = torch.where(self._via_home, self._home_ee_z, ez)

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
