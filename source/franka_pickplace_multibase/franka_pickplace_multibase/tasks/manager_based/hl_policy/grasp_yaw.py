# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-aligned grasp yaw for top-down Franka picks.

Why object-aligned yaw
----------------------
For elongated objects the gripper must not descend with an arbitrary wrist
rotation.  Finger *pads* should run parallel to the object's **long axis** so
the fingers close across the **short width**.  Closing at the wrong angle
pushes or slips the object instead of trapping it.

Franka ``panda_hand`` convention (verified in sim at ``pitch = π``, top-down)
-----------------------------------------------------------------------------
With ``quat_from_euler_xyz(roll=0, pitch=π, yaw=ψ)`` the **visible** finger
closing direction in the table plane requires a ``+π/2`` wrist offset relative
to the object long-axis heading.  Equivalently, fingers close along EE local **X**
(not Y) in this Isaac Lab Franka setup.

Therefore, for an object whose long axis has heading ``object_yaw`` in world XY:

* ``target_yaw = object_yaw + π/2`` (global correction applied in :func:`compute_grasp_yaw`)
* Finger closing direction is perpendicular to the long axis (across the short width)

An extra ``yaw_offset`` (typically ``0`` or ``π/2``) can be added per object on the
symmetry-fold path only; all grasps receive a uniform ``+π/2`` wrist correction.

Long-axis estimation priority
-----------------------------
1. **Rectangular boxes** (``footprint_x ≠ footprint_y``): score local X/Y/Z by
   ``‖proj_xy‖ × extent`` and use the best **in-plane** axis (never body-yaw +
   catalog π/2, which aligned the gripper **along** the long edge).
2. **Catalog elongation axis** (mustard bottle): used when footprint is round and
   the axis lies in the table plane.
3. Body yaw from the object quaternion (symmetry-folded fallback when upright).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from isaaclab.utils.math import euler_xyz_from_quat, normalize, quat_apply


# Uniform wrist correction: detected long-axis yaw is 90° short (closes along length).
GRASP_YAW_GLOBAL_OFFSET_RAD: float = math.pi / 2


class LongAxisSource(str, Enum):
    """How the object long axis was estimated."""

    CATALOG = "catalog"
    FOOTPRINT = "footprint"
    FOOTPRINT_BOX = "footprint_box"  # weighted local X/Y/Z for rectangular boxes
    BODY_YAW = "body_yaw"


class ClosingAxis(str, Enum):
    """Which EE local axis is the finger closing direction at ``yaw=0, pitch=π``."""

    EE_X = "ee_x"
    EE_Y = "ee_y"


@dataclass(frozen=True)
class GripperFrameConfig:
    """Gripper-frame convention for mapping object yaw → wrist yaw.

    Attributes:
        closing_axis: EE axis along which fingers open/close at neutral yaw.
        yaw_offset:   Extra radians added after alignment (use ``0`` or ``π/2``
                      to flip convention without code changes).
        upright_xy_norm_min: Minimum ‖long_axis_xy‖ to trust projected long axis;
                      below this the body-yaw fallback is used (upright cylinder).
    """

    closing_axis: ClosingAxis = ClosingAxis.EE_X
    yaw_offset: float = 0.0
    upright_xy_norm_min: float = 0.35

    @property
    def closing_axis_offset(self) -> float:
        """Wrist-yaw correction when closing is along EE X instead of EE Y."""
        return math.pi / 2 if self.closing_axis == ClosingAxis.EE_X else 0.0


@dataclass
class GraspYawDebug:
    """Per-env debug payload returned by :func:`compute_grasp_yaw`."""

    object_yaw: torch.Tensor          # (N,) long-axis heading in world XY (rad)
    target_yaw: torch.Tensor          # (N,) commanded wrist yaw (rad)
    long_axis_w: torch.Tensor         # (N, 3) unit long axis in world frame
    closing_axis_w: torch.Tensor      # (N, 3) unit closing direction in world XY
    width_alignment: torch.Tensor     # (N,) |dot(closing, long)|; 0 is across width, 1 is along length
    source: list[str]                 # len N, LongAxisSource value per env


def normalize_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Wrap yaw to ``[-π, π]``."""
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))


def _fold_yaw_symmetry(yaw: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """Fold yaw into ``[-sym/2, sym/2]``; ``sym=0`` → neutral wrist."""
    neutral_mask = sym <= 0.0
    safe_sym = torch.where(neutral_mask, torch.ones_like(sym), sym)
    folded = (yaw + 0.25 * safe_sym) % safe_sym - 0.25 * safe_sym
    return torch.where(neutral_mask, torch.zeros_like(yaw), folded)


def _fold_long_axis_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Fold to ``[-π/2, π/2]`` — long axis has 180° heading ambiguity only."""
    return 0.5 * torch.atan2(torch.sin(2.0 * yaw), torch.cos(2.0 * yaw))


def estimate_long_axis_local(
    long_axis_local: torch.Tensor,
    footprint_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(unit_long_local, has_explicit)`` for each env.

    Args:
        long_axis_local: ``(N, 3)`` catalog long axis (may be zero).
        footprint_xy:    ``(N, 2)`` object local ``(size_x, size_y)`` in metres.
    """
    n = long_axis_local.shape[0]
    device = long_axis_local.device
    has_catalog = torch.norm(long_axis_local, dim=-1) > 1e-6

    fp_x = footprint_xy[:, 0]
    fp_y = footprint_xy[:, 1]
    has_fp = (fp_x > 1e-6) & (fp_y > 1e-6)

    # Major footprint dimension → local long axis (PCA equivalent for AABB).
    long_from_fp = torch.zeros(n, 3, device=device)
    long_from_fp[:, 0] = torch.where(fp_x >= fp_y, 1.0, 0.0)
    long_from_fp[:, 1] = torch.where(fp_x >= fp_y, 0.0, 1.0)

    use_fp = has_fp & (~has_catalog)
    axis = torch.where(has_catalog.unsqueeze(-1), long_axis_local, long_from_fp)
    valid = has_catalog | has_fp
    axis = normalize(axis)
    return axis, valid


def _box_world_axes(
    obj_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(world_axes, xy_norm)`` for local X/Y/Z."""
    n = obj_quat.shape[0]
    device = obj_quat.device
    dtype = obj_quat.dtype
    cardinals = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, 3, 3)
    flat = cardinals.reshape(n * 3, 3)
    quat_rep = obj_quat.repeat_interleave(3, dim=0)
    world_axes = normalize(quat_apply(quat_rep, flat)).reshape(n, 3, 3)
    xy_norm = torch.norm(world_axes[:, :, :2], dim=-1)
    return world_axes, xy_norm


def _box_grasp_heading(
    obj_quat: torch.Tensor,
    footprint_xy: torch.Tensor,
    upright_height: torch.Tensor,
    frame: GripperFrameConfig,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Pick the dominant in-plane box axis for top-down grasp yaw.

    Always uses the best **horizontal** local axis (X/Y/Z) scored by
    ``‖proj_xy‖ × extent``.  This avoids the old body-yaw + catalog-π/2 path,
    which often collapsed to the **long-axis heading** and made the gripper
    close **along** the length instead of across the width.
    """
    n = obj_quat.shape[0]
    device = obj_quat.device

    fp_x = footprint_xy[:, 0]
    fp_y = footprint_xy[:, 1]
    z_extent = upright_height.clamp(min=1e-6)

    world_axes, xy_norm = _box_world_axes(obj_quat)
    scores = xy_norm * torch.stack([fp_x, fp_y, z_extent], dim=-1)

    in_plane = xy_norm >= frame.upright_xy_norm_min
    scores = torch.where(in_plane, scores, torch.full_like(scores, -1.0))
    best_idx = scores.argmax(dim=-1)
    arange = torch.arange(n, device=device)
    long_w = world_axes[arange, best_idx]
    long_yaw = torch.atan2(long_w[:, 1], long_w[:, 0])

    # If nothing projects into the table plane, fall back to the larger footprint
    # edge even when nearly vertical (still better than body-yaw + π/2).
    none_in_plane = ~in_plane.any(dim=-1)
    if none_in_plane.any():
        long_local_y = fp_y >= fp_x
        fallback_idx = torch.where(long_local_y, 1, 0).long()
        fb_w = world_axes[arange, fallback_idx]
        long_w = torch.where(none_in_plane.unsqueeze(-1), fb_w, long_w)
        long_yaw = torch.where(
            none_in_plane,
            torch.atan2(fb_w[:, 1], fb_w[:, 0]),
            long_yaw,
        )

    return long_w, long_yaw, True


def _pick_best_long_axis_world(
    obj_quat: torch.Tensor,
    long_axis_local: torch.Tensor,
    footprint_xy: torch.Tensor,
    upright_height: torch.Tensor,
    frame: GripperFrameConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pick the object long axis projected into the table XY plane.

    * **Rectangular boxes** — best in-plane local axis (X/Y/Z) by extent score;
      never body-yaw + catalog π/2 (that path closed **along** the long edge).
    * **Round / catalog objects** (mustard bottle) — catalog elongation axis
      when its XY projection is strong; body-yaw when upright.
    """
    n = obj_quat.shape[0]
    device = obj_quat.device
    dtype = obj_quat.dtype

    fp_x = footprint_xy[:, 0]
    fp_y = footprint_xy[:, 1]
    has_fp = (fp_x > 1e-6) & (fp_y > 1e-6)
    is_box = has_fp & ((fp_x - fp_y).abs() > 0.003)

    has_catalog = torch.norm(long_axis_local, dim=-1) > 1e-6
    catalog_w = normalize(quat_apply(obj_quat, normalize(long_axis_local)))
    catalog_xy = torch.norm(catalog_w[:, :2], dim=-1)

    long_w = torch.zeros(n, 3, device=device, dtype=dtype)
    xy_norm_best = torch.zeros(n, device=device, dtype=dtype)
    upright_mask = torch.zeros(n, dtype=torch.bool, device=device)

    if is_box.any():
        box_long, _box_yaw, _ = _box_grasp_heading(
            obj_quat, footprint_xy, upright_height, frame,
        )
        long_w = torch.where(is_box.unsqueeze(-1), box_long, long_w)
        xy_norm_best = torch.where(
            is_box,
            torch.norm(box_long[:, :2], dim=-1),
            xy_norm_best,
        )

    is_round = has_catalog & (~is_box)
    if is_round.any():
        upright_catalog = is_round & (catalog_xy < frame.upright_xy_norm_min)
        use_catalog_lying = is_round & (~upright_catalog) & (catalog_xy >= frame.upright_xy_norm_min)
        long_w = torch.where(use_catalog_lying.unsqueeze(-1), catalog_w, long_w)
        xy_norm_best = torch.where(use_catalog_lying, catalog_xy, xy_norm_best)
        upright_mask = torch.where(is_round, upright_catalog | upright_mask, upright_mask)

    # Legacy footprint fallback for non-box objects without catalog axis.
    needs_fp_fallback = (~is_box) & (~has_catalog) & has_fp
    if needs_fp_fallback.any():
        cardinals = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, 3, 3)
        flat = cardinals.reshape(n * 3, 3)
        quat_rep = obj_quat.repeat_interleave(3, dim=0)
        world_axes = normalize(quat_apply(quat_rep, flat)).reshape(n, 3, 3)
        xy_norm = torch.norm(world_axes[:, :, :2], dim=-1)
        long_local_y = fp_y >= fp_x
        primary_idx = torch.where(long_local_y, 1, 0).long()
        arange = torch.arange(n, device=device)
        fp_long = world_axes[arange, primary_idx]
        fp_xy = xy_norm[arange, primary_idx]
        long_w = torch.where(needs_fp_fallback.unsqueeze(-1), fp_long, long_w)
        xy_norm_best = torch.where(needs_fp_fallback, fp_xy, xy_norm_best)
        fp_upright = needs_fp_fallback & (fp_xy < frame.upright_xy_norm_min)
        upright_mask = upright_mask | fp_upright

    sources: list[str] = []
    for i in range(n):
        if is_box[i]:
            sources.append(LongAxisSource.FOOTPRINT_BOX.value)
        elif upright_mask[i]:
            sources.append(LongAxisSource.BODY_YAW.value)
        elif is_round[i] and catalog_xy[i] >= frame.upright_xy_norm_min:
            sources.append(LongAxisSource.CATALOG.value)
        elif has_fp[i]:
            sources.append(LongAxisSource.FOOTPRINT.value)
        else:
            sources.append(LongAxisSource.BODY_YAW.value)

    return long_w, upright_mask, is_box, sources


def object_long_axis_world(
    obj_quat: torch.Tensor,
    long_axis_local: torch.Tensor,
) -> torch.Tensor:
    """Rotate a local long-axis unit vector into the world frame."""
    return normalize(quat_apply(obj_quat, long_axis_local))


def compute_object_yaw_from_pose(
    obj_quat: torch.Tensor,
    long_axis_local: torch.Tensor,
    footprint_xy: torch.Tensor,
    upright_height: torch.Tensor,
    frame: GripperFrameConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Estimate object long-axis heading and supporting vectors.

    Returns:
        ``object_yaw``   – ``(N,)`` heading of the long axis in world XY (rad).
        ``long_axis_w``  – ``(N, 3)`` unit long axis in world frame.
        ``upright_mask`` – ``(N,)`` bool, long-axis XY projection is degenerate.
        ``sources``      – list of :class:`LongAxisSource` names, len ``N``.
    """
    long_w, upright_mask, is_box, sources = _pick_best_long_axis_world(
        obj_quat, long_axis_local, footprint_xy, upright_height, frame,
    )

    long_yaw = torch.atan2(long_w[:, 1], long_w[:, 0])
    body_yaw, _, _ = euler_xyz_from_quat(obj_quat)
    object_yaw = torch.where(
        is_box,
        long_yaw,
        torch.where(upright_mask, body_yaw, long_yaw),
    )

    return object_yaw, long_w, upright_mask, sources


def compute_grasp_yaw(
    obj_quat: torch.Tensor,
    sym: torch.Tensor,
    yaw_off: torch.Tensor,
    long_axis_local: torch.Tensor,
    footprint_xy: torch.Tensor | None = None,
    upright_height: torch.Tensor | None = None,
    frame: GripperFrameConfig | None = None,
) -> tuple[torch.Tensor, GraspYawDebug]:
    """Compute wrist yaw for a top-down grasp aligned with object shape.

    Args:
        obj_quat:          ``(N, 4)`` object orientation (wxyz).
        sym:               ``(N,)`` yaw symmetry period (rad); ``0`` = round.
        yaw_off:           ``(N,)`` manual/catalog yaw bias (rad).
        long_axis_local:   ``(N, 3)`` catalog elongation axis (may be zero).
        footprint_xy:      ``(N, 2)`` optional local footprint for PCA fallback.
        frame:             Gripper frame convention; defaults to Franka EE-X closing.

    Returns:
        ``(target_yaw, debug)`` – wrist yaw in world frame, normalized to ``[-π, π]``.
    """
    if frame is None:
        frame = GripperFrameConfig()
    if footprint_xy is None:
        footprint_xy = torch.zeros(obj_quat.shape[0], 2, device=obj_quat.device)
    if upright_height is None:
        upright_height = torch.zeros(obj_quat.shape[0], device=obj_quat.device)

    object_yaw, long_w, upright_mask, sources = compute_object_yaw_from_pose(
        obj_quat, long_axis_local, footprint_xy, upright_height, frame,
    )

    fp_x = footprint_xy[:, 0]
    fp_y = footprint_xy[:, 1]
    has_fp = (fp_x > 1e-6) & (fp_y > 1e-6)
    is_box = has_fp & ((fp_x - fp_y).abs() > 0.003)
    has_catalog = torch.norm(long_axis_local, dim=-1) > 1e-6

    # Long-axis path: align to detected heading; global +π/2 applied below for all grasps.
    long_aligned_path = is_box | ((~upright_mask) & has_catalog)

    aligned_long = object_yaw + frame.yaw_offset
    target_long = normalize_yaw(_fold_long_axis_yaw(aligned_long))

    # Upright / round / legacy: symmetry-fold body yaw; catalog offset applies here.
    sym_input = object_yaw + yaw_off + frame.yaw_offset
    target_sym = normalize_yaw(_fold_yaw_symmetry(sym_input, sym))

    target_yaw = torch.where(long_aligned_path, target_long, target_sym)
    target_yaw = normalize_yaw(
        target_yaw + torch.full_like(target_yaw, GRASP_YAW_GLOBAL_OFFSET_RAD)
    )

    closing_axis_w = closing_direction_world_from_target(target_yaw, frame)
    long_xy = normalize(torch.cat([long_w[:, :2], torch.zeros_like(long_w[:, 2:3])], dim=-1))
    width_alignment = (closing_axis_w * long_xy).sum(dim=-1).abs()

    debug = GraspYawDebug(
        object_yaw=object_yaw,
        target_yaw=target_yaw,
        long_axis_w=long_w,
        closing_axis_w=closing_axis_w,
        width_alignment=width_alignment,
        source=sources,
    )
    return target_yaw, debug


# Backward-compatible scalar path (sym-yaw only, no footprint).
def compute_grasp_yaw_symmetry(
    obj_quat: torch.Tensor,
    sym: torch.Tensor,
    yaw_off: torch.Tensor,
) -> torch.Tensor:
    """Legacy symmetry-only yaw (round / unknown footprint objects)."""
    _, _, yaw = euler_xyz_from_quat(obj_quat)
    return normalize_yaw(_fold_yaw_symmetry(yaw + yaw_off, sym))


def closing_direction_world_from_target(
    target_yaw: torch.Tensor,
    frame: GripperFrameConfig | None = None,
) -> torch.Tensor:
    """Unit vector in world XY for the finger closing direction at ``target_yaw``."""
    if frame is None:
        frame = GripperFrameConfig()
    heading = normalize_yaw(
        target_yaw + torch.full_like(target_yaw, 0.5 * math.pi - frame.closing_axis_offset)
    )
    return torch.stack(
        [
            torch.cos(heading),
            torch.sin(heading),
            torch.zeros_like(heading),
        ],
        dim=-1,
    )
