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

* ``closing_along_ee_x`` (default): ``target_yaw = object_yaw + π/2``
* ``closing_along_ee_y``:           ``target_yaw = object_yaw``

An extra ``yaw_offset`` (typically ``0`` or ``π/2``) can be added per object or
globally when the USD mesh frame differs from the catalog assumption.

Long-axis estimation priority
-----------------------------
1. **Rectangular boxes** (``footprint_x ≠ footprint_y``): score local X/Y/Z by
   ``‖proj_xy‖ × (fp_x, fp_y, upright_height)`` — handles object0 upright vs lying.
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


def _pick_best_long_axis_world(
    obj_quat: torch.Tensor,
    long_axis_local: torch.Tensor,
    footprint_xy: torch.Tensor,
    upright_height: torch.Tensor,
    frame: GripperFrameConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pick the object long axis projected into the table XY plane.

    * **Rectangular boxes** (``footprint_x ≠ footprint_y``): score each local body
      axis by ``‖proj_xy‖ × extent`` where extent is ``(fp_x, fp_y, upright_h)``.
      This handles object0 sugar box both upright (long = local Y) and lying
      (long = local Z height axis) without a fixed catalog axis.
    * **Round / catalog objects** (mustard bottle): use catalog elongation axis
      when its XY projection is strong; fall back to body yaw when upright.
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

    cardinals = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, 3, 3)
    flat = cardinals.reshape(n * 3, 3)
    quat_rep = obj_quat.repeat_interleave(3, dim=0)
    world_axes = normalize(quat_apply(quat_rep, flat)).reshape(n, 3, 3)
    xy_norm = torch.norm(world_axes[:, :, :2], dim=-1)

    z_extent = upright_height.clamp(min=1e-6)
    weights = torch.stack([fp_x, fp_y, z_extent], dim=-1)
    scores = xy_norm * weights
    best_body_idx = scores.argmax(dim=-1)
    arange = torch.arange(n, device=device)
    long_from_body = world_axes[arange, best_body_idx]
    body_best_xy = xy_norm[arange, best_body_idx]

    long_w = long_from_body
    xy_norm_best = body_best_xy

    is_round = has_catalog & (~is_box)
    upright_catalog = is_round & (catalog_xy < frame.upright_xy_norm_min)
    use_catalog_lying = is_round & (~upright_catalog) & (catalog_xy >= frame.upright_xy_norm_min)
    long_w = torch.where(use_catalog_lying.unsqueeze(-1), catalog_w, long_w)
    xy_norm_best = torch.where(use_catalog_lying, catalog_xy, xy_norm_best)

    upright_mask = upright_catalog | (xy_norm_best < frame.upright_xy_norm_min)

    sources: list[str] = []
    for i in range(n):
        if upright_mask[i]:
            sources.append(LongAxisSource.BODY_YAW.value)
        elif is_box[i]:
            sources.append(LongAxisSource.FOOTPRINT_BOX.value)
        elif use_catalog_lying[i]:
            sources.append(LongAxisSource.CATALOG.value)
        elif has_fp[i]:
            sources.append(LongAxisSource.FOOTPRINT.value)
        else:
            sources.append(LongAxisSource.BODY_YAW.value)

    has_any = is_box | has_catalog | has_fp
    return long_w, upright_mask, has_any, sources


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
    long_w, upright_mask, _has_any, sources = _pick_best_long_axis_world(
        obj_quat, long_axis_local, footprint_xy, upright_height, frame,
    )

    long_yaw = torch.atan2(long_w[:, 1], long_w[:, 0])
    body_yaw, _, _ = euler_xyz_from_quat(obj_quat)
    object_yaw = torch.where(upright_mask, body_yaw, long_yaw)

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
        frame:             Gripper frame convention; defaults to Franka EE-Y closing.

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

    has_axis = (
        (torch.norm(long_axis_local, dim=-1) > 1e-6)
        | (
            (footprint_xy[:, 0] > 1e-6)
            & (footprint_xy[:, 1] > 1e-6)
            & ((footprint_xy[:, 0] - footprint_xy[:, 1]).abs() > 0.003)
        )
    )

    # Bias from catalog manual offset + optional per-object frame toggle (0 or π/2).
    bias = yaw_off + frame.yaw_offset
    lying_elongated = has_axis & (~upright_mask)

    # Lying elongated: align pads with long axis, close across short width.
    aligned = object_yaw + bias + torch.where(
        lying_elongated,
        torch.full_like(object_yaw, frame.closing_axis_offset),
        torch.zeros_like(object_yaw),
    )
    folded_long = normalize_yaw(_fold_long_axis_yaw(aligned))

    # Upright / round / legacy: symmetry-fold body yaw.
    sym_input = object_yaw + bias
    folded_sym = normalize_yaw(_fold_yaw_symmetry(sym_input, sym))

    target_yaw = torch.where(lying_elongated, folded_long, folded_sym)

    # Closing direction in world XY (unit vector, z=0).
    closing_heading = object_yaw + 0.5 * math.pi
    closing_axis_w = torch.stack(
        [
            torch.cos(closing_heading),
            torch.sin(closing_heading),
            torch.zeros_like(closing_heading),
        ],
        dim=-1,
    )

    debug = GraspYawDebug(
        object_yaw=object_yaw,
        target_yaw=target_yaw,
        long_axis_w=long_w,
        closing_axis_w=closing_axis_w,
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
