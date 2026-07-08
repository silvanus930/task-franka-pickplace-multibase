#!/usr/bin/env python3
"""Smoke tests for grasp long-axis / wrist-yaw logic."""

from __future__ import annotations

import math
import sys

import torch

from isaaclab.utils.math import quat_from_euler_xyz

from franka_pickplace_multibase.tasks.manager_based.hl_policy.grasp_yaw import (
    GripperFrameConfig,
    LongAxisSource,
    compute_grasp_yaw,
    normalize_yaw,
)


def _euler_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> torch.Tensor:
    return quat_from_euler_xyz(
        torch.deg2rad(torch.tensor([roll_deg])),
        torch.deg2rad(torch.tensor([pitch_deg])),
        torch.deg2rad(torch.tensor([yaw_deg])),
    )


def _assert_closing_perpendicular(dbg, tol_deg: float = 5.0) -> None:
    long_heading = math.atan2(dbg.long_axis_w[0, 1].item(), dbg.long_axis_w[0, 0].item())
    closing_heading = math.atan2(
        dbg.closing_axis_w[0, 1].item(), dbg.closing_axis_w[0, 0].item()
    )
    delta = abs(normalize_yaw(torch.tensor(closing_heading - long_heading)).item())
    assert abs(delta - math.pi / 2) < math.radians(tol_deg), (
        f"expected 90°, got {math.degrees(delta):.1f}° "
        f"(width_align={dbg.width_alignment[0].item():.2f})"
    )
    assert dbg.width_alignment[0].item() < 0.5


def test_upright_box_uses_in_plane_axis() -> None:
    """Upright sugar-box-like pose: in-plane Y (3.5 cm) not body-yaw + π/2."""
    frame = GripperFrameConfig()
    q = _euler_quat(0.0, 0.0, 30.0)
    fp = torch.tensor([[0.025, 0.035]])
    sym = torch.tensor([math.pi])
    yaw_off = torch.tensor([math.pi / 2])  # catalog auto offset — must be ignored for boxes

    _target, dbg = compute_grasp_yaw(
        q, sym, yaw_off, torch.zeros(1, 3), fp, torch.tensor([0.07]), frame,
    )
    assert dbg.source[0] == LongAxisSource.FOOTPRINT_BOX.value
    _assert_closing_perpendicular(dbg)


def test_lying_box_closing_perpendicular_to_long() -> None:
    """Lying box: finger closing direction ⊥ long-axis heading in world XY."""
    frame = GripperFrameConfig()
    q = _euler_quat(90.0, 0.0, 0.0)
    fp = torch.tensor([[0.025, 0.035]])
    sym = torch.tensor([math.pi])
    yaw_off = torch.tensor([math.pi / 2])

    _target, dbg = compute_grasp_yaw(
        q, sym, yaw_off, torch.zeros(1, 3), fp, torch.tensor([0.07]), frame,
    )
    assert dbg.source[0] == LongAxisSource.FOOTPRINT_BOX.value
    _assert_closing_perpendicular(dbg)


def test_long_path_ignores_catalog_pi_half() -> None:
    """Box long-axis path must not add catalog π/2 on top of closing offset."""
    frame = GripperFrameConfig()
    q = _euler_quat(90.0, 0.0, 0.0)
    fp = torch.tensor([[0.025, 0.035]])
    sym = torch.tensor([math.pi])

    t_no_off, _ = compute_grasp_yaw(
        q, sym, torch.zeros(1), torch.zeros(1, 3), fp, torch.tensor([0.07]), frame,
    )
    t_with_off, _ = compute_grasp_yaw(
        q, sym, torch.tensor([math.pi / 2]), torch.zeros(1, 3), fp, torch.tensor([0.07]), frame,
    )
    assert abs(normalize_yaw(t_no_off - t_with_off).item()) < 0.05


def test_scenario5_sugar_spawn_quat() -> None:
    """Scenario 5 object0 spawn: gripper must close across width, not along length."""
    frame = GripperFrameConfig()
    # From envhub scenario 5 (w, x, y, z)
    q = torch.tensor([[0.1078, 0.0, 0.0, -0.9942]])
    fp = torch.tensor([[0.025, 0.035]])
    sym = torch.tensor([math.pi])
    yaw_off = torch.tensor([math.pi / 2])

    target, dbg = compute_grasp_yaw(
        q, sym, yaw_off, torch.zeros(1, 3), fp, torch.tensor([0.07]), frame,
    )
    assert dbg.source[0] == LongAxisSource.FOOTPRINT_BOX.value
    _assert_closing_perpendicular(dbg)

    long_heading = math.degrees(
        math.atan2(dbg.long_axis_w[0, 1].item(), dbg.long_axis_w[0, 0].item())
    )
    target_deg = math.degrees(target[0].item())
    # Long axis ≈ -78°; target wrist yaw should be ≈ +12° (not ≈ -78° parallel).
    assert abs(normalize_yaw(target - torch.tensor(long_heading * math.pi / 180)).item()
               - math.pi / 2) < 0.15, (
        f"target {target_deg:.1f}° too close to long heading {long_heading:.1f}°"
    )


def main() -> int:
    test_upright_box_uses_in_plane_axis()
    test_lying_box_closing_perpendicular_to_long()
    test_long_path_ignores_catalog_pi_half()
    test_scenario5_sugar_spawn_quat()
    print("grasp_yaw tests OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
