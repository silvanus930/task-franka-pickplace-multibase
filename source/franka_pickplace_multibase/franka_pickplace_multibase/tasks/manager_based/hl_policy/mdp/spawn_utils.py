# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pose-offset sampling for HL cube spawn and goal reset."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch


def sample_xyz_offsets(
    num_envs: int,
    pose_range: dict[str, tuple[float, float]],
    device: torch.device,
) -> torch.Tensor:
    """Sample ``(N, 3)`` XYZ offsets – same logic as ``reset_root_state_uniform``."""
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    ranges = torch.tensor(range_list, device=device)
    return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 3), device=device)
