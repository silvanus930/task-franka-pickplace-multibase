# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for the LL EE-tracking environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def override_pose_z_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: tuple[float, float],
    z_range: tuple[float, float],
    num_steps: int,
) -> tuple[float, float] | object:
    """Switch ``ee_pose`` Z sampling to a table-biased range after ``num_steps``."""
    if env.common_step_counter > num_steps:
        return z_range
    return mdp.modify_term_cfg.NO_CHANGE
