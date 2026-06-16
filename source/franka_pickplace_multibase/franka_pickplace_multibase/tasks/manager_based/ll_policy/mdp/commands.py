# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command terms for the LL policy environment.

GripperCommand / GripperCommandCfg
-----------------------------------
Samples a binary gripper target (0 = open, 1 = close) once per episode at
environment reset.  The timer is set far beyond any episode length so it never
fires mid-episode; resampling happens only via the reset path in CommandManager.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class GripperCommand(CommandTerm):
    """Holds a binary gripper target that is resampled at each episode reset.

    Output tensor shape: (num_envs, 1)
      0.0  — open
      1.0  — close

    The fraction of close episodes is controlled by ``cfg.close_prob``
    (default 0.5 → 50 % open, 50 % close).  This ratio should match the
    expected duty cycle of the High-Level policy:

      close_prob = 0.5   balanced (default)
      close_prob = 0.35  more open-gripper episodes
      close_prob = 0.65  more close-gripper episodes
    """

    cfg: GripperCommandCfg

    def __init__(self, cfg: GripperCommandCfg, env):
        super().__init__(cfg, env)
        # Initialise all envs to "open".
        self._grip_target: torch.Tensor = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "GripperCommand\n"
        msg += f"\tCommand dimension: {self.command_dim}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Binary gripper target per env, shape (N, 1)."""
        return self._grip_target

    @property
    def command_dim(self) -> int:
        return 1

    # ------------------------------------------------------------------
    # Implementation of abstract CommandTerm methods
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Bernoulli(close_prob) sample: each reset env gets open or close independently."""
        n = env_ids.shape[0]
        self._grip_target[env_ids, 0] = (
            torch.rand(n, device=self.device) < self.cfg.close_prob
        ).float()

    def _update_command(self) -> None:
        """No mid-step update required — command is static within an episode."""
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class GripperCommandCfg(CommandTermCfg):
    """Configuration for :class:`GripperCommand`.

    Attributes:
        close_prob: Probability of sampling a *close* command at each episode
            reset.  The complementary probability ``1 - close_prob`` produces
            an *open* episode.  Default: 0.5 (balanced).
    """

    class_type: type = GripperCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    close_prob: float = 0.5
