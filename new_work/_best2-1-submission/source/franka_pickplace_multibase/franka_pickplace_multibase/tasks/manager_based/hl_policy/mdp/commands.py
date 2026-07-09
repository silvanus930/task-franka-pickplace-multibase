# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HL pick-and-place command terms.

``HLPoseCommand`` drives the ``ee_pose`` command slot: each step it queries
the ``PickPlacePlanner`` for the *endpoint* of the current stage and writes
that static pose in robot-base frame.  This matches the piecewise-constant
command distribution the frozen LL policy was trained on.

Multi-object / container support
---------------------------------
``HLPoseCommandCfg.cube_names`` lists all objects in pick order.
``HLPoseCommand`` stores per-env, per-object goal tensors
``goal_pos_w (N, M, 3)`` and ``goal_quat_w (N, M, 4)``.  Each step it
gathers the current-object slice using ``planner._task_idx`` and passes it
to the planner together with per-object grasp metadata tensors.

Per-object grasp metadata (``grasp_z_offsets``, ``grasp_syms``,
``grasp_yaw_offsets``) are stored as ``(M,)`` tensors and selected by
``_task_idx`` before being passed to ``planner.step()``.

In container mode (``container_drop=True``) goal Z is set to the container
drop height and the marker switches to a cuboid visualising the bin opening.

At episode reset the event ``reset_scattered_objects_into_container`` calls
``set_goals_from_strategy(env_ids, goal_pos, goal_rot)`` to populate the
per-object goals.

``HLGripCommand`` mirrors the grip value from ``HLPoseCommand._grip_command``
into the ``grip_cmd`` command slot.  It must be declared *after* ``ee_pose``
in ``HLCommandsCfg`` so the pose term always updates first.
"""

from __future__ import annotations

import logging
import math
import os as _os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, subtract_frame_transforms

from ..classical_planner import STAGE_NAMES, Stage, PickPlacePlanner
from .spawn_utils import sample_xyz_offsets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LOG = logging.getLogger(__name__)


class HLPoseCommand(CommandTerm):
    """EE pose command driven by the ``PickPlacePlanner`` state machine.

    Supports sequential multi-object pick-and-place: ``cube_names`` lists all
    objects in pick order; the planner's ``_task_idx`` selects the current
    object each step.

    Per-object grasp metadata (``grasp_z_offsets``, ``grasp_syms``,
    ``grasp_yaw_offsets``) are stored as ``(M,)`` tensors and sliced by
    ``_task_idx`` each step before being forwarded to ``planner.step()``.

    Outputs ``(N, 7)`` in robot-base frame ``[pos | quat(wxyz)]``, matching
    the format of ``UniformPoseCommand`` so the frozen LL policy runs unchanged.
    """

    cfg: HLPoseCommandCfg

    def __init__(self, cfg: HLPoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.robot_name]
        self._body_idx: int      = self.robot.find_bodies(cfg.body_name)[0][0]
        self._wrist_joint_idx: int = self.robot.find_joints(cfg.wrist_joint_name)[0][0]

        self._cubes: list[RigidObject] = [
            env.scene[name] for name in cfg.cube_names
        ]
        self._M: int = len(self._cubes)

        # Typed-scenario mode: num_active < M means each episode activates a
        # subset of the catalog objects via _active_catalog_indices (N, M_active).
        _num_active = cfg.num_active if (cfg.num_active > 0 and cfg.num_active < self._M) else self._M
        self._typed_mode: bool = (_num_active < self._M)
        self._M_active: int    = _num_active
        # Honour the NEPHER_PICK_ORDER permutation even in non-typed mode (see
        # _update_command else branch). Default-on ("far" order); disabled only
        # when NEPHER_PICK_ORDER=none.
        self._pick_order_active: bool = _os.environ.get("NEPHER_PICK_ORDER", "far") != "none"

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

        # Per-env per-object goal tensors (N, M_active, 3/4).
        # In typed mode M_active < M; in standard mode M_active == M.
        self.goal_pos_w  = torch.zeros(self.num_envs, self._M_active, 3, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, self._M_active, 4, device=self.device)
        self.goal_quat_w[:, :, 0] = 1.0
        # Snapshot of goal positions at reset (the "home" goals before any bin drift).
        # Used for live-bin retargeting (NEPHER_LIVE_BIN): goals follow the container.
        self._goal_home_w = torch.zeros_like(self.goal_pos_w)
        self._live_bin = bool(_os.environ.get("NEPHER_LIVE_BIN"))

        # Per-object grasp metadata tensors (M,) – indexed by catalog index in
        # typed mode, or by pick-slot index in standard mode.
        M = self._M
        _pad = lambda lst, default: lst + [default] * max(0, M - len(lst))
        self._grasp_z_offsets  = torch.tensor(
            _pad(cfg.grasp_z_offsets,   cfg.grasp_z_offset_default), dtype=torch.float32, device=self.device
        )  # (M,)
        self._grasp_syms       = torch.tensor(
            _pad(cfg.grasp_syms,        cfg.grasp_sym_default),       dtype=torch.float32, device=self.device
        )  # (M,)
        self._grasp_yaw_offsets = torch.tensor(
            _pad(cfg.grasp_yaw_offsets, cfg.grasp_yaw_offset_default), dtype=torch.float32, device=self.device
        )  # (M,)
        self._grasp_offset_locals = torch.tensor(
            _pad(cfg.grasp_offset_locals, cfg.grasp_offset_local_default), dtype=torch.float32, device=self.device
        )  # (M, 3)

        # Per-env active catalog indices (N, M_active).
        # Default: identity mapping — slot m uses catalog object m.
        self._active_catalog_indices = torch.arange(
            self._M_active, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()  # (N, M_active)

        self._init_goals(torch.arange(self.num_envs, device=self.device))

        self._grip_command = torch.zeros(self.num_envs, 1, device=self.device)
        # `cmd_w` is the smoothed carrot pose sent to the frozen LL policy.
        self._cmd_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # `endpoint_w` is the static planner endpoint used for transition checks.
        self._endpoint_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_catalog_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._marker_pos_w  = torch.zeros(self.num_envs, 3, device=self.device)
        self._marker_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._marker_quat_w[:, 0] = 1.0
        self._marker_surface_z = cfg.table_surface_z + 0.5 * cfg.marker_thickness

        self.planner = PickPlacePlanner(
            num_envs          = self.num_envs,
            device            = self.device,
            num_objects       = self._M_active,
            hand_tcp_offset_z = cfg.hand_tcp_offset_z,
            pre_approach_z    = cfg.pre_approach_z,
            carry_z           = cfg.carry_z,
            grasp_z_offset    = cfg.grasp_z_offset_default,
            release_z_offset  = cfg.release_z_offset,
            retract_approach_z = cfg.retract_approach_z,
            pos_tol           = cfg.pos_tol,
            ang_tol           = cfg.ang_tol,
            pos_tol_approach  = cfg.pos_tol_approach,
            ang_tol_approach  = cfg.ang_tol_approach,
            pos_tol_grasp     = cfg.pos_tol_grasp,
            ang_tol_grasp     = cfg.ang_tol_grasp,
            pos_tol_transport = cfg.pos_tol_transport,
            ang_tol_transport = cfg.ang_tol_transport,
            pos_tol_place     = cfg.pos_tol_place,
            pos_tol_retract   = cfg.pos_tol_retract,
            ang_tol_retract   = cfg.ang_tol_retract,
            min_stage_dur     = cfg.min_stage_dur,
            grasp_hold_s      = cfg.grasp_hold_s,
            release_hold_s    = cfg.release_hold_s,
            max_retries       = cfg.max_retries,
            min_carry_cube_z  = cfg.min_carry_cube_z,
            grasp_secure_xy_tol = cfg.grasp_secure_xy_tol,
            wrist_soft_limit    = cfg.wrist_soft_limit,
            yaw_switch_cooldown = cfg.yaw_switch_cooldown,
            yaw_k_max           = cfg.yaw_k_max,
            pitch_cmd           = cfg.pitch_cmd,
            pitch_transport     = cfg.pitch_transport,
            container_drop      = cfg.container_drop,
            place_yaw_gate      = cfg.place_yaw_gate,
            max_step            = cfg.max_step,
            pre_grasp_settle_s   = cfg.pre_grasp_settle_s,
            pre_grasp_settle_ang = cfg.pre_grasp_settle_ang,
            lift_anchor_radius   = cfg.lift_anchor_radius,
            place_settle_s       = cfg.place_settle_s,
            place_settle_max     = cfg.place_settle_max,
            stall_pos_tol        = cfg.stall_pos_tol,
            stall_time_s         = cfg.stall_time_s,
            retreat_steps        = cfg.retreat_steps,
            retreat_z            = cfg.retreat_z,
            max_reach_retries    = cfg.max_reach_retries,
            carry_drop_gap       = cfg.carry_drop_gap,
            stage_escape_s       = cfg.stage_escape_s,
            stage_escape_pos_mult = cfg.stage_escape_pos_mult,
            place_verify_xy      = cfg.place_verify_xy,
            place_verify_yaw     = cfg.place_verify_yaw,
            max_place_retries    = cfg.max_place_retries,
            max_lower_retries    = cfg.max_lower_retries,
            hurry_after_s        = cfg.hurry_after_s,
            hurry_scale          = cfg.hurry_scale,
            container_retract_xy_offset = cfg.container_retract_xy_offset,
            skip_then_revisit         = cfg.skip_then_revisit,
            pre_grasp_yaw_tol         = cfg.pre_grasp_yaw_tol,
            freeze_yaw_before_descend = cfg.freeze_yaw_before_descend,
            grasp_yaw_flip_enabled    = cfg.grasp_yaw_flip_enabled,
            grasp_yaw_flip_after_retries = cfg.grasp_yaw_flip_after_retries,
            grasp_yaw_flip_rad        = cfg.grasp_yaw_flip_rad,
            grasp_miss_retreat        = cfg.grasp_miss_retreat,
            descend_xy_gate_z_margin  = cfg.descend_xy_gate_z_margin,
            keep_grasp_yaw_container  = cfg.keep_grasp_yaw_container,
            release_at_container_center = cfg.release_at_container_center,
            safe_release_above_rim    = cfg.safe_release_above_rim,
            table_z                   = cfg.table_surface_z,
        )

        self.metrics["position_error"]    = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["track_ok"]          = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage"]             = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage_elapsed_s"]   = torch.zeros(self.num_envs, device=self.device)
        self.metrics["retry_count"]       = torch.zeros(self.num_envs, device=self.device)
        self.metrics["task_idx"]          = torch.zeros(self.num_envs, device=self.device)

        self._step_count   = 0
        self._stuck_warned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_grasp_stuck_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._descend_stuck_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_stuck_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_miss_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._place_miss_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._skip_schedule_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._container_displaced_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if cfg.enable_log and not _LOG.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        if cfg.enable_log:
            _LOG.info(
                "[HL] planner params  grasp_z_bias=%.3f  retry_depth=%.3f  table_z=%.3f  "
                "grasp_z_clamp=%d  min_grasp_z=%.3f  pre_grasp_yaw_tol=%.3f  "
                "pre_grasp_settle_ang=%.3f  release_hold=%.3f  post_release_wait=%.3f  "
                "release_z_bias=%.3f  release_at_container_center=%d",
                self.planner.grasp_z_bias,
                self.planner.retry_depth_boost,
                self.planner.table_z,
                int(self.planner.enable_grasp_z_clamp),
                self.planner.min_grasp_z,
                self.planner.pre_grasp_yaw_tol,
                self.planner.pre_grasp_settle_ang,
                self.planner.release_hold_s,
                self.planner.post_release_wait_s,
                self.planner.release_z_bias,
                int(self.planner.release_at_container_center),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_goals(self, env_ids: torch.Tensor) -> None:
        """Initialise goal tensors from ``goal_pos_defaults`` and random yaw."""
        if env_ids.numel() == 0:
            return
        for m in range(self._M_active):
            default = self.cfg.goal_pos_defaults[m] if m < len(self.cfg.goal_pos_defaults) else self.cfg.goal_pos_defaults[-1]
            base = torch.tensor(default, dtype=torch.float32, device=self.device)
            offsets = sample_xyz_offsets(env_ids.numel(), {
                "x": self.cfg.ranges.pos_x,
                "y": self.cfg.ranges.pos_y,
                "z": self.cfg.ranges.pos_z,
            }, self.device)
            self.goal_pos_w[env_ids, m] = self._env.scene.env_origins[env_ids] + base + offsets
            goal_yaw = torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.ranges.yaw)
            zeros = torch.zeros_like(goal_yaw)
            self.goal_quat_w[env_ids, m] = quat_from_euler_xyz(zeros, zeros, goal_yaw)

    def set_goals_from_strategy(
        self,
        env_ids: torch.Tensor,
        goal_pos: torch.Tensor,   # (N, M_active, 3) robot-local frame
        goal_rot: torch.Tensor,   # (N, M_active, 4) wxyz
    ) -> None:
        """Store per-env per-object goals supplied by a scenario strategy or scatter event."""
        origins = self._env.scene.env_origins[env_ids]  # (N, 3)
        M = goal_pos.shape[1]
        self.goal_pos_w[env_ids, :M]  = origins.unsqueeze(1) + goal_pos
        self.goal_quat_w[env_ids, :M] = goal_rot
        # Snapshot home goals for live-bin retargeting.
        self._goal_home_w[env_ids, :M] = self.goal_pos_w[env_ids, :M]

    def set_active_objects_from_typed_scenario(
        self,
        env_ids: torch.Tensor,
        active_catalog_indices: torch.Tensor,   # (N, M_active) long
    ) -> None:
        """Update per-env active catalog assignments for the typed-scenario path.

        Called by ``reset_typed_objects_from_scenario`` after each episode reset.
        ``active_catalog_indices[n, m]`` is the catalog object index (0–C-1)
        that occupies pick slot ``m`` for environment ``n``.
        """
        self._active_catalog_indices[env_ids] = active_catalog_indices

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            "HLPoseCommand (PickPlacePlanner)\n"
            f"\tObjects          : {self.cfg.cube_names}\n"
            f"\tGoal defaults    : {self.cfg.goal_pos_defaults}\n"
            f"\tGoal ranges      : {self.cfg.ranges}\n"
            f"\tContainer drop   : {self.cfg.container_drop}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._log_episode_counters(ids)
        # Grasp metadata and initial cube orientation for slot 0 at reset.
        # Full N-size tensors required so planner.reset() can index with `ids`.
        if self._typed_mode:
            # In typed mode, index by the catalog object assigned to slot 0.
            arange_n = torch.arange(self.num_envs, device=self.device)
            cat_idx_slot0 = self._active_catalog_indices[:, 0]  # (N,)
            all_quats = torch.stack([c.data.root_quat_w for c in self._cubes], dim=1)
            cube_quat_w = all_quats[arange_n, cat_idx_slot0]  # (N, 4)
            sym0     = self._grasp_syms[cat_idx_slot0]          # (N,)
            yaw_off0 = self._grasp_yaw_offsets[cat_idx_slot0]   # (N,)
        elif self._pick_order_active:
            # Non-typed mode but a pick-order permutation is active: seed reset
            # metadata from the object that now occupies slot 0.
            arange_n = torch.arange(self.num_envs, device=self.device)
            cat0 = self._active_catalog_indices[:, 0]  # (N,)
            all_quats = torch.stack([c.data.root_quat_w for c in self._cubes], dim=1)
            cube_quat_w = all_quats[arange_n, cat0]
            sym0     = self._grasp_syms[cat0]
            yaw_off0 = self._grasp_yaw_offsets[cat0]
        else:
            cube_quat_w = self._cubes[0].data.root_quat_w
            sym0     = self._grasp_syms[0].unsqueeze(0).expand(self.num_envs)
            yaw_off0 = self._grasp_yaw_offsets[0].unsqueeze(0).expand(self.num_envs)

        self.planner.reset(
            ids,
            ee_pos_w      = self.robot.data.body_pos_w[:,  self._body_idx],
            ee_quat_w     = self.robot.data.body_quat_w[:, self._body_idx],
            cube_quat_w   = cube_quat_w,
            grasp_sym     = sym0,
            grasp_yaw_off = yaw_off0,
        )
        self._grip_command[ids]  = 0.0
        self._stuck_warned[ids] = False
        self._episode_started[ids] = True
        self._pre_grasp_stuck_count[ids] = 0
        self._descend_stuck_count[ids] = 0
        self._grasp_stuck_count[ids] = 0
        self._grasp_miss_count[ids] = 0
        self._place_miss_count[ids] = 0
        self._skip_schedule_count[ids] = 0
        self._container_displaced_count[ids] = 0

        if self.cfg.enable_log:
            for i in ids.tolist():
                _LOG.info(
                    "[HL] env %d reset  cube0_w=%s  goal0_w=%s",
                    i,
                    _fmt_xyz(self._cubes[0].data.root_pos_w[i]),
                    _fmt_xyz(self.goal_pos_w[i, 0]),
                )

    def _update_command(self) -> None:
        # Live-bin retargeting: shift goals by the container's drift from home so
        # placement follows a shoved bin (objects still land in the live bin).
        if self._live_bin and getattr(self.cfg, "container_drop", False):
            try:
                container = self._env.scene["container"]
                if hasattr(container.data, "root_pos_w") and hasattr(self._env, "_hl_container_home_w"):
                    drift = container.data.root_pos_w[:, :2] - self._env._hl_container_home_w[:, :2]
                    self.goal_pos_w[:, :, :2] = self._goal_home_w[:, :, :2] + drift.unsqueeze(1)
            except (KeyError, AttributeError):
                pass

        ee_pos_w  = self.robot.data.body_pos_w[:,  self._body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._body_idx]

        task_idx = self.planner._task_idx              # (N,), values 0 to M_active-1
        arange   = torch.arange(self.num_envs, device=self.device)

        if self._typed_mode:
            # Gather positions/quaternions from ALL catalog objects.
            all_pos  = torch.stack([c.data.root_pos_w  for c in self._cubes], dim=1)  # (N, C, 3)
            all_quat = torch.stack([c.data.root_quat_w for c in self._cubes], dim=1)  # (N, C, 4)
            # Map pick slot → catalog index for each env.
            cat_for_task = self._active_catalog_indices[arange, task_idx]             # (N,)
            current_cube_pos  = all_pos[arange,  cat_for_task]   # (N, 3)
            current_cube_quat = all_quat[arange, cat_for_task]   # (N, 4)
            self._current_catalog_idx.copy_(cat_for_task)
            # Grasp metadata indexed by catalog index (not pick slot).
            cur_grasp_z_off   = self._grasp_z_offsets[cat_for_task]    # (N,)
            cur_grasp_sym     = self._grasp_syms[cat_for_task]          # (N,)
            cur_grasp_yaw_off = self._grasp_yaw_offsets[cat_for_task]  # (N,)
            cur_grasp_offset_local = self._grasp_offset_locals[cat_for_task]  # (N, 3)
        else:
            cube_pos_all  = torch.stack([c.data.root_pos_w  for c in self._cubes], dim=1)  # (N, M, 3)
            cube_quat_all = torch.stack([c.data.root_quat_w for c in self._cubes], dim=1)  # (N, M, 4)
            # PICK-ORDER: in non-typed mode (all catalog objects active) the
            # planner normally targets objects in fixed slot order (task_idx).
            # When NEPHER_PICK_ORDER is set the reset event stores a permutation
            # in _active_catalog_indices; honour it here so the reorder actually
            # drives which object is grasped at each step. Identity when unset.
            if self._pick_order_active:
                sel = self._active_catalog_indices[arange, task_idx]   # (N,) permuted catalog idx
            else:
                sel = task_idx
            current_cube_pos  = cube_pos_all[arange,  sel]
            current_cube_quat = cube_quat_all[arange, sel]
            self._current_catalog_idx.copy_(sel)
            # Grasp metadata indexed by the (possibly permuted) slot.
            cur_grasp_z_off   = self._grasp_z_offsets[sel]    # (N,)
            cur_grasp_sym     = self._grasp_syms[sel]          # (N,)
            cur_grasp_yaw_off = self._grasp_yaw_offsets[sel]  # (N,)
            cur_grasp_offset_local = self._grasp_offset_locals[sel]  # (N, 3)

        current_goal_pos  = self.goal_pos_w[arange,  task_idx]
        current_goal_quat = self.goal_quat_w[arange, task_idx]

        wrist_angle = self.robot.data.joint_pos[:, self._wrist_joint_idx]
        cmd_pos_w, end_quat_w, grip = self.planner.step(
            cube_pos_w    = current_cube_pos,
            cube_quat_w   = current_cube_quat,
            goal_pos_w    = current_goal_pos,
            goal_quat_w   = current_goal_quat,
            ee_pos_w      = ee_pos_w,
            ee_quat_w     = ee_quat_w,
            dt            = self._env.step_dt,
            wrist_angle   = wrist_angle,
            grasp_z_off   = cur_grasp_z_off,
            grasp_sym     = cur_grasp_sym,
            grasp_yaw_off = cur_grasp_yaw_off,
            grasp_offset_local = cur_grasp_offset_local,
            catalog_idx   = self._current_catalog_idx,
        )

        end_pos_b, end_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w,
            cmd_pos_w, end_quat_w,
        )
        self.pose_command_b[:, :3] = end_pos_b
        self.pose_command_b[:, 3:] = end_quat_b
        self._grip_command[:, 0]   = grip
        self._cmd_pos_w.copy_(cmd_pos_w)
        self._endpoint_pos_w.copy_(self.planner._target_pos)

        if self.cfg.enable_log:
            self._log_planner_events(ee_pos_w, current_cube_pos)

        self._step_count += 1

    def _update_metrics(self) -> None:
        ee_pos_w = self.robot.data.body_pos_w[:, self._body_idx]
        p = self.planner

        self.metrics["position_error"]    = torch.norm(ee_pos_w - self._endpoint_pos_w, dim=-1)
        self.metrics["orientation_error"] = p._ang_err
        self.metrics["track_ok"]          = p._track_ok.float()
        self.metrics["stage"]             = p.stage.float()
        self.metrics["stage_elapsed_s"]   = p._elapsed
        self.metrics["retry_count"]       = p._retry_count.float()
        self.metrics["task_idx"]          = p._task_idx.float()

        if self.cfg.enable_log and self._step_count % self.cfg.log_interval == 0:
            self._log_status_snapshot()

    def _log_planner_events(
        self,
        ee_pos_w: torch.Tensor,
        current_cube_pos: torch.Tensor,
    ) -> None:
        p = self.planner

        if p._grasp_miss.any():
            self._grasp_miss_count += p._grasp_miss.long()
            for i in torch.where(p._grasp_miss)[0].tolist():
                cz = current_cube_pos[i, 2].item()
                req_lift = p._required_lift_z[i].item()
                cat_idx = int(self._current_catalog_idx[i].item())
                obj_name = self.cfg.cube_names[cat_idx] if 0 <= cat_idx < len(self.cfg.cube_names) else "unknown"
                task_idx = int(p._task_idx[i].item())
                reason = "carry_missed" if bool(p._grasp_miss_is_carry[i].item()) else "grasp_failed"
                off = p._cur_grasp_offset_local[i]
                retry_local = p._retry_pattern_local[i]
                max_retries = int(p._cur_max_grasp_retries[i].item())
                _LOG.warning(
                    "[HL] env %d [obj %d/%d] grasp_miss  reason=%s  catalog_idx=%d  object_name=%s  "
                    "cube_z=%.3f  required_lift_z=%.3f  retry=%d/%d  target_grasp_z=%.3f  ee_z=%.3f  "
                    "grasp_offset_local=(%.3f, %.3f, %.3f)  retry_pattern={idx:%d, dxy_local:(%.3f, %.3f), dz:%.3f, dyaw:%.4f}  "
                    "extra_depth=%.3f  -> PRE_GRASP",
                    i, task_idx, self._M_active - 1, reason, cat_idx, obj_name,
                    cz, req_lift, p._retry_count[i].item(), max_retries,
                    p._target_grasp_z[i].item(), ee_pos_w[i, 2].item(),
                    off[0].item(), off[1].item(), off[2].item(),
                    int(p._retry_pattern_idx[i].item()),
                    retry_local[0].item(), retry_local[1].item(), retry_local[2].item(),
                    p._retry_pattern_dyaw[i].item(),
                    p._catalog_extra_depth[i].item(),
                )

        if getattr(p, "_skip_event", None) is not None and p._skip_event.any():
            self._skip_schedule_count += p._skip_event.long()
            for i in torch.where(p._skip_event)[0].tolist():
                placed = int(p._object_placed[i].sum().item())
                deferred = int(p._object_deferred[i].sum().item())
                abandoned = int(p._object_abandoned[i].sum().item())
                cat_idx = int(self._current_catalog_idx[i].item())
                obj_name = self.cfg.cube_names[cat_idx] if 0 <= cat_idx < len(self.cfg.cube_names) else "unknown"
                all_patterns_tried = bool(p._retry_count[i].item() >= p._cur_max_grasp_retries[i].item())
                _LOG.warning(
                    "[HL] env %d skip_schedule  placed=%d deferred=%d abandoned=%d  catalog_idx=%d  "
                    "object_name=%s  all_retry_patterns_tried=%d  next_obj=%d",
                    i, placed, deferred, abandoned, cat_idx, obj_name, int(all_patterns_tried),
                    int(p._task_idx[i].item()),
                )

        if p._place_miss.any():
            self._place_miss_count += p._place_miss.long()
            for i in torch.where(p._place_miss)[0].tolist():
                _LOG.warning(
                    "[HL] env %d [obj %d/%d] place_miss  retry=%d/%d  -> re-pick",
                    i, int(p._task_idx[i].item()), self._M_active - 1,
                    p._place_retries[i].item(), p.max_place_retries,
                )

        for i in torch.where(p._stage_changed)[0].tolist():
            stage    = int(p.stage[i].item())
            task_idx = int(p._task_idx[i].item())
            _LOG.info(
                "[HL] env %d [obj %d/%d] stage -> %d %s  endpoint_w=%s  cmd_w=%s  ee_w=%s  grip=%.0f",
                i, task_idx, self._M_active - 1,
                stage, STAGE_NAMES[stage], _fmt_xyz(self._endpoint_pos_w[i]),
                _fmt_xyz(self._cmd_pos_w[i]), _fmt_xyz(ee_pos_w[i]), self._grip_command[i, 0].item(),
            )
            if stage == int(Stage.RETRACT):
                _LOG.info(
                    "[HL] env %d retract_state  retract_phase=%s  release_xy=%s  current_xy=(%.3f, %.3f)  "
                    "current_z=%.3f  clear_z=%.3f",
                    i,
                    "xy_clear" if bool(p._retract_phase_xy_clear[i].item()) else "up_only",
                    _fmt_xy(p._retract_release_xy[i]),
                    ee_pos_w[i, 0].item(), ee_pos_w[i, 1].item(),
                    ee_pos_w[i, 2].item(),
                    p._retract_clear_z[i].item(),
                )
            self._stuck_warned[i] = False

        stuck = (~p._track_ok) & (p._stuck_elapsed > self.cfg.stuck_warn_s) & ~self._stuck_warned
        for i in torch.where(stuck)[0].tolist():
            stage    = int(p.stage[i].item())
            task_idx = int(p._task_idx[i].item())
            if stage == 0:
                self._pre_grasp_stuck_count[i] += 1
            elif stage == 1:
                self._descend_stuck_count[i] += 1
            elif stage == 2:
                self._grasp_stuck_count[i] += 1
            _LOG.warning(
                "[HL] env %d [obj %d/%d] stuck  stage=%d %s  t=%.2fs  "
                "endpoint_err=%.4f  cmd_err=%.4f  ang_err=%.4f  (tol pos=%.3f ang=%.3f)  "
                "endpoint_w=%s  cmd_w=%s  ee_w=%s  retry=%d",
                i, task_idx, self._M_active - 1,
                stage, STAGE_NAMES[stage], p._stuck_elapsed[i].item(),
                p._endpoint_err[i].item(), p._cmd_err[i].item(), p._ang_err[i].item(),
                p._pos_tol_eff[i].item(), p._ang_tol_eff[i].item(),
                _fmt_xyz(self._endpoint_pos_w[i]), _fmt_xyz(self._cmd_pos_w[i]), _fmt_xyz(ee_pos_w[i]),
                int(p._retry_count[i].item()),
            )
            if stage == int(Stage.RETRACT):
                _LOG.warning(
                    "[HL] env %d retract_state  retract_phase=%s  release_xy=%s  current_xy=(%.3f, %.3f)  "
                    "current_z=%.3f  clear_z=%.3f",
                    i,
                    "xy_clear" if bool(p._retract_phase_xy_clear[i].item()) else "up_only",
                    _fmt_xy(p._retract_release_xy[i]),
                    ee_pos_w[i, 0].item(), ee_pos_w[i, 1].item(),
                    ee_pos_w[i, 2].item(),
                    p._retract_clear_z[i].item(),
                )
            endpoint_err_now = torch.norm(ee_pos_w[i] - self._endpoint_pos_w[i]).item()
            if abs(endpoint_err_now - p._pos_err[i].item()) > 0.10 and endpoint_err_now < 0.05:
                _LOG.warning(
                    "[HL] env %d pos_err_guard  stored=%.4f  recomputed_endpoint_err=%.4f  "
                    "endpoint_w=%s  ee_w=%s",
                    i,
                    p._pos_err[i].item(),
                    endpoint_err_now,
                    _fmt_xyz(self._endpoint_pos_w[i]),
                    _fmt_xyz(ee_pos_w[i]),
                )
            self._stuck_warned[i] = True

    def _log_status_snapshot(self) -> None:
        p = self.planner
        for i in _log_env_ids(self.num_envs, self.cfg.log_env_id):
            stage    = int(p.stage[i].item())
            task_idx = int(p._task_idx[i].item())
            _LOG.info(
                "[HL] env %d [obj %d/%d]  stage=%d %s  t=%.2fs  pos_err=%.4f  "
                "cmd_err=%.4f  ang_err=%.4f  track_ok=%d  retries=%d  cube_z=%.3f",
                i, task_idx, self._M_active - 1,
                stage, STAGE_NAMES[stage], p._elapsed[i].item(),
                p._endpoint_err[i].item(), p._cmd_err[i].item(), p._ang_err[i].item(),
                int(p._track_ok[i].item()), p._retry_count[i].item(),
                self._get_cube_z_w(i, task_idx),
            )

    def _log_episode_counters(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.enable_log:
            return
        p = self.planner
        for i in env_ids.tolist():
            if not bool(self._episode_started[i].item()) or p._episode_t[i].item() <= 0.0:
                continue
            container_displaced = self._detect_container_displaced(i)
            if container_displaced:
                self._container_displaced_count[i] = 1
            _LOG.info(
                "[HL] env %d episode_counters  pre_grasp_stuck_count=%d  descend_stuck_count=%d  "
                "grasp_stuck_count=%d  grasp_miss_count=%d  place_miss_count=%d  "
                "container_displaced_count=%d  skip_schedule_count=%d  objects_placed=%d",
                i,
                int(self._pre_grasp_stuck_count[i].item()),
                int(self._descend_stuck_count[i].item()),
                int(self._grasp_stuck_count[i].item()),
                int(self._grasp_miss_count[i].item()),
                int(self._place_miss_count[i].item()),
                int(self._container_displaced_count[i].item()),
                int(self._skip_schedule_count[i].item()),
                int(p._object_placed[i].sum().item()),
            )
            self._episode_started[i] = False

    def _detect_container_displaced(self, env_idx: int) -> bool:
        try:
            container = self._env.scene["container"]
        except KeyError:
            return False
        if not hasattr(container, "data") or not hasattr(container.data, "root_pos_w"):
            return False
        if not hasattr(self._env, "_hl_container_home_w") or not hasattr(self._env, "_hl_container_home_quat"):
            return False
        pos_w = container.data.root_pos_w[env_idx]
        quat_w = container.data.root_quat_w[env_idx]
        home_pos = self._env._hl_container_home_w[env_idx]
        home_quat = self._env._hl_container_home_quat[env_idx]
        xy_disp = torch.norm(pos_w[:2] - home_pos[:2]).item()
        _, _, yaw = euler_xyz_from_quat(quat_w.unsqueeze(0))
        _, _, home_yaw = euler_xyz_from_quat(home_quat.unsqueeze(0))
        yaw_disp = torch.abs(torch.atan2(torch.sin(yaw - home_yaw), torch.cos(yaw - home_yaw))).item()
        return (xy_disp > 0.02) or (yaw_disp > 0.1)

    def _get_cube_z_w(self, env_idx: int, task_idx_val: int) -> float:
        """Return the world-Z of the current task object for a given env (for logging)."""
        if self._typed_mode:
            cat_idx = int(self._active_catalog_indices[env_idx, task_idx_val].item())
            return self._cubes[cat_idx].data.root_pos_w[env_idx, 2].item()
        return self._cubes[task_idx_val].data.root_pos_w[env_idx, 2].item()

    def _goal_marker_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Visualise the current task's goal (flat marker at table/bin height)."""
        arange   = torch.arange(self.num_envs, device=self.device)
        task_idx = self.planner._task_idx
        cur_goal_pos  = self.goal_pos_w[arange,  task_idx]
        cur_goal_quat = self.goal_quat_w[arange, task_idx]

        self._marker_pos_w[:, :2] = cur_goal_pos[:, :2]
        self._marker_pos_w[:, 2]  = self._marker_surface_z

        _, _, yaw = euler_xyz_from_quat(cur_goal_quat)
        zeros = torch.zeros_like(yaw)
        self._marker_quat_w[:] = quat_from_euler_xyz(zeros, zeros, yaw)
        return self._marker_pos_w, self._marker_quat_w

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not hasattr(self, "goal_pose_visualizer"):
            return
        marker_pos, marker_quat = self._goal_marker_pose_w()
        self.goal_pose_visualizer.visualize(marker_pos, marker_quat)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_xyz(v: torch.Tensor) -> str:
    return f"({v[0].item():.3f}, {v[1].item():.3f}, {v[2].item():.3f})"


def _fmt_xy(v: torch.Tensor) -> str:
    return f"({v[0].item():.3f}, {v[1].item():.3f})"


def _log_env_ids(num_envs: int, log_env_id: int) -> list[int]:
    if log_env_id < 0:
        return list(range(num_envs))
    return [min(log_env_id, num_envs - 1)]


def make_goal_marker_cfg(cube_size: float, thickness: float) -> VisualizationMarkersCfg:
    """Thin rectangular pad sized to the goal footprint, lying flat on the table."""
    return VisualizationMarkersCfg(
        markers={
            "pad": sim_utils.CuboidCfg(
                size=(cube_size, cube_size, thickness),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.85, 0.30),
                    emissive_color=(0.0, 0.35, 0.08),
                    opacity=0.60,
                ),
            ),
        },
    ).replace(prim_path="/Visuals/HL/goal_pose")


def make_container_marker_cfg(half_x: float, half_y: float, thickness: float) -> VisualizationMarkersCfg:
    """Thin rectangle showing the container opening on the table surface."""
    return VisualizationMarkersCfg(
        markers={
            "bin_opening": sim_utils.CuboidCfg(
                size=(half_x * 2.0, half_y * 2.0, thickness),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.45, 0.90),
                    emissive_color=(0.02, 0.10, 0.30),
                    opacity=0.55,
                ),
            ),
        },
    ).replace(prim_path="/Visuals/HL/goal_pose")


HL_GOAL_MARKER_CFG = make_goal_marker_cfg(cube_size=0.04, thickness=0.002)


@configclass
class HLPoseCommandCfg(CommandTermCfg):
    """Configuration for :class:`HLPoseCommand`."""

    class_type: type = HLPoseCommand

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = True

    robot_name: str = "robot"
    body_name:  str = "panda_hand"

    # List of object scene-dict names in pick_order.
    cube_names: list[str] = ["cube"]

    # Default goal positions per object.
    goal_pos_defaults: list[tuple[float, float, float]] = [(0.55, -0.30, 0.055)]

    # Typed-scenario mode: set to a positive integer < len(cube_names) to enable.
    # In typed mode, each episode only ``num_active`` of the ``len(cube_names)``
    # catalog objects are active; the rest are parked by the reset event.
    # ``0`` (default) disables typed mode: all cube_names objects are tracked.
    num_active: int = 0

    # Goal marker
    table_surface_z: float = 0.03
    marker_thickness: float = 0.002
    cube_size_xy: float = 0.04

    # Per-object grasp metadata (length M; padded with defaults if shorter).
    grasp_z_offsets:    list[float] = []   # TCP below object centre at grasp (m)
    grasp_syms:         list[float] = []   # grasp yaw symmetry (rad); 0=rotationally symmetric
    grasp_yaw_offsets:  list[float] = []   # constant yaw offset (rad)
    grasp_offset_locals: list[tuple[float, float, float]] = []  # object-local xyz bias

    # Scalar defaults used when the per-object lists are shorter than M.
    grasp_z_offset_default:   float = -0.01
    grasp_sym_default:        float = math.pi / 2
    grasp_yaw_offset_default: float = 0.0
    grasp_offset_local_default: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Container drop mode: disables yaw gate; goal Z = bin rim target.
    container_drop: bool = False

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (-0.10, 0.10)
        pos_y: tuple[float, float] = (-0.20, 0.20)
        pos_z: tuple[float, float] = (0.0, 0.0)
        yaw:   tuple[float, float] = (-3.14159, 3.14159)

    ranges: Ranges = Ranges()

    goal_pose_visualizer_cfg = HL_GOAL_MARKER_CFG

    # PickPlacePlanner parameters
    hand_tcp_offset_z: float = 0.107
    pre_approach_z:    float = 0.10
    carry_z:           float = 0.22   # raised to clear container rim (~11 cm)
    grasp_z_offset:    float = -0.01  # kept for backward compat; use grasp_z_offset_default
    release_z_offset:  float = -0.020
    retract_approach_z: float = 0.07
    pos_tol:           float = 0.045
    ang_tol:           float = 0.45
    pos_tol_approach:  float = 0.055
    ang_tol_approach:  float = 0.40
    pos_tol_grasp:     float = 0.080
    ang_tol_grasp:     float = 0.70
    pos_tol_transport: float = 0.045
    ang_tol_transport: float = 0.45
    pos_tol_place:     float = 0.055  # relaxed for container drop
    pos_tol_retract:   float = 0.045
    ang_tol_retract:   float = 0.45
    min_stage_dur:     float = 0.12
    grasp_hold_s:      float = 0.40
    release_hold_s:    float = 0.20
    max_retries:       int   = 3
    min_carry_cube_z:  float = 0.08
    grasp_secure_xy_tol: float = 0.06
    wrist_joint_name:    str   = "panda_joint7"
    wrist_soft_limit:    float = 2.5
    yaw_switch_cooldown: int   = 20
    yaw_k_max:           int   = 0
    pitch_cmd:           float = 3.14159265
    pitch_transport:     float = 3.14159265
    place_yaw_gate:      float = 10.0   # effectively disabled in container mode
    max_step:            float = 0.065
    pre_grasp_settle_s:   float = 1.5
    pre_grasp_settle_ang: float = 0.75
    lift_anchor_radius:   float = 0.05
    place_settle_s:       float = 0.3
    place_settle_max:     float = 0.8
    stall_pos_tol:        float = 0.06
    stall_time_s:         float = 4.0
    retreat_steps:        int   = 25
    retreat_z:            float = 0.40
    max_reach_retries:    int   = 0
    carry_drop_gap:       float = 0.04
    stage_escape_s:       float = 1.5
    stage_escape_pos_mult: float = 3.0
    place_verify_xy:      float = 0.18   # bin half-extent; disabled for on-table yaw check
    place_verify_yaw:     float = 10.0   # disabled in container mode
    max_place_retries:    int   = 1
    max_lower_retries:    int   = 1
    hurry_after_s:        float = 25.0
    hurry_scale:          float = 0.6
    container_retract_xy_offset: tuple[float, float] = (-0.05, 0.12)

    # Skip-then-revisit + grasp approach hardening.
    skip_then_revisit: bool = True
    pre_grasp_yaw_tol: float = 0.35
    freeze_yaw_before_descend: bool = True
    grasp_yaw_flip_enabled: bool = True
    grasp_yaw_flip_after_retries: int = 1
    grasp_yaw_flip_rad: float = math.pi / 2
    grasp_miss_retreat: bool = True
    descend_xy_gate_z_margin: float = 0.08
    keep_grasp_yaw_container: bool = True
    release_at_container_center: bool = False
    safe_release_above_rim: bool = True

    enable_log:    bool  = False
    log_interval:  int   = 100
    log_env_id:    int   = 0
    stuck_warn_s:  float = 3.0


# ---------------------------------------------------------------------------


class HLGripCommand(CommandTerm):
    """Mirrors the grip value from ``HLPoseCommand`` into the ``grip_cmd`` slot."""

    cfg: HLGripCommandCfg

    def __init__(self, cfg: HLGripCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._grip_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._grip_command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self._grip_command[env_ids] = 0.0

    def _update_command(self) -> None:
        pose_term: HLPoseCommand = self._env.command_manager.get_term(self.cfg.pose_cmd_name)
        self._grip_command.copy_(pose_term._grip_command)

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class HLGripCommandCfg(CommandTermCfg):
    """Configuration for :class:`HLGripCommand`."""

    class_type: type = HLGripCommand

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    pose_cmd_name: str = "ee_pose"
