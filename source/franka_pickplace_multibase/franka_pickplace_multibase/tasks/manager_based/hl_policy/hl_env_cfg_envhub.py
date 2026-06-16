# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EnvHub integration for the HL pick-and-place environment.

Loads a manipulation preset from the Nepher envhub and wires its scene
(table, objects, lighting, workspace) into the HL policy environment.

Multi-object sequential pick-and-place
--------------------------------------
When the preset is a :class:`PickAndPlacePresetCfg` with a
:class:`PrebakedScenarioStrategy`, this module:

* Sorts ``preset.objects`` by ``pick_order``.
* Wires all object names into ``HLPoseCommandCfg.cube_names``.
* Passes the strategy into the ``reset_objects_and_goals`` event.
* Wires all object scene-entity configs into the
  ``all_objects_reached_goals`` termination.
* Falls back to single-object ``cube_reached_goal`` / ``reset_cube_and_goal_poses``
  when the preset has no strategy (legacy / random-goal presets).

Usage::

    python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0
"""

from __future__ import annotations

from typing import Any

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from franka_pickplace_multibase.tasks.manager_based.hl_policy.hl_env_cfg import (
    GOAL_POS_DEFAULT,
    HLEnvCfg,
)
from franka_pickplace_multibase.tasks.manager_based.ll_policy.ll_env_cfg import LLSceneCfg
import franka_pickplace_multibase.tasks.manager_based.hl_policy.mdp as mdp


# ---------------------------------------------------------------------------
# Scene construction helpers
# ---------------------------------------------------------------------------


def _create_scene_class(base_class: type, name: str, **attrs: Any) -> type:
    """Dynamically create a ``@configclass`` scene subclass with extra attributes."""
    annotations: dict[str, type] = {}
    for attr_name, val in attrs.items():
        annotations[attr_name] = AssetBaseCfg
    class_attrs: dict[str, Any] = {"__annotations__": annotations, **attrs}
    return configclass(type(name, (base_class,), class_attrs))


def _build_envhub_scene(
    base_scene: InteractiveSceneCfg,
    preset: Any,
) -> InteractiveSceneCfg:
    """Construct a ``LLSceneCfg``-based scene augmented with preset content.

    - Objects from ``preset.get_object_cfgs()`` are injected as new attrs.
    - Lights from ``preset.get_light_cfgs()`` override the default dome light.
    - Table from ``preset.get_table_cfg()`` overrides the default table (if set).
    """
    attrs: dict[str, Any] = {}

    for obj_name, obj_cfg in preset.get_object_cfgs().items():
        attrs[obj_name] = obj_cfg

    for light_name, light_cfg in preset.get_light_cfgs().items():
        key = light_name.lower().replace("-", "_").replace(" ", "_")
        attrs[key] = light_cfg

    table_cfg = preset.get_table_cfg()
    if table_cfg is not None:
        attrs["table"] = table_cfg

    SceneCls = _create_scene_class(LLSceneCfg, "PresetHLSceneCfg", **attrs)
    env_spacing = max(base_scene.env_spacing, getattr(preset, "env_spacing", 2.5))
    return SceneCls(num_envs=base_scene.num_envs, env_spacing=env_spacing)


# ---------------------------------------------------------------------------
# HL Envhub environment configurations
# ---------------------------------------------------------------------------


@configclass
class HLEnvCfg_Envhub(HLEnvCfg):
    """HL pick-and-place environment backed by a Nepher envhub manipulation preset.

    ``env_id`` and ``scene_id`` identify the envhub preset.  On ``__post_init__``
    the preset is fetched and applied:

    - Scene rebuilt from ``LLSceneCfg`` + preset objects / lights / table.
    - For :class:`PickAndPlacePresetCfg` presets with a strategy:
        - Objects sorted by ``pick_order``; all wired into ``cube_names``.
        - Strategy passed to ``reset_objects_and_goals`` event.
        - ``all_objects_reached_goals`` termination wired with all objects.
    - Fallback (single-object / random-goal presets):
        - Legacy ``reset_cube_and_goal_poses`` + ``cube_reached_goal`` path.
    """

    env_id: str = "franka-pickplace-base-sample"
    """Nepher envhub environment identifier."""

    scene_id: str | int = 0
    """Scene index or name inside the envhub environment manifest."""

    _preset: Any = None

    def __post_init__(self) -> None:
        scene_id_value = self.scene_id
        super().__post_init__()
        self.scene_id = scene_id_value
        self._apply_envhub_preset()

    # ------------------------------------------------------------------

    def _apply_envhub_preset(self) -> None:
        """Fetch the preset and rewire all env components."""
        if not self.env_id or self._preset is not None:
            return

        from nepher import load_env, load_scene

        env_manifest = load_env(self.env_id, category="manipulation")
        preset = load_scene(env_manifest, self.scene_id, category="manipulation")
        self._preset = preset

        # ---- Scene ----
        self.scene = _build_envhub_scene(self.scene, preset)

        # ---- Detect preset type ----
        objects = getattr(preset, "objects", []) or []
        goals   = getattr(preset, "goals",   []) or []
        strategy = getattr(preset, "position_strategy", None)

        if strategy is not None and len(objects) > 0:
            # Multi-object pick-and-place with PrebakedScenarioStrategy.
            self._wire_multi_object(preset, objects, goals, strategy)
        else:
            # Legacy single-object fallback.
            self._wire_single_object(preset, objects, goals)

        # ---- Episode length ----
        if hasattr(preset, "max_episode_length_s"):
            self.episode_length_s = preset.max_episode_length_s

    # ------------------------------------------------------------------
    # Multi-object wiring (PickAndPlacePresetCfg + strategy)
    # ------------------------------------------------------------------

    def _wire_multi_object(self, preset, objects, goals, strategy) -> None:
        """Wire multi-object sequential pick-and-place from a PrebakedScenarioStrategy."""
        # Sort objects by pick_order.
        sorted_objects = sorted(objects, key=lambda o: getattr(o, "pick_order", 0))
        sorted_names   = [o.name for o in sorted_objects]
        M = len(sorted_names)

        # Derive thresholds from the first goal descriptor (uniform for now).
        goal_map = {g.target_object: g for g in goals} if goals else {}
        first_goal = goal_map.get(sorted_names[0])
        pos_thresh  = getattr(first_goal, "success_threshold_pos",   0.02) if first_goal else 0.02
        ang_thresh  = getattr(first_goal, "success_threshold_ang",   0.10) if first_goal else 0.10
        yaw_thresh  = getattr(first_goal, "success_threshold_yaw",   0.10) if first_goal else 0.10
        grip_thresh = getattr(first_goal, "grip_open_threshold",     0.8)  if first_goal else 0.8
        dwell_s     = getattr(first_goal, "success_dwell_s",         0.5)  if first_goal else 0.5

        # ---- Commands ----
        # Reset container_drop: EnvHub uses precise point-goal placement, not bin drop.
        self.commands.ee_pose.container_drop     = False
        self.commands.ee_pose.cube_names         = sorted_names
        self.commands.ee_pose.goal_pos_defaults  = [(0.60, 0.0, 0.055)] * M

        # ---- Events: replace with multi-object version ----
        self.events.reset_cube_and_goal_poses = EventTerm(
            func=mdp.reset_objects_and_goals,
            mode="reset",
            params={
                "object_cfgs":       [SceneEntityCfg(name) for name in sorted_names],
                "scenario_strategy": strategy,
                "pose_cmd_name":     "ee_pose",
            },
        )

        # ---- Terminations: replace with multi-object version ----
        self.terminations.cube_at_goal = DoneTerm(
            func=mdp.all_objects_reached_goals,
            params={
                "cube_cfgs":           [SceneEntityCfg(name) for name in sorted_names],
                "robot_cfg":           SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
                "pose_cmd_name":       "ee_pose",
                "pos_threshold":       pos_thresh,
                "ang_threshold":       ang_thresh,
                "yaw_threshold":       yaw_thresh,
                "grip_open_threshold": grip_thresh,
                "success_dwell_s":     dwell_s,
                "enable_log":          True,
            },
        )

        # cube_fell: monitor only the first object (primary failure signal).
        # any_object_fell takes object_cfgs list, not a single asset_cfg.
        self.terminations.cube_fell = DoneTerm(
            func=mdp.any_object_fell,
            params={
                "object_cfgs":    [SceneEntityCfg(sorted_names[0])],
                "minimum_height": -0.05,
            },
        )

    # ------------------------------------------------------------------
    # Single-object fallback (legacy random-goal presets)
    # ------------------------------------------------------------------

    def _wire_single_object(self, preset, objects, goals) -> None:
        """Wire the legacy single-object pick-and-place path."""
        from nepher.env_cfgs.manipulation.preset_mani_cfg import (
            ManipulationGoalCfg,
            ManipulationObjectCfg,
        )

        # Resolve primary object / goal.
        if goals:
            goal = goals[0]
            primary_obj = next(
                (o for o in objects if o.name == goal.target_object),
                objects[0] if objects else None,
            )
        else:
            goal = ManipulationGoalCfg(
                type="place",
                target_object=objects[0].name if objects else "cube",
                goal_pos_default=GOAL_POS_DEFAULT,
            )
            primary_obj = objects[0] if objects else ManipulationObjectCfg(name="cube", usd_path="")

        # ---- Commands ----
        # Reset container_drop: EnvHub uses precise point-goal placement, not bin drop.
        self.commands.ee_pose.container_drop    = False
        self.commands.ee_pose.cube_names        = [primary_obj.name]
        self.commands.ee_pose.goal_pos_defaults = [goal.goal_pos_default or GOAL_POS_DEFAULT]
        if goal.goal_pose_range is not None:
            r = goal.goal_pose_range
            self.commands.ee_pose.ranges.pos_x = r.get("x", (-0.10, 0.10))
            self.commands.ee_pose.ranges.pos_y = r.get("y", (-0.20, 0.20))
            self.commands.ee_pose.ranges.pos_z = r.get("z", (0.0, 0.0))
        self.commands.ee_pose.ranges.yaw = goal.goal_yaw_range

        # ---- Events: replace entirely (the base now uses scatter event with different params) ----
        pose_range = primary_obj.spawn_range if primary_obj.spawn_range is not None else {
            "x": (-0.10, 0.10), "y": (-0.20, 0.20), "z": (0.0, 0.0),
        }
        self.events.reset_cube_and_goal_poses = EventTerm(
            func=mdp.reset_cube_and_goal_poses,
            mode="reset",
            params={
                "pose_range":       pose_range,
                "goal_pos_default": goal.goal_pos_default or GOAL_POS_DEFAULT,
                "pose_cmd_name":    "ee_pose",
                "yaw_range":        goal.goal_yaw_range,
                "cube_cfg":         SceneEntityCfg(primary_obj.name),
            },
        )

        # ---- Terminations ----
        self.terminations.cube_at_goal = DoneTerm(
            func=mdp.cube_reached_goal,
            params={
                "cube_cfg":            SceneEntityCfg(primary_obj.name),
                "robot_cfg":           SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
                "pose_cmd_name":       "ee_pose",
                "pos_threshold":       goal.success_threshold_pos,
                "ang_threshold":       goal.success_threshold_ang,
                "yaw_threshold":       goal.success_threshold_yaw,
                "grip_open_threshold": goal.grip_open_threshold,
                "success_dwell_s":     goal.success_dwell_s,
            },
        )

        # cube_fell: any_object_fell takes object_cfgs list.
        self.terminations.cube_fell = DoneTerm(
            func=mdp.any_object_fell,
            params={
                "object_cfgs":    [SceneEntityCfg(primary_obj.name)],
                "minimum_height": -0.05,
            },
        )


@configclass
class HLEnvCfg_Envhub_PLAY(HLEnvCfg_Envhub):
    """Play / evaluation variant: fewer envs, no observation noise."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
