# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EnvHub integration for the HL pick-and-place environment.

Loads a manipulation preset from the Nepher envhub and wires its scene
(table, objects, lighting, workspace) into the HL policy environment.

Three wiring paths are supported:

Typed-scenario (``franka-pickplace-multibase-sample`` default)
--------------------------------------------------------------
When the preset exposes a :class:`TypedPrebakedScenarioStrategy` (detected
via ``hasattr(strategy, 'get_active_indices')``):

* All catalog object names are wired into ``HLPoseCommandCfg.cube_names``.
* ``num_active`` is set so the command term enters typed mode.
* ``reset_typed_objects_from_scenario`` event parks inactive objects and
  places active ones each episode; updates per-env active-catalog indices
  and goals.
* ``all_objects_reached_goals`` is wired with ``typed_mode=True`` so it
  uses per-env active indices to check only the active slots.
* Grasp metadata lists (length ``num_catalog``, indexed by catalog) wired
  from ``OBJECT_CATALOG`` in ``mdp.object_assets``.

Standard multi-object (``franka-pickplace-base-sample``)
--------------------------------------------------------
When the preset has a :class:`PrebakedScenarioStrategy` (no
``get_active_indices``):

* Objects sorted by ``pick_order``; all names wired into ``cube_names``.
* ``reset_objects_and_goals`` event used.
* Standard ``all_objects_reached_goals`` (no typed_mode).

Single-object fallback
----------------------
When the preset has no strategy (legacy / random-goal presets).

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
    HLSafeTerminationsCfg,
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
    extra_attrs: dict[str, Any] | None = None,
) -> InteractiveSceneCfg:
    """Construct a ``LLSceneCfg``-based scene augmented with preset content.

    - Objects from ``preset.get_object_cfgs()`` are injected as new attrs.
    - Lights from ``preset.get_light_cfgs()`` override the default dome light.
    - Table from ``preset.get_table_cfg()`` overrides the default table (if set).
    - ``extra_attrs`` — additional scene entities (e.g. container for the typed
      pick-and-place path) merged after the preset objects.
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

    if extra_attrs:
        attrs.update(extra_attrs)

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

    env_id: str = "franka-pickplace-multibase-sample"
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

        # ---- Detect preset type early (needed for scene construction) ----
        objects  = getattr(preset, "objects",  []) or []
        goals    = getattr(preset, "goals",    []) or []
        strategy = getattr(preset, "position_strategy", None)
        is_typed = (strategy is not None and len(objects) > 0
                    and hasattr(strategy, "get_active_indices"))

        # ---- Scene ----
        # For the typed-scenario path (franka-pickplace-multibase-sample), this
        # is still a container-drop task, so the KLT bin must be in the scene.
        # For all other envhub paths the scene has no container.
        extra_scene_attrs: dict[str, Any] | None = None
        if is_typed:
            from franka_pickplace_multibase.tasks.manager_based.hl_policy.mdp.object_assets import (
                CONTAINER_CFG,
                make_container_rigid_cfg,
            )
            extra_scene_attrs = {
                "container": make_container_rigid_cfg("{ENV_REGEX_NS}/Container", CONTAINER_CFG),
            }

        self.scene = _build_envhub_scene(self.scene, preset, extra_attrs=extra_scene_attrs)

        # For non-typed envhub paths the scene has no container → disable
        # container-specific terminations inherited from HLTerminationsCfg.
        # For the typed path the container IS present, so they remain active.
        if not is_typed:
            self.terminations.container_fell = None
            self.terminations.container_displaced = None

        if strategy is not None and len(objects) > 0:
            if is_typed:
                # Typed-scenario path: N-type catalog, M active per episode.
                self._wire_typed_multi_object(preset, objects, strategy)
            else:
                # Standard multi-object with PrebakedScenarioStrategy.
                self._wire_multi_object(preset, objects, goals, strategy)
        else:
            # Legacy single-object fallback.
            self._wire_single_object(preset, objects, goals)

        # ---- Episode length ----
        if hasattr(preset, "max_episode_length_s"):
            self.episode_length_s = preset.max_episode_length_s

    # ------------------------------------------------------------------
    # Typed-scenario wiring (TypedPrebakedScenarioStrategy)
    # ------------------------------------------------------------------

    def _wire_typed_multi_object(self, preset, objects, strategy) -> None:
        """Wire the typed-scenario path for franka-pickplace-multibase-sample.

        This path keeps the **container-drop task** from the base ``HLEnvCfg``
        (objects are dropped into the KLT bin), but uses an 8-type YCB catalog
        where each of the 30 benchmark scenarios picks 5 active types and assigns
        them deterministic spawn positions.

        The container is already in the envhub scene (added in
        ``_apply_envhub_preset`` before this method is called).  Container
        terminations (``container_fell``, ``container_displaced``) inherited
        from ``HLTerminationsCfg`` remain active.

        Grasp metadata (indexed by catalog index, not pick slot) is read from
        :data:`mdp.object_assets.OBJECT_CATALOG`.
        """
        from franka_pickplace_multibase.tasks.manager_based.hl_policy.mdp.object_assets import (
            CONTAINER_CFG,
            OBJECT_CATALOG,
        )
        from franka_pickplace_multibase.tasks.manager_based.hl_policy.hl_env_cfg import (
            GOAL_POS_DEFAULT,
            _DROP_Z_LOCAL,
        )

        num_active  = strategy.num_active
        num_catalog = strategy.num_catalog

        # Catalog names in catalog-index order (catalog_0 … catalog_N).
        catalog_names = sorted(
            [o.name for o in objects],
            key=lambda n: int(n.split("_")[-1]),
        )

        # Grasp metadata indexed by catalog index (length = num_catalog = 8).
        catalog_grasp_z   = [obj.grasp_z_offset              for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_grasp_sym = [obj.grasp_sym                   for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_grasp_yaw = [obj.effective_grasp_yaw_offset() for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_upright   = [obj.upright_height              for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_grasp_off = [obj.grasp_offset_local           for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_grasp_long_axis = [obj.grasp_long_axis_local   for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_footprint_xy = [obj.footprint_xy               for obj in OBJECT_CATALOG[:num_catalog]]
        catalog_frame_yaw_off = [obj.grasp_yaw_frame_offset    for obj in OBJECT_CATALOG[:num_catalog]]

        # ---- Commands: keep container-drop mode (same as base HLEnvCfg) ----
        # container_drop=True is the default from HLCommandsCfg; do NOT set False.
        self.commands.ee_pose.cube_names        = catalog_names
        self.commands.ee_pose.num_active        = num_active
        self.commands.ee_pose.goal_pos_defaults = [GOAL_POS_DEFAULT] * num_active
        self.commands.ee_pose.grasp_z_offsets   = catalog_grasp_z
        self.commands.ee_pose.grasp_syms        = catalog_grasp_sym
        self.commands.ee_pose.grasp_yaw_offsets = catalog_grasp_yaw
        self.commands.ee_pose.upright_heights   = catalog_upright
        self.commands.ee_pose.grasp_offset_locals = catalog_grasp_off
        self.commands.ee_pose.grasp_long_axis_locals = catalog_grasp_long_axis
        self.commands.ee_pose.footprint_xys = catalog_footprint_xy
        self.commands.ee_pose.grasp_yaw_frame_offsets = catalog_frame_yaw_off
        # goal_pose_visualizer_cfg stays as _MARKER_CFG (container opening marker)

        # ---- Events: typed-scenario object reset + container placement ----
        _C = CONTAINER_CFG
        self.events.reset_cube_and_goal_poses = EventTerm(
            func=mdp.reset_typed_objects_from_scenario,
            mode="reset",
            params={
                "all_object_cfgs":           [SceneEntityCfg(name) for name in catalog_names],
                "scenario_strategy":         strategy,
                "pose_cmd_name":             "ee_pose",
                "parking_pos":               (20.0, 0.0, 0.5),
                # Container pose is read per-env from scenarios.json; no
                # runtime sampling parameters are needed here.
                "container_cfg":             SceneEntityCfg("container"),
                "container_interior_half_x": _C.interior_half_x,
                "container_interior_half_y": _C.interior_half_y,
                "container_drop_z_local":    _DROP_Z_LOCAL,
                "center_drop":               True,
            },
        )

        # ---- Terminations: container success (typed — check only active objects) ----
        self.terminations.cube_at_goal = DoneTerm(
            func=mdp.objects_in_container,
            params={
                "object_cfgs":                [SceneEntityCfg(name) for name in catalog_names],
                "robot_cfg":                  SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
                "pose_cmd_name":              "ee_pose",
                "container_cfg":              SceneEntityCfg("container"),
                "container_pos_world_offset": (_C.pos[0], _C.pos[1], _C.pos[2]),
                "container_interior_half_x":  _C.interior_half_x,
                "container_interior_half_y":  _C.interior_half_y,
                "grip_open_threshold":        0.8,
                "success_dwell_s":            1.0,
                "enable_log":                 True,
                "typed_mode":                 True,
                "num_active":                 num_active,
            },
        )

        # cube_fell: monitor all catalog objects; parked ones at local z=0.5 are
        # safely above the -0.05 threshold so they never trigger this term.
        self.terminations.cube_fell = DoneTerm(
            func=mdp.any_object_fell,
            params={
                "object_cfgs":    [SceneEntityCfg(name) for name in catalog_names],
                "minimum_height": -0.05,
            },
        )

        # object_dropped: use catalog names and typed active-object lookup.
        self.terminations.object_dropped = DoneTerm(
            func=mdp.object_dropped_mid_carry,
            params={
                "object_cfgs":       [SceneEntityCfg(name) for name in catalog_names],
                "pose_cmd_name":     "ee_pose",
                "drop_height_world": 0.10,
                "enable_log":        True,
            },
        )
        # container_fell and container_displaced remain active.
        # Scenario spawn positions are validated against the fixed container
        # position with 0.10 m clearance, so no physics ejection occurs on reset.

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

        # object_dropped: override to use preset object names (base uses object0..4).
        self.terminations.object_dropped = DoneTerm(
            func=mdp.object_dropped_mid_carry,
            params={
                "object_cfgs":       [SceneEntityCfg(name) for name in sorted_names],
                "pose_cmd_name":     "ee_pose",
                "drop_height_world": 0.10,
                "enable_log":        True,
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

        # object_dropped: override to use the actual object name (base uses object0..4).
        self.terminations.object_dropped = DoneTerm(
            func=mdp.object_dropped_mid_carry,
            params={
                "object_cfgs":       [SceneEntityCfg(primary_obj.name)],
                "pose_cmd_name":     "ee_pose",
                "drop_height_world": 0.10,
                "enable_log":        True,
            },
        )


@configclass
class HLEnvCfg_Envhub_PLAY(HLEnvCfg_Envhub):
    """Play / evaluation variant: full 30-scenario benchmark, no observation noise.

    ``play.py --num_envs`` may still override this for quick local smoke tests
    after Hydra config construction.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 30
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class HLEnvCfg_Envhub_PLAY_OpportunisticPlace(HLEnvCfg_Envhub_PLAY):
    """Play variant with opportunistic in-bin placement during transport.

  When an object lands inside the container during CARRY/LOWER/RELEASE, the
  planner marks it placed and advances instead of recycling to PRE_GRASP.
  ``object_dropped_mid_carry`` is also suppressed for objects already in the bin.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.ee_pose.opportunistic_container_place = True


@configclass
class HLEnvCfg_Envhub_PLAY_OpportunisticPlace_VIDEO(HLEnvCfg_Envhub_PLAY_OpportunisticPlace):
    """Multi-env video capture: opportunistic placement, HL logging, long episode.

    Default ``num_envs=4`` for parallel smoke testing; eval-nav overrides via yaml.
    Uses relaxed ``HLSafeTerminationsCfg`` so incidental container bumps do not
    cut the recording short. Not for official benchmark scoring.
    """

    terminations: HLSafeTerminationsCfg = HLSafeTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.commands.ee_pose.enable_log = True
        self.commands.ee_pose.log_env_id = -1  # log all envs when num_envs > 1


@configclass
class HLEnvCfg_Envhub_SAFE_PLAY(HLEnvCfg_Envhub):
    """Safe diagnostic EnvHub variant for production-readiness debugging.

    This variant keeps object/container fall and drop checks active, but relaxes
    incidental container displacement. It is intentionally separate from
    ``HLEnvCfg_Envhub_PLAY`` so official 30-env benchmark evaluation remains
    strict and comparable.
    """

    terminations: HLSafeTerminationsCfg = HLSafeTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 45.0
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.max_retries = 2
        self.commands.ee_pose.log_env_id = 0
