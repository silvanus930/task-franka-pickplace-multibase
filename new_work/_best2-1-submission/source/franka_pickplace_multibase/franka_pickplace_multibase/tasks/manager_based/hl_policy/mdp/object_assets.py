# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Catalog of graspable USD objects and container geometry for the HL container task.

Each :class:`GraspObjectCfg` encodes the object's USD path, scale, and
per-object grasp parameters used by :class:`~..classical_planner.PickPlacePlanner`:

* ``grasp_z_offset``  – TCP Z relative to object *centre* at grasp depth.
  Negative means the TCP descends below the object centre (for taller objects the
  centre is higher so the gripper grabs mid-height).
* ``grasp_sym``       – gripper yaw symmetry in radians.  The planner folds the
  object's world yaw into ``[-sym/2, sym/2]``.  Use ``math.pi`` for 180° symmetry
  (long-axis objects: box, bottle, banana), ``math.pi / 2`` for 90° symmetry
  (square cross-section cans), and ``0.0`` for rotationally symmetric objects
  (round can cross-section grasped from top) which use a neutral wrist angle.
* ``grasp_yaw_offset``– constant yaw added to the snapped grasp yaw before
  sending to the EE (object-frame alignment).  Zero for most objects.
* ``footprint_radius``– approximate radius (m) used for non-overlap scatter.

:class:`ContainerCfg` describes the physical KLT bin: its USD path, world position
on the table, approximate interior half-extents, and Z levels used for drop
targeting and success checking.
"""

from __future__ import annotations

import math
import os as _os
from dataclasses import dataclass, field
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils


# ---------------------------------------------------------------------------
# Per-object grasp metadata
# ---------------------------------------------------------------------------


@dataclass
class GraspObjectCfg:
    """Metadata for a single graspable rigid object in the container task."""

    name: str
    """Scene attribute name used in HLSceneCfg (e.g. ``'object0'``)."""

    usd_path: str
    """Absolute Nucleus USD path."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Uniform scale applied at spawn."""

    # ---- Grasp geometry ----
    grasp_z_offset: float = -0.01
    """TCP Z below object *centre* at grasp depth (m).

    The planner computes ``z_grasp = object_centre_z + grasp_z_offset + hand_tcp_offset``.
    Use a more negative value for taller objects so the gripper catches the
    mid-height instead of the very top.
    """

    grasp_sym: float = math.pi / 2
    """Grasp yaw symmetry (rad).  ``0.0`` → neutral wrist (round cross-section)."""

    grasp_yaw_offset: float = 0.0
    """Constant yaw offset added after symmetry-folding (rad).

    This is a *manual* bias.  For elongated objects, prefer setting
    ``footprint_xy`` instead and let :meth:`effective_grasp_yaw_offset`
    compute the correct value automatically.
    """

    footprint_xy: tuple[float, float] = (0.0, 0.0)
    """Horizontal footprint ``(local_x_size, local_y_size)`` in metres (after scale).

    Used to automatically align the gripper with the **short axis** of the
    object so the fingers can close around it successfully.

    The panda gripper's fingers open along the object's **local-Y direction**
    in world frame when ``grasp_yaw_offset = 0``.  Therefore:

    * ``local_y_size < local_x_size`` — short axis is **local Y** →
      ``grasp_yaw_offset = 0`` is correct (no auto adjustment).
    * ``local_x_size < local_y_size`` — short axis is **local X** →
      ``+π/2`` is added so the fingers rotate onto the local-X direction.
    * ``(0, 0)`` — dimensions unknown; :meth:`effective_grasp_yaw_offset`
      falls back to the manual ``grasp_yaw_offset``.
    """

    footprint_radius: float = 0.05
    """Approximate object footprint radius (m) for non-overlap scatter."""

    grasp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Conservative object-local grasp offset applied in world frame by the planner.

    This lets us bias grasps for specific shapes without changing observations,
    actions, or the checkpoint format. Defaults remain neutral until tuned.
    """

    # ---- Spawn ----
    default_pos: tuple[float, float, float] = (0.45, 0.0, 0.055)
    """Default world-local position used for scene initialisation."""

    def effective_grasp_yaw_offset(self) -> float:
        """Return the grasp yaw offset that aligns the gripper with the short axis.

        If ``footprint_xy`` is specified (both components > 0), the offset is
        computed automatically:

        * Short axis = **local Y** (``y < x``): returns ``grasp_yaw_offset`` unchanged.
        * Short axis = **local X** (``x < y``): returns ``grasp_yaw_offset + π/2``.
        * Square footprint (``x == y``): returns ``grasp_yaw_offset`` unchanged.

        Falls back to the manual ``grasp_yaw_offset`` when ``footprint_xy`` is
        ``(0, 0)`` (dimensions not specified).
        """
        x, y = self.footprint_xy
        if x <= 0.0 or y <= 0.0:
            return self.grasp_yaw_offset
        if x < y:
            # Local X is the short axis; rotate π/2 so fingers span local X.
            return self.grasp_yaw_offset + math.pi / 2
        return self.grasp_yaw_offset


# ---------------------------------------------------------------------------
# Object catalog
# ---------------------------------------------------------------------------

_YCB_P = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"  # physics-baked S3 subset

OBJECT_CATALOG: list[GraspObjectCfg] = [
    # 004 sugar box – ~7 cm tall × 3.5 cm × 2.5 cm at 0.7 scale.
    # Long axis along local Y (3.5 cm), short axis along local X (2.5 cm).
    # footprint_xy: x < y → effective_grasp_yaw_offset adds π/2 so fingers
    # span the narrow (local-X) face.  180° yaw symmetry.
    GraspObjectCfg(
        name="object0",
        usd_path=f"{_YCB_P}/004_sugar_box.usd",
        scale=(0.7, 0.7, 0.7),
        grasp_z_offset=-0.03,
        grasp_sym=math.pi,
        grasp_yaw_offset=0.0,
        footprint_xy=(0.025, 0.035),  # local x=2.5 cm (short), local y=3.5 cm (long)
        footprint_radius=0.04,
        grasp_offset_local=(0.0, 0.0, 0.0),
        default_pos=(0.60, 0.10, 0.05),
    ),
    # 006 mustard bottle – ~13 cm tall, FLATTENED OVAL cross-section (NOT round):
    # narrow squeeze axis ~3.5 cm, wide face ~5-6 cm. Treating it as round (neutral
    # wrist) means the grasp lands on a RANDOM axis vs the bottle's spawn yaw -> on the
    # WIDE axis a finger clips & tips it (yaw-dependent CUBE_FELL, fails 1-2/3 reps).
    # FIX (env-gated): grasp_sym=pi + oval footprint -> planner ALIGNS the gripper to
    # span the NARROW axis every time (like the box objects). NEPHER_MUST_SYM/FX/FY
    # default to the old round values (no behaviour change).
    GraspObjectCfg(
        name="object1",
        usd_path=f"{_YCB_P}/006_mustard_bottle.usd",
        scale=(0.7, 0.7, 0.7),
        grasp_z_offset=-0.04,
        grasp_sym=math.pi,            # OVAL: 180deg-symmetric; align grasp to bottle axis
        grasp_yaw_offset=0.0,
        footprint_xy=(0.035, 0.055),  # narrow X (3.5cm squeeze axis) < wide Y -> fingers
                                      # span the NARROW axis (auto +pi/2). Lift 0.86->0.91.
        footprint_radius=0.03,
        grasp_offset_local=(0.0, 0.0, 0.0),
        default_pos=(0.75, -0.10, 0.065),
    ),
    # 005 tomato soup can – ~7 cm tall × ~4.8 cm diam at 0.7 scale.
    # Round cross-section → neutral wrist.
    # footprint_xy equal → no auto π/2 adjustment.
    GraspObjectCfg(
        name="object2",
        usd_path=f"{_YCB_P}/005_tomato_soup_can.usd",
        scale=(0.7, 0.7, 0.7),
        grasp_z_offset=-0.02,
        grasp_sym=0.0,
        grasp_yaw_offset=0.0,
        footprint_xy=(0.048, 0.048),  # circular footprint
        footprint_radius=0.03,
        grasp_offset_local=(0.0, 0.0, 0.0),
        default_pos=(0.70, 0.15, 0.04),
    ),
    # 003 cracker box – ~12.9 cm × 9.0 cm × 3.6 cm at 0.6 scale (reduced from 0.7).
    # Long axis along local Y (9.0 cm), short axis along local X (3.6 cm).
    # footprint_xy: x < y → effective_grasp_yaw_offset adds π/2.  180° sym.
    GraspObjectCfg(
        name="object3",
        usd_path=f"{_YCB_P}/003_cracker_box.usd",
        scale=(0.4, 0.4, 0.4),
        grasp_z_offset=-0.030,
        grasp_sym=math.pi,
        grasp_yaw_offset=0.0,
        footprint_xy=(0.036, 0.090),  # local x=3.6 cm (short), local y=9.0 cm (long)
        footprint_radius=0.040,
        grasp_offset_local=(0.0, 0.0, 0.0),
        default_pos=(0.65, -0.15, 0.050),
    ),
    # DexCube – ~2.2 cm cube at 0.55 scale.
    # Square footprint → 90° symmetry, no auto π/2 adjustment needed.
    GraspObjectCfg(
        name="object4",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.55, 0.55, 0.55),
        grasp_z_offset=-0.01,
        grasp_sym=math.pi / 2,
        grasp_yaw_offset=0.0,
        footprint_xy=(0.022, 0.022),  # square footprint
        footprint_radius=0.03,
        grasp_offset_local=(0.0, 0.0, 0.010),
        default_pos=(0.80, 0.05, 0.04),
    ),
]


# ---------------------------------------------------------------------------
# Container geometry
# ---------------------------------------------------------------------------


@dataclass
class ContainerGeomCfg:
    """Physical geometry of the container used for drop targeting and success checking."""

    # Default world-local position of the container on the table (robot-frame offset).
    # Used as a fallback when no per-episode randomisation range is provided.
    pos: tuple[float, float, float] = (0.55, -0.23, 0.03)
    """Container centre at table surface height (default / fallback)."""

    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion (w, x, y, z) orientation of the container."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale applied to the KLT bin USD."""

    interior_half_x: float = 0.14
    """Interior half-extent along container USD local X (m)."""

    interior_half_y: float = 0.10
    """Interior half-extent along container USD local Y (m)."""

    def table_interior_half_extents(self) -> tuple[float, float]:
        """Interior half-extents in robot/table XY (table +X aligns with container +Y)."""
        return self.interior_half_y, self.interior_half_x

    floor_z: float = 0.035
    """Approximate world Z of the bin floor (slightly above table surface)."""

    rim_z: float = 0.11
    """Approximate world Z of the bin rim top."""

    drop_height_above_rim: float = 0.02
    """Additional height above rim for the LOWER/RELEASE target (drop, not press)."""

    retract_xy_offset_table: tuple[float, float] = (-0.05, 0.12)
    """Table-frame XY shift during RETRACT (toward robot, away from bin centre)."""

    # ---- Per-episode randomisation ----
    pos_range_x: tuple[float, float] = (0.45, 0.62)
    """Robot-local X range for sampling the container centre each episode."""

    pos_range_y: tuple[float, float] = (-0.25, -0.12)
    """Robot-local Y range for sampling the container centre each episode."""

    # ---- Displacement termination ----
    max_displacement: float = 0.02
    """Maximum allowed XY displacement (m) from the episode-start position.

    If the container is pushed more than this far from where it spawned the
    episode terminates (container_displaced termination term).
    """

    # ---- Object placement clearance ----
    object_clearance: float = 0.10
    """Minimum XY clearance (m) between each object spawn and the container exterior."""

    object_spacing: float = 0.03
    """Minimum extra gap (m) between object footprints when scatter-sampling."""


# Default container config used by HLSceneCfg and the scatter event.
CONTAINER_CFG = ContainerGeomCfg()


def container_to_table_interior_half_extents(
    interior_half_x: float,
    interior_half_y: float,
) -> tuple[float, float]:
    """Map container-local interior half-extents to robot/table XY axes."""
    return interior_half_y, interior_half_x


# Table-frame slot pattern as fractions of the usable interior (±1 = edge at margin).
_DROP_SLOT_FRAC_TABLE: tuple[tuple[float, float], ...] = (
    (-0.55, -0.75),
    (0.55, -0.75),
    (0.0, 0.0),
    (-0.55, 0.75),
    (0.55, 0.75),
)


def container_drop_slot_offsets_table(
    num_slots: int,
    half_table_x: float,
    half_table_y: float,
    margin: float = 0.75,
) -> list[tuple[float, float]]:
    """Return table-frame XY offsets from bin centre for each object's drop slot."""
    hx = half_table_x * margin
    hy = half_table_y * margin
    offsets: list[tuple[float, float]] = []
    for m in range(num_slots):
        if m < len(_DROP_SLOT_FRAC_TABLE):
            fx, fy = _DROP_SLOT_FRAC_TABLE[m]
        else:
            angle = (m - len(_DROP_SLOT_FRAC_TABLE) + 1) * 2.399963
            fx = 0.45 * math.cos(angle)
            fy = 0.45 * math.sin(angle)
        offsets.append((fx * hx, fy * hy))
    return offsets


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

_RIGID_BODY_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    # More velocity iterations resolve multi-body contact islands (an object
    # dropped onto one already in the bin) more accurately.  This also makes the
    # finger-object grasp contact stiffer, so it helps grip rather than hurting
    # it.
    solver_velocity_iteration_count=4,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    # Keep this high: the position-controlled gripper builds grip force through
    # the penetration-recovery impulse on the squeezed object, so capping it low
    # weakens the grasp and the object slips on LIFT.  Drop bounce is handled by
    # the scene-level bounce_threshold_velocity instead (no effect on grip).
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_CONTAINER_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
    max_angular_velocity=50.0,
    max_linear_velocity=50.0,
    max_depenetration_velocity=1.0,
    disable_gravity=False,
    linear_damping=10.0,
    angular_damping=10.0,
)


def make_object_rigid_cfg(prim_path: str, obj: GraspObjectCfg) -> RigidObjectCfg:
    """Build a :class:`RigidObjectCfg` for a catalog object.

    All objects in :data:`OBJECT_CATALOG` use USDs that already carry
    ``PhysicsRigidBodyAPI`` / ``PhysicsCollisionAPI`` / ``PhysicsMassAPI``
    schemas (either from ``Axis_Aligned_Physics`` on S3, or from local USDA
    wrapper files in ``assets/YCB_physics/``).  Only ``rigid_props`` (solver
    tuning) is overridden here — the physics APIs come from the USD itself.

    Args:
        prim_path: Scene prim path template (e.g. ``"{ENV_REGEX_NS}/Object0"``).
        obj:       Catalog entry from :data:`OBJECT_CATALOG`.

    Returns:
        Fully configured :class:`RigidObjectCfg` ready to be added to the scene.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=obj.default_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj.usd_path,
            scale=obj.scale,
            rigid_props=_RIGID_BODY_PROPS,
        ),
    )


def make_container_asset_cfg(prim_path: str, geom: ContainerGeomCfg) -> AssetBaseCfg:
    """Build a static (kinematic) :class:`AssetBaseCfg` for the KLT bin container.

    The bin has collision but no rigid-body dynamics — it stays fixed on the table.

    Args:
        prim_path: Scene prim path template.
        geom:      Container geometry config.

    Returns:
        :class:`AssetBaseCfg` with a :class:`~isaaclab.sim.UsdFileCfg` spawn.
    """
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=geom.pos,
            rot=geom.rot,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=geom.scale,
        ),
    )


def make_container_rigid_cfg(prim_path: str, geom: ContainerGeomCfg) -> RigidObjectCfg:
    """Build a dynamic :class:`RigidObjectCfg` for the KLT bin container.

    The bin is a heavy rigid body with high damping and friction so it stays
    in place under normal operation but *can* be displaced by a direct collision,
    enabling the ``container_displaced`` termination term.

    Args:
        prim_path: Scene prim path template.
        geom:      Container geometry config.

    Returns:
        :class:`RigidObjectCfg` with mass, damping, and friction configured.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=geom.pos,
            rot=geom.rot,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=geom.scale,
            rigid_props=_CONTAINER_RIGID_PROPS,
            # Heavier bin (a real KLT bin with contents is heavy) so incidental
            # arm/object contact cannot shove it. Env-var tunable for sweeps.
            mass_props=MassPropertiesCfg(mass=float(_os.environ.get("NEPHER_CONT_MASS", "5.0"))),
        ),
    )
