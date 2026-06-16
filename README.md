# Franka Pick-and-Place Multibase — Hierarchical Policy Training

**Developed by Nepher Robotics — contact@nepher.ai**

Isaac Lab external project for hierarchical Franka robotic arm pick-and-place training across multiple base environments.

![Franka Pick-and-Place simulation](docs/assets/franka-base-image.png)

## Architecture

```
HL Policy  — ClassicalPickPlacePlanner state machine
    ↓  waypoints (x, y, z, quat, grip)
LL Policy  — goal-conditioned EE tracker  [Phase 1 — trained]
    ↓  target (x, y, z, rx, ry, rz, grip)
DifferentialIK + BinaryGripper
    ↓
Franka Panda (7-DOF + gripper)
```

## Low-Level Policy

| | |
|---|---|
| **Action** | 7D IK-Rel `(Δx,Δy,Δz,Δrx,Δry,Δrz)` + 1D binary gripper |
| **Observation** | 41D: joint pos/vel, EE pose, target EE pose, grip cmd, gripper pos, last action |
| **Command** | `ee_pose` (resampled every 4 s) + `grip_cmd` (per-episode) |
| **Reward** | Coarse L2 + fine tanh for position & orientation; soft gripper match; smoothness |
| **Curriculum** | Action-rate / joint-velocity penalties ramped 0.001 → 0.01/0.005 over 10 k iters |
| **Network** | MLP [256, 128, 64], ELU, PPO (RSL-RL) |

## Install

```bash
python -m pip install -e source/franka_pickplace_multibase
```

## Train

```bash
cd scripts/rsl_rl
python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --num_envs 4096
python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --resume
```

Checkpoints → `logs/rsl_rl/franka_ll_ee_tracking/<timestamp>/`.

## Evaluate (LL)

```bash
cd scripts/rsl_rl
python play.py --task=Nepher-Franka-PickPlace-LL-Play-v0
```

`play.py` copies the latest checkpoint into `best_policy/best_policy.pt` and exports TorchScript + ONNX to `best_policy/exported/` for HL use.

---

## High-Level Classical Pipeline — Container Task

9-stage `PickPlacePlanner` drives the HL commands; the frozen LL policy executes them.
Five varied YCB objects are scattered on the table each episode; the arm picks each one
sequentially and drops it into a KLT bin container in the corner.

| | |
|---|---|
| **Stages** | PRE_GRASP → DESCEND → GRASP → LIFT → CARRY → LOWER → RELEASE → RETRACT → DONE |
| **Transition** | EE error < stage tolerance + minimum dwell |
| **Objects** | 5 × YCB: sugar box, mustard bottle, tomato soup can, cracker box, DexCube |
| **Container** | KLT bin (small), placed at table corner (0.55, −0.30) |
| **Grasp metadata** | Per-object: grasp Z offset, yaw symmetry, yaw offset |
| **Goal** | All objects inside bin (XY footprint + Z range check); no yaw/upright requirement |
| **Episode** | ~35 s (5 objects × ~7 s budget each) |

Object catalog (`mdp/object_assets.py`):

| Scene name | USD | Grasp symmetry |
|-----------|-----|----------------|
| `object0` | `Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd` | 180° (long axis) |
| `object1` | `Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd` | rotationally symmetric |
| `object2` | `Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd` | rotationally symmetric |
| `object3` | `Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd` | 180° (long axis) |
| `object4` | `Props/Blocks/DexCube/dex_cube_instanceable.usd` | 90° (square) |

```bash
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-Play-v0
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-Play-v0 --video --video_length 600
```

---

## Multi-Base Integration (EnvHub)

`Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0` — same LL checkpoint and planner, scene loaded from a Nepher manipulation preset (`HLEnvCfg_Envhub`, `env_id` / `scene_id`).

Default preset `franka-pickplace-base-sample`: SeattleLabTable, 3 × DexCube, 30 deterministic scenarios (`env_id % 30`).

```bash
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0
```

Evaluated via `eval-nav/configs/task-franka-pickplace-multibase.yaml` (`num_envs: 30`, scoring v1).

### Termination contract (eval-nav)

| Signal | Meaning | Triggered by |
|---|---|---|
| `task_completed` | All objects reached their goal | `cube_at_goal` / `all_objects_reached_goals` |
| `task_failed` | Object or container failure, or timeout | `cube_fell`, `object_dropped`, `container_fell`, `container_displaced`, `time_out` |
