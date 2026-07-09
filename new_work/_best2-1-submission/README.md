# Franka Pick-and-Place Multibase — Hierarchical Policy Training

**Developed by Nepher Robotics — contact@nepher.ai**

Isaac Lab external project for hierarchical Franka robotic arm pick-and-place training across multiple base environments.

![Franka Pick-and-Place simulation](docs/assets/franka-pickplace-multi.gif)

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

## Multi-Base Integration (EnvHub)

`Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0` — same LL checkpoint and planner, scene loaded from a Nepher manipulation preset (`HLEnvCfg_Envhub`, `env_id` / `scene_id`).

Default preset `franka-pickplace-multibase-sample`: SeattleLabTable, 8-type YCB catalog (5 of 8 active per episode), 30 typed deterministic scenarios (`env_id % 30`).  Override with `--preset franka-pickplace-base-sample` to use the 3 × DexCube fixed-catalog preset.

```bash
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0
```

Evaluated via `eval-nav/configs/task-franka-pickplace-multibase.yaml` (`num_envs: 30`, `max_episode_steps: 1575`, `max_episode_time_s: 35.0`, scoring v2).

### Termination contract (eval-nav)

| Signal | Meaning | Triggered by |
|---|---|---|
| `task_completed` | All objects reached their goal | `cube_at_goal` / `all_objects_reached_goals` |
| `task_failed` | Object or container failure, or timeout | `cube_fell`, `object_dropped`, `container_fell`, `container_displaced`, `time_out` |
