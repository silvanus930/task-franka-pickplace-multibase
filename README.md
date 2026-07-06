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

For HL pick-and-place, the LL policy must track wrist orientation accurately;
position-only reaching is not enough to secure grasps. After SafePlay reports
`PLANNER_GRASP_FAILED`, continue or restart LL training with the orientation-
weighted reward config:

```bash
cd scripts/rsl_rl
python train.py --task=Nepher-Franka-PickPlace-LL-v0 --headless --num_envs 4096 --max_iterations 5000 --run_name ll_orientation_grasp_v1
```

## Evaluate (LL)

```bash
cd scripts/rsl_rl
python play.py --task=Nepher-Franka-PickPlace-LL-Play-v0
```

`play.py` copies the latest checkpoint into `best_policy/best_policy.pt` and exports TorchScript + ONNX to `best_policy/exported/` for HL use.

---

## Production-Safety Gates

Use the safe diagnostic task before official scoring. It keeps drop/fall
failures active, relaxes incidental container displacement, runs one env by
default, and reports true success/failure/timeout metrics instead of PPO reward.

```bash
cd scripts/rsl_rl
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0 --headless --max_episodes 5 --max_steps 1500
```

Only move to the strict benchmark task after SafePlay can complete episodes
without repeated `PRE_GRASP stuck`, `object_dropped`, or `container_fell`
failures.

```bash
cd scripts/rsl_rl
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0 --headless --num_envs 30 --max_episodes 90
```

For video recording in headless mode, pass `--video`; `play.py` automatically
enables cameras for Isaac Lab rendering.

```bash
cd scripts/rsl_rl
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubSafePlay-v0 --headless --video --video_length 300
```

## Multi-Base Integration (EnvHub)

`Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0` — same LL checkpoint and planner, scene loaded from a Nepher manipulation preset (`HLEnvCfg_Envhub`, `env_id` / `scene_id`).

Default preset `franka-pickplace-multibase-sample`: SeattleLabTable, 8-type YCB catalog (5 of 8 active per episode), 30 typed deterministic scenarios (`env_id % 30`). Override with `--preset franka-pickplace-base-sample` to use the 3 × DexCube fixed-catalog preset.

```bash
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0
python play.py --task=Nepher-Franka-PickPlace-HL-Multibase-EnvhubPlay-v0 --preset franka-pickplace-base-sample
```

Evaluated via `eval-nav/configs/task-franka-pickplace-multibase.yaml` (`num_envs: 30`, `max_episode_steps: 1575`, `max_episode_time_s: 35.0`, scoring v2).

### Termination contract (eval-nav)

| Signal | Meaning | Triggered by |
|---|---|---|
| `task_completed` | All objects reached their goal | `cube_at_goal` / `all_objects_reached_goals` |
| `task_failed` | Object or container failure, or timeout | `cube_fell`, `object_dropped`, `container_fell`, `container_displaced`, `time_out` |
