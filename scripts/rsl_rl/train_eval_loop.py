#!/usr/bin/env python3
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chunked LL training with periodic eval and early stopping.

Trains for ``--eval-every`` iterations, runs eval-nav on the latest checkpoint,
records score/success-rate, and stops when eval score stops improving.

Each train and eval step runs in a fresh subprocess (Isaac Sim cannot stay up
across both in one process).

Example — HL-in-the-loop finetune with fast eval every 500 iters::

    # Finetune task lives in task-franka-pickplace-multibase (not _best/_best2):
    pip install -e ../../source/franka_pickplace_multibase

    cd task-franka-pickplace-multibase/scripts/rsl_rl

    python train_eval_loop.py \\
        --method hl_finetune_v1 \\
        --task Nepher-Franka-PickPlace-HL-LL-Finetune-v0 \\
        --checkpoint ../../best_policy/best_policy.pt \\
        --eval-every 500 \\
        --patience 3 \\
        --fast-eval

For empty-table LL pretraining (always registered), use::

    --task Nepher-Franka-PickPlace-LL-v0

Compare methods later via ``logs/train_eval_runs/compare_methods.py`` or the
``history.json`` files under each method run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STACK_ROOT = PROJECT_ROOT.parent
EVAL_NAV_ROOT = STACK_ROOT / "eval-nav"
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
EVAL_SCRIPT = EVAL_NAV_ROOT / "scripts" / "evaluate.py"

DEFAULT_TASK = "Nepher-Franka-PickPlace-HL-LL-Finetune-v0"
FALLBACK_TASK = "Nepher-Franka-PickPlace-LL-v0"
DEFAULT_EVAL_CONFIG = EVAL_NAV_ROOT / "configs" / "task-franka-pickplace-multibase.yaml"
FAST_EVAL_CONFIG = EVAL_NAV_ROOT / "configs" / "task-franka-pickplace-multibase-fast.yaml"


def _list_franka_tasks() -> list[str]:
    import gymnasium as gym
    import franka_pickplace_multibase.tasks  # noqa: F401

    return sorted(k for k in gym.registry if "Franka-PickPlace" in k)


def _ensure_task_registered(task: str) -> None:
    """Fail fast with a helpful message when the gym task is not registered."""
    import gymnasium as gym
    import franka_pickplace_multibase  # noqa: F401

    pkg_path = Path(franka_pickplace_multibase.__file__).resolve().parent.parent
    try:
        gym.spec(task)
        print(f"[INFO] Task registered: {task}")
        print(f"[INFO] franka_pickplace_multibase from: {pkg_path}")
        return
    except gym.error.NameNotFound:
        available = _list_franka_tasks()
        msg = (
            f"Gym task not registered: {task}\n"
            f"Installed package: {pkg_path}\n"
            f"Available Franka tasks: {available}\n\n"
            "Fix:\n"
            "  pip install -e /root/leatherback-stack/task-franka-pickplace-multibase/source/franka_pickplace_multibase\n\n"
            "Note: _best / _best2 do NOT register HL-LL-Finetune. "
            "Use task-franka-pickplace-multibase for finetune, or pass "
            f"--task {FALLBACK_TASK} for empty-table LL training."
        )
        raise SystemExit(msg) from None


def _checkpoint_iter(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(log_dir: Path) -> Path | None:
    ckpts = sorted(log_dir.glob("model_*.pt"), key=_checkpoint_iter)
    return ckpts[-1] if ckpts else None


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n[CMD] {printable}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd or PROJECT_ROOT))


def _train_chunk(
    *,
    task: str,
    log_dir: Path,
    checkpoint: Path | None,
    chunk: int,
    num_envs: int | None,
    save_interval: int,
    resume: bool,
    extra_train_args: list[str],
) -> None:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"--task={task}",
        "--headless",
        f"--log_dir={log_dir}",
        f"--iteration_chunk={chunk}",
        f"--save_interval={save_interval}",
    ]
    if num_envs is not None:
        cmd.append(f"--num_envs={num_envs}")
    if resume:
        cmd.append("--resume")
        if checkpoint is not None:
            cmd.append(f"--checkpoint={checkpoint}")
    elif checkpoint is not None:
        # Seed weights from a baseline without treating the run as resumed.
        cmd.append(f"--checkpoint={checkpoint}")
    cmd.extend(extra_train_args)
    _run(cmd, cwd=SCRIPT_DIR)


def _eval_checkpoint(
    *,
    checkpoint: Path,
    eval_config: Path,
    result_path: Path,
) -> dict:
    env = os.environ.copy()
    env["NEPHER_EVAL_IN_PROCESS"] = "1"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(eval_config),
        "--checkpoint",
        str(checkpoint),
        "--headless",
        "--result-path",
        str(result_path),
        "--quiet",
    ]
    _run(cmd, env=env, cwd=EVAL_NAV_ROOT)
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics: dict = {}
    run_dir = payload.get("log_dir")
    if run_dir:
        results_json = Path(run_dir) / "results.json"
        if results_json.is_file():
            with results_json.open("r", encoding="utf-8") as f:
                run_payload = json.load(f)
            metrics = run_payload.get("metrics", {}) or {}

    return {
        "score": float(payload.get("score", 0.0)),
        "success_rate": metrics.get("success_rate"),
        "successful_episodes": metrics.get("successful_episodes"),
        "total_episodes": metrics.get("total_episodes"),
        "eval_run_dir": run_dir,
        "raw": payload,
    }


def _write_history(run_dir: Path, history: list[dict]) -> None:
    path = run_dir / "history.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Updated {path}")


def _write_summary(run_dir: Path, history: list[dict], *, method: str, stopped_reason: str) -> None:
    best = max(history, key=lambda h: h["score"]) if history else None
    lines = [
        f"# Train/Eval Loop — {method}",
        "",
        f"Stopped: {stopped_reason}",
        f"Run directory: {run_dir}",
        "",
        "## Checkpoints",
        "",
        "| Iter | Score | Success rate | Checkpoint |",
        "|------|-------|--------------|------------|",
    ]
    for row in history:
        sr = row.get("success_rate")
        sr_txt = f"{sr * 100:.1f}%" if sr is not None else "n/a"
        lines.append(
            f"| {row['iteration']} | {row['score']:.4f} | {sr_txt} | `{row['checkpoint']}` |"
        )
    if best:
        lines.extend(
            [
                "",
                "## Best",
                "",
                f"- Iteration: **{best['iteration']}**",
                f"- Score: **{best['score']:.4f}**",
                f"- Checkpoint: `{best['checkpoint']}`",
            ]
        )
    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked LL training with eval and early stopping.")
    parser.add_argument("--method", required=True, help="Method label (used in run dir name).")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID for training.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Baseline checkpoint to start from (e.g. best_policy/best_policy.pt).",
    )
    parser.add_argument("--eval-every", type=int, default=500, help="Train this many iters between evals.")
    parser.add_argument("--max-total-iters", type=int, default=5000, help="Hard cap on total training iters.")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many consecutive evals without score improvement.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum score increase to count as improvement.",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="eval-nav YAML config (default: official; use --fast-eval for quick loop).",
    )
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help="Use fast eval config (6 envs, 1 episode) for quicker feedback.",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Run one full official eval on the best checkpoint at the end.",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Training num_envs override.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run dir to resume a prior train/eval loop.",
    )
    parser.add_argument(
        "extra_train_args",
        nargs="*",
        help="Extra args forwarded to train.py (e.g. --seed 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_config = Path(args.eval_config) if args.eval_config else (FAST_EVAL_CONFIG if args.fast_eval else DEFAULT_EVAL_CONFIG)
    if not eval_config.is_file():
        print(f"Error: eval config not found: {eval_config}", file=sys.stderr)
        return 1

    _ensure_task_registered(args.task)

    baseline_ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if baseline_ckpt and not baseline_ckpt.is_file():
        print(f"Error: checkpoint not found: {baseline_ckpt}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = PROJECT_ROOT / "logs" / "train_eval_runs" / f"{args.method}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log_dir = run_dir / "train"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "method": args.method,
        "task": args.task,
        "baseline_checkpoint": str(baseline_ckpt) if baseline_ckpt else None,
        "eval_every": args.eval_every,
        "max_total_iters": args.max_total_iters,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "eval_config": str(eval_config),
        "fast_eval": args.fast_eval,
        "started_at": stamp,
    }
    (run_dir / "loop_config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    history: list[dict] = []
    if (run_dir / "history.json").is_file():
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    best_score = max((h["score"] for h in history), default=float("-inf"))
    patience_left = args.patience
    stopped_reason = "max_total_iters"

    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Eval config: {eval_config}")
    print(f"[INFO] Existing checkpoints in train dir: {bool(list(train_log_dir.glob('model_*.pt')))}")

    iters_trained_in_loop = sum(
        h.get("chunk_iters", args.eval_every) for h in history
    )

    while True:
        if iters_trained_in_loop >= args.max_total_iters:
            stopped_reason = "max_total_iters"
            break

        latest_before = _find_latest_checkpoint(train_log_dir)
        resume_train = latest_before is not None
        load_ckpt = latest_before if resume_train else baseline_ckpt

        chunk = min(args.eval_every, args.max_total_iters - iters_trained_in_loop)
        if chunk <= 0:
            stopped_reason = "max_total_iters"
            break

        try:
            _train_chunk(
                task=args.task,
                log_dir=train_log_dir,
                checkpoint=load_ckpt,
                chunk=chunk,
                num_envs=args.num_envs,
                save_interval=args.eval_every,
                resume=resume_train or load_ckpt is not None,
                extra_train_args=args.extra_train_args,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Training chunk failed (exit {exc.returncode}).", file=sys.stderr)
            stopped_reason = "train_failed"
            break

        latest_ckpt = _find_latest_checkpoint(train_log_dir)
        if latest_ckpt is None:
            print("[ERROR] No checkpoint written after training chunk.", file=sys.stderr)
            stopped_reason = "no_checkpoint"
            break

        iteration = _checkpoint_iter(latest_ckpt)
        result_path = eval_dir / f"eval_iter_{iteration:05d}.json"
        eval_payload = _eval_checkpoint(
            checkpoint=latest_ckpt,
            eval_config=eval_config,
            result_path=result_path,
        )
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "chunk_iters": chunk,
            "checkpoint": str(latest_ckpt),
            "score": eval_payload["score"],
            "success_rate": eval_payload["success_rate"],
            "successful_episodes": eval_payload["successful_episodes"],
            "total_episodes": eval_payload["total_episodes"],
            "eval_result": str(result_path),
            "eval_run_dir": eval_payload.get("eval_run_dir"),
        }
        history.append(row)
        _write_history(run_dir, history)
        iters_trained_in_loop += chunk

        print(
            f"[EVAL] iter={iteration} score={row['score']:.4f} "
            f"success_rate={row['success_rate']} best={best_score:.4f} patience_left={patience_left}"
        )

        if row["score"] > best_score + args.min_delta:
            best_score = row["score"]
            patience_left = args.patience
            best_dir = run_dir / "best"
            best_dir.mkdir(exist_ok=True)
            best_copy = best_dir / f"model_{iteration}.pt"
            shutil.copy2(latest_ckpt, best_copy)
            deploy_ckpt = PROJECT_ROOT / "best_policy" / "best_policy.pt"
            deploy_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_ckpt, deploy_ckpt)
            print(f"[INFO] New best -> {best_copy} (deployed to {deploy_ckpt})")
        else:
            patience_left -= 1
            print(f"[INFO] No improvement (delta <= {args.min_delta}). Patience remaining: {patience_left}")
            if patience_left <= 0:
                stopped_reason = "early_stop_no_improvement"
                break

    _write_summary(run_dir, history, method=args.method, stopped_reason=stopped_reason)

    best_row = max(history, key=lambda h: h["score"]) if history else None
    if args.final_eval and best_row:
        print("\n[INFO] Running final full official eval on best checkpoint...")
        final_result = eval_dir / "final_official_eval.json"
        _eval_checkpoint(
            checkpoint=Path(best_row["checkpoint"]),
            eval_config=DEFAULT_EVAL_CONFIG,
            result_path=final_result,
        )
        print(f"[INFO] Final official eval -> {final_result}")

    print(f"\n[DONE] stopped={stopped_reason} best_score={best_score:.4f} run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
