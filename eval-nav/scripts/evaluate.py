#!/usr/bin/env python3
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Command-line interface for navigation evaluation.

Launch Isaac Sim Simulator first.

Examples
--------
    NEPHER_EVAL_IN_PROCESS=1 python scripts/evaluate.py \\
        --config configs/task-franka-pickplace-multibase.yaml \\
        --checkpoint /path/to/model_5400.pt \\
        --headless

    # Saves full stdout to logs/.../final_5400.txt (tag derived from checkpoint name).
"""

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).parent.parent))


def _resolve_checkpoint_path(raw: str) -> Path:
    """Resolve a checkpoint CLI argument to an absolute existing file."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _checkpoint_log_tag(checkpoint: Path) -> str:
    """Derive ``final_<tag>.txt`` name from a checkpoint filename.

    ``model_5400.pt`` -> ``5400``; ``best_policy.pt`` -> ``best_policy``.
    """
    stem = checkpoint.stem
    if stem.startswith("model_"):
        return stem.removeprefix("model_")
    return stem


class _Tee:
    """Duplicate writes to the original stream and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


@contextmanager
def _tee_output(log_path: Path):
    """Mirror stdout/stderr to ``log_path`` for the duration of the context."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        stdout_tee = _Tee(sys.stdout, log_file)
        stderr_tee = _Tee(sys.stderr, log_file)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_tee, stderr_tee
        try:
            yield log_file
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def _write_eval_header(log_file, *, config_path: Path, config_dict: dict, checkpoint: Path | None) -> None:
    """Write a reproducible header (config + checkpoint) at the top of full logs."""
    header = {
        "config_file": str(config_path.resolve()),
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        **config_dict,
    }
    log_file.write("# Evaluation run configuration\n")
    yaml.dump(header, log_file, default_flow_style=False, allow_unicode=True)
    log_file.write("\n")
    log_file.flush()


def _apply_launcher_flags_from_config_yaml() -> None:
    """Apply Isaac Sim launcher flags from the eval YAML before AppLauncher starts.

    - ``enable_cameras: true`` or ``video: true`` -> ``--enable_cameras``
    - CLI ``--video`` -> ``--enable_cameras``
    """
    if "--video" in sys.argv and "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if not pre_args.config:
        return
    path = Path(pre_args.config)
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f) or {}
        needs_cameras = bool(data.get("enable_cameras") or data.get("video"))
        if needs_cameras and "--enable_cameras" not in sys.argv:
            sys.argv.append("--enable_cameras")
    except Exception:
        pass


_apply_launcher_flags_from_config_yaml()

parser = argparse.ArgumentParser(
    description="Evaluate IsaacLab navigation environments",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to evaluation configuration YAML file",
)

parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress console output",
)

parser.add_argument(
    "--result-path",
    type=str,
    default=None,
    help="Absolute path for evaluation_result.json output (default: cwd)",
)

parser.add_argument(
    "--checkpoint",
    "--model",
    dest="checkpoint",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "LL policy checkpoint (.pt). Overrides policy_path in the YAML. "
        "Full eval log is saved to <log_dir>/final_<tag>.txt where tag is "
        "derived from the filename (e.g. model_5400.pt -> final_5400.txt)."
    ),
)

parser.add_argument(
    "--full-log",
    type=str,
    default=None,
    metavar="PATH",
    help="Override path for the full eval transcript (default: <log_dir>/final_<tag>.txt).",
)

parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record evaluation video to <log_dir>/videos/eval/ (implies --enable_cameras).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from eval_nav import EvalConfig, EvaluationReporter, NavigationEvaluator


def main():
    """Main entry point for evaluation CLI.

    Returns:
        dict: Dictionary containing core evaluation results with keys:
            - score: float (final evaluation score)
            - log_dir: str (path to the run directory where results are saved)
            - metadata: dict (evaluation metadata as JSON-serializable dict)
            - summary: str (reporter's human-readable summary)
            - full_log: str (path to the full transcript log file)
    """
    config_path = Path(args_cli.config).expanduser().resolve()
    try:
        config = EvalConfig.from_yaml(str(config_path))
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if args_cli.video:
        config.video = True
        config.enable_cameras = True

    checkpoint_path: Path | None = None
    if args_cli.checkpoint:
        try:
            checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        config.policy_path = str(checkpoint_path)

    if not config.log_dir:
        raise ValueError("log_dir must be specified in config YAML")

    log_dir = Path(config.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (Path.cwd() / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.full_log:
        full_log_path = Path(args_cli.full_log).expanduser().resolve()
    elif checkpoint_path is not None:
        full_log_path = log_dir / f"final_{_checkpoint_log_tag(checkpoint_path)}.txt"
    elif config.policy_path:
        full_log_path = log_dir / f"final_{_checkpoint_log_tag(Path(config.policy_path))}.txt"
    else:
        full_log_path = log_dir / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with _tee_output(full_log_path) as log_file:
        _write_eval_header(
            log_file,
            config_path=config_path,
            config_dict=config.to_dict(),
            checkpoint=checkpoint_path or (Path(config.policy_path) if config.policy_path else None),
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = log_dir / f"eval_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        original_log_dir = config.log_dir
        config.log_dir = str(run_dir)

        in_process = os.environ.get("NEPHER_EVAL_IN_PROCESS", "0") == "1"
        evaluator = NavigationEvaluator(
            config,
            checkpoint_path=config.policy_path,
            subprocess_mode=in_process,
        )

        if config.policy_path:
            print(f"[INFO] Policy checkpoint specified: {config.policy_path}")
            print("[INFO] Policy will be loaded when first environment is created")
        else:
            print("[INFO] No policy checkpoint specified, using random actions")
        print(f"[INFO] Full eval log: {full_log_path}")

        results = evaluator.evaluate(policy=None)
        reporter = EvaluationReporter(results)

        log_json_path = run_dir / "results.json"
        log_summary_path = run_dir / "summary.txt"

        reporter.save_json(log_json_path)
        reporter.save_summary(log_summary_path)

        run_config_path = run_dir / "config.yaml"
        config.log_dir = original_log_dir
        with open(run_config_path, "w", encoding="utf-8", errors="replace") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

        run_full_log = run_dir / "full.txt"
        shutil.copy2(full_log_path, run_full_log)

        if not args_cli.quiet:
            reporter.print_summary()
            print(f"\nResults saved to log directory: {run_dir}")
            print(f"  - JSON: {log_json_path}")
            print(f"  - Summary: {log_summary_path}")
            print(f"  - Config: {run_config_path}")
            print(f"  - Full log: {full_log_path}")
            print(f"  - Run copy: {run_full_log}")
            print(f"  - NumPy state logs: {run_dir}/*.npy")
            if config.video:
                print(f"  - Video: {run_dir}/videos/eval/")

        result = {
            "score": results.get("score", 0),
            "log_dir": str(run_dir),
            "full_log": str(full_log_path),
            "metadata": results.get("metadata", {}),
            "summary": reporter.generate_summary(),
        }

        if args_cli.result_path:
            result_json_path = Path(args_cli.result_path).expanduser().resolve()
            result_json_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            result_json_path = Path("evaluation_result.json")
        try:
            with open(result_json_path, "w", encoding="utf-8", errors="replace") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if not args_cli.quiet:
                print(f"  - Result: {result_json_path}")
        except IOError as e:
            print(f"[WARNING] Failed to save result JSON: {e}", file=sys.stderr)

        print(f"\n[INFO] Evaluation result: {result}")

        if results.get("status") != "SUCCESS":
            sys.exit(1)

        return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()

