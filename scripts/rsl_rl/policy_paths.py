# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve LL policy checkpoints via ``best_policy/``.

On play / evaluation, the latest RSL-RL checkpoint under ``logs/`` is copied
into ``best_policy/best_policy.pt`` (fixed name).  Downstream code always loads from
that path so HL evaluation does not depend on timestamped log run folders.
"""

from __future__ import annotations

import os
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

BEST_POLICY_DIR = os.path.join(PROJECT_ROOT, "best_policy")
BEST_POLICY_CHECKPOINT = os.path.join(BEST_POLICY_DIR, "best_policy.pt")
BEST_POLICY_EXPORT_DIR = os.path.join(BEST_POLICY_DIR, "exported")


def log_root_path(experiment_name: str) -> str:
    """Absolute path to ``logs/rsl_rl/<experiment_name>/`` at the project root."""
    return os.path.join(PROJECT_ROOT, "logs", "rsl_rl", experiment_name)


def _find_latest_in_logs(experiment_name: str, load_run: str, load_checkpoint: str) -> str | None:
    from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

    root = log_root_path(experiment_name)
    if not os.path.isdir(root):
        return None
    try:
        return get_checkpoint_path(root, load_run, load_checkpoint)
    except ValueError:
        return None


def sync_best_policy(
    experiment_name: str,
    load_run: str,
    load_checkpoint: str,
    *,
    explicit_checkpoint: str | None = None,
) -> str:
    """Copy the chosen LL checkpoint into ``best_policy/best_policy.pt`` and return that path.

    Resolution order for the source checkpoint:

    1. ``explicit_checkpoint`` when provided (``--checkpoint``).
    2. Latest match under ``logs/rsl_rl/<experiment_name>/``.
    3. Existing ``best_policy/best_policy.pt`` if no logs checkpoint is available.

    Raises:
        FileNotFoundError: When no checkpoint can be resolved.
    """
    os.makedirs(BEST_POLICY_DIR, exist_ok=True)

    source: str | None = None
    if explicit_checkpoint:
        if os.path.isfile(explicit_checkpoint):
            source = os.path.abspath(explicit_checkpoint)
        else:
            from isaaclab.utils.assets import retrieve_file_path

            source = retrieve_file_path(explicit_checkpoint)
    else:
        source = _find_latest_in_logs(experiment_name, load_run, load_checkpoint)

    if source is not None:
        source = os.path.abspath(source)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Checkpoint file not found: {source}")
        if os.path.samefile(source, BEST_POLICY_CHECKPOINT):
            print(f"[INFO] Using best_policy checkpoint (already in place): {BEST_POLICY_CHECKPOINT}")
            return BEST_POLICY_CHECKPOINT
        shutil.copy2(source, BEST_POLICY_CHECKPOINT)
        print(f"[INFO] Synced LL policy to best_policy: {source} -> {BEST_POLICY_CHECKPOINT}")
        return BEST_POLICY_CHECKPOINT

    if os.path.isfile(BEST_POLICY_CHECKPOINT):
        print(f"[INFO] Using existing best_policy checkpoint: {BEST_POLICY_CHECKPOINT}")
        return BEST_POLICY_CHECKPOINT

    log_root = log_root_path(experiment_name)
    raise FileNotFoundError(
        "No LL policy checkpoint found. Train the LL policy first, or place a checkpoint at "
        f"{BEST_POLICY_CHECKPOINT}. Expected training logs under: {log_root}"
    )
