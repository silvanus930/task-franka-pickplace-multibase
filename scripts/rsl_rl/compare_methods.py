#!/usr/bin/env python3
"""Compare train/eval loop runs across methods.

Usage::

    python compare_methods.py logs/train_eval_runs/*/
    python compare_methods.py logs/train_eval_runs/hl_finetune_v1_* logs/train_eval_runs/hl_finetune_v2_*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_run(run_dir: Path) -> dict | None:
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return None
    best = max(history, key=lambda h: h["score"])
    return {
        "method_dir": str(run_dir),
        "method": run_dir.name,
        "evals": len(history),
        "best_score": best["score"],
        "best_iter": best["iteration"],
        "best_success_rate": best.get("success_rate"),
        "best_checkpoint": best.get("checkpoint"),
        "last_score": history[-1]["score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare train/eval loop method runs.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing history.json")
    args = parser.parse_args()

    rows = []
    for raw in args.run_dirs:
        run_dir = Path(raw).expanduser().resolve()
        row = _load_run(run_dir)
        if row is None:
            print(f"[WARN] Skipping {run_dir} (no history.json)", file=sys.stderr)
            continue
        rows.append(row)

    if not rows:
        print("No valid runs found.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["best_score"], reverse=True)
    print("| Rank | Method | Best score | Best iter | Success rate | Evals |")
    print("|------|--------|------------|-----------|--------------|-------|")
    for i, row in enumerate(rows, start=1):
        sr = row["best_success_rate"]
        sr_txt = f"{sr * 100:.1f}%" if sr is not None else "n/a"
        print(
            f"| {i} | `{row['method']}` | {row['best_score']:.4f} | "
            f"{row['best_iter']} | {sr_txt} | {row['evals']} |"
        )
    print(f"\nBest method: {rows[0]['method']} (score={rows[0]['best_score']:.4f})")
    print(f"Checkpoint: {rows[0]['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
