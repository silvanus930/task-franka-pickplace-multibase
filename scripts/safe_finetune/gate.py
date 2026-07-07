#!/usr/bin/env python3
"""Compare official eval score against the configured baseline gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a checkpoint against the baseline score.")
    parser.add_argument("result_json", type=Path, help="evaluation_result.json from eval-nav")
    parser.add_argument("--baseline-score", type=float, default=0.122)
    parser.add_argument("--baseline-successes", type=int, default=11)
    parser.add_argument("--total-episodes", type=int, default=90)
    args = parser.parse_args()

    data = json.loads(args.result_json.read_text())
    score = float(data["score"])
    summary = data.get("summary", "")
    successes = None
    for line in summary.splitlines():
        if "Successful:" in line:
            try:
                successes = int(line.split("Successful:")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

    passed = score >= args.baseline_score
    print("=" * 60)
    print("Safe finetune gate")
    print("=" * 60)
    print(f"  Result file : {args.result_json}")
    print(f"  Score       : {score:.4f}")
    if successes is not None:
        print(f"  Successes   : {successes}/{args.total_episodes}")
    print(f"  Baseline    : {args.baseline_score:.4f} ({args.baseline_successes}/{args.total_episodes})")
    print(f"  Verdict     : {'PASS — promote checkpoint' if passed else 'FAIL — keep current baseline'}")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
