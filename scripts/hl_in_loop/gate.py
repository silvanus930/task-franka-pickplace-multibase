#!/usr/bin/env python3
"""Compare official eval score against the HL-in-loop baseline gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate an HL-in-loop checkpoint against baseline.")
    parser.add_argument("result_json", type=Path, help="evaluation_result.json from eval-nav")
    parser.add_argument("--baseline-score", type=float, default=0.131)
    parser.add_argument("--baseline-successes", type=int, default=13)
    parser.add_argument("--min-successes", type=int, default=None,
                        help="Optional stricter promotion bar (e.g. 15 for v4).")
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
    if args.min_successes is not None and successes is not None:
        passed = passed and successes >= args.min_successes
    print("=" * 60)
    print("HL-in-loop finetune gate")
    print("=" * 60)
    print(f"  Result file : {args.result_json}")
    print(f"  Score       : {score:.4f}")
    if successes is not None:
        print(f"  Successes   : {successes}/{args.total_episodes}")
    print(f"  Baseline    : {args.baseline_score:.4f} ({args.baseline_successes}/{args.total_episodes})")
    if args.min_successes is not None:
        print(f"  Min promote : {args.min_successes}/{args.total_episodes} successes")
    print(f"  Verdict     : {'PASS — promote checkpoint' if passed else 'FAIL — keep current baseline'}")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
