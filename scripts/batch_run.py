"""Batch runner for DirecTune.

Usage:
    python scripts/batch_run.py --csv problems/kb_level1/_batch.csv --first 10 --iters 3
    python scripts/batch_run.py --csv problems/kb_level1/_batch.csv --filter "gemm" --iters 5
"""

import argparse
import asyncio
import csv
import re
import subprocess
import sys
import os
from pathlib import Path


def numeric_key(name: str) -> int:
    """Extract leading number from problem name for sorting."""
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 9999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to _batch.csv")
    parser.add_argument("--first", type=int, default=0, help="Run first N problems (sorted numerically)")
    parser.add_argument("--last", type=int, default=0, help="Run last N problems")
    parser.add_argument("--filter", type=str, default="", help="Only run problems matching this substring")
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config file")
    parser.add_argument("--output", type=str, default="output/kb_level1", help="Output root directory")
    parser.add_argument("--iters", type=int, default=3, help="Override iterations")
    parser.add_argument("--breadth", type=int, default=2, help="Override breadth")
    parser.add_argument("--num-samples", type=int, default=1, help="Override num_samples")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # Read and sort problems
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        problems = list(reader)

    problems.sort(key=lambda r: numeric_key(r["name"]))

    # Filter
    if args.filter:
        problems = [p for p in problems if args.filter.lower() in p["name"].lower()]

    if args.first > 0:
        problems = problems[:args.first]
    if args.last > 0:
        problems = problems[-args.last:]

    print(f"Running {len(problems)} problems:")
    for p in problems:
        print(f"  {p['name']}")

    if args.dry_run:
        return

    # Run sequentially
    success = 0
    failed = []

    for i, p in enumerate(problems):
        name = p["name"]
        out_dir = os.path.join(args.output, name)
        cmd = [
            "python", "main.py",
            "--config", args.config,
            "--problem", p["problem"],
            "--initial", p["initial_solution"],
            "--iters", str(args.iters),
            "--breadth", str(args.breadth),
            "--num-samples", str(args.num_samples),
            "--output", out_dir,
        ]

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(problems)}] {name}")
        print(f"{'='*60}")

        try:
            result = subprocess.run(cmd, timeout=3600)
            if result.returncode == 0:
                success += 1
                print(f"  → PASSED")
            else:
                failed.append(name)
                print(f"  → FAILED (exit code {result.returncode})")
        except subprocess.TimeoutExpired:
            failed.append(name)
            print(f"  → TIMEOUT")

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(problems)} passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
