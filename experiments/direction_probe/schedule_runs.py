#!/usr/bin/env python3
"""Create free/guided run schedules for direction probing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_DIRECTIONS = [
    "tiling_blocking",
    "parallelism_occupancy",
    "memory_access",
    "vectorization",
    "reduction_strategy",
    "fusion_or_fission",
    "algorithmic_rewrite",
    "data_layout_indexing",
    "specialization_fast_path",
    "precision_dtype",
    "autotune_parameters",
    "correctness_boundary_fix",
]


def load_directions(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_DIRECTIONS)
    text = Path(path).read_text()
    directions: list[str] = []
    in_directions = False
    for line in text.splitlines():
        if re.match(r"^directions:\s*$", line):
            in_directions = True
            continue
        if not in_directions:
            continue
        match = re.match(r"^\s{2}([a-zA-Z0-9_]+):\s*$", line)
        if match:
            direction = match.group(1)
            if direction != "other":
                directions.append(direction)
    return directions or list(DEFAULT_DIRECTIONS)


def make_steps(steps: int, assigned: list[str | None]) -> list[dict]:
    return [
        {"step": i + 1, "assigned_direction": assigned[i] if i < len(assigned) else None}
        for i in range(steps)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="selected_tasks.json")
    parser.add_argument("--taxonomy", default="", help="direction_taxonomy.yaml")
    parser.add_argument("--steps", type=int, default=3, help="Steps per run")
    parser.add_argument("--free-runs", type=int, default=1, help="Free runs per task")
    parser.add_argument("--guided-runs", type=int, default=1, help="Guided runs per task")
    parser.add_argument("--out", required=True, help="Output run_schedule.json")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    directions = load_directions(args.taxonomy)
    schedule: list[dict] = []
    direction_index = 0

    for task in tasks:
        task_id = task["task_id"]
        for run_idx in range(args.free_runs):
            schedule.append({
                "task_id": task_id,
                "task_type": task.get("task_type", "other"),
                "name": task.get("name", task_id),
                "problem": task["problem"],
                "initial_solution": task.get("initial_solution"),
                "run_id": f"{task_id}_free_{run_idx}",
                "selection_mode": "free",
                "steps": make_steps(args.steps, [None] * args.steps),
            })
        for run_idx in range(args.guided_runs):
            assigned: list[str] = []
            for _ in range(args.steps):
                assigned.append(directions[direction_index % len(directions)])
                direction_index += 1
            schedule.append({
                "task_id": task_id,
                "task_type": task.get("task_type", "other"),
                "name": task.get("name", task_id),
                "problem": task["problem"],
                "initial_solution": task.get("initial_solution"),
                "run_id": f"{task_id}_guided_{run_idx}",
                "selection_mode": "guided",
                "steps": make_steps(args.steps, assigned),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schedule, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(schedule)} runs to {out}")
    print(f"Directions: {', '.join(directions)}")


if __name__ == "__main__":
    main()
