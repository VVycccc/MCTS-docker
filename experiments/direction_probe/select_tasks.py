#!/usr/bin/env python3
"""Select representative KernelBench tasks for direction probing.

This intentionally uses lightweight heuristics so it can run before the full
experiment machinery exists. It reads one or more DirecTune `_batch.csv` files
and writes a JSON task list consumed by `schedule_runs.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


TASK_TYPE_ORDER = [
    "attention_like",
    "convolution_like",
    "matmul_like",
    "reduction",
    "normalization",
    "indexing",
    "sort_topk_selection",
    "elementwise",
    "mixed_fused",
    "other",
]


def numeric_key(name: str) -> tuple[int, str]:
    match = re.match(r"(\d+)", name)
    return (int(match.group(1)) if match else 9999, name)


def infer_task_type(name: str, problem_path: str = "") -> str:
    text = f"{name} {problem_path}".lower()
    if any(k in text for k in ["attention", "attn", "scaled_dot", "scaleddot"]):
        return "attention_like"
    if any(k in text for k in ["conv", "convolution"]):
        return "convolution_like"
    if any(k in text for k in ["matmul", "gemm", "bmm", "mm_", "linear", "matrix_multiplication"]):
        return "matmul_like"
    if any(k in text for k in ["sum", "mean", "max", "min", "prod", "reduction", "argmax", "argmin"]):
        return "reduction"
    if any(k in text for k in ["norm", "batchnorm", "layernorm", "groupnorm", "rmsnorm"]):
        return "normalization"
    if any(k in text for k in ["gather", "scatter", "index", "embedding", "where", "mask"]):
        return "indexing"
    if any(k in text for k in ["sort", "topk", "top_k", "select", "median"]):
        return "sort_topk_selection"
    if any(k in text for k in ["relu", "gelu", "sigmoid", "tanh", "softmax", "add", "mul", "div", "sub", "clamp", "exp"]):
        return "elementwise"
    if any(k in text for k in ["fused", "fusion", "module", "model"]):
        return "mixed_fused"
    return "other"


def read_tasks(csv_paths: Iterable[Path]) -> list[dict]:
    tasks: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for csv_path in csv_paths:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                problem = row.get("problem", "").strip()
                initial = row.get("initial_solution", "").strip()
                if not name or not problem:
                    continue
                key = (name, problem)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append({
                    "task_id": name,
                    "name": name,
                    "problem": problem,
                    "initial_solution": initial,
                    "task_type": infer_task_type(name, problem),
                    "source_csv": str(csv_path),
                })
    tasks.sort(key=lambda t: numeric_key(t["name"]))
    return tasks


def stratified_select(tasks: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(tasks):
        return tasks

    buckets: dict[str, list[dict]] = {k: [] for k in TASK_TYPE_ORDER}
    for task in tasks:
        buckets.setdefault(task["task_type"], []).append(task)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    # Round-robin across task types to avoid one family dominating the probe set.
    while len(selected) < limit:
        progressed = False
        for task_type in TASK_TYPE_ORDER:
            bucket = buckets.get(task_type, [])
            while bucket and bucket[0]["task_id"] in selected_ids:
                bucket.pop(0)
            if bucket and len(selected) < limit:
                task = bucket.pop(0)
                selected.append(task)
                selected_ids.add(task["task_id"])
                progressed = True
        if not progressed:
            break

    selected.sort(key=lambda t: numeric_key(t["name"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, help="Path to a _batch.csv file. Repeatable.")
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to select. <=0 means all.")
    parser.add_argument("--filter", default="", help="Only keep task names containing this substring.")
    parser.add_argument("--out", required=True, help="Output selected_tasks.json")
    args = parser.parse_args()

    tasks = read_tasks([Path(p) for p in args.csv])
    if args.filter:
        needle = args.filter.lower()
        tasks = [t for t in tasks if needle in t["name"].lower()]
    selected = stratified_select(tasks, args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for task in selected:
        counts[task["task_type"]] = counts.get(task["task_type"], 0) + 1
    print(f"Wrote {len(selected)} tasks to {out}")
    for task_type in TASK_TYPE_ORDER:
        if counts.get(task_type):
            print(f"  {task_type}: {counts[task_type]}")


if __name__ == "__main__":
    main()
