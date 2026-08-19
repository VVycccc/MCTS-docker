#!/usr/bin/env python3
"""Aggregate direction probe JSONL records into CSV summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIRECTIONS = [
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
    "other",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return records


def rate(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def recommendation(row: dict[str, Any]) -> str:
    attempts = row.get("attempt_count", 0) or 0
    natural_frequency = row.get("natural_frequency") or 0.0
    applicability_rate = row.get("applicability_rate")
    improve_rate = row.get("improve_rate")

    if attempts < 10:
        return "needs_more_samples"
    if applicability_rate is not None and applicability_rate < 0.2:
        return "low_priority"
    if improve_rate is not None and improve_rate >= 0.30:
        if natural_frequency < 0.05:
            return "underused_but_effective"
        return "high_priority"
    if improve_rate is not None and improve_rate >= 0.15:
        return "medium"
    if natural_frequency >= 0.15 and (improve_rate is None or improve_rate < 0.10):
        return "overused_by_agent"
    return "low_priority"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def aggregate_direction(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    free_attempted = [r for r in records if r.get("selection_mode") == "free" and r.get("attempted", True)]
    total_free = len(free_attempted)
    natural_counts = Counter(r.get("direction") or "other" for r in free_attempted)

    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        guided_assigned = [
            r for r in records
            if r.get("selection_mode") == "guided"
            and (r.get("assigned_direction") == direction or r.get("direction") == direction)
        ]
        applicable = [r for r in guided_assigned if r.get("applicability") == "applicable"]
        attempted = [r for r in guided_assigned if r.get("attempted")]
        compile_pass = [r for r in attempted if r.get("compile_pass") is True]
        correctness_pass = [r for r in attempted if r.get("correctness_pass") is True]
        benchmark_pass = [r for r in attempted if r.get("benchmark_pass") is True]
        improved = [r for r in attempted if r.get("improved") is True]
        speedups = [float(r["speedup"]) for r in attempted if r.get("correctness_pass") is True and r.get("speedup") is not None]

        row = {
            "direction": direction,
            "natural_count": natural_counts.get(direction, 0),
            "natural_frequency": rate(natural_counts.get(direction, 0), total_free),
            "assigned_count": len(guided_assigned),
            "applicable_count": len(applicable),
            "applicability_rate": rate(len(applicable), len(guided_assigned)),
            "attempt_count": len(attempted),
            "compile_rate": rate(len(compile_pass), len(attempted)),
            "correctness_rate": rate(len(correctness_pass), len(attempted)),
            "benchmark_rate": rate(len(benchmark_pass), len(attempted)),
            "improve_rate": rate(len(improved), len(attempted)),
            "median_speedup": median(speedups),
        }
        row["recommendation"] = recommendation(row)
        rows.append(row)
    return rows


def aggregate_task_type(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record.get("task_type") or "unknown"].append(record)

    rows = []
    for task_type, group in sorted(by_type.items()):
        attempted = [r for r in group if r.get("attempted", True)]
        improved = [r for r in attempted if r.get("improved") is True]
        speedups = [float(r["speedup"]) for r in attempted if r.get("correctness_pass") is True and r.get("speedup") is not None]
        direction_counts = Counter(r.get("direction") or "other" for r in attempted)
        rows.append({
            "task_type": task_type,
            "records": len(group),
            "attempts": len(attempted),
            "improvements": len(improved),
            "improve_rate": rate(len(improved), len(attempted)),
            "median_speedup": median(speedups),
            "top_direction": direction_counts.most_common(1)[0][0] if direction_counts else None,
        })
    return rows


def aggregate_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(r.get("failure_type") or "success_or_unset" for r in records)
    total = sum(counts.values())
    return [
        {"failure_type": kind, "count": count, "frequency": rate(count, total)}
        for kind, count in counts.most_common()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="records.jsonl path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    records = read_jsonl(Path(args.records))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_rows = aggregate_direction(records)
    task_type_rows = aggregate_task_type(records)
    failure_rows = aggregate_failures(records)

    write_csv(out_dir / "direction_stats.csv", direction_rows, [
        "direction", "natural_count", "natural_frequency", "assigned_count", "applicable_count",
        "applicability_rate", "attempt_count", "compile_rate", "correctness_rate",
        "benchmark_rate", "improve_rate", "median_speedup", "recommendation",
    ])
    write_csv(out_dir / "task_type_stats.csv", task_type_rows, [
        "task_type", "records", "attempts", "improvements", "improve_rate", "median_speedup", "top_direction",
    ])
    write_csv(out_dir / "failure_stats.csv", failure_rows, ["failure_type", "count", "frequency"])

    print(f"Aggregated {len(records)} records into {out_dir}")


if __name__ == "__main__":
    main()
