#!/usr/bin/env python3
"""Generate a markdown report for a direction probe run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if 0 <= number <= 1:
        return f"{number:.3f}"
    return f"{number:.3f}" if not number.is_integer() else str(int(number))


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(fmt(row.get(col)) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing aggregate CSVs")
    parser.add_argument("--out", default="", help="Optional report path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "direction_probe_report.md"

    direction_rows = read_csv(run_dir / "direction_stats.csv")
    task_rows = read_csv(run_dir / "task_type_stats.csv")
    failure_rows = read_csv(run_dir / "failure_stats.csv")

    natural = sorted(direction_rows, key=lambda r: float(r.get("natural_frequency") or 0), reverse=True)
    effectiveness = sorted(direction_rows, key=lambda r: float(r.get("improve_rate") or 0), reverse=True)
    applicability = sorted(direction_rows, key=lambda r: float(r.get("applicability_rate") or 0), reverse=True)

    lines = [
        "# Direction Probe Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Natural frequency from free records",
        "",
        table(natural, ["direction", "natural_count", "natural_frequency", "recommendation"]),
        "",
        "## Guided applicability",
        "",
        table(applicability, ["direction", "assigned_count", "applicable_count", "applicability_rate"]),
        "",
        "## Guided effectiveness",
        "",
        table(effectiveness, ["direction", "attempt_count", "compile_rate", "correctness_rate", "improve_rate", "median_speedup", "recommendation"]),
        "",
        "## Task type summary",
        "",
        table(task_rows, ["task_type", "records", "attempts", "improvements", "improve_rate", "median_speedup", "top_direction"]),
        "",
        "## Failure distribution",
        "",
        table(failure_rows, ["failure_type", "count", "frequency"]),
        "",
        "## Notes",
        "",
        "- Treat rows with `needs_more_samples` as inconclusive.",
        "- `underused_but_effective` means free runs rarely chose the direction, but guided attempts had strong improvement rate.",
        "- `overused_by_agent` means free runs often chose the direction, but guided effectiveness was weak.",
    ]

    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
