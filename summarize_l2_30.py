#!/usr/bin/env python3
"""Summarize the fixed 30-problem L2 MCTS experiment.

The input directory must contain one final_results.json per completed problem.
Missing directories/results remain in the 30-problem denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / "l2_selected_30_20260726.txt"


def selected_problems() -> list[str]:
    return [
        line.strip()
        for line in SELECTION.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def family(name: str) -> str:
    lower = name.lower()
    if "convtranspose" in lower:
        return "ConvTranspose"
    if "conv" in lower:
        return "Conv"
    if "gemm" in lower or "matmul" in lower or "bmm" in lower:
        return "GEMM/Matmul"
    if "norm" in lower:
        return "Norm"
    return "Other"


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def geomean(values: list[float]) -> float | None:
    vals = [v for v in values if finite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def read_row(root: Path, problem: str) -> dict:
    final = root / problem / "final_results.json"
    row = {
        "problem": problem,
        "family": family(problem),
        "status": "missing",
        "strict_triton": False,
        "pytorch_latency_ms": None,
        "seed_latency_ms": None,
        "champion_latency_ms": None,
        "speedup_vs_pytorch": None,
        "speedup_vs_seed": None,
        "tree_nodes": None,
        "max_depth": None,
        "failure_class": "missing_final",
    }
    if not final.exists():
        return row
    try:
        data = json.loads(final.read_text())
    except Exception as exc:
        row["status"] = "invalid"
        row["failure_class"] = f"invalid_json:{type(exc).__name__}"
        return row
    row["pytorch_latency_ms"] = data.get("baseline_latency_ms")
    row["seed_latency_ms"] = data.get("seed_latency_ms")
    row["champion_latency_ms"] = data.get("champion_latency_ms")
    row["speedup_vs_pytorch"] = data.get("speedup_vs_pytorch")
    row["speedup_vs_seed"] = data.get("speedup_vs_seed")
    row["strict_triton"] = bool(data.get("strict_triton"))
    row["status"] = data.get("status", "success" if row["strict_triton"] else "invalid")
    row["failure_class"] = None if row["strict_triton"] else "non_strict_final"
    candidates = data.get("final_candidates") or []
    if candidates:
        row["champion_latency_ms"] = row["champion_latency_ms"] or candidates[0].get("latency_ms")
        row["strict_triton"] = row["strict_triton"] or "@triton.jit" in (candidates[0].get("code") or "")
    iterations = data.get("iterations") or []
    if iterations:
        row["tree_nodes"] = sum(x.get("num_candidates", 0) for x in iterations if isinstance(x, dict)) or None
    if not row["strict_triton"] and row["failure_class"] is None:
        row["failure_class"] = "non_strict_final"
    if row["strict_triton"]:
        row["status"] = "success"
        if finite(row["pytorch_latency_ms"]) and finite(row["champion_latency_ms"]):
            row["speedup_vs_pytorch"] = row["pytorch_latency_ms"] / row["champion_latency_ms"]
        if finite(row["seed_latency_ms"]) and finite(row["champion_latency_ms"]):
            row["speedup_vs_seed"] = row["seed_latency_ms"] / row["champion_latency_ms"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output or root / "l2_30_results.json"
    if not output.is_absolute():
        output = ROOT / output

    rows = [read_row(root, p) for p in selected_problems()]
    strict = [r for r in rows if r["strict_triton"]]
    summary = {
        "total": len(rows),
        "strict_triton": len(strict),
        "coverage": len(strict) / len(rows) if rows else 0,
        "geomean_speedup_vs_pytorch_conditional": geomean([r["speedup_vs_pytorch"] for r in strict]),
        "geomean_speedup_vs_seed_conditional": geomean([r["speedup_vs_seed"] for r in strict]),
        "failure_counts": {},
        "family": {},
    }
    for row in rows:
        if row["failure_class"]:
            key = row["failure_class"]
            summary["failure_counts"][key] = summary["failure_counts"].get(key, 0) + 1
    for fam in sorted({r["family"] for r in rows}):
        group = [r for r in strict if r["family"] == fam]
        summary["family"][fam] = {
            "total": sum(r["family"] == fam for r in rows),
            "strict_triton": len(group),
            "geomean_vs_pytorch": geomean([r["speedup_vs_pytorch"] for r in group]),
            "geomean_vs_seed": geomean([r["speedup_vs_seed"] for r in group]),
        }
    payload = {"summary": summary, "rows": rows}
    output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    fields = list(rows[0])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
