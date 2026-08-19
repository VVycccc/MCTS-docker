#!/usr/bin/env python3
"""Evaluate one generated candidate kernel for the direction probe experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triton_backend import anti_pytorch_check, load_problem, profile  # noqa: E402


IMPROVEMENT_THRESHOLD = 1.05


def read_text(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text()


def profile_to_payload(result: Any) -> dict[str, Any]:
    return {
        "compiled": bool(result.compiled),
        "runnable": bool(result.runnable),
        "correct": bool(result.correct),
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


def failure_type(compiled: bool, correct: bool, benchmark_pass: bool, speedup: float | None, error: str | None) -> str | None:
    if not compiled:
        return "compile_error"
    if not correct:
        if error and "timed out" in error.lower():
            return "timeout"
        if error and "runtime" in error.lower():
            return "runtime_error"
        return "correctness_mismatch"
    if not benchmark_pass:
        return "benchmark_error"
    if speedup is None:
        return "benchmark_error"
    if speedup < 1.0:
        return "performance_regression"
    if speedup < IMPROVEMENT_THRESHOLD:
        return "no_measurable_improvement"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="Problem JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate Python file defining run(...)")
    parser.add_argument("--baseline", default="", help="Optional baseline/initial Python file defining run(...)")
    parser.add_argument("--out", required=True, help="Output result.json")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    problem = load_problem(args.problem)
    candidate_source = Path(args.candidate).read_text()
    candidate_violation = anti_pytorch_check(candidate_source)

    baseline_payload = None
    baseline_latency = None
    if args.baseline:
        baseline_source = Path(args.baseline).read_text()
        baseline_result = profile(
            baseline_source,
            problem,
            warmup=args.warmup,
            reps=args.reps,
            timeout_seconds=args.timeout_seconds,
            device=0,
            rel_tol=args.rel_tol,
        )
        baseline_payload = profile_to_payload(baseline_result)
        baseline_latency = baseline_result.latency_ms

    if candidate_violation:
        candidate_payload = {
            "compiled": False,
            "runnable": False,
            "correct": False,
            "latency_ms": None,
            "error": f"Anti-PyTorch violation: {candidate_violation}",
        }
    else:
        candidate_result = profile(
            candidate_source,
            problem,
            warmup=args.warmup,
            reps=args.reps,
            timeout_seconds=args.timeout_seconds,
            device=0,
            rel_tol=args.rel_tol,
        )
        candidate_payload = profile_to_payload(candidate_result)

    candidate_latency = candidate_payload["latency_ms"]
    benchmark_pass = candidate_latency is not None
    speedup = None
    if baseline_latency and candidate_latency:
        speedup = float(baseline_latency / candidate_latency)

    compiled = bool(candidate_payload["compiled"])
    correct = bool(candidate_payload["correct"])
    failure = failure_type(compiled, correct, benchmark_pass, speedup, candidate_payload.get("error"))
    improved = bool(compiled and correct and benchmark_pass and speedup is not None and speedup >= IMPROVEMENT_THRESHOLD)

    payload = {
        "schema_version": "kernel_direction_probe_eval_v1",
        "problem": args.problem,
        "candidate": args.candidate,
        "baseline": args.baseline or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "improvement_threshold": IMPROVEMENT_THRESHOLD,
        "baseline_result": baseline_payload,
        "candidate_result": candidate_payload,
        "compile_pass": compiled,
        "correctness_pass": correct,
        "benchmark_pass": benchmark_pass,
        "latency_before_ms": baseline_latency,
        "latency_after_ms": candidate_latency,
        "speedup": speedup,
        "improved": improved,
        "kept": improved,
        "failure_type": failure,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
