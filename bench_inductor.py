"""Measure torch.compile (inductor) latencies on the DirecTune problem sets.

Uses the same CUDA-event timing convention as triton_backend.benchmark()
(warmup + timed reps, L2 cache clear between reps) so inductor numbers are
directly comparable with the paper's eager/champion numbers.

Memory strategy: the driver runs each problem in a fresh subprocess (clean
CUDA context, no cross-problem retention) and the correctness check is
sample-based — eager output is sampled then freed before the compiled run,
so peak memory is input + one output instead of input + clones + two outputs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python bench_inductor.py \
      --problems 40_layernorm 14_Gemm_Divide_Sum_Scaling ... \
      [--level 1|2] [--modes default max-autotune] [--out out.json]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent

PROBLEM_DIR = {
    1: REPO / "problems" / "kb_level1",
    2: REPO / "problems" / "kb_level2",
}

SAMPLE_ELEMS = 1_000_000  # correctness sample size (flat strided slice)


def _load_reference(source: str):
    ns: dict = {}
    exec(compile(source, "<reference>", "exec"), ns)
    return ns["run"], ns


def _make_inputs(problem, resolve_problem_inputs, device=0):
    resolved, err = resolve_problem_inputs(problem)
    if err:
        raise RuntimeError(f"input resolve: {err}")
    inputs = []
    for spec in resolved or []:
        dtype = getattr(torch, spec["dtype"])
        shape = spec.get("shape")
        t = (
            torch.randn((), dtype=dtype, device=device)
            if shape is None
            else torch.randn(*shape, dtype=dtype, device=device)
        )
        if spec.get("zero_input", False):
            t.zero_()
        inputs.append(t)
    return inputs


_CACHE_CLEAR = None


def _cache_clear(device=0):
    global _CACHE_CLEAR
    if _CACHE_CLEAR is None:
        _CACHE_CLEAR = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)
    return _CACHE_CLEAR


def bench_like_harness(fn, args, warmup=10, reps=100, device=0):
    """Mirror triton_backend.benchmark: L2 clear + CUDA events + median."""
    cache_clear = _cache_clear(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        cache_clear.zero_()
        fn(*args)
    torch.cuda.synchronize(device)
    cache_clear.zero_()
    start.record()
    fn(*args)
    end.record()
    torch.cuda.synchronize(device)
    single_ms = start.elapsed_time(end)
    if single_ms > 10:
        reps = min(reps, max(20, int(1000 / max(single_ms, 0.1))))
    times = []
    for _ in range(reps):
        cache_clear.zero_()
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    return float(np.median(times)), single_ms


def _sample(out):
    """Small flat sample of an output tensor (or tuple of them)."""
    outs = out if isinstance(out, tuple) else (out,)
    samples = []
    for t in outs:
        if isinstance(t, torch.Tensor) and t.numel() > SAMPLE_ELEMS:
            step = max(1, t.numel() // SAMPLE_ELEMS)
            samples.append(t.flatten()[::step].detach().cpu())
        elif isinstance(t, torch.Tensor):
            samples.append(t.detach().cpu())
        else:
            samples.append(t)
    return samples


def _check_sampled(fn, inputs, ref_samples):
    got = fn(*inputs)
    got_samples = _sample(got)
    del got
    torch.cuda.empty_cache()
    if len(got_samples) != len(ref_samples):
        return False
    for a, b in zip(got_samples, ref_samples):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                return False
            if not torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-3):
                return False
        elif a != b:
            return False
    return True


def run_single(name: str, level: int, modes: list[str]) -> dict:
    sys.path.insert(0, str(REPO))
    from triton_backend import (  # noqa: E402
        _patch_device_in_source,
        load_problem,
        resolve_problem_inputs,
    )

    ppath = PROBLEM_DIR[level] / f"{name}.json"
    if not ppath.is_file():
        return {"error": f"problem json not found: {ppath}"}
    problem = load_problem(str(ppath))
    ref_src = _patch_device_in_source(problem.reference)
    try:
        eager_fn, ns = _load_reference(ref_src)
    except Exception as e:
        return {"error": f"reference exec: {e}"}

    inputs = _make_inputs(problem, resolve_problem_inputs)
    rec: dict = {"level": level}

    # eager reference latency + correctness reference samples
    try:
        with torch.no_grad():
            eager_ms, _ = bench_like_harness(eager_fn, inputs)
            ref_out = eager_fn(*inputs)
            ref_samples = _sample(ref_out)
            del ref_out
        rec["eager_ms"] = eager_ms
        torch.cuda.empty_cache()
    except Exception as e:
        rec["eager_error"] = f"{type(e).__name__}: {e}"[:200]
        return rec

    for mode in modes:
        try:
            t0 = time.time()
            compiled = torch.compile(eager_fn, mode=mode)
            with torch.no_grad():
                ok = _check_sampled(compiled, inputs, ref_samples)
                lat_ms, single = bench_like_harness(compiled, inputs)
            rec[mode] = {
                "latency_ms": lat_ms,
                "first_iter_ms": single,
                "correct": bool(ok),
                "compile_wall_s": round(time.time() - t0, 1),
            }
        except Exception as e:
            rec[mode] = {"error": f"{type(e).__name__}: {e}"[:300]}
        finally:
            compiled = None
            torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+")
    ap.add_argument("--single", type=str, default=None, help="internal: run one problem in this process")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--modes", nargs="+", default=["default", "max-autotune"])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.single:
        rec = run_single(args.single, args.level, args.modes)
        Path(args.out).write_text(json.dumps({args.single: rec}, indent=1))
        print(f"[{args.single}] " + " ".join(
            f"{m}={rec.get(m, {}).get('latency_ms', rec.get(m, {}).get('error', rec.get('error', '?')))}"
            for m in args.modes
        ), flush=True)
        return

    torch.backends.cuda.matmul.allow_tf32 = True  # driver process: nothing GPU-side

    out_path = Path(args.out)
    out: dict = {}
    if out_path.is_file():
        out = json.loads(out_path.read_text())

    todo = [p for p in args.problems if p not in out or any(
        "error" in out[p].get(m, {}) for m in args.modes
    )]
    tmp_dir = out_path.parent / "single"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name in todo:
        tmp = tmp_dir / f"{name}.json"
        r = subprocess.run(
            [sys.executable, __file__, "--single", name,
             "--level", str(args.level), "--modes", *args.modes,
             "--out", str(tmp)],
            capture_output=True, text=True, timeout=1800,
            env=None,
        )
        if tmp.is_file():
            out.update(json.loads(tmp.read_text()))
        else:
            out[name] = {"error": f"subprocess failed rc={r.returncode}: {r.stderr[-300:]}"}
        out_path.write_text(json.dumps(out, indent=1))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
