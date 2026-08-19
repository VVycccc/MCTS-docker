"""Hardware profiler abstraction layer.

Provides a unified interface for GPU hardware performance counter profiling
with backend-specific implementations (NVIDIA NCU, AMD rocprof, etc.).

All backends return a flat dict of semantically-named metrics — callers never
need to know raw vendor-specific metric names.
"""

from __future__ import annotations

import os
import sys

from abc import ABC, abstractmethod
from typing import Any

from triton_backend import _normalize_shape, resolve_problem_inputs


# ---------------------------------------------------------------------------
# Unified metric key definitions (semantic, vendor-agnostic)
# ---------------------------------------------------------------------------

METRIC_KEYS = {
    "compute_util_pct":         "Compute unit utilization (%)",
    "memory_bw_util_pct":       "DRAM bandwidth utilization (%)",
    "l1_hit_rate_pct":          "L1 cache hit rate (%)",
    "l2_hit_rate_pct":          "L2 cache hit rate (%)",
    "occupancy_pct":            "Warp/wavefront occupancy (%)",
    "stall_memory_pct":         "Stall on memory dependency (%)",
    "stall_sync_pct":           "Stall on barrier/sync (%)",
    "stall_other_pct":          "Stall on other dependencies (%)",
    "registers_per_thread":     "Registers per thread",
    "tensor_core_util_pct":     "Tensor/matrix core utilization (%)",
}


def summarize_profile_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Derive a compact bottleneck summary from unified hardware metrics.

    Returns a small vendor-agnostic diagnosis block suitable for prompts and
    logs.  Missing metrics simply reduce confidence; callers can fall back to
    raw metrics only when this returns None.
    """
    if not metrics:
        return None

    compute = metrics.get("compute_util_pct")
    memory = metrics.get("memory_bw_util_pct")
    occupancy = metrics.get("occupancy_pct")
    registers = metrics.get("registers_per_thread")
    tensor = metrics.get("tensor_core_util_pct")
    l1_hit = metrics.get("l1_hit_rate_pct")
    l2_hit = metrics.get("l2_hit_rate_pct")
    stall_mem = metrics.get("stall_memory_pct")
    stall_sync = metrics.get("stall_sync_pct")
    stall_other = metrics.get("stall_other_pct")

    evidence: list[str] = []
    actions: list[str] = []
    bottleneck = "mixed"
    confidence = "medium"

    if memory is not None and memory >= 70:
        evidence.append(f"memory_bw_util_pct={memory:.1f} is high")
    if stall_mem is not None and stall_mem >= 30:
        evidence.append(f"stall_memory_pct={stall_mem:.1f} indicates memory stalls")
    if occupancy is not None and occupancy < 50:
        evidence.append(f"occupancy_pct={occupancy:.1f} is low")
    if registers is not None and registers >= 128:
        evidence.append(f"registers_per_thread={registers} is high")
    if tensor is not None and tensor < 20:
        evidence.append(f"tensor_core_util_pct={tensor:.1f} is low")
    if compute is not None and compute < 60:
        evidence.append(f"compute_util_pct={compute:.1f} is low")
    if l2_hit is not None and l2_hit < 60:
        evidence.append(f"l2_hit_rate_pct={l2_hit:.1f} is low")
    if l1_hit is not None and l1_hit < 80:
        evidence.append(f"l1_hit_rate_pct={l1_hit:.1f} is low")

    if (
        memory is not None and memory >= 70
        and stall_mem is not None and stall_mem >= 30
    ):
        bottleneck = "memory_bound"
        confidence = "high"
        actions = [
            "increase data reuse with larger or better-shaped tiles",
            "improve global memory coalescing or vectorized loads",
            "use more software pipelining / num_stages to hide memory latency",
        ]
    elif (
        occupancy is not None and occupancy < 50
        and registers is not None and registers >= 128
    ):
        bottleneck = "register_limited"
        confidence = "high"
        actions = [
            "reduce accumulator footprint or split large tiles",
            "simplify the kernel to reduce live values",
            "trade some ILP for higher occupancy",
        ]
    elif (
        compute is not None and compute >= 60
        and tensor is not None and tensor < 20
    ):
        bottleneck = "tensor_core_underused"
        confidence = "medium"
        actions = [
            "restructure math around tl.dot / MMA-friendly tiles",
            "increase tile sizes enough to amortize launch overhead",
            "check whether data layout blocks tensor-core friendly access",
        ]
    elif (
        compute is not None and compute < 60
        and memory is not None and memory < 60
    ):
        bottleneck = "underutilized"
        confidence = "medium"
        actions = [
            "focus on the dominant stall reason before retuning hyperparameters",
            "increase useful work per program instance or reduce launch overhead",
            "improve occupancy and instruction scheduling",
        ]
    elif stall_sync is not None and stall_sync >= 15:
        bottleneck = "sync_bound"
        confidence = "medium"
        actions = [
            "reduce unnecessary synchronization",
            "pipeline stages to overlap work",
            "use warp-local or lockstep-friendly organization where possible",
        ]
    elif stall_other is not None and stall_other >= 20:
        bottleneck = "dependency_bound"
        confidence = "medium"
        actions = [
            "reduce register dependencies and long dependency chains",
            "simplify arithmetic or change instruction ordering",
            "consider modest unrolling only if register pressure stays acceptable",
        ]
    elif memory is not None and compute is not None:
        bottleneck = "compute_bound" if compute >= memory else "memory_tilted"
        confidence = "low"
        actions = [
            "adjust tiling based on the stronger resource signal",
            "confirm bottleneck with occupancy, cache-hit, and stall metrics",
        ]

    return {
        "bottleneck": bottleneck,
        "confidence": confidence,
        "headline": bottleneck.replace("_", " "),
        "evidence": evidence[:4],
        "suggested_actions": actions[:3],
    }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class HardwareProfiler(ABC):
    """Abstract hardware profiler.

    Each backend implements two methods:

        available() → bool
            Returns True if the profiler tool and permissions are available.

        profile(source, problem, config) → dict | None
            Profiles a correct Triton kernel and returns a flat dict of
            semantically-named metrics.  Returns None on any failure (callers
            fall back silently to latency-only feedback).
    """

    @abstractmethod
    def available(self) -> bool:
        """Check if this profiler backend is usable right now."""
        ...

    @abstractmethod
    def profile(self, source: str, problem, config: dict) -> dict | None:
        """Collect hardware metrics for a kernel.  Returns None on failure."""
        ...


# ---------------------------------------------------------------------------
# NoOp (disabled / fallback)
# ---------------------------------------------------------------------------

class NoOpProfiler(HardwareProfiler):
    def available(self) -> bool:
        return False

    def profile(self, source: str, problem, config: dict) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# NVIDIA NCU Profiler
# ---------------------------------------------------------------------------

class NcuProfiler(HardwareProfiler):
    """Profiles Triton kernels with NVIDIA Nsight Compute.

    Requirements (all checked in available()):
      - ``ncu`` binary on PATH
      - ``sudo -n ncu`` works (passwordless sudo configured)
      - pandas + numpy available for CSV parsing
    """

    # Raw NCU metric name → unified key
    METRIC_MAP: dict[str, str] = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed":       "compute_util_pct",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed":     "memory_bw_util_pct",
        "l1tex__t_sector_hit_rate.pct":                           "l1_hit_rate_pct",
        "lts__t_sector_hit_rate.pct":                             "l2_hit_rate_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active":      "occupancy_pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "stall_memory_pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct":   "stall_sync_pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "stall_other_pct",
        "launch__registers_per_thread":                           "registers_per_thread",
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": "tensor_core_util_pct",
    }

    # Additional raw metrics to include alongside the mapped ones
    EXTRA_RAW_METRICS = [
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
        "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
        "smsp__sass_average_branch_targets_threads_uniform.pct",
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    ]

    def available(self) -> bool:
        import shutil
        import subprocess
        if shutil.which("ncu") is None:
            return False
        try:
            result = subprocess.run(
                ["sudo", "-n", "ncu", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def profile(self, source: str, problem, config: dict) -> dict | None:
        import os
        import re
        import uuid
        import tempfile
        import traceback

        if not self.available():
            return None

        # --- extract kernel names ---
        kernel_names = re.findall(
            r'@triton\.jit\s*(?:\([^)]*\))?\s*\n\s*def\s+(\w+)\s*\(',
            source,
        )
        kernel_names = list(dict.fromkeys(kernel_names))

        # --- temp directory ---
        tag = uuid.uuid4().hex[:8]
        tmp_dir = os.path.join(tempfile.gettempdir(), f"DirecTune_ncu_{tag}")
        os.makedirs(tmp_dir, exist_ok=True)

        csv_path = os.path.join(tmp_dir, "metrics.csv")
        wrapper_path = os.path.join(tmp_dir, "bench.py")

        try:
            # --- write kernel source ---
            module_name = f"kernel_{tag}"
            kernel_path = os.path.join(tmp_dir, f"{module_name}.py")
            with open(kernel_path, "w") as f:
                f.write(source)
            os.chmod(kernel_path, 0o644)

            # --- write bench wrapper ---
            wrapper_code = _build_ncu_bench_wrapper(tmp_dir, module_name, problem)
            if not wrapper_code:
                return None  # shape validation failed, skip NCU
            with open(wrapper_path, "w") as f:
                f.write(wrapper_code)
            os.chmod(wrapper_path, 0o755)

            # --- python path (must be absolute for sudo ncu) ---
            profiler_cfg = config.get("hw_profiler", {})
            conda_python = profiler_cfg.get(
                "ncu_python",
                os.environ.get("DT_NCU_PYTHON", sys.executable),
            )
            conda_bin = os.path.dirname(conda_python)

            # --- run NCU (subprocess directly, avoid run_ncu.py's sys.exit) ---
            import subprocess as _sp
            import shutil as _sh

            ncu_bin = _sh.which("ncu") or "/usr/bin/ncu"
            repeat = profiler_cfg.get("repeat", 100)

            # Build the same metrics list as run_ncu.py
            from run_ncu import METRICS as NCU_METRICS_STR

            # Build kernel name filter
            name_filter: list[str] = []
            if kernel_names:
                if len(kernel_names) == 1:
                    name_filter = [f"--kernel-name={kernel_names[0]}"]
                else:
                    pattern = "|".join(re.escape(k) for k in kernel_names)
                    name_filter = [f"--kernel-name=::regex:^({pattern})(\\(|$)"]
            else:
                print("[NcuProfiler] No explicit @triton.jit kernel names found; profiling all launched kernels and selecting the hottest rows")

            env = os.environ.copy()
            env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

            cmd = [
                "sudo", "-n", ncu_bin,
                "--csv", "--page=raw",
                "--kernel-name-base=demangled",
                "--target-processes=all",
                "--replay-mode=kernel",
                "--profile-from-start=on",
                f"--log-file={csv_path}",
                f"--metrics={NCU_METRICS_STR}",
                "--launch-skip=0", "--launch-count=10",
            ] + name_filter + [
                conda_python, wrapper_path, "--repeat", str(repeat),
            ]

            print(f"[NcuProfiler] running: {' '.join(cmd)}")
            proc = _sp.run(cmd, env=env, capture_output=True, text=True, timeout=300)

            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-500:]
                print(f"[NcuProfiler] ncu returned {proc.returncode}: {stderr_tail}")
                # NCU often returns non-zero on warnings; try to parse CSV anyway

            # --- parse results ---
            from run_ncu import load_ncu_metrics

            df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",))
            if df is None or df.empty:
                print("[NcuProfiler] No metrics collected from NCU output")
                return None

            if not kernel_names and "sm__cycles_active.avg" in df.columns:
                df = df.sort_values("sm__cycles_active.avg", ascending=False).head(5)
            elif kernel_names and "Kernel Name" in df.columns:
                filtered = df[
                    df["Kernel Name"].astype(str).apply(
                        lambda name: any(k in name for k in kernel_names)
                    )
                ]
                if not filtered.empty:
                    df = filtered

            # --- map to unified keys ---
            # load_ncu_metrics returns wide-format DataFrame:
            #   each row = one kernel launch, each column = one metric
            # We average across all launches for the same kernel, then map keys.
            metrics: dict[str, Any] = {}

            for col in df.columns:
                if col == "Kernel Name":
                    continue
                values = df[col].dropna()
                if len(values) == 0:
                    continue
                avg_val = float(values.mean())

                # Map to unified key, or keep raw name for extra metrics
                unified = self.METRIC_MAP.get(col, col)
                metrics[unified] = round(avg_val, 1)

            print(f"[NcuProfiler] Collected {len(metrics)} metrics for "
                  f"{len(kernel_names)} kernel(s)")
            return metrics

        except Exception as e:
            print(f"[NcuProfiler] Profiling failed: {e}")
            traceback.print_exc()
            return None

        finally:
            import time
            time.sleep(0.3)
            try:
                for fpath in [kernel_path, wrapper_path, csv_path]:
                    if os.path.exists(fpath):
                        os.unlink(fpath)
                if os.path.exists(tmp_dir):
                    os.rmdir(tmp_dir)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Bench wrapper helper (shared by all backends that need a standalone script)
# ---------------------------------------------------------------------------

def _build_ncu_bench_wrapper(tmp_dir: str, module_name: str, problem) -> str:
    """Generate a standalone Python script that imports and runs the kernel.

    The script creates random inputs matching the problem spec, warms up,
    then runs the kernel in a loop (driven by ``--repeat`` arg).
    """
    input_lines = []
    resolved_inputs, input_err = resolve_problem_inputs(problem)
    if input_err:
        print(f"[NcuProfiler] Skipping NCU: {input_err}")
        return ""
    for spec in resolved_inputs or []:
        shape = spec["shape"]
        norm_shape, err = _normalize_shape(shape)
        if err:
            print(f"[NcuProfiler] Skipping NCU: non-concrete shape {shape} ({err})")
            return ""
        dtype = spec["dtype"]
        name = spec["name"]
        if norm_shape is None:
            input_lines.append(
                f"{name} = torch.randn((), dtype=torch.{dtype}, device='cuda')"
            )
        else:
            shape_repr = repr(norm_shape)
            input_lines.append(
                f"{name} = torch.randn({shape_repr}, dtype=torch.{dtype}, device='cuda')"
            )

    input_args = ", ".join(spec["name"] for spec in problem.inputs)

    lines = [
        "#!/usr/bin/env python3",
        '"""NCU bench wrapper — auto-generated."""',
        "import argparse",
        "import sys",
        "import torch",
        "",
        f"sys.path.insert(0, {repr(tmp_dir)})",
        f"from {module_name} import run",
        "",
    ]
    for il in input_lines:
        lines.append(il)
    lines.append("")
    lines.append("for _ in range(10):")
    lines.append(f"    run({input_args})")
    lines.append("")
    lines.append("parser = argparse.ArgumentParser()")
    lines.append("parser.add_argument('--repeat', type=int, default=100)")
    lines.append("args = parser.parse_args()")
    lines.append("")
    lines.append("for _ in range(args.repeat):")
    lines.append(f"    run({input_args})")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_profiler(config: dict) -> HardwareProfiler:
    """Build a HardwareProfiler instance from the config dict.

    Reads ``hw_profiler.backend`` from config:
      - ``"ncu"``   → NcuProfiler
      - ``"noop"``  → NoOpProfiler (default if key missing)
    """
    profiler_cfg = config.get("hw_profiler", {})
    backend = profiler_cfg.get("backend", "noop")

    if backend == "ncu":
        return NcuProfiler()
    elif backend == "noop":
        return NoOpProfiler()
    else:
        print(f"[HardwareProfiler] Unknown backend '{backend}', using noop")
        return NoOpProfiler()
