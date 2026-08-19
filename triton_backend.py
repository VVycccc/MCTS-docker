"""Minimal Triton backend: compile, benchmark, correctness check."""

import os
import re
import sys
import math
import time
import json
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Anti-PyTorch-fallback detection (from KernelAgent)
# ---------------------------------------------------------------------------

def _strip_comments_and_strings(code: str) -> str:
    return re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#.*)', "", code)


DISALLOWED_TORCH_PATTERNS: list[tuple[re.Pattern, str]] = [
    # AKG output format uses class ModelNew(torch.nn.Module) — allow nn.Module, nn.Parameter
    # but BLOCK torch.nn.functional compute shortcuts and torch high-level compute ops.
    (re.compile(r"\bimport\s+torch\.nn\.functional\s+as\s+F\b"), "aliasing torch.nn.functional as F is not allowed"),
    (re.compile(r"\btorch\.nn\.functional\b"), "torch.nn.functional usage is not allowed"),
    (re.compile(r"\bF\.(relu|sigmoid|tanh|softmax|gelu|mish|silu|swish|leaky_relu|elu|mish|hardtanh|hardswish|log_softmax|logsumexp|max_pool|avg_pool|adaptive_avg_pool|layer_norm|batch_norm|group_norm|instance_norm|interpolate|cross_entropy|nll_loss|conv)\s*\("),
     "torch.nn.functional compute calls are not allowed"),
    (re.compile(r"\btorch\.conv"), "torch convolution helpers are not allowed"),
    (re.compile(r"\btorch\.(relu|sigmoid|tanh|softmax|gelu|mish|hardtanh|max_pool|avg_pool)[A-Za-z0-9_]*\("),
     "PyTorch activation/pooling helpers are not allowed"),
    (re.compile(r"\btorch\.ops\.aten\b"), "torch.ops.aten.* calls are not allowed"),
    (re.compile(r"\btorch\.(matmul|mm|bmm)\s*\("), "PyTorch matmul/mm/bmm ops are not allowed"),
    (re.compile(r"\.(matmul|mm|bmm)\s*\("), "Tensor.matmul/mm/bmm methods are not allowed"),
    (re.compile(r"\btorch\.einsum\s*\("), "torch.einsum is not allowed"),
    (re.compile(r"\.einsum\s*\("), "Tensor.einsum is not allowed"),
    (re.compile(r"\bimport\s+inspect\b"), "inspect-based reflection is not allowed"),
    (re.compile(r"\binspect\.(stack|currentframe|getouterframes)\s*\("), "inspect introspection is not allowed"),
    (re.compile(r"\bsys\._getframe\s*\("), "sys._getframe is not allowed"),
    (re.compile(r"\.f_locals\b|\.f_globals\b"), "frame locals/globals access is not allowed"),
    (re.compile(r"\bglobals\s*\("), "globals() is not allowed in kernels"),
    (re.compile(r"\blocals\s*\("), "locals() is not allowed in kernels"),
]


def anti_pytorch_check(code: str) -> str | None:
    """Scan generated code for PyTorch compute patterns. Returns violation or None."""
    sanitized = _strip_comments_and_strings(code)
    for pattern, message in DISALLOWED_TORCH_PATTERNS:
        if pattern.search(sanitized):
            return message
    return None


# ---------------------------------------------------------------------------
def _patch_device_in_source(source: str) -> str:
    """Patch torch.randn/ones/zeros/empty calls to include device=_dev.

    Detects L1-style references where ``def run(x):`` creates tensors without
    ``device=``, causing CPU/CUDA mismatches when inputs are on GPU.
    """
    if not ("def run(" in source and any(
        fn in source for fn in ("torch.randn(", "torch.ones(", "torch.zeros(", "torch.empty(")
    )):
        return source

    import re as _re
    # Extract actual first parameter name from def run(XXX, ...)
    first_param = "x"  # default
    m = _re.search(r'def run\((\w+)', source)
    if m:
        first_param = m.group(1)
    lines = source.split("\n")
    new_lines = []
    injected = False
    for line in lines:
        new_lines.append(line)
        if not injected and line.strip().startswith("def run("):
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * (indent + 4) + f"_dev = {first_param}.device")
            injected = True
    source = "\n".join(new_lines)
    source = _re.sub(
        r'torch\.randn\((\[[^\]]+\])\)',
        r'torch.randn(\1, device=_dev)',
        source,
    )
    source = _re.sub(
        r'torch\.ones\(\*(\[[^\]]+\])\)',
        r'torch.ones(*\1, device=_dev)',
        source,
    )
    source = _re.sub(
        r'torch\.zeros\((\d+)\)',
        r'torch.zeros(\1, device=_dev)',
        source,
    )
    source = _re.sub(
        r'torch\.zeros\(\*(\[[^\]]+\])\)',
        r'torch.zeros(*\1, device=_dev)',
        source,
    )
    source = _re.sub(
        r'torch\.empty\(\*(\[[^\]]+\])\)',
        r'torch.empty(*\1, device=_dev)',
        source,
    )
    return source
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """Kernel problem specification."""
    name: str
    inputs: list[dict]   # [{"name": "A", "shape": [1024, 512], "dtype": "float16"}, ...]
    outputs: list[dict]  # [{"name": "C", "shape": [1024, 512], "dtype": "float16"}, ...]
    reference: str       # Python source of reference implementation, must define ref(*inputs) -> outputs


def adaptive_rel_tol(problem: "Problem") -> float:
    """容差按数值特性自适应。
    - level2（nn.Module，TF32 + 大 K 累积误差）：1e-2
    - level1 大归约维（任一 input 维 > 1e5，fp32 累加 10万+ 次误差）：1e-2
    - level1 小算子：1e-3（严格抓 elementwise bug）

    大 K 放宽正当性：fp32 累加 N 次的物理误差 ~sqrt(N)*eps，N=1e5 时已超 1e-3；
    1e-2 仍能区分精度极限(<0.01)与真 bug(通常>0.05)。详见 work-log 2026-06-21。
    """
    ref = problem.reference or ""
    if "nn.Module" in ref or "Model(" in ref:
        return 1e-2
    max_dim = 0
    for inp in (problem.inputs or []):
        for d in (inp.get("shape") or []):
            if isinstance(d, int) and d > max_dim:
                max_dim = d
    return 1e-2 if max_dim > 100_000 else 1e-3


# 全局 token 用量累积（验证 skill 瘦身等优化的实际 token 节省）
TOKEN_USAGE = {"prompt": 0, "completion": 0, "calls": 0}


def record_usage(resp) -> None:
    """Accumulate LLM token usage from a completion response."""
    u = getattr(resp, "usage", None)
    if u:
        TOKEN_USAGE["prompt"] += getattr(u, "prompt_tokens", 0) or 0
        TOKEN_USAGE["completion"] += getattr(u, "completion_tokens", 0) or 0
        TOKEN_USAGE["calls"] += 1


@dataclass
class ValidationResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    failure_kind: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    compiled: bool = False
    runnable: bool = False
    correct: bool = False
    latency_ms: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2norm_allclose(v_k: np.ndarray, v_r: np.ndarray, rel_tol: float = 1e-5) -> bool:
    a = v_k.astype(np.float64)
    b = v_r.astype(np.float64)
    return bool(np.linalg.norm(a - b) < rel_tol * np.linalg.norm(b))


def _extract_expanded_reference(source: str) -> str:
    """返回完整 reference（含 Model 类 + run）。
    通用 run() 调 Model.forward()，需要 Model 类，不能截断 expanded 段（截断会丢 Model）。"""
    return source


def _normalize_shape(shape: Any) -> tuple[list[int] | None, str | None]:
    """Convert a shape spec to concrete ints when possible."""
    if shape is None:
        return None, None
    if not isinstance(shape, (list, tuple)):
        return None, f"shape must be a list/tuple or null, got {type(shape).__name__}"
    normalized: list[int] = []
    for dim in shape:
        if isinstance(dim, bool):
            return None, f"invalid bool dimension {dim!r}"
        if isinstance(dim, int):
            normalized.append(dim)
            continue
        if isinstance(dim, float) and dim.is_integer():
            normalized.append(int(dim))
            continue
        return None, f"non-concrete dimension {dim!r}"
    return normalized, None


_INPUT_METADATA_CACHE: dict[int, list[dict[str, Any]] | None] = {}


def _infer_input_metadata_from_get_inputs(reference_source: str, timeout_seconds: int = 60) -> list[dict[str, Any]] | None:
    """Infer input tensor metadata by executing reference get_inputs() in a subprocess.

    Some converted KernelBench JSON files use ``shape: null`` for tensors whose
    real shapes are only available from get_inputs().  Return metadata only so
    callers can distinguish real scalars from unresolved tensor shapes.
    """
    if "def get_inputs" not in (reference_source or ""):
        return None
    cache_key = hash(reference_source)
    if cache_key in _INPUT_METADATA_CACHE:
        return _INPUT_METADATA_CACHE[cache_key]

    runner = r'''
import json
import runpy
import sys
import torch

path = sys.argv[1]
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
ns = runpy.run_path(path)
get_inputs = ns.get("get_inputs")
if get_inputs is None:
    print(json.dumps({"ok": False, "error": "missing get_inputs"}))
    raise SystemExit(0)
with torch.no_grad():
    values = get_inputs()
if not isinstance(values, (list, tuple)):
    values = [values]
meta = []
for v in values:
    if isinstance(v, torch.Tensor):
        dtype = str(v.dtype).replace("torch.", "")
        meta.append({"is_tensor": True, "shape": list(v.shape), "ndim": int(v.dim()), "dtype": dtype})
    else:
        meta.append({"is_tensor": False, "shape": None, "ndim": None, "dtype": type(v).__name__})
print(json.dumps({"ok": True, "inputs": meta}))
'''
    inferred: list[dict[str, Any]] | None = None
    with tempfile.TemporaryDirectory(prefix="DirecTune_inputs_") as tmpdir:
        tmp = Path(tmpdir)
        ref_path = tmp / "reference.py"
        runner_path = tmp / "infer_inputs.py"
        ref_path.write_text(reference_source)
        runner_path.write_text(runner)
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(ref_path)],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={
                    **os.environ,
                    "MKL_THREADING_LAYER": "GNU",
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                },
            )
            if proc.returncode == 0:
                lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
                if lines:
                    payload = json.loads(lines[-1])
                    if payload.get("ok") and isinstance(payload.get("inputs"), list):
                        inferred = payload["inputs"]
        except Exception:
            inferred = None

    _INPUT_METADATA_CACHE[cache_key] = inferred
    return inferred


def resolve_problem_inputs(problem: Problem) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return input specs with null tensor shapes resolved from reference get_inputs().

    Concrete JSON shapes remain authoritative.  Null shapes are kept as None for
    real scalar tensors, or replaced with concrete shapes when get_inputs()
    proves the corresponding input is non-scalar.
    """
    metadata = None
    if any(spec.get("shape") is None for spec in problem.inputs):
        metadata = _infer_input_metadata_from_get_inputs(problem.reference)

    resolved_inputs: list[dict[str, Any]] = []
    for idx, spec in enumerate(problem.inputs):
        shape, err = _normalize_shape(spec.get("shape"))
        if err:
            return None, f"Input {spec.get('name', '?')} has invalid shape: {err}"
        resolved = dict(spec)
        if shape is None and metadata and idx < len(metadata):
            m = metadata[idx]
            if m.get("is_tensor") and (m.get("ndim") or 0) > 0:
                shape = [int(d) for d in (m.get("shape") or [])]
                if m.get("dtype") and not resolved.get("dtype"):
                    resolved["dtype"] = m["dtype"]
        resolved["shape"] = shape
        resolved_inputs.append(resolved)
    return resolved_inputs, None


def validate_problem_shapes(problem: Problem) -> tuple[bool, str | None]:
    """Validate that problem input shapes are concrete enough for generation-time tests."""
    for spec in problem.inputs:
        shape, err = _normalize_shape(spec.get("shape"))
        if err:
            return False, f"Input {spec.get('name', '?')} has invalid shape: {err}"
    return True, None


def compare_outputs(
    kernel_outputs: tuple[Any, ...],
    ref_outputs: tuple[Any, ...],
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-5,
) -> tuple[bool, str | None]:
    """Compare outputs using the same policy as check_correctness."""
    if len(kernel_outputs) != len(ref_outputs):
        return False, f"Output count mismatch: kernel={len(kernel_outputs)} ref={len(ref_outputs)}"

    for i, (vk, vr) in enumerate(zip(kernel_outputs, ref_outputs)):
        vk_np = vk.detach().cpu().numpy() if isinstance(vk, torch.Tensor) else np.array(vk)
        vr_np = vr.detach().cpu().numpy() if isinstance(vr, torch.Tensor) else np.array(vr)

        if vk_np.shape != vr_np.shape:
            return False, f"Output {i} shape mismatch: kernel={vk_np.shape} ref={vr_np.shape}"

        diff = np.abs(vk_np - vr_np)
        max_abs = float(np.max(diff))
        l2 = float(np.linalg.norm((vk_np - vr_np).astype(np.float64)))
        l2_ref = float(np.linalg.norm(vr_np.astype(np.float64)))
        l2_rel = l2 / l2_ref if l2_ref > 0 else l2

        if l2_rel < rel_tol or max_abs < abs_tol:
            continue

        return False, f"Output {i} value mismatch: max_diff={max_abs:.6f} l2_rel={l2_rel:.6f}"

    return True, None


def _load_source(source: str, entry_point: str = "run") -> tuple[Callable | None, str | None]:
    """Dynamically load a function from Python source code. Returns (callable, error).

    Writes source to a temp file then imports it, because @triton.jit requires
    the source to be in a real file, not exec'd from a string.
    """
    import importlib.util
    import tempfile
    import uuid

    # Write to temp file (Triton requires file-based source)
    tmpdir = tempfile.gettempdir()
    module_name = f"_DirecTune_{uuid.uuid4().hex[:8]}"
    filepath = os.path.join(tmpdir, f"{module_name}.py")
    with open(filepath, "w") as f:
        f.write(source)

    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return None, "Failed to create module spec from file"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        try:
            os.unlink(filepath)
        except OSError:
            pass
        return None, f"Source execution failed: {e}"

    fn = getattr(module, entry_point, None)
    if fn is None:
        return None, f"No `{entry_point}` function found in source"
    return fn, None


# ---------------------------------------------------------------------------
# num_warps injection (FlagTree triton 3.6 bug guard)
# ---------------------------------------------------------------------------

def _num_warps_for_block(block: int | None) -> int:
    """Infer num_warps from a tile BLOCK size. FlagTree 3.6.0 has a compiler
    bug where consecutive compiles of kernels with different BLOCK sizes (and
    default num_warps inference) can return wrong results. Forcing explicit
    num_warps on every launch avoids it.
    """
    if block is None or block <= 0:
        return 4
    if block <= 1024:
        return 1
    if block <= 2048:
        return 2
    if block <= 4096:
        return 4
    if block <= 8192:
        return 8
    return 16


def _ensure_num_warps(source: str) -> str:
    """Inject explicit num_warps into @triton.jit kernel launches missing it.

    Walks the AST for `kernel[grid](...)` subscript-call patterns (Triton's
    kernel launch syntax) and adds `num_warps=<inferred>` when the call has no
    num_warps kwarg. Inference: scan the launch's keyword args for the largest
    BLOCK-like constexpr (names containing BLOCK / with value a power of 2 ≥ 64);
    falls back to num_warps=4. Also handles @triton.autotune Configs lacking
    num_warps (adds a default). Only re-emits via ast.unparse when a change was
    made; otherwise returns source unchanged to avoid reformatting noise.

    This guards against the FlagTree 3.6.0 multi-specialization bug. Under
    stock triton (forge) it's a no-op (called only when triton_backend==forge_tle).
    """
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # let the real compile error surface later

    changed = False

    def _looks_like_block(name: str) -> bool:
        up = name.upper()
        return up.startswith("BLOCK") or up in {"BM", "BN", "BK", "BT"}

    def _infer_from_kwargs(kwargs) -> int:
        best = None
        for kw in kwargs:
            if not isinstance(kw, ast.keyword):
                continue
            if isinstance(kw.arg, str) and _looks_like_block(kw.arg):
                try:
                    val = ast.literal_eval(kw.value)
                except Exception:
                    val = None
                if isinstance(val, int) and val > 0 and (val & (val - 1)) == 0 and val >= 64:
                    if best is None or val > best:
                        best = val
        return _num_warps_for_block(best)

    for node in ast.walk(tree):
        # kernel[grid](...) launches: Call whose func is a Subscript
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            kwargs = node.keywords
            has_nw = any(isinstance(k, ast.keyword) and k.arg == "num_warps" for k in kwargs)
            if not has_nw:
                nw = _infer_from_kwargs(kwargs)
                kwargs.append(ast.keyword(arg="num_warps", value=ast.Constant(value=nw)))
                changed = True

        # @triton.autotune(Config(...)) — add num_warps to Configs lacking it
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "autotune":
            for arg in node.args:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    for elt in arg.elts:
                        if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name) \
                                and elt.func.id == "Config":
                            kw_names = {k.arg for k in elt.keywords if isinstance(k, ast.keyword)}
                            if "num_warps" not in kw_names and "num_stages" not in kw_names:
                                # only add when no num_warps at all; pick a safe default
                                elt.keywords.append(ast.keyword(
                                    arg="num_warps", value=ast.Constant(value=4)))
                                changed = True

    if not changed:
        return source
    try:
        import ast as _ast
        return _ast.unparse(tree)
    except Exception:
        return source


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def compile_triton(source: str) -> tuple[Callable | None, str | None]:
    """Try to load the `run` function from source. Returns (callable, error)."""
    return _load_source(source, "run")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    fn: Callable,
    *args: Any,
    warmup: int = 10,
    reps: int = 100,
    device: int = 0,
) -> float:
    """Measure latency in milliseconds using CUDA events with L2 cache clearing."""
    # L2 cache clearing buffer (256 MB)
    cache_clear = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        cache_clear.zero_()
        fn(*args)

    # Adaptive reps: 慢 kernel 减少重复次数，避免 search 阶段验证累积超时
    # （40ms kernel 的 100 reps ≈ 4.4s → 25 reps ≈ 1s；快 kernel 不受影响）
    torch.cuda.synchronize(device)
    cache_clear.zero_()
    start.record()
    fn(*args)
    end.record()
    torch.cuda.synchronize(device)
    single_ms = start.elapsed_time(end)
    if single_ms > 10:
        reps = min(reps, max(20, int(1000 / max(single_ms, 0.1))))

    # Timed iterations
    times = []
    for _ in range(reps):
        cache_clear.zero_()
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    return float(np.median(times))


def _validation_script() -> str:
    return r'''
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch


def _load_fn(source_path: str, entry_point: str):
    spec = importlib.util.spec_from_file_location(Path(source_path).stem, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, entry_point, None)
    if fn is None:
        raise RuntimeError(f"No `{entry_point}` function found in {source_path}")
    return fn


def _compare_outputs(kernel_outputs, ref_outputs, rel_tol: float, abs_tol: float):
    if len(kernel_outputs) != len(ref_outputs):
        return False, f"Output count mismatch: kernel={len(kernel_outputs)} ref={len(ref_outputs)}"
    # 分块对比：避免 materialize 全 diff/float 拷贝（huge tensor 如 [4096,393216]=6GB 会让
    # (vk-vr).abs() + .float() 多分配 ~3× 张量 → 24GB GPU OOM）。每块 1M 元素(~4MB)增量累加。
    CHUNK = 1 << 20
    for i, (vk, vr) in enumerate(zip(kernel_outputs, ref_outputs)):
        vk_t = vk if isinstance(vk, torch.Tensor) else torch.as_tensor(vk)
        vr_t = vr if isinstance(vr, torch.Tensor) else torch.as_tensor(vr)
        if vk_t.shape != vr_t.shape:
            return False, f"Output {i} shape mismatch: kernel={tuple(vk_t.shape)} ref={tuple(vr_t.shape)}"
        n = vk_t.numel()
        if n == 0:
            continue
        flat_vk = vk_t.reshape(-1)
        flat_vr = vr_t.reshape(-1)
        max_abs = 0.0
        sq_diff = 0.0
        sq_ref = 0.0
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            ck = flat_vk[s:e].float()
            cr = flat_vr[s:e].float()
            d = ck - cr
            max_abs = max(max_abs, float(d.abs().max().item()))
            sq_diff += float((d * d).sum().item())
            sq_ref += float((cr * cr).sum().item())
        l2 = sq_diff ** 0.5
        l2_ref = sq_ref ** 0.5
        l2_rel = l2 / l2_ref if l2_ref > 0 else l2
        if l2_rel < rel_tol or max_abs < abs_tol:
            continue
        return False, f"Output {i} value mismatch: max_diff={max_abs:.6f} l2_rel={l2_rel:.6f}"
    return True, None


def main():
    cfg_path = sys.argv[1]
    cfg = json.loads(Path(cfg_path).read_text())
    kernel_fn = _load_fn(cfg["kernel_path"], "run")
    ref_fn = None
    for entry_point in ("ref", "run"):
        try:
            ref_fn = _load_fn(cfg["reference_path"], entry_point)
            break
        except Exception:
            continue
    if ref_fn is None:
        raise RuntimeError("No `ref` or `run` function found in reference source")

    device = cfg.get("device", "cuda")

    # Fix random seed so reference and kernel generate matching weights
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    # Load weights if provided
    weights = None
    weights_path = cfg.get("weights_path")
    if weights_path and Path(weights_path).exists():
        weights = torch.load(weights_path, map_location=device, weights_only=True)

    inputs = []
    for spec in cfg["inputs"]:
        shape = spec["shape"]
        dtype = getattr(torch, spec["dtype"])
        if shape is None:
            t = torch.randn((), dtype=dtype, device=device)
        elif spec.get("random", True):
            t = torch.randn(*shape, dtype=dtype, device=device)
        else:
            t = torch.zeros(*shape, dtype=dtype, device=device)
        if spec.get("zero_input", False):
            t.zero_()
        inputs.append(t)

    try:
        if weights is not None:
            ref_outputs = ref_fn(*inputs, weights=weights)
        else:
            ref_outputs = ref_fn(*inputs)
    except Exception as e:
        payload = {"success": False, "failure_kind": "reference_fail", "message": f"Reference execution failed: {e}", "traceback": traceback.format_exc()}
        print(json.dumps(payload))
        return 0

    if not isinstance(ref_outputs, tuple):
        ref_outputs = (ref_outputs,)

    try:
        if weights is not None:
            try:
                kernel_outputs = kernel_fn(*inputs, weights=weights)
            except TypeError:
                # Kernel doesn't accept weights kwarg — fall back
                kernel_outputs = kernel_fn(*inputs)
        else:
            kernel_outputs = kernel_fn(*inputs)
    except Exception as e:
        payload = {"success": False, "failure_kind": "runtime", "message": f"Kernel execution failed: {e}", "traceback": traceback.format_exc()}
        print(json.dumps(payload))
        return 0

    if not isinstance(kernel_outputs, tuple):
        kernel_outputs = (kernel_outputs,)

    ok, message = _compare_outputs(kernel_outputs, ref_outputs, cfg["rel_tol"], cfg["abs_tol"])
    if not ok:
        payload = {"success": False, "failure_kind": "numerical_mismatch", "message": message}
        print(json.dumps(payload))
        return 0

    payload = {"success": True, "failure_kind": "", "message": "PASS"}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def run_isolated_validation(
    kernel_source: str,
    reference_source: str,
    problem: Problem,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-5,
    timeout_seconds: int = 60,
    device: str = "cuda",
    weights: dict | None = None,
    config: dict | None = None,
) -> ValidationResult:
    """Validate kernel vs reference in an isolated subprocess using temp files.

    If *weights* is provided, each value should be a torch.Tensor.  The dict is
    saved to a ``.pt`` file in the temp directory and the path is included in
    the runner config so the reference function receives it as a ``weights=``
    keyword argument.
    """
    # forge_tle / FlagTree triton 3.6: force explicit num_warps to dodge the
    # multi-specialization compiler bug before writing kernel source to disk.
    if config and config.get("triton_backend", "forge") == "forge_tle":
        kernel_source = _ensure_num_warps(kernel_source)
    with tempfile.TemporaryDirectory(prefix="DirecTune_validate_") as tmpdir:
        tmp = Path(tmpdir)
        kernel_path = tmp / "kernel.py"
        reference_path = tmp / "reference.py"
        runner_path = tmp / "runner.py"
        config_path = tmp / "config.json"
        weights_path = tmp / "weights.pt"

        kernel_path.write_text(kernel_source)
        # Ensure reference has standard imports (Fuser may omit them)
        if "import torch.nn.functional" not in reference_source:
            reference_source = "import torch\nimport torch.nn.functional as F\n" + reference_source
        reference_source = _patch_device_in_source(reference_source)
        reference_path.write_text(reference_source)
        runner_path.write_text(_validation_script())

        # Save weights if provided
        if weights:
            torch.save(weights, weights_path)

        resolved_inputs, input_err = resolve_problem_inputs(problem)
        if input_err:
            return ValidationResult(
                success=False,
                failure_kind="invalid_shape",
                stderr=input_err,
                details={"input": "?"},
            )
        normalized_inputs = resolved_inputs or []

        config_data = {
            "kernel_path": str(kernel_path),
            "reference_path": str(reference_path),
            "inputs": normalized_inputs,
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
            "device": device,
        }
        if weights:
            config_data["weights_path"] = str(weights_path)
        config_path.write_text(json.dumps(config_data))

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(config_path)],
                cwd=os.getcwd(),  # 项目根：让 reference 里的相对 _weights_path（level1）能解析（level2 用绝对路径）
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={
                    **os.environ,
                    "MKL_THREADING_LAYER": "GNU",
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                },
            )
        except subprocess.TimeoutExpired as e:
            return ValidationResult(
                success=False,
                failure_kind="timeout",
                stdout=e.stdout or "",
                stderr=e.stderr or f"Validation timed out after {timeout_seconds}s",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            return ValidationResult(
                success=False,
                failure_kind="subprocess_error",
                stdout=stdout,
                stderr=stderr or f"validation subprocess exited {proc.returncode}",
            )

        payload_line = ""
        for line in reversed(stdout.splitlines()):
            if line.strip():
                payload_line = line.strip()
                break
        if not payload_line:
            return ValidationResult(
                success=False,
                failure_kind="malformed_result",
                stdout=stdout,
                stderr=stderr or "validation subprocess produced no JSON result",
            )

        try:
            payload = json.loads(payload_line)
        except json.JSONDecodeError:
            return ValidationResult(
                success=False,
                failure_kind="malformed_result",
                stdout=stdout,
                stderr=stderr or f"Invalid JSON result: {payload_line[:200]}",
            )

        return ValidationResult(
            success=bool(payload.get("success")),
            stdout=stdout,
            stderr=stderr,
            failure_kind=str(payload.get("failure_kind") or ("" if payload.get("success") else "unknown")),
            details={
                **({"message": payload.get("message")} if payload.get("message") is not None else {}),
                **{k: v for k, v in payload.items() if k not in {"success", "failure_kind", "message"}},
            },
        )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def check_correctness(
    fn: Callable,
    problem: Problem,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-5,
) -> tuple[bool, str | None]:
    """Run kernel and reference, compare outputs.

    Uses both relative AND absolute tolerance:
    - rel_tol=1e-3: catches elementwise ops and minor regressions
      (standard for TF32/TensorCore operations where mantissa is 10 bits)
    - abs_tol=1e-5: catches large absolute errors even when rel passes
    A value is considered correct if EITHER check passes.
    """
    # Build reference function (tries "ref" first, then "run")
    ref_source = _extract_expanded_reference(problem.reference)
    ref_source = _patch_device_in_source(ref_source)

    ref_fn, _ = _load_source(ref_source, "ref")
    if ref_fn is None:
        ref_fn, _ = _load_source(ref_source, "run")
    if ref_fn is None:
        return False, "No `ref` or `run` function found in problem.reference."

    # Generate inputs
    resolved_inputs, input_err = resolve_problem_inputs(problem)
    if input_err:
        return False, input_err
    inputs = []
    for spec in resolved_inputs or []:
        shape = spec.get("shape")
        dtype = getattr(torch, spec["dtype"])
        if shape is None:
            # Scalar input
            t = torch.randn((), dtype=dtype, device="cuda")
        elif spec.get("random", True):
            t = torch.randn(*shape, dtype=dtype, device="cuda")
        else:
            t = torch.zeros(*shape, dtype=dtype, device="cuda")
        inputs.append(t)

    # Run reference
    try:
        ref_outputs = ref_fn(*inputs)
    except Exception as e:
        return False, f"Reference execution failed: {e}"

    if not isinstance(ref_outputs, tuple):
        ref_outputs = (ref_outputs,)

    # Run kernel
    try:
        kernel_outputs = fn(*inputs)
    except Exception as e:
        return False, f"Kernel execution failed: {e}"

    if not isinstance(kernel_outputs, tuple):
        kernel_outputs = (kernel_outputs,)

    ok, message = compare_outputs(kernel_outputs, ref_outputs, rel_tol=rel_tol, abs_tol=abs_tol)
    if not ok:
        return False, message

    # Clean up CUDA cache between correctness checks
    torch.cuda.empty_cache()

    return True, None


# ---------------------------------------------------------------------------
# Profile (compile + benchmark + correctness)
# ---------------------------------------------------------------------------

def profile(
    source: str,
    problem: Problem,
    warmup: int = 10,
    reps: int = 100,
    timeout_seconds: float = 300,
    device: int = 0,
    rel_tol: float = 1e-3,
    config: dict | None = None,
) -> ProfileResult:
    """Full profiling pipeline: compile → correctness → benchmark."""
    import signal

    result = ProfileResult()

    # forge_tle / FlagTree triton 3.6: force explicit num_warps on launches to
    # dodge the multi-specialization compiler bug. No-op under stock triton.
    if config and config.get("triton_backend", "forge") == "forge_tle":
        source = _ensure_num_warps(source)

    # Step 1: Compile (patch device first for L1-style references)
    source = _patch_device_in_source(source)
    fn, err = compile_triton(source)
    if fn is None:
        result.error = f"Compile error: {err}"
        return result
    result.compiled = True

    # Step 2: Correctness (with alarm-based timeout for long-running checks)
    ok: bool = False
    msg: str | None = None

    def _on_timeout(signum, frame):
        raise TimeoutError("Correctness check timed out")

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(int(timeout_seconds))
    try:
        ok, msg = check_correctness(fn, problem, rel_tol=rel_tol)
    except TimeoutError:
        result.error = "Timeout during correctness check"
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return result
    except Exception as e:
        result.error = f"Correctness check error: {e}"
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    result.correct = ok
    result.runnable = True
    if not ok:
        result.error = msg
        return result

    # Step 3: Benchmark
    try:
        resolved_inputs, input_err = resolve_problem_inputs(problem)
        if input_err:
            result.error = f"Benchmark shape error: {input_err}"
            return result
        inputs = []
        for spec in resolved_inputs or []:
            shape = spec.get("shape")
            dtype = getattr(torch, spec["dtype"])
            if shape is None:
                t = torch.randn((), dtype=dtype, device=device)
            else:
                t = torch.randn(*shape, dtype=dtype, device=device)
            if spec.get("zero_input", False):
                t.zero_()
            inputs.append(t)

        lat = benchmark(fn, *inputs, warmup=warmup, reps=reps, device=device)
        result.latency_ms = lat
    except Exception as e:
        result.error = f"Benchmark failed: {e}"

    # Clean up CUDA cache between profile calls
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Problem I/O
# ---------------------------------------------------------------------------

def load_problem(path: str) -> Problem:
    with open(path) as f:
        data = json.load(f)
    reference = data["reference"]
    # 部分早期转换的 JSON 把权重写成绝对路径（/home/xxx/..._weights.pt）或项目根
    # 相对路径，迁移到别的机器/目录后失效。权重文件约定与 problem JSON 同目录，
    # 这里统一重写为按 JSON 所在目录解析的绝对路径，下游（generator shim、隔离
    # 验证子进程）拿到即可直接 torch.load。
    ref_dir = Path(path).resolve().parent
    reference = re.sub(
        r'(_weights_path\s*=\s*)(["\'])([^"\']*\.pt)\2',
        lambda m: f'{m.group(1)}{m.group(2)}{ref_dir / Path(m.group(3)).name}{m.group(2)}',
        reference,
    )
    return Problem(
        name=data["name"],
        inputs=data["inputs"],
        outputs=data["outputs"],
        reference=reference,
    )
