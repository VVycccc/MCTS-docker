"""Fallback kernel generation (v3 single-step + AKG LangGraph frontend).

Default seed generation lives in ``naive_seed_gen.py`` (pure LLM, 0 AKG).
This module is only invoked when ``gen_mode`` is explicitly set to ``v3`` or
``akg`` — i.e. the blind-spot fallback path where naive generation fails
(e.g. attention / SDPA, where AKG can write what naive cannot; see
work-log 2026-07-16 blind ablation). The default path never imports this.

v3 = single LLM call + compile/validate + simple error-feedback retry.
akg = full vendored AKG LangGraph frontend (coder_only_workflow), with a
.pt-weights shim so ModelNew aligns with the reference's frozen weights.
"""
import asyncio
import os
import re
import sys
import time
from pathlib import Path

import torch
from openai import AsyncOpenAI

from triton_backend import (
    Problem,
    _normalize_shape,
    adaptive_rel_tol,
    anti_pytorch_check,
    compile_triton,
    resolve_problem_inputs,
    run_isolated_validation,
    validate_problem_shapes,
)
from fix_code_gen import truncate_error_log

# ── AKG vendored frontend (only touched on the akg fallback path) ──────
_AKG_ROOT = str(Path(__file__).resolve().parent / "akg_frontend")
if _AKG_ROOT not in sys.path:
    sys.path.insert(0, _AKG_ROOT)

_akg_imports_ok = False
try:
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.op.config.config_validator import load_config as akg_load_config
    from akg_agents.core.async_pool.task_pool import TaskPool as AkgTaskPool
    from akg_agents.core.worker.manager import register_local_worker as akg_register_worker
    _akg_imports_ok = True
except ImportError as e:
    print(f"[Generator] AKG frontend not available: {e}")
    LangGraphTask = None  # type: ignore
    akg_load_config = None  # type: ignore
    AkgTaskPool = None  # type: ignore
    akg_register_worker = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _extract_code(text: str) -> str | None:
    """Extract Python code from LLM output.

    Supports two formats:
    1. Markdown-wrapped: ```python ... ```
    2. Pure code: no markdown wrapping (AKG convention)
    """
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if text.strip():
        return text.strip()
    return None


def _render(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


async def _chat(client, model, system, user, temperature=0.7, max_tokens=20000):
    """Single LLM call. Returns (content, meta) — meta unused on v3 path."""
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system} if system else None,
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return content, {}
        except Exception as e:
            if attempt == 2:
                print(f"  [chat] LLM call failed after 3 retries: {e}")
                return None, {}
            await asyncio.sleep(2 ** attempt)
    return None, {}


def _extract_expanded_reference(problem: Problem) -> str:
    """完整 reference（含 Model 类 + run）。验证用，不能截断。"""
    return problem.reference


# ---------------------------------------------------------------------------
# v3 worker: single LLM call → compile → validate → simple error-feedback retry
# ---------------------------------------------------------------------------

async def _worker_v3(
    worker_id: int,
    system: str,
    user: str,
    problem: Problem,
    model_cfg: dict,
    success_event: asyncio.Event,
    max_refinements: int,
    temperature: float,
    weights: dict | None = None,
    config: dict | None = None,
) -> dict | None:
    """Single LLM call → compile → validate → simple error feedback retry."""
    client = AsyncOpenAI(base_url=model_cfg["url"], api_key=model_cfg["api_key"])
    reference_source = _extract_expanded_reference(problem)
    model = model_cfg["model"]

    code, _ = await _chat(client, model, system, user, temperature)
    if code is None:
        return None
    code = _extract_code(code)
    if code is None:
        return None

    for rnd in range(max_refinements + 1):
        if success_event.is_set():
            return None

        # Anti-PyTorch check
        violation = anti_pytorch_check(code)
        if violation:
            if rnd >= max_refinements:
                return None
            retry_user = f"""你的代码触发了反 PyTorch 检测：{violation}

【要求】
- 不要调用任何 torch.xxx 算子（如 torch.matmul, torch.relu, F.softmax 等）
- 所有计算必须用 Triton kernel（tl.xxx）实现
- 完整重写整个 kernel
- 只输出 ```python 代码块"""
            new_code, _ = await _chat(client, model, "", retry_user, temperature=0.3)
            if new_code:
                extracted = _extract_code(new_code)
                if extracted:
                    code = extracted
            continue

        # Compile
        fn, err = compile_triton(code)
        if fn is None:
            if rnd >= max_refinements:
                return None
            compile_err = truncate_error_log(err or "unknown", max_len=3000)
            print(f"  [W{worker_id} r{rnd}] compile error: {compile_err[:120]}")
            retry_user = f"""编译失败，请分析错误并重写整个 kernel。

【编译错误】
{compile_err}

【要求】
- 完整重写 Triton kernel
- 检查 @triton.jit 装饰器、tl.constexpr 参数、类型注解
- 检查 tl.load/tl.store 的参数顺序和 mask 用法
- 只输出 ```python 代码块"""
            new_code, _ = await _chat(client, model, "", retry_user, temperature=0.3)
            if new_code:
                extracted = _extract_code(new_code)
                if extracted:
                    code = extracted
            continue

        # Validate
        validation = run_isolated_validation(
            kernel_source=code,
            reference_source=reference_source,
            problem=problem,
            rel_tol=adaptive_rel_tol(problem),
            abs_tol=1e-5,
            timeout_seconds=120,
            device="cuda",
            weights=weights,
            config=config,
        )
        if not validation.success:
            fail_msg = str(validation.details.get('message') or validation.stderr or validation.stdout)
            fail_msg = truncate_error_log(fail_msg, max_len=3000)
            fail_kind = validation.failure_kind or "unknown"
            print(f"  [W{worker_id} r{rnd}] validation fail ({fail_kind}): {fail_msg[:120]}")
            if rnd >= max_refinements:
                return None

            retry_hints = []
            if fail_kind == "runtime":
                if "out of resource: shared memory" in fail_msg:
                    retry_hints.append("Shared memory 溢出 → 减小 BLOCK_SIZE 或使用更小的 tile。当前硬件限制 101KB/SM。")
                elif "CUDA out of memory" in fail_msg or "out of memory" in fail_msg:
                    retry_hints.append("显存不足 → 减少同时持有的张量，逐块计算并及时释放。")
                elif "illegal memory access" in fail_msg or "out of bounds" in fail_msg:
                    retry_hints.append("越界访问 → 检查 mask 和边界条件，确保 offset + BLOCK 不超过张量维度。")
                else:
                    retry_hints.append("运行时错误 → 仔细检查索引计算、mask 条件、数据类型匹配。")
            elif fail_kind == "compile_error":
                retry_hints.append("编译错误 → 检查 Triton 语法、constexpr 参数、tl.load/store 调用是否正确。")
            elif fail_kind == "numerical_mismatch":
                retry_hints.append("数值不匹配 → 检查计算公式、dtype 转换、数值稳定性（如 softmax 用 online 算法）。")
            elif fail_kind in ("subprocess_error", "reference_fail"):
                retry_hints.append("验证脚本错误 → 检查 kernel 输入输出 shape/dtype 是否与 reference 一致。")

            retry_user = f"""你的 Triton kernel 验证失败，请分析错误并重写整个 kernel。

【错误类型】{fail_kind}
【完整错误】
{fail_msg}

【修复建议】
{chr(10).join(f'- {h}' for h in retry_hints)}

【要求】
- 完整重写 Triton kernel，不要只是微调
- 如果涉及 tiling，确保 BLOCK_SIZE 适配 101KB shared memory 限制
- 注意边界条件、mask、dtype 转换
- 只输出 ```python 代码块"""
            new_code, _ = await _chat(client, model, "", retry_user, temperature=0.3)
            if new_code:
                extracted = _extract_code(new_code)
                if extracted:
                    code = extracted
            continue

        # Success — benchmark
        success_event.set()
        lat = None
        try:
            inputs = []
            resolved_inputs, input_err = resolve_problem_inputs(problem)
            if input_err:
                print(f"  [W{worker_id}] Benchmark skipped: {input_err}")
                lat = None
            else:
                for spec in resolved_inputs or []:
                    shape = spec.get("shape")
                    dtype = getattr(torch, spec["dtype"])
                    if shape is None:
                        t = torch.randn((), dtype=dtype, device="cuda")
                    else:
                        t = torch.randn(*shape, dtype=dtype, device="cuda")
                    inputs.append(t)
                else:
                    from triton_backend import benchmark
                    lat = benchmark(fn, *inputs)
        except Exception:
            lat = None
        finally:
            torch.cuda.empty_cache()

        return {
            "worker_id": worker_id,
            "code": code,
            "latency_ms": lat,
            "refinement_rounds": rnd,
        }

    return None


# ---------------------------------------------------------------------------
# Public entry: generate_kernel (v3 / akg fallback paths only)
# ---------------------------------------------------------------------------

async def generate_kernel(
    problem: Problem,
    baseline_code: str,
    optimization_plan: str,
    config: dict,
    weights: dict | None = None,
) -> dict | None:
    """Fallback kernel generation. Only called when gen_mode is v3 or akg
    (the default naive path lives in naive_seed_gen.gen_seed).

    - v3: single LLM call + simple error feedback retry.
    - akg: full AKG LangGraph frontend (coder_only_workflow) with .pt shim.
    """
    ok, shape_err = validate_problem_shapes(problem)
    if not ok:
        print(f"[Generator] Invalid problem shapes: {shape_err}")
        return None

    gen_mode = config.get("gen_mode", "akg")
    prompt_dir = Path(config["prompt_dir"])
    model_cfg = config.get("model_frontend", config["model"])
    num_workers = config.get("gen_workers", 1)
    max_refinements = config.get("gen_refinement_rounds", 4)
    temperature = config.get("gen_temperature", 0.7)

    if gen_mode == "akg":
        return await _generate_kernel_akg(
            problem, baseline_code, optimization_plan, config,
            weights=weights,
        )
    # v3 (and any unknown mode defaults to v3)
    return await _generate_kernel_v3(
        problem, baseline_code, optimization_plan, config,
        prompt_dir, model_cfg, num_workers, max_refinements, temperature,
        weights=weights,
    )


async def _generate_kernel_v3(
    problem, baseline_code, optimization_plan, config,
    prompt_dir, model_cfg, num_workers, max_refinements, temperature,
    weights: dict | None = None,
) -> dict | None:
    """Single-step generation with simple error feedback."""
    system = _read(str(prompt_dir / "generate_v3_system.txt"))
    if "{triton_api_ref}" in system:
        ref_path = prompt_dir / "triton_api_reference.md"
        if ref_path.exists():
            system = system.replace("{triton_api_ref}", _read(str(ref_path)))

    user_template = _read(str(prompt_dir / "generate_v3_user.txt"))
    user = _render(
        user_template,
        optimization_plan=optimization_plan,
        kernel_code=baseline_code,
        rag_examples="",
    )

    success_event = asyncio.Event()
    tasks = [
        _worker_v3(
            i, system, user,
            problem, model_cfg, success_event,
            max_refinements, temperature,
            weights=weights,
            config=config,
        )
        for i in range(num_workers)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        if r is not None:
            print(f"[Generator v3] Worker {r['worker_id']} won in {r['refinement_rounds']} refinement rounds, latency={r['latency_ms']}")
            return r
    print("[Generator v3] All workers failed")
    return None


# ---------------------------------------------------------------------------
# AKG mode: full vendored LangGraphTask (replaces pipeline)
# ---------------------------------------------------------------------------

def _extract_akg_task_desc(reference: str) -> str:
    """Extract the AKG-compatible task description from a Level 2 reference.

    Strips DirecTune-specific expanded reference sections so AKG's
    KernelVerifier sees only the original nn.Module + get_inputs/get_init_inputs.
    """
    markers = [
        "\n# --- EXPANDED REFERENCE ---",
        "\n# Frozen weights",
        "\nimport torch as _torch",
    ]
    for marker in markers:
        idx = reference.find(marker)
        if idx > 0:
            reference = reference[:idx]
    return reference.strip()


async def _generate_kernel_akg(
    problem, baseline_code, optimization_plan, config,
    weights: dict | None = None,
) -> dict | None:
    """Generate a Triton kernel using the full AKG LangGraph frontend.

    Creates a LangGraphTask with coder_only_workflow, registers a local CUDA
    worker, and runs the graph until success or timeout. AKG handles skill
    selection, j2 prompt rendering, Coder→CodeChecker→Verifier→Conductor→
    FixCodeGen routing, safety nets.
    """
    if not _akg_imports_ok:
        print("[Generator AKG] AKG frontend imports failed — falling back to v3")
        return None

    model_cfg = config.get("model_frontend", config.get("model"))
    if not model_cfg:
        model_cfg = {"url": "https://api.openai.com/v1", "model": "gpt-4", "api_key": ""}
    timeout_per_gen = config.get("timeout_seconds", 600)

    # Bridge DirecTune config.yaml → AKG env vars (AKG reads AIKG_* for API config)
    _prev_akg_env = {}
    for _key in ("AIKG_BASE_URL", "AIKG_API_KEY", "AIKG_MODEL_NAME",
                 "AKG_AGENTS_BASE_URL", "AKG_AGENTS_API_KEY", "AKG_AGENTS_MODEL_NAME"):
        _prev_akg_env[_key] = os.environ.get(_key, "")
    os.environ["AIKG_BASE_URL"] = model_cfg.get("url", "")
    os.environ["AIKG_API_KEY"] = model_cfg.get("api_key", "")
    os.environ["AIKG_MODEL_NAME"] = model_cfg.get("model", "")
    os.environ["AKG_AGENTS_BASE_URL"] = model_cfg.get("url", "")
    os.environ["AKG_AGENTS_API_KEY"] = model_cfg.get("api_key", "")
    os.environ["AKG_AGENTS_MODEL_NAME"] = model_cfg.get("model", "")

    task_desc = _extract_akg_task_desc(problem.reference)
    op_name = f"accel_{problem.name}"
    print(f"[Generator AKG] op_name={op_name}, task_desc={len(task_desc)} chars")

    try:
        await akg_register_worker([0], backend="cuda", arch="a100")
        akg_config = akg_load_config("triton_cuda", backend="cuda")
        task = LangGraphTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id="0",
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            config=akg_config,
            framework="torch",
            workflow="coder_only_workflow",
        )
        pool = AkgTaskPool()
        pool.create_task(task.run)
        t0 = time.time()
        results = await asyncio.wait_for(
            pool.wait_all(), timeout=timeout_per_gen,
        )
        elapsed = time.time() - t0

        for _, success, _final_state in results:
            if success:
                final_code = _final_state.get("coder_code", "")
                if not final_code:
                    print(f"[Generator AKG] Success but no coder_code in state")
                    return None

                print(f"[Generator AKG] PASSED in {elapsed:.1f}s, code={len(final_code)} chars")

                # AKG outputs ModelNew class; shim with def run() for our verifier.
                # 用 .pt frozen 权重覆盖 ModelNew 的 manual_seed(0) 权重（按参数顺序+形状
                # copy，绕过 ModelNew 改层名导致 load_state_dict key 不匹配），使 ModelNew
                # 权重与 v5 reference(.pt seed42) 对齐，run_isolated_validation 能通过。
                if "class ModelNew" in final_code and "def run(" not in final_code:
                    import re as _re
                    _wp_match = _re.search(r"_weights_path\s*=\s*['\"]([^'\"]+)['\"]", problem.reference)
                    _wp = _wp_match.group(1) if _wp_match else ""
                    final_code = final_code + f"""

# --- DirecTune shim (.pt weights aligned, cached) ---
{task_desc}

_weights_path = "{_wp}"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        torch.manual_seed(0)
        _MODEL = ModelNew(*get_init_inputs())
        _ref = Model(*get_init_inputs())
        _ref.load_state_dict(torch.load(_weights_path, map_location='cpu', weights_only=True))
        _rp = list(_ref.parameters()); _np = list(_MODEL.parameters())
        for _pn, _pr in zip(_np, _rp):
            if _pn.shape == _pr.shape:
                _pn.data.copy_(_pr.data)
        _MODEL = _MODEL.to(x.device).eval()
    return _MODEL(x, *args)
"""

                lat = None
                try:
                    fn, err = compile_triton(final_code)
                    if fn is None:
                        print(f"[Generator AKG] Compile error after AKG success: {err[:120]}")
                    else:
                        inputs = []
                        resolved_inputs, input_err = resolve_problem_inputs(problem)
                        if input_err:
                            print(f"[Generator AKG] Benchmark skipped: {input_err}")
                        else:
                            for spec in resolved_inputs or []:
                                shape = spec.get("shape")
                                dtype = getattr(torch, spec["dtype"])
                                if shape is None:
                                    t = torch.randn((), dtype=dtype, device="cuda")
                                else:
                                    t = torch.randn(*shape, dtype=dtype, device="cuda")
                                inputs.append(t)
                            else:
                                from triton_backend import benchmark
                                lat = benchmark(fn, *inputs)
                        print(f"[Generator AKG] latency={lat}")
                except Exception:
                    pass
                finally:
                    torch.cuda.empty_cache()

                return {
                    "worker_id": 0,
                    "code": final_code,
                    "latency_ms": lat,
                    "refinement_rounds": _final_state.get("step_count", 0),
                }
            else:
                err = _final_state.get("verifier_error", "") if isinstance(_final_state, dict) else str(_final_state)
                print(f"[Generator AKG] FAILED in {elapsed:.1f}s: {err[:200]}")

    except asyncio.TimeoutError:
        print(f"[Generator AKG] TIMEOUT after {timeout_per_gen}s")
    except Exception as e:
        print(f"[Generator AKG] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return None
