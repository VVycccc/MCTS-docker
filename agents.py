"""Planner, Executor, Summarizer LLM agents.

Each agent reads .txt prompts, renders {var} templates, calls the LLM,
and returns structured results. Prompt paths and model config come from
the config dict — nothing is hardcoded.
"""

import re
import json
import asyncio
import time
import uuid
import traceback
from pathlib import Path

from openai import AsyncOpenAI

from triton_backend import Problem, ProfileResult, profile as triton_profile, anti_pytorch_check, adaptive_rel_tol, record_usage
from hardware_profiler import create_profiler, HardwareProfiler, summarize_profile_metrics
from fix_code_gen import parse_modifications, DiffApplier, truncate_error_log

# Module-level profiler cache (lazily created once per process)
_profiler: HardwareProfiler | None = None


def _get_profiler(config: dict) -> HardwareProfiler:
    global _profiler
    if _profiler is None:
        _profiler = create_profiler(config)
    return _profiler


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _render(template: str, **kwargs) -> str:
    """Replace {var} placeholders in template with kwargs values."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _extract_code(text: str) -> str | None:
    """Extract the Triton kernel code from an LLM response.

    Model-robust (ds4-flash 实测比 GLM-5.2 更易偏离格式，逐级兜底)：
    1. ```python ... ``` (或 ``` ... ```) 代码块；
    2. 响应整体未加围栏，但含代码特征（import triton / def run / @triton.jit）→ 视为纯代码返回；
    3. 都没命中 → None。
    """
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "import triton" in text or "def run" in text or "@triton" in text:
        return text.strip()
    return None


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

async def _chat(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 20000,
    timeout: float = 600,
) -> str | None:
    """Single async chat completion with retries."""
    for attempt in range(5):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            record_usage(resp)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 4:
                print(f"[LLM] Failed after 5 retries: {e}")
                return None
            await asyncio.sleep(2 ** attempt)


async def _gather_llm(client, model, system, user, n: int, temperature: float = 0.7) -> list[str]:
    """Fire N parallel LLM calls, return non-None responses."""
    tasks = [_chat(client, model, system, user, temperature) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def _format_hw_profile(latency: float | None, hw_metrics: dict | None) -> str:
    """Format latency + optional hardware metrics for planner/summarizer prompts."""
    summary = summarize_profile_metrics(hw_metrics)
    if not hw_metrics:
        return f"latency: {latency} ms" if latency is not None else "latency: unknown"

    lines = [f"latency: {latency} ms" if latency is not None else "latency: unknown"]
    if summary:
        lines.extend([
            "",
            "=== Bottleneck Summary ===",
            f"  bottleneck: {summary.get('bottleneck', 'unknown')}",
            f"  confidence: {summary.get('confidence', 'unknown')}",
        ])
        for item in summary.get("evidence", []):
            lines.append(f"  evidence: {item}")
        for item in summary.get("suggested_actions", []):
            lines.append(f"  suggested_action: {item}")

    lines.extend(["", "=== Hardware Metrics ==="])
    rendered_keys: set[str] = set()
    preferred_order = [
        "compute_util_pct", "memory_bw_util_pct",
        "occupancy_pct", "tensor_core_util_pct",
        "l1_hit_rate_pct", "l2_hit_rate_pct",
        "stall_memory_pct", "stall_sync_pct", "stall_other_pct",
        "registers_per_thread",
    ]

    for key in preferred_order:
        val = hw_metrics.get(key)
        if val is None:
            continue
        rendered_keys.add(key)
        lines.append(f"  {key}: {val:.1f}" if isinstance(val, float) else f"  {key}: {val}")

    for key, val in hw_metrics.items():
        if key in rendered_keys or key.startswith("_"):
            continue
        lines.append(f"  {key}: {val:.1f}" if isinstance(val, float) else f"  {key}: {val}")

    lines.append("========================")
    return "\n".join(lines)


def _format_problem(problem: Problem) -> str:
    """Format Problem as human-readable string for LLM prompts."""
    lines = [f"Name: {problem.name}"]
    lines.append("\nInputs:")
    for inp in problem.inputs:
        shape_str = "scalar" if inp["shape"] is None else f"[{', '.join(map(str, inp['shape']))}]"
        lines.append(f"  {inp['name']}: {shape_str} ({inp['dtype']})")
    lines.append("\nOutputs:")
    for out in problem.outputs:
        shape_str = "scalar" if out["shape"] is None else f"[{', '.join(map(str, out['shape']))}]"
        lines.append(f"  {out['name']}: {shape_str} ({out['dtype']})")
    lines.append(f"\nReference Implementation:\n{problem.reference}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direction classification + skill context
# ---------------------------------------------------------------------------

# Problem JSON op_type vocabulary -> normalized internal name.
_OP_TYPE_TO_SKILL = {
    "gemm": "matmul",
    "reduction": "reduce",
}


def _derive_op_type(problem: Problem, config: dict) -> str:
    """Derive the operator type for direction classification.

    The Problem dataclass drops op_type from the JSON (triton_backend.load_problem),
    so we re-read the problem JSON file for the true op_type, then map vocab.
    """
    # 1. Explicit override in config takes priority.
    ot = config.get("op_type")
    if ot:
        return _OP_TYPE_TO_SKILL.get(ot, ot)
    # 2. Re-read the problem JSON file.
    problem_path = config.get("problem")
    if problem_path:
        try:
            with open(problem_path) as f:
                data = json.load(f)
            raw = data.get("op_type", "")
            if raw:
                return _OP_TYPE_TO_SKILL.get(raw, raw)
        except Exception:
            pass
    return ""


def _load_skill_context_dynamic(config: dict, problem: Problem) -> str:
    """Search-layer skill context. v6 默认 OFF（0 skill）。

    work-log 2026-07-15 step B 消融：5 题 skill_off 全部 >= skill_on（50_conv2d off 65.5x
    vs on 1.0x），证明 skill 在搜索层是负资产。故 v6 默认不注入 skill。
    skill_mode=on 时走 akg skill catalog（消融对照用），默认 off 不触发任何外部 import。
    """
    if config.get("skill_mode", "off") != "on":
        return ""
    try:
        import sys as _sys
        akg_root = str(Path(__file__).resolve().parent / "akg_frontend")
        if akg_root not in _sys.path:
            _sys.path.insert(0, akg_root)
        from akg_agents.op.autoresearch.agent.skill_adapter import _get_catalog
        from akg_agents.op.skill import OperatorSkillCatalog
        catalog = _get_catalog()
        op_type = _derive_op_type(problem, config)
        skills = catalog.filter_by_context(
            dsl="triton_cuda", backend="cuda", framework="torch",
            hardware="a100", operator_type=op_type)
        if skills:
            rendered = OperatorSkillCatalog.render_as_markdown(skills)
            names = [getattr(s, "name", "?") for s in skills]
            print(f"  [dynamic-skill] op_type={op_type!r} -> {len(skills)} skills: {names}")
            return rendered
        print(f"  [dynamic-skill] no skills matched op_type={op_type!r}")
    except Exception as e:
        print(f"  [dynamic-skill] catalog failed: {e!r}")
    return ""


# ---------------------------------------------------------------------------
# Unified editing agent — merges planner + executor into a single agent
# that analyzes the bottleneck and emits an incremental search/replace patch
# in one inference. Falls back to full rewrite only when the patch fails to
# apply or verification keeps failing.
# ---------------------------------------------------------------------------

def _build_executor_system(config: dict) -> str:
    """Full-rewrite fallback system prompt (reused by the unified editor)."""
    prompt_dir = Path(config["prompt_dir"])
    system = _read(str(prompt_dir / "executor_system.txt"))

    # Expand {triton_api_ref} if present
    if "{triton_api_ref}" in system:
        ref_path = prompt_dir / "triton_api_reference.md"
        if ref_path.exists():
            system = system.replace("{triton_api_ref}", _read(str(ref_path)))
    # forge_tle 后端额外注入 TLE API 参考；否则替换为空串（移除占位符）。
    tle_ref = ""
    if config.get("triton_backend", "forge") == "forge_tle":
        tle_path = prompt_dir / "tle_api_reference.md"
        if tle_path.exists():
            tle_ref = _read(str(tle_path))
    system = system.replace("{tle_api_ref}", tle_ref)
    # forge_tle 软规则：强制显式 num_warps（full-rewrite 路径也要）
    if config.get("triton_backend", "forge") == "forge_tle":
        system += (
            "\n\n# num_warps 强制规则（forge_tle / FlagTree triton 3.6 专属）\n\n"
            "FlagTree triton 3.6.0 在同进程连续编译不同 BLOCK 的 kernel 时存在编译器 bug（可能返回错误结果）。"
            "所有 @triton.jit kernel 启动**必须显式指定 num_warps**：BLOCK≤1024→1, ≤2048→2, ≤4096→4, "
            "≤8192→8, >8192→16。写法 `kernel[grid](...args, BLOCK=block, num_warps=4)`。"
            "后端会 AST 检测未指定 num_warps 的启动并注入默认值，但你应显式写对。\n"
        )
    return system


def _build_executor_user(problem_str: str, kernel_code: str, plan: str, config: dict) -> str:
    path = Path(config["prompt_dir"]) / "executor_user.txt"
    template = _read(str(path))
    return _render(
        template,
        problem_code=problem_str,
        kernel_code=kernel_code,
        optimization_plan=plan,
    )


def _verify_code(code: str, problem: Problem, timeout: int, config: dict | None = None) -> tuple[ProfileResult | None, str | None]:
    """Anti-PyTorch pre-check + compile/correctness/benchmark verification.

    Returns (ProfileResult, None) on success, or (None, error_str) on failure.
    Centralizes the verification so the incremental and full-rewrite paths
    share identical checks.
    """
    ap_err = anti_pytorch_check(code)
    if ap_err:
        return None, f"[anti-pytorch] {ap_err}"
    # 后端一致性：forge 后端不应出现 tle import（free_explore 幻觉防护）
    if config and config.get("triton_backend", "forge") != "forge_tle":
        if "triton.experimental.tle" in code:
            return None, ("[backend] 代码引用了 tle (triton.experimental.tle)，"
                          "但当前后端不是 forge_tle，无法编译。请改用 stock triton API。")
    pr = triton_profile(code, problem, timeout_seconds=timeout, rel_tol=adaptive_rel_tol(problem), config=config)
    if pr.compiled and pr.correct:
        return pr, None
    if not pr.compiled:
        return None, f"[compile] {pr.error}"
    if not pr.correct:
        return None, f"[validation] {pr.error}"
    return None, pr.error or "verification failed"


def _build_unified_system(
    config: dict, problem: Problem,
) -> str:
    """System prompt for the unified editor: dynamic skill + triton ref."""
    prompt_dir = Path(config["prompt_dir"])
    template = _read(str(prompt_dir / "unified_editor_system.txt"))
    skill_ctx = _load_skill_context_dynamic(config, problem)
    triton_ref = ""
    ref_path = prompt_dir / "triton_api_reference.md"
    if ref_path.exists():
        triton_ref = _read(str(ref_path))
    # forge_tle 后端额外注入 TLE API 参考；forge 后端留空（模板占位符替换为空串）。
    tle_ref = ""
    if config.get("triton_backend", "forge") == "forge_tle":
        tle_path = prompt_dir / "tle_api_reference.md"
        if tle_path.exists():
            tle_ref = _read(str(tle_path))
    out = _render(
        template,
        triton_api_ref=triton_ref,
        tle_api_ref=tle_ref,
        skill_context=skill_ctx,
    )
    # forge_tle 软规则：强制显式 num_warps（FlagTree 3.6 多 specialization bug 规避）
    if config.get("triton_backend", "forge") == "forge_tle":
        out += (
            "\n\n# num_warps 强制规则（forge_tle / FlagTree triton 3.6 专属）\n\n"
            "FlagTree triton 3.6.0 在同进程连续编译不同 BLOCK 的 kernel 时存在编译器 bug（可能返回错误结果）。"
            "所有 @triton.jit kernel 启动**必须显式指定 num_warps**：BLOCK≤1024→1, ≤2048→2, ≤4096→4, "
            "≤8192→8, >8192→16。写法 `kernel[grid](...args, BLOCK=block, num_warps=4)`。"
            "后端会 AST 检测未指定 num_warps 的启动并注入默认值，但你应显式写对。\n"
        )
    return out


def _build_unified_user(
    kernel_code: str, problem_str: str, profile_str: str, config: dict,
    direction_directive: str = "",
) -> str:
    path = Path(config["prompt_dir"]) / "unified_editor_user.txt"
    template = _read(str(path))
    return _render(
        template,
        definition_str=problem_str,
        kernel_code=kernel_code,
        profile=profile_str,
        direction_directive=direction_directive,
    )


def _parse_unified_response(resp: str | None) -> tuple[list, str]:
    """Parse the unified editor's JSON into (modifications, change_description).

    Reuses fix_code_gen.parse_modifications for the modifications list, and
    separately extracts change_description/summary for experience seeding.
    """
    mods = parse_modifications(resp) if resp else []
    change_desc = ""
    if resp:
        try:
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                change_desc = data.get("change_description") or data.get("summary") or ""
        except Exception:
            pass
    return mods, change_desc


def _build_unified_result(
    baseline_code: str, plan_id: str, branch_id: str, code: str, pr: ProfileResult,
    change_desc: str, edit_mode: str, match_levels: dict,
    baseline_latency_map: dict, config: dict, problem: Problem,
) -> dict:
    """Build a success result entry (executor-result-compatible shape)."""
    entry: dict = {
        "baseline_code": baseline_code,
        "plan_id": plan_id,
        "branch_id": branch_id,
        "sample_id": "0",
        "code": code,
        "compiled": pr.compiled,
        "correct": pr.correct,
        "runnable": pr.runnable,
        "change_description": change_desc,
        "edit_mode": edit_mode,
        "match_levels": match_levels,
    }
    if pr.error:
        entry["error"] = pr.error
    if pr.latency_ms is not None:
        entry["latency_ms"] = pr.latency_ms
        bl = baseline_latency_map.get(baseline_code)
        if bl:
            entry["speedup"] = bl / pr.latency_ms
    if pr.compiled and pr.correct and pr.latency_ms is not None:
        try:
            profiler = _get_profiler(config)
            hw = profiler.profile(code, problem, config)
            if hw:
                entry["hw_metrics"] = hw
        except Exception as e:
            print(f"  [unified] NCU profile failed: {e!r}")
    return entry


async def _unified_edit_one(
    ci: int, ri: int, resp: str | None, baseline_code: str,
    problem: Problem, config: dict, client: AsyncOpenAI, model: str,
    baseline_latency_map: dict, problem_str: str, full_rewrite_system: str,
    branch_id: str | None = None,
) -> dict:
    """Process one sampled patch: try incremental, fall back to full rewrite.

    `resp` is the already-sampled incremental LLM response. `full_rewrite_system`
    is the executor system prompt (with dynamic skill prepended) reused across
    full-rewrite retries. `branch_id` carries the optimization-direction label
    in direction-organized mode (e.g. ``dir_5_algo``); when None it falls back
    to ``str(ci)`` (the legacy parent-index label).
    """
    plan_id = f"ue_{ci}_{ri}"
    bid = branch_id if branch_id is not None else str(ci)
    # In direction mode bid is a direction label (e.g. "dir_5_algo_equiv");
    # in legacy mode it equals str(ci) and duplicates the ue_{ci}_ prefix, so
    # only tag the log when it carries new info.
    tag = f" [{bid}]" if bid != str(ci) else ""
    timeout = config.get("timeout_seconds", 300)
    max_full = config.get("unified_fail_threshold", 3)
    change_desc = ""
    match_levels: dict = {}
    last_error = ""
    last_code: str | None = None

    # --- Phase 1: incremental search/replace patch ---
    mods, change_desc = _parse_unified_response(resp)
    if mods:
        diff = DiffApplier.apply_modifications(baseline_code, mods, raw_llm_output=resp or "")
        match_levels = dict(getattr(diff, "match_levels", {}) or {})
        if diff.success and diff.modified_code:
            new_code = diff.modified_code
            last_code = new_code
            pr, err = _verify_code(new_code, problem, timeout, config=config)
            if pr is not None:
                print(f"  [unified] ue_{ci}_{ri}{tag} ✓ incremental (match={match_levels})")
                return _build_unified_result(
                    baseline_code, plan_id, bid, new_code, pr, change_desc,
                    "incremental", match_levels, baseline_latency_map, config, problem,
                )
            last_error = err or "verification failed"
        else:
            last_error = "patch apply failed: " + ("; ".join(diff.errors) if diff.errors else "no match")
    elif resp:
        last_error = "no modifications parsed from response"
    else:
        last_error = "LLM returned None"

    # --- Phase 2: full-rewrite fallback (rule-based, no Conductor LLM) ---
    for full in range(max_full):
        user = _build_executor_user(
            problem_str, baseline_code,
            change_desc or "Rewrite the kernel to fix the error and apply the intended optimization.",
            config,
        )
        if last_code is not None and last_error:
            user += (
                f"\n\n# 上次修改失败，请修正\n## 失败的代码\n```python\n{last_code[:3000]}\n```\n"
                f"## 错误信息\n{truncate_error_log(last_error)}"
            )
        resp2 = await _chat(client, model, full_rewrite_system, user, temperature=0.3)
        if resp2 is None:
            last_error = "LLM call failed (full rewrite)"
            continue
        new_code = _extract_code(resp2)
        if new_code is None:
            last_error = "No code block found in full-rewrite response"
            continue
        last_code = new_code
        pr, err = _verify_code(new_code, problem, timeout, config=config)
        if pr is not None:
            print(f"  [unified] ue_{ci}_{ri}{tag} ✓ full_rewrite (after {full + 1} attempt(s))")
            return _build_unified_result(
                baseline_code, plan_id, bid, new_code, pr, change_desc,
                "full_rewrite", {}, baseline_latency_map, config, problem,
            )
        last_error = err or "verification failed"

    print(f"  [unified] ue_{ci}_{ri}{tag} ✗ failed: {last_error[:120] if last_error else 'unknown'}")
    return {
        "baseline_code": baseline_code, "plan_id": plan_id, "branch_id": bid,
        "sample_id": "0", "error": f"All {max_full} full-rewrite retries exhausted: {last_error}",
        "change_description": change_desc, "edit_mode": "full_rewrite", "match_levels": match_levels,
    }


# Default direction set (8 directions, v6). 优先级序 = 预期收益从高到低，作分类器失败兜底。
# 5→8 重构（work-log 2026-07-09 §极简重构 + dir_probe REPORT §4）：补 3 个 dir_probe 暴露的
# 结构性盲区维度 —— ⑥ timing_overlap（原埋在①③）、⑦ reduction_struct（原散①③⑤）、
# ⑧ control_flow_spec（原无归属）。仍按变换类型分桶，不分层。
_DEFAULT_DIRECTIONS: list[dict] = [
    {"id": 5, "name": "algo_equiv", "directive": "本次聚焦算法等价变换：寻找可预计算/降维的冗余计算（如 GEMM+Sum→matvec 预计算权重和、softmax→online softmax、重复子表达式预计算、im2col+GEMM）。若无可变换结构则跳过。"},
    {"id": 7, "name": "reduction_struct", "directive": "本次聚焦归约结构：reduction_axis_blocking（归约维分块）、split_two_stage_reduction（block 归约→global 归约两阶段）、online_reduction（online softmax/online top-k 一遍）、reduction_tree_layout（2D tile + tl.sum 轴选择树形归约）。仅当 kernel 含沿某维的归约/scan 时适用，否则跳过。注意 split_two_stage 在 memory-bound 正收益、compute-bound（split-K）可能倒退。"},
    {"id": 3, "name": "mem_layout", "directive": "本次聚焦访存 & 布局：合并全局访问、shared memory tiling、消除 bank conflict、转置/contiguous 专门化、消中间张量（避免 permute+contiguous 重建）。注意强制 contiguous 在单次读场景可能倒退。"},
    {"id": 6, "name": "timing_overlap", "directive": "本次聚焦时序/访存重叠：num_stages 软件流水线、double_buffer 多缓冲、async_copy 预取（cp.async）、compute/memory overlap（load 下一块同时算当前块）。仅适用于含多 tile 循环且 load+compute 可重叠的 kernel（GEMM K-loop、attention 多阶段），单 block reduction/elementwise 无重叠空间则跳过。"},
    {"id": 2, "name": "precision_tc", "directive": "本次聚焦精度 & Tensor Core：tl.dot 用 allow_tf32=True、考虑 fp16/bf16 输入、tile≥32。仅当算子含 matmul/dot 时应用，否则空改。"},
    {"id": 4, "name": "fusion", "directive": "本次聚焦算子融合：合并相邻 op（减中间写回显存）、epilogue 融进 GEMM/conv 主 kernel、dematerialization（中间量不落盘）。若为单 op kernel 无可合并项则跳过；GEMM 主导的 fusion 收益小。"},
    {"id": 8, "name": "control_flow_spec", "directive": "本次聚焦控制流/特化：mask_simplify（简化边界 mask）、early_exit_skip（block-uniform early-return 跳过无效程序）、constexpr/shape_spec（constexpr 化形状去动态分支）、stride_specialization（contiguous 快路径）。低杠杆但 autotune 友好，无明确特化空间则跳过。"},
    {"id": 1, "name": "tile_config", "directive": "本次聚焦 tiling/并行配置：调整 BLOCK_M/N/K、num_warps、GROUP_M、program_mapping、persistent_kernel。优先级最低，autotune 常更可靠。"},
]

# forge_tle 后端对 ⑥ timing_overlap 的增补：TLE 异步访存 + 共享内存重叠是 ⑥ 在 FlagTree
# triton 3.6 上的具体实现手段（stock triton 3.7 下 ⑥ 就是 num_stages/double_buffer）。
# forge_tle 模式下追加到分类器 prompt，让 LLM 在判 ⑥ 适用时把 TLE API 纳入 directive。
# tle_async_smem 不再单独成方向（v5 的 id 6 已让位给 timing_overlap）。
_TIMING_OVERLAP_TLE_BRIEF: str = (
    "\n\n# 方向 ⑥ timing_overlap 的 forge_tle 增补（TLE 异步访存）\n\n"
    "在 forge_tle 后端（FlagTree triton 3.6 + TLE）下，方向 ⑥ timing_overlap 除 stock triton 的 "
    "num_stages/double_buffer 外，还可使用 TLE API 显式控制访存重叠：\n"
    "- `tle.load(is_async=True)`：发起异步加载 hint\n"
    "- `tle.gpu.alloc(scope=smem)` + `tle.gpu.local_ptr`：显式 smem 缓冲 + 任意形状指针 view（addrspace=3）\n"
    "- `tle.extract_tile/insert_tile`：register/smem 子 tile 切片\n"
    "判定 ⑥ 适用（含多 tile 循环、load+compute 可重叠）且后端为 forge_tle 时，directive 应优先建议 TLE 异步重叠。\n"
    "## directive 示例（forge_tle 下 ⑥）\n"
    "本次聚焦 TLE 异步访存与共享内存重叠：用 tle.load(is_async=True) 发起异步加载，tle.gpu.alloc(scope=smem)+tle.gpu.local_ptr 分配 smem 缓冲，将下一个 tile 的 load 与当前 tile 的 compute 重叠隐藏延迟。仅适用于含多 tile 循环的 kernel（GEMM K-loop、attention 多 tile），单 block reduction/elementwise 无重叠空间则跳过。\n"
)


def _get_available_directions(config: dict) -> list[dict]:
    """方向全集（v6，固定 8 方向，不分后端增减）。

    ①-⑧ = tile_config / precision_tc / mem_layout / fusion / algo_equiv /
    timing_overlap / reduction_struct / control_flow_spec（按 _DEFAULT_DIRECTIONS 序）。
    forge_tle 后端不再 insert 独立方向——TLE 异步重叠是 ⑥ timing_overlap 的实现手段，
    通过 _TIMING_OVERLAP_TLE_BRIEF 注入分类器 prompt（见 determine_applicable_directions）。
    """
    return [dict(d) for d in _DEFAULT_DIRECTIONS]
    return base


async def determine_applicable_directions(
    problem: Problem, config: dict,
    latency: float | None, hw_metrics: dict | None,
) -> list[dict]:
    """Classify which optimization directions (①-⑧, v6) apply to this operator.
    Returns ``[{id, name, directive, reason}]`` for applicable directions,
    ordered by expected payoff (highest first).

    One LLM call (temp ``direction_classifier_temp``). Computed **once per
    episode** — operator semantics are stable across iterations, and the
    per-iteration bottleneck is already fed to ``unified_editor`` via
    ``_format_hw_profile`` independently. The NCU bottleneck (when available)
    only adjusts priority ordering, not applicability, so this works under the
    ``noop`` profiler too (judges from op_type + reference alone).

    Self-contained: builds its own client from ``model_backend`` (same pattern
    as ``unified_editor``). Falls back to the full available direction set
    (8 directions) on LLM/parse failure.
    """
    model_cfg = config.get("model_backend", config["model"])
    client = AsyncOpenAI(base_url=model_cfg["url"], api_key=model_cfg["api_key"])

    available = _get_available_directions(config)
    is_tle = config.get("triton_backend", "forge") == "forge_tle"

    prompt_dir = Path(config["prompt_dir"])
    template = _read(str(prompt_dir / "direction_classifier_system.txt"))
    system = _render(
        template,
        op_type_str=_derive_op_type(problem, config) or "(未提供)",
        op_reference=_format_problem(problem),
        skill_context=_load_skill_context_dynamic(config, problem),
        bottleneck_summary=_format_hw_profile(latency, hw_metrics),
    )
    # forge_tle 后端向分类器 system prompt 追加 ⑥ timing_overlap 的 TLE 增补，
    # 让 LLM 在判 ⑥ 适用时把 TLE 异步 API 纳入 directive。forge 后端不追加——
    # ⑥ 退化为 stock triton 的 num_stages/double_buffer（directive 默认文本已含）。
    if is_tle:
        system += _TIMING_OVERLAP_TLE_BRIEF

    temp = config.get("direction_classifier_temp", 0.3)
    user = "请根据上述算子与 profile，判定适用的优化方向并按指定 JSON 格式输出。"
    resp = await _chat(client, model_cfg["model"], system, user, temperature=temp)
    if not resp:
        print(f"  [directions] classifier LLM call failed — using default {len(available)} directions")
        return [dict(d) for d in available]

    try:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            raise ValueError("no JSON object in response")
        raw = json.loads(m.group(0)).get("directions", [])
    except Exception as e:
        print(f"  [directions] classifier parse failed ({e!r}) — using default {len(available)} directions")
        return [dict(d) for d in available]

    valid_names = {d["name"] for d in available}
    out: list[dict] = []
    for d in raw:
        if not isinstance(d, dict) or not d.get("applicable", True):
            continue
        name = str(d.get("name", "")).strip()
        if name not in valid_names:
            continue
        directive = str(d.get("directive", "")).strip()
        if not directive:  # fall back to this direction's default directive
            directive = next(x["directive"] for x in available if x["name"] == name)
        out.append({"id": d.get("id"), "name": name, "directive": directive,
                    "reason": str(d.get("reason", ""))})

    if not out:
        print(f"  [directions] classifier returned no applicable directions — using default {len(available)}")
        return [dict(d) for d in available]
    return out


async def unified_editor(
    candidates: list[dict],
    experiences: list[dict],
    problem: Problem,
    config: dict,
    applicable_directions: list[dict] | None = None,
    deadline: float = 0.0,
) -> list[dict]:
    """Unified editing agent (v5): merges planner + executor.

    For each candidate, samples `breadth` incremental search/replace patches
    in parallel (diversity via temperature), then verifies each. Falls back to
    full rewrite when a patch fails to apply or verification fails. Returns
    executor-result-shaped dicts directly consumable by select_candidates.

    In-search `experiences` (short-term memory) are appended to the system
    prompt.

    Direction-organized frontier (方向 0, opt-in): when ``applicable_directions``
    is provided and ``config.direction_organized_frontier`` is true, sampling
    switches from ``breadth`` identical temperature-sampled patches to one
    patch per applicable direction (each steered by a direction-specific
    directive) plus an optional free-exploration branch. The direction label
    rides on ``branch_id`` so ``select_candidates_by_direction`` can keep one
    patch per direction.
    """
    model_cfg = config.get("model_backend", config["model"])
    client = AsyncOpenAI(base_url=model_cfg["url"], api_key=model_cfg["api_key"])
    system = _build_unified_system(config, problem)

    # Inject in-search experiences into system prompt
    if experiences:
        exp_lines = ["\n# This Episode's Optimization Experiences\n"]
        for i, exp in enumerate(experiences):
            exp_lines.append(f"\n## Experience {i + 1}")
            exp_lines.append(exp.get("summary", json.dumps(exp)))
        system += "\n".join(exp_lines)

    # Full-rewrite fallback reuses executor_system + dynamic skill prepended.
    full_rewrite_system = _build_executor_system(config)
    skill_ctx = _load_skill_context_dynamic(config, problem)
    if skill_ctx:
        full_rewrite_system = skill_ctx + "\n\n---\n\n" + full_rewrite_system

    problem_str = _format_problem(problem)
    breadth = config.get("breadth", 4)
    temperature = config.get("unified_temperature", 0.7)
    direction_mode = bool(config.get("direction_organized_frontier")) and bool(applicable_directions)
    free_explore = direction_mode and config.get("direction_free_explore", True)

    baseline_latency_map = {}
    for c in candidates:
        if c.get("latency_ms"):
            baseline_latency_map[c["code"]] = c["latency_ms"]

    results = []
    for ci, candidate in enumerate(candidates):
        # deadline 检查（循环间隙）：单次 LLM/verify 不可中断，但 candidate 之间能 break。
        # 避免 unified_editor 在 iter 内部卡死导致 main.py 的 time_budget 早停形同虚设。
        if deadline and time.time() > deadline:
            print(f"  [unified] deadline reached before candidate {ci}, "
                  f"returning {len(results)} results so far")
            break
        baseline_code = candidate.get("code", "")
        latency = candidate.get("latency_ms")
        hw_metrics = candidate.get("hw_metrics")
        profile_str = _format_hw_profile(latency, hw_metrics)

        # Phase 1: parallel sampling. Direction mode samples one patch per
        # applicable direction (+ optional free_explore); legacy mode samples
        # `breadth` identical temperature-sampled patches.
        if direction_mode:
            users: list[str] = []
            labels: list[str] = []
            for d in applicable_directions:
                users.append(_build_unified_user(
                    baseline_code, problem_str, profile_str, config,
                    direction_directive=d["directive"]))
                labels.append(f"dir_{d['id']}_{d['name']}")
            if free_explore:
                users.append(_build_unified_user(
                    baseline_code, problem_str, profile_str, config,
                    direction_directive=""))
                labels.append("free_explore")
            print(f"  [unified] candidate {ci}: directional sampling {len(users)} patches "
                  f"({len(applicable_directions)} dirs, free_explore={'on' if free_explore else 'off'}) (temp={temperature})")
            raw = await asyncio.gather(*[
                _chat(client, model_cfg["model"], system, u, temperature) for u in users
            ])
            pairs: list[tuple] = [(r, lab) for r, lab in zip(raw, labels) if r is not None]
        else:
            user = _build_unified_user(baseline_code, problem_str, profile_str, config)
            print(f"  [unified] candidate {ci}: sampling {breadth} incremental patches (temp={temperature})")
            responses = await _gather_llm(
                client, model_cfg["model"], system, user, n=breadth, temperature=temperature,
            )
            pairs = [(r, None) for r in responses]

        # Phase 2: verify + fallback (sequential — GPU/CUDA ops are not concurrency-safe).
        # branch_id=None (legacy) → _unified_edit_one falls back to str(ci).
        for ri, (resp, label) in enumerate(pairs):
            # deadline 检查（patch 之间）：已采到的 patch 不再验证，提前返回。
            if deadline and time.time() > deadline:
                print(f"  [unified] deadline reached at candidate {ci} patch {ri}, "
                      f"returning {len(results)} results so far")
                break
            result = await _unified_edit_one(
                ci, ri, resp, baseline_code, problem, config, client, model_cfg["model"],
                baseline_latency_map, problem_str, full_rewrite_system,
                branch_id=label,
            )
            results.append(result)

    return results
