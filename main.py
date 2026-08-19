"""Entry point: read config, run beam search, save results.

Usage:
    python main.py --config config.yaml

Batch experiments:
    for c in configs/*.yaml; do
        python main.py --config "$c" --output "results/$(basename $c .yaml)"
    done
"""

import argparse
import asyncio
import json
import os
import re
import sys
import traceback

import torch
import yaml
from pathlib import Path

# 行缓冲：print 实时落盘，进程被 shell timeout 杀时不丢 "Iteration"/traceback 等进度日志。
# 历史 run.log 只剩 AKG logging 的 INFO 行（它 flush），main.py 自己的 print 全被 stdout
# 全缓冲吞掉，debug 困难。reconfigure 不依赖 run 脚本记得加 `python -u`。
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass  # 非 TextIOWrapper（如重定向到非 tty）时 no-op

from triton_backend import Problem, ProfileResult, _normalize_shape, profile as triton_profile, load_problem, adaptive_rel_tol, TOKEN_USAGE
from search import select_candidates, select_candidates_by_direction, update_experiences
from agents import unified_editor, determine_applicable_directions, _derive_op_type, _get_profiler
from direction_store import load_stats, record_iteration, merge_run, save_stats, get_op_stats, format_op_stats
from mcts import run_mcts


# ---------------------------------------------------------------------------
# Experience builder for unified mode — replaces the summarizer LLM.
# ---------------------------------------------------------------------------

def _build_experiences_from_results(
    results: list[dict], candidates: list[dict], config: dict,
) -> list[dict]:
    """Build episode-local experiences from unified_editor results.

    Each result's change_description becomes the experience summary; speedup
    drives the positive/negative split that update_experiences expects.
    Output shape: {speedup, summary, title}.
    """
    baseline_latencies = {c.get("code", ""): c.get("latency_ms") for c in candidates}
    experiences = []
    for r in results:
        latency = r.get("latency_ms")
        change_desc = r.get("change_description", "")
        edit_mode = r.get("edit_mode", "?")
        if latency is not None:
            bl = baseline_latencies.get(r.get("baseline_code", ""))
            speedup = (bl / latency) if bl else 1.0
            summary = change_desc or f"{edit_mode} edit, latency={latency:.4f}ms"
            title = f"{edit_mode}: {speedup:.2f}x"
        else:
            speedup = 0.0
            err = str(r.get("error", ""))[:200]
            summary = f"[FAILED {edit_mode}] {change_desc}: {err}" if change_desc else f"[FAILED {edit_mode}] {err}"
            title = f"failed {edit_mode}"
        experiences.append({"speedup": speedup, "summary": summary, "title": title})
    return experiences


# ---------------------------------------------------------------------------
# Reusable search episode runner
# ---------------------------------------------------------------------------

async def run_search_episode(
    initial_candidate: dict,
    problem: Problem,
    config: dict,
    experiences: list[dict] | None = None,
    episode_id: int = 0,
    episode_output_dir: str | None = None,
) -> dict:
    """Run one search episode: multiple beam-search iterations over one kernel.

    This is the core one-shot search primitive (also called directly by
    ``run_l1_simple.py`` / ``run_l1_batch.py``).  It runs the configured
    ``config["iters"]`` iterations of the Planner → Executor → Summarizer
    (or unified_editor) cycle, writing per-iteration checkpoints and returning
    the final candidates together with the updated experience queue.

    Parameters
    ----------
    initial_candidate:
        Seed kernel with keys ``code``, ``solution_path``, ``latency_ms``.
    problem:
        Parsed problem definition (inputs, outputs, reference, …).
    config:
        Full configuration dict (iters, breadth, num_samples, output paths, …).
    experiences:
        Pre-loaded in-search experience queue.  Carries short-term memory
        across iterations within this run.
    episode_id:
        Zero-based index used for checkpoint naming.  0 = single-shot.
    episode_output_dir:
        Directory for artefacts.  Defaults to ``config["output_dir"]``.

    Returns
    -------
    dict with keys:
        candidates (list[dict])   – final top-K candidate kernels
        experiences (list[dict])  – updated experience queue
        results (list[dict])      – per-iteration summary records
        best_latency_ms (float)   – latency of the best candidate (or None)
    """
    import torch as _torch
    import gc as _gc

    candidates = [initial_candidate]
    if experiences is None:
        experiences = []

    output_dir = episode_output_dir or config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # C: 时间预算从 episode 开头算（含二次生成 v3），避免 GLM-5.2 generator 慢导致 search 没到早停就被 shell timeout
    import time as _time
    search_start = _time.time()
    time_budget = config.get("search_time_budget", 0)  # 秒，0=不限
    # deadline 传给 unified_editor，让它在 candidate/patch 循环间隙检查超时——单次 LLM 调用
    # 2-5min 不可中断，但循环间隙能 break，避免 iter 内部卡死导致 time_budget 形同虚设。
    deadline = (search_start + time_budget) if time_budget else 0.0

    # seed 已在 main pipeline 生成（统一入口，1×）；run_search_episode 纯搜索，不再 in-episode 生成。

    all_results: list[dict] = []

    search_mode = config.get("search_mode", "mcts")  # mcts (default) | unified
    print(f"\n[search_mode={search_mode}]")

    # mcts 分支：委托给 mcts.run_mcts，整树搜索后返回同 shape 结果（champion=全树最优，
    # seed 永在树中 → carry-forward 治 work-log TODO [A]）。与 unified 分流，
    # 不共享下方 iter 循环（MCTS 的 rollout 循环在 run_mcts 内部）。
    if search_mode == "mcts":
        # 方向分类在 run_mcts 内部按 mcts_use_directions 决定；这里若 main 层已开
        # direction_organized_frontier 则预算一次传入，避免重复调用。
        pre_directions = None
        if config.get("direction_organized_frontier") or config.get("mcts_use_directions", True):
            pre_directions = await determine_applicable_directions(
                problem, config,
                candidates[0].get("latency_ms"),
                candidates[0].get("hw_metrics"),
            )
            print(f"[directions] {len(pre_directions)} applicable: "
                  f"{[d['name'] for d in pre_directions]}")
        mcts_result = await run_mcts(
            initial_candidate=candidates[0],
            problem=problem,
            config=config,
            experiences=experiences,
            episode_id=episode_id,
            episode_output_dir=output_dir,
            applicable_directions=pre_directions,
        )
        return mcts_result

    # 方向 0：按优化方向组织 beam 前沿（opt-in）。每 episode 1 次 LLM 分类，判定算子适用
    # ①-⑧ 哪些方向 + 每方向一句采样指令。仅 unified 模式生效。
    applicable_directions = None
    if search_mode == "unified" and config.get("direction_organized_frontier"):
        applicable_directions = await determine_applicable_directions(
            problem, config,
            candidates[0].get("latency_ms"),
            candidates[0].get("hw_metrics"),
        )
        print(f"[directions] {len(applicable_directions)} applicable: "
              f"{[d['name'] for d in applicable_directions]}")

    # 持久化层（direction-experience-loop TODO-1）：direction 模式下按 op_type 记录各方向
    # 结局统计（sampled/passed/survived/best_speedup_vs_seed），跨 run 累积。仅记账+持久化+log，
    # 不改搜索行为（消费 = TODO-2 经验排序 / TODO-3 注入分类器 prompt，是独立下一步）。
    run_outcomes: dict = {}
    op_type: str | None = None
    stats_db: dict | None = None
    if applicable_directions:
        op_type = _derive_op_type(problem, config)
        stats_db = load_stats(config.get("direction_stats_path", "direction_stats.json"))
        print(format_op_stats(get_op_stats(stats_db, op_type)))

    for iteration in range(config["iters"]):
        # C: 超时间预算早停，保留已产出 champion，避免 shell timeout (rc=124)
        if time_budget and _time.time() - search_start > time_budget:
            print(f"  ⏰ Search time budget ({time_budget}s) exceeded at iter {iteration+1}/{config['iters']}, early stop (champion preserved)")
            break
        print(f"\n=== Iteration {iteration + 1}/{config['iters']} === (mode={search_mode})")

        # unified editor merges planner + executor into one agent.
        results = await unified_editor(
            candidates, experiences, problem, config,
            applicable_directions=applicable_directions,
            deadline=deadline,
        )
        print(f"  UnifiedEditor: {len(results)} edits attempted")
        # no per-iteration summarizer LLM — experiences come from change_description.
        new_experiences = _build_experiences_from_results(results, candidates, config)
        print(f"  Experiences: {len(new_experiences)} (from change_descriptions)")
        num_plans = len(results)  # checkpoint-compat field

        # Log successful and failed results
        successes = [r for r in results if r.get("latency_ms")]
        errors = [r for r in results if r.get("error")]
        if successes:
            best = min(successes, key=lambda r: r["latency_ms"])
            print(f"  Best latency this iter: {best['latency_ms']:.4f} ms (speedup: {best.get('speedup', 'N/A')})")
        if errors:
            first_errors = errors[:3]
            for e in first_errors:
                err_msg = str(e.get("error", ""))[:120]
                print(f"  Error: {e.get('plan_id', '?')} — {err_msg}")

        # unified-mode edit-mode breakdown
        if successes:
            inc = sum(1 for r in successes if r.get("edit_mode") == "incremental")
            fr = sum(1 for r in successes if r.get("edit_mode") == "full_rewrite")
            print(f"  Edit modes (successful): incremental={inc}, full_rewrite={fr}")

        # 4. Update experience queue
        experiences = update_experiences(
            experiences, new_experiences,
            capacity=config.get("exp_n", 16),
            topk=config.get("topk", 8),
        )

        # 5. Select candidates for next iteration
        if applicable_directions:
            # 方向 0：每方向留最快 patch，按分类器优先级截断到 direction_max_width，
            # free_explore 豁免截断。不按 latency 截断——早期 latency 是差的假设价值代理，
            # 会误杀需多轮才能解锁的高收益方向（如 ⑤ algo-equiv）。
            new_candidates = select_candidates_by_direction(
                results, direction_max_width=config.get("direction_max_width", 3))
        else:
            new_candidates = select_candidates(results, k=config.get("topk_candidates", 2))

        # 持久化层：聚样本轮各方向结局（direction 模式，按 branch_id）
        if applicable_directions:
            record_iteration(run_outcomes, results, new_candidates, initial_candidate.get("latency_ms"))

        # Fallback: keep previous candidates if nothing new compiles
        if new_candidates:
            candidates = new_candidates
        else:
            print("  No successful new candidates, keeping previous baseline")

        # Force cleanup between iterations to prevent CUDA OOM
        _gc.collect()
        _torch.cuda.empty_cache()

        # Save iteration checkpoint
        iter_data = {
            "episode": episode_id,
            "iteration": iteration + 1,
            "search_mode": search_mode,
            "num_plans": num_plans,
            "num_results": len(results),
            "num_experiences": len(new_experiences),
            "candidates": [
                {
                    "solution_path": c.get("solution_path", ""),
                    "latency_ms": c.get("latency_ms"),
                    "hw_metrics": c.get("hw_metrics"),
                }
                for c in candidates
            ],
        }
        all_results.append(iter_data)

        ckpt_path = Path(output_dir) / f"checkpoint_iter{iteration + 1}.json"
        with open(ckpt_path, "w") as f:
            json.dump(iter_data, f, indent=2)

        print(f"  → Checkpoint saved to {ckpt_path}")

    best_latency = None
    valid = [c.get("latency_ms") for c in candidates if c.get("latency_ms")]
    if valid:
        best_latency = min(valid)

    # 持久化层：把本次 run 的方向结局合并进全局 DB（direction 模式）
    if applicable_directions and run_outcomes:
        stats_db = merge_run(stats_db, op_type, run_outcomes)
        stats_path = config.get("direction_stats_path", "direction_stats.json")
        save_stats(stats_db, stats_path)
        print(f"[direction-stats] saved run outcomes for op_type={op_type} "
              f"({len(run_outcomes)} directions) → {stats_path}")

    return {
        "candidates": candidates,
        "experiences": experiences,
        "results": all_results,
        "best_latency_ms": best_latency,
    }


async def main():
    parser = argparse.ArgumentParser(description="DirecTune")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--problem", type=str, default=None, help="Override problem path")
    parser.add_argument("--initial", type=str, default=None, help="Override initial solution path")
    parser.add_argument("--rounds", type=int, default=None, help="Total optimization rounds")
    parser.add_argument("--breadth", type=int, default=None, help="Override breadth")
    parser.add_argument("--num-samples", type=int, default=None, help="Override num_samples")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Triton 后端校验（forge = stock triton 3.7 | forge_tle = FlagTree triton 3.6 + TLE）---
    # compile_triton/_load_source 在主进程内 importlib 加载并执行 @triton.jit，TLE kernel 顶部
    # `import triton.experimental.tle.language` 要求主进程就是 FlagTree。主进程 forge + 子进程
    # forge_tle 拆分不可行（子进程用 sys.executable 继承主进程 env）。所以 forge_tle 模式必须
    # 整个 DirecTune 跑在 forge_tle env 下，启动时校验。
    triton_backend = config.get("triton_backend", "forge")
    import triton as _triton
    _triton_ver = _triton.__version__
    if triton_backend == "forge_tle":
        if not _triton_ver.startswith("3.6"):
            print(f"[backend] ERROR: triton_backend=forge_tle requires FlagTree triton 3.6.x, "
                  f"but got triton {_triton_ver} (sys.executable={sys.executable}).")
            print("[backend]   请 `conda activate forge_tle` 后重试。")
            sys.exit(1)
        try:
            import triton.experimental.tle.language as _tle  # noqa: F401
        except Exception as e:
            print(f"[backend] ERROR: triton_backend=forge_tle 但 import tle 失败: {e!r}")
            print("[backend]   请 `conda activate forge_tle` 后重试。")
            sys.exit(1)
        print(f"[backend] forge_tle 激活 (triton {_triton_ver}, TLE 可用) — ⑥ timing_overlap 增补 TLE 异步访存")
    else:  # forge
        if _triton_ver.startswith("3.6"):
            print(f"[backend] WARNING: triton_backend=forge 但检测到 triton {_triton_ver} (疑似 FlagTree)。"
                  f"  forge 模式应在 forge env (stock triton 3.7) 下运行,⑥ timing_overlap 退化为 num_stages/double_buffer。")
        else:
            print(f"[backend] forge 激活 (triton {_triton_ver}) — 8 方向 ①-⑧（⑥ timing_overlap = num_stages/double_buffer）")

    if args.output:
        config["output_dir"] = args.output
    if args.problem:
        config["problem"] = args.problem
    if args.initial:
        config["initial_solution"] = args.initial
    if args.breadth is not None:
        config["breadth"] = args.breadth
    if args.num_samples is not None:
        config["num_samples"] = args.num_samples

    # Resolve rounds: --rounds > config rounds (default 15).
    # run_search_episode reads config["iters"] as the per-run iteration count.
    total_rounds = args.rounds or config.get("rounds") or 15
    config["iters"] = total_rounds

    os.makedirs(config["output_dir"], exist_ok=True)

    # Load problem
    problem = load_problem(config["problem"])
    print(f"Problem: {problem.name}")

    # Load initial solution
    initial_path = config["initial_solution"]
    with open(initial_path) as f:
        initial_code = f.read()

    # --- 统一入口（单路径）---
    # baseline def run → 1× seed 生成 → search。Fuser 退役（fuser.py 留代码不删）。
    # gen_mode（v6 默认改）：naive = 纯 LLM 手写 naive seed（naive_seed_gen.py，0 AKG 依赖，
    #   仅 triton_api_ref + 朴素约束 prompt；2026-07-15 L1 验证 5/5 成功 naiveness 1.00，
    #   优于 v3/akg 的"预支 TC/tile"）。保留 v3/akg 作 fallback（gen_mode 配置切换）：
    #   v3 = AKG 风格重写（Designer→Coder→Conductor→FixCodeGen+skill+RAG）；akg = 真跑 akg_frontend（level2 .pt 权重）。
    gen_mode = config.get("gen_mode", "naive")
    # 旧 level 路由保留：若显式配 v3/akg 则按 level 自动选（向后兼容老 config）。
    if gen_mode in ("v3", "akg"):
        gen_mode = "v3" if "level1" in config.get("problem", "") else "akg"

    # Profile the PyTorch reference independently from the initial candidate.
    # --initial may already be a Triton seed and must never redefine the
    # baseline used for paper speedups.
    print("Profiling PyTorch reference baseline...")
    baseline_rtol = adaptive_rel_tol(problem)
    baseline_result = triton_profile(
        problem.reference,
        problem,
        timeout_seconds=config.get("timeout_seconds", 300),
        rel_tol=baseline_rtol,
        config=config,
    )
    baseline_hw_metrics = None
    if baseline_result.latency_ms is not None:
        profiler = _get_profiler(config)
        baseline_hw_metrics = profiler.profile(problem.reference, problem, config)
    if baseline_result.latency_ms:
        print(f"PyTorch baseline latency: {baseline_result.latency_ms:.4f} ms")
        if baseline_hw_metrics:
            print(f"Baseline HW metrics collected: {len(baseline_hw_metrics)} fields")
    else:
        print(f"PyTorch baseline error: {baseline_result.error}")
        sys.exit(1)
    torch.cuda.empty_cache()

    # Generate or load the Triton seed independently. A pre-generated --initial
    # seed is profiled through the same benchmark path as other candidates.
    seed_code = initial_code
    seed_latency = None
    if "@triton.jit" not in initial_code:
        print(f"Generating Triton seed (gen_mode={gen_mode})...")
        try:
            if gen_mode == "naive":
                # v6 default: pure LLM naive seed (0 AKG).
                from naive_seed_gen import gen_seed as _naive_gen
                from agents import _derive_op_type
                op_type = _derive_op_type(problem, config) or ""
                gen_result = await _naive_gen(problem, op_type, config)
            else:
                from generator import generate_kernel as _gen
                gen_config = dict(config)
                gen_config["gen_mode"] = gen_mode
                plan = "Generate a correct, high-performance Triton kernel for this computation"
                gen_result = await _gen(problem, initial_code, plan, gen_config)
            if gen_result and gen_result.get("code") and "@triton.jit" in gen_result.get("code", ""):
                seed_code = gen_result["code"]
                seed_latency = gen_result.get("latency_ms")
                nv = gen_result.get("naiveness")
                nv_str = f", naiveness={nv['score']:.2f}" if nv else ""
                print(f"Generator ({gen_mode}) → {len(seed_code)} chars, latency={seed_latency}ms{nv_str}")
            else:
                print(f"Generator ({gen_mode}) failed to produce Triton seed — aborting this problem")
                sys.exit(2)
        except Exception as e:
            print(f"Generator failed: {e} — aborting this problem")
            sys.exit(2)
        torch.cuda.empty_cache()
    else:
        print("Initial code is already Triton — profiling seed separately")

    if seed_latency is None:
        seed_result = triton_profile(
            seed_code,
            problem,
            timeout_seconds=config.get("timeout_seconds", 300),
            rel_tol=baseline_rtol,
            config=config,
        )
        if not seed_result.latency_ms:
            print(f"Triton seed error: {seed_result.error}")
            sys.exit(2)
        seed_latency = seed_result.latency_ms
    print(f"Triton seed latency: {seed_latency:.4f} ms")
    torch.cuda.empty_cache()

    initial_candidate = {
        "code": seed_code,
        "solution_path": initial_path,
        "latency_ms": seed_latency,
        "hw_metrics": baseline_hw_metrics,
    }

    # Run one-shot search. run_search_episode is the reusable primitive (also
    # called directly by run_l1_simple.py / run_l1_batch.py); config["iters"]=rounds.
    #
    # try/except 保护：search 阶段可能因 LLM 超时 / CUDA OOM / SIGTERM(shell timeout) 等抛异常
    # 或被取消。search 失败时只允许用 Triton seed 兜底写 final_results.json；reference seed
    # 已在生成阶段禁止，避免 PyTorch 路径混入 Triton 实验结果。
    episode_result = None
    try:
        episode_result = await run_search_episode(
            initial_candidate=initial_candidate,
            problem=problem,
            config=config,
            experiences=[],
            episode_id=0,
            episode_output_dir=config["output_dir"],
        )
    except Exception as e:
        print(f"[main] run_search_episode failed: {e!r} — falling back to seed champion")
        traceback.print_exc()

    # Champion = fastest correct Triton candidate. The Triton seed is carried in candidates,
    # so when no edit improves, the seed is retained (no regression).
    # search 失败时用 initial_candidate 兜底，保证 final_results 永远有产物。
    if episode_result and episode_result.get("candidates"):
        candidates = episode_result["candidates"]
    else:
        candidates = [initial_candidate]
    valid = [c for c in candidates if c.get("latency_ms") is not None and "@triton.jit" in c.get("code", "")]
    if valid:
        champion = min(valid, key=lambda c: c["latency_ms"])
    else:
        print("No valid Triton champion produced — aborting this problem")
        sys.exit(2)
    candidates = [champion]

    champion_latency = champion.get("latency_ms")
    baseline_latency = baseline_result.latency_ms
    speedup_vs_pytorch = (baseline_latency / champion_latency
                          if baseline_latency and champion_latency else None)
    speedup_vs_seed = (seed_latency / champion_latency
                       if seed_latency and champion_latency else None)

    # Final output keeps the candidate/tree payload and also stores the three
    # independently measured latencies needed for attribution.
    final_path = Path(config["output_dir"]) / "final_results.json"
    final_data = {
        "config": {k: v for k, v in config.items() if k != "model"},
        "baseline_latency_ms": baseline_latency,
        "seed_latency_ms": seed_latency,
        "champion_latency_ms": champion_latency,
        "speedup_vs_pytorch": speedup_vs_pytorch,
        "speedup_vs_seed": speedup_vs_seed,
        "strict_triton": "@triton.jit" in champion.get("code", ""),
        "status": "success",
        "iterations": episode_result["results"] if episode_result else [],
        "final_candidates": [
            {
                "code": c.get("code", ""),
                "latency_ms": c.get("latency_ms"),
                "solution_path": c.get("solution_path", ""),
                "hw_metrics": c.get("hw_metrics"),
            }
            for c in candidates
        ],
    }
    with open(final_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"\n=== Token usage ===")
    print(f"  prompt={TOKEN_USAGE['prompt']} completion={TOKEN_USAGE['completion']} total={TOKEN_USAGE['prompt']+TOKEN_USAGE['completion']} calls={TOKEN_USAGE['calls']}")
    print(f"\n=== Done ===")
    print(f"Results saved to {final_path}")
    for c in candidates:
        print(f"  Final candidate latency: {c.get('latency_ms', 'N/A')} ms")


if __name__ == "__main__":
    asyncio.run(main())
