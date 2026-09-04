#!/usr/bin/env python3
"""对照试验 AKG 臂驱动：上游 master clone 生成 kernel → 统一 harness 评测。

资源对齐（方案 A，best-of-N 等资源包）：
- 预算轴 = LLM 调用数 + tokens 总量（来自 MCTS 臂同题的 resource_usage 回填，
  两个 --budget-* 同时设置，任一先到即停）；--wall-clock 兜底防挂死。
- 每次迭代 = 一次独立的 AKG coder_only_workflow（子进程隔离，防状态泄漏），
  通过验证的 kernel 全部留档；预算耗尽后取 harness 延迟最优者为该臂成绩。
- 上报延迟只认统一 harness（triton_backend.profile_isolated，与 MCTS 臂同一份
  代码）；AKG 内部 Verifier/profile_res 仅作其流水线组成部分记录。
- 记账 hook 挂 AKG 自己的 core_v2.LLMClient._update_token_stats（含 reasoning
  token），Conductor 路由 / FixCodeGen 修复的调用全部计入。

两层结构：
  驱动模式（默认）  ：循环起子进程跑 --single 迭代，聚合 usage、追踪 best
  --single 迭代模式 ：一次 coder_only_workflow + harness 评测 + iteration json

用法（forge env）：
  python scripts/run_akg_arm.py \
      --problem problems/kb_level2/9_Matmul_Subtract_Multiply_ReLU.json \
      --output-dir output/ab_vs_akg/akg/9_Matmul_Subtract_Multiply_ReLU \
      --budget-calls 60 --budget-tokens 900000
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MCTS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AKG_ROOT = "/home/wangyichen/akg/akg_agents/python"

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "hook": "none"}


def install_usage_hooks():
    """记账 hook：patch AKG 自己的 core_v2.LLMClient 记账漏斗（复用其 usage
    解析，含 reasoning token）；openai sdk 作兜底。"""
    try:
        from akg_agents.core_v2.llm.client import LLMClient

        orig_update = LLMClient._update_token_stats
        orig_generate = LLMClient.generate

        def upd(self, usage):
            orig_update(self, usage)
            if usage:
                USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                USAGE["completion_tokens"] += usage.get("completion_tokens", 0) or 0

        async def generate(self, *a, **kw):
            USAGE["calls"] += 1
            return await orig_generate(self, *a, **kw)

        LLMClient._update_token_stats = upd
        LLMClient.generate = generate
        USAGE["hook"] = "core_v2.LLMClient"
        return
    except Exception as e:
        print(f"[hook] LLMClient hook unavailable: {e}", flush=True)
    try:
        from openai.resources.chat import completions as occ

        orig_create = occ.Completions.create

        def create(self, *a, **kw):
            r = orig_create(self, *a, **kw)
            USAGE["calls"] += 1
            u = getattr(r, "usage", None)
            if u:
                USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            return r

        occ.Completions.create = create
        USAGE["hook"] = "openai.Completions.create"
    except Exception as e:
        print(f"[hook] openai hook unavailable: {e}", flush=True)


def bridge_model_config(shared_cfg: dict):
    """把 model_frontend 写进 AIKG_*/AKG_AGENTS_*（新版 settings 兼容旧变量）。"""
    mf = shared_cfg.get("model_frontend") or shared_cfg.get("model") or {}
    for k in ("AIKG_BASE_URL", "AIKG_API_KEY", "AIKG_MODEL_NAME",
              "AKG_AGENTS_BASE_URL", "AKG_AGENTS_API_KEY", "AKG_AGENTS_MODEL_NAME"):
        os.environ.pop(k, None)
    os.environ["AIKG_BASE_URL"] = mf.get("url", "")
    os.environ["AIKG_API_KEY"] = mf.get("api_key", "")
    os.environ["AIKG_MODEL_NAME"] = mf.get("model", "")
    os.environ["AKG_AGENTS_BASE_URL"] = mf.get("url", "")
    os.environ["AKG_AGENTS_API_KEY"] = mf.get("api_key", "")
    os.environ["AKG_AGENTS_MODEL_NAME"] = mf.get("model", "")
    forge_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = forge_bin + os.pathsep + os.environ.get("PATH", "")
    print(f"[bridge] AKG model → {mf.get('model')} @ {mf.get('url')}", flush=True)


def extract_task_desc(reference: str) -> str:
    """L2: 截取 AKG 兼容段（Model class + get_inputs/get_init_inputs），
    去掉 DirecTune 的 expanded reference / .pt 权重段。L1: 原样返回。"""
    for marker in ("\n# --- EXPANDED REFERENCE ---", "\n# Frozen weights",
                   "\nimport torch as _torch"):
        idx = reference.find(marker)
        if idx > 0:
            reference = reference[:idx]
    return reference.strip()


def shim_modelnew_to_run(code: str, task_desc: str, reference: str) -> str:
    """AKG 产出 ModelNew class → 加 def run() 适配统一 harness。
    .pt frozen 权重按参数顺序+形状 zip copy（移植自老 DirecTune generator.py）。"""
    if "class ModelNew" not in code or "def run(" in code:
        return code
    wp = re.search(r"_weights_path\s*=\s*['\"]([^'\"]+)['\"]", reference)
    wp = wp.group(1) if wp else ""
    return code + f"""

# --- DirecTune shim (.pt weights aligned, cached) ---
{task_desc}

_weights_path = "{wp}"
_MODEL = None
def run(x, *args):
    global _MODEL
    if _MODEL is None:
        import torch
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


def run_single(problem, task_desc, out_dir: Path, timeout: float, it: int) -> dict:
    """一次独立 AKG 生成（--single 子进程内执行）。"""
    return asyncio.run(_run_single_async(problem, task_desc, out_dir, timeout, it))


async def _run_single_async(problem, task_desc, out_dir: Path, timeout: float,
                            it: int) -> dict:
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.core.async_pool.task_pool import TaskPool
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.core.worker.manager import register_local_worker

    await register_local_worker([0], backend="cuda", arch="a100")
    cfg = load_config("triton_cuda", backend="cuda")
    log_dir = out_dir / "akg_logs" / f"iter{it:02d}"
    cfg["log_dir"] = str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg["profile_settings"] = {"warmup_times": 5, "run_times": 50}

    task = LangGraphTask(
        op_name=f"akg_{problem.name}_it{it}",
        task_desc=task_desc,
        task_id=str(it),
        dsl="triton_cuda",
        backend="cuda",
        arch="a100",
        config=cfg,
        framework="torch",
        workflow="coder_only_workflow",
        task_type="profile",
        bench_type="kernelbench",
    )
    pool = TaskPool()
    pool.create_task(task.run)
    t0 = time.time()
    results = await asyncio.wait_for(pool.wait_all(), timeout=timeout)
    elapsed = time.time() - t0
    for _, success, final_state in results:
        fs = final_state or {}
        return {"success": bool(success), "elapsed_s": round(elapsed, 1),
                "coder_code": fs.get("coder_code", ""),
                "verifier_result": fs.get("verifier_result"),
                "profile_res": fs.get("profile_res")}
    return {"success": False, "elapsed_s": round(elapsed, 1), "coder_code": ""}


def single_main(args):
    """--single：一次 AKG 生成 + shim + 统一 harness 评测。"""
    import yaml
    shared = yaml.safe_load(open(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.akg_root)      # akg_agents (上游 master clone)
    sys.path.insert(0, str(MCTS_ROOT))     # triton_backend（统一 harness）

    bridge_model_config(shared)
    install_usage_hooks()

    import triton_backend as tb
    problem = tb.load_problem(args.problem)
    rel_tol = tb.adaptive_rel_tol(problem)
    task_desc = extract_task_desc(problem.reference)

    akg = run_single(problem, task_desc, out_dir, args.timeout, args.iteration)
    code = akg.get("coder_code", "")

    it_result = {
        "iteration": args.iteration, "arm": "akg", "problem": args.problem,
        "model": shared["model_frontend"]["model"],
        "usage": dict(USAGE),
        "akg": {k: v for k, v in akg.items() if k != "coder_code"},
        "strict_triton": "@triton.jit" in code,
    }
    if code:
        (out_dir / f"kernel_raw_it{args.iteration:02d}.py").write_text(code)
        final_code = shim_modelnew_to_run(code, task_desc, problem.reference)
        (out_dir / f"kernel_final_it{args.iteration:02d}.py").write_text(final_code)
        harness = {"compiled": False, "correct": False, "latency_ms": None}
        try:
            pr = tb.profile_isolated(final_code, problem, timeout_seconds=300,
                                     rel_tol=rel_tol, config={"isolated_verify": True})
            harness = {"compiled": bool(pr.compiled), "correct": bool(pr.correct),
                       "latency_ms": pr.latency_ms,
                       "error": None if pr.compiled and pr.correct else (pr.error or "")[:400]}
        except Exception as e:
            harness["error"] = str(e)[:400]
        it_result["harness"] = harness

        # baseline（PyTorch reference，供 speedup_vs_pytorch）
        base_lat = None
        try:
            fn, err = tb.compile_triton(problem.reference)
            if fn is not None:
                import torch
                inputs = []
                for spec in problem.inputs:
                    shape = spec.get("shape")
                    dtype = getattr(torch, spec["dtype"])
                    t = torch.randn((), dtype=dtype, device="cuda") if shape is None \
                        else torch.randn(*shape, dtype=dtype, device="cuda")
                    inputs.append(t)
                else:
                    base_lat = tb.benchmark(fn, *inputs, warmup=10, reps=100,
                                            device="cuda")
        except Exception as e:
            print(f"[baseline] failed: {e}", flush=True)
        it_result["baseline_latency_ms"] = base_lat
    it_path = args.iteration_json
    Path(it_path).write_text(json.dumps(it_result, indent=1, default=str))
    print(f"=== iteration done === it={args.iteration} harness={it_result.get('harness')} "
          f"usage={USAGE}", flush=True)


def driver_main(args):
    """驱动模式：best-of-N 迭代直到预算耗尽，取 harness 最优。"""
    import yaml
    shared = yaml.safe_load(open(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mf = shared.get("model_frontend") or {}

    budget_calls = args.budget_calls or 0
    budget_tokens = args.budget_tokens or 0
    if not budget_calls and not budget_tokens:
        sys.exit("ERROR: --budget-calls / --budget-tokens 至少设一个（0/缺省=不限）")

    tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    best = None       # {"latency_ms", "kernel", "iteration"}
    baseline = None
    iters = []
    t0 = time.time()
    it = 0
    stop_reason = "budget-exhausted"
    while True:
        it += 1
        rem_wall = args.wall_clock - (time.time() - t0)
        if rem_wall <= 60:
            stop_reason = "wall-clock"
            break
        it_json = out_dir / f"iteration_{it:02d}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single",
               "--problem", args.problem, "--output-dir", str(out_dir),
               "--config", args.config, "--akg-root", args.akg_root,
               "--timeout", str(min(args.timeout, rem_wall)),
               "--iteration", str(it), "--iteration-json", str(it_json)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
        try:
            subprocess.run(cmd, check=True, env=env,
                           stdout=open(out_dir / f"iter{it:02d}.log", "w"),
                           stderr=subprocess.STDOUT, timeout=rem_wall + 120)
            itd = json.loads(it_json.read_text())
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            print(f"[iter {it}] crashed: {e}", flush=True)
            itd = {"usage": USAGE, "harness": {}, "crashed": True}
        u = itd.get("usage", {})
        for k in tot:
            tot[k] += u.get(k, 0)
        h = itd.get("harness") or {}
        iters.append({"iteration": it, "usage": u, "harness": h,
                      "strict_triton": itd.get("strict_triton")})

        if h.get("correct") and h.get("latency_ms") and \
           (best is None or h["latency_ms"] < best["latency_ms"]):
            best = {"latency_ms": h["latency_ms"], "iteration": it,
                    "kernel": f"kernel_final_it{it:02d}.py"}
        if baseline is None and itd.get("baseline_latency_ms"):
            baseline = itd["baseline_latency_ms"]

        if budget_calls and tot["calls"] >= budget_calls:
            stop_reason = f"budget-calls ({tot['calls']}/{budget_calls})"
            break
        if budget_tokens and (tot["prompt_tokens"] + tot["completion_tokens"]) >= budget_tokens:
            stop_reason = (f"budget-tokens "
                           f"({tot['prompt_tokens'] + tot['completion_tokens']}/{budget_tokens})")
            break
        if it >= args.max_iters:
            stop_reason = f"max-iters ({it})"
            break

    result = {
        "arm": "akg", "problem": args.problem, "model": mf.get("model"),
        "budget": {"calls": budget_calls, "tokens": budget_tokens,
                   "wall_clock": args.wall_clock},
        "usage_total": dict(tot), "iterations": len(iters),
        "iters_detail": iters, "stop_reason": stop_reason,
        "best": best, "baseline_latency_ms": baseline,
        "harness": ({"latency_ms": best["latency_ms"], "correct": True} if best
                    else {"latency_ms": None, "correct": False}),
        "passed": best is not None,
        "speedup_vs_pytorch": (baseline / best["latency_ms"])
        if baseline and best else None,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if best:
        champion_src = out_dir / best["kernel"]
        if champion_src.exists():
            (out_dir / "champion.py").write_text(champion_src.read_text())
            result["champion_file"] = "champion.py"
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"=== done === iters={len(iters)} best={best} usage={tot} "
          f"stop={stop_reason}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default=str(MCTS_ROOT / "config_ab_shared.yaml"))
    ap.add_argument("--akg-root", default=DEFAULT_AKG_ROOT)
    ap.add_argument("--timeout", type=float, default=1800.0, help="单次 AKG 生成超时(秒)")
    ap.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES")
    # 驱动模式（best-of-N 预算）
    ap.add_argument("--budget-calls", type=int, default=0)
    ap.add_argument("--budget-tokens", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=200, help="迭代数安全上限")
    ap.add_argument("--wall-clock", type=float, default=14400.0, help="兜底墙钟(秒)")
    # 迭代模式（由驱动模式内部调用）
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--iteration-json", default="")
    args = ap.parse_args()

    if args.single:
        single_main(args)
    else:
        driver_main(args)


if __name__ == "__main__":
    main()
