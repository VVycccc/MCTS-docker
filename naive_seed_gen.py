"""naive_seed_gen.py — L1 级验证：纯 LLM 手写 naive seed（仅 + triton_api_ref，0 AKG 依赖）。

验证目标（work-log「最小需求消融」L1）：GLM-5.2 在没有任何 AKG skill / prompt / RAG 的条件下，
仅凭 reference + triton_api_ref + 朴素约束 prompt，能否写出能编译+正确+足够 naive 的 Triton seed。

用法：
    conda activate forge
    cd /home/wangyichen/DirecTune-MCTS
    python naive_seed_gen.py            # 跑默认题集
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
PROJECT = str(Path(__file__).resolve().parent)
sys.path.insert(0, PROJECT)

from openai import AsyncOpenAI
import torch
from triton_backend import (
    Problem, load_problem, profile as triton_profile,
    anti_pytorch_check, adaptive_rel_tol, record_usage,
)

# 验证题集：覆盖 GLM 能力圈内 + 盲区边缘，用 dir_probe problems（含 op_type）
PROBLEMS = [
    ("01_matmul",    "01_square_matrix_multiplication.json", "gemm"),
    ("23_softmax",   "23_softmax.json",                      "softmax"),
    ("40_layernorm", "40_layernorm.json",                    "normalization"),
    ("47_reduction", "47_reduction.json",                    "reduction"),
    ("50_conv2d",    "50_conv_standard_2d.json",             "conv2d"),
]
DIR_PROBE = os.environ.get("DT_DIR_PROBE", str(Path(__file__).resolve().parent.parent / "dir_probe"))

# naiveness 评分（正则检测 seed 是否预支了优化——预支越多分越低）
_NAIVE_CHECKS = [
    ("allow_tf32",   r"allow_tf32\s*=\s*True",          False, "用了 TC，precision_tc 方向无 headroom"),
    ("autotune",     r"@triton\.autotune",               False, "用了 autotune，tile_config 方向无 headroom"),
    ("num_stages",   r"num_stages\s*=",                   False, "用了软件流水，timing_overlap 方向无 headroom"),
    ("big_block",    r"BLOCK[_A-Z]*\s*=\s*(128|256|512)", False, "用了大 tile（≥128）"),
    ("small_block",  r"BLOCK[_A-Z]*\s*=\s*(32|64)",       True,  "用了小 tile（朴素）"),
]


def naiveness_score(code: str) -> dict:
    """检测 seed 是否朴素。返回 {checks, score, notes}。score 越高越朴素（0-1）。"""
    notes = []
    good = 0
    total = 0
    for name, pat, want_present, _desc in _NAIVE_CHECKS:
        total += 1
        found = bool(re.search(pat, code))
        if want_present:
            ok = found
            if not ok:
                notes.append(f"{name}: 期望有但没找到")
        else:
            ok = not found
            if not ok:
                notes.append(f"{name}: 预支了优化（{name}）")
        if ok:
            good += 1
    return {"score": good / total if total else 0, "good": good, "total": total, "notes": notes}


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _render(template: str, **kwargs) -> str:
    r = template
    for k, v in kwargs.items():
        r = r.replace("{" + k + "}", str(v))
    return r


def _extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _debug_dir(config: dict) -> Path | None:
    """Directory used to persist naive seed generation attempts for post-mortem."""
    out = config.get("output_dir")
    if not out:
        return None
    d = Path(out) / "naive_seed_debug"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_debug_text(dbg: Path | None, name: str, text: str) -> None:
    if dbg is None:
        return
    try:
        (dbg / name).write_text(text or "", encoding="utf-8")
    except Exception as e:
        print(f"  [naive-debug] failed to write {name}: {e}")


def _write_debug_json(dbg: Path | None, name: str, data: dict) -> None:
    if dbg is None:
        return
    try:
        (dbg / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"  [naive-debug] failed to write {name}: {e}")


async def gen_one(client, model, system, user, temp=0.7) -> str | None:
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                temperature=temp, max_tokens=20000, timeout=600,
            )
            record_usage(resp)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                print(f"  [LLM] failed: {e}")
                return None
            await asyncio.sleep(2 ** attempt)


async def gen_seed(problem: Problem, op_type: str, config: dict, max_retries=2) -> dict:
    """L1: 纯 LLM 手写 naive seed。system=naive_seed_system + triton_api_ref，0 AKG。

    返回 shape 兼容 generator.generate_kernel：成功 {"code","latency_ms","refinement_rounds"}，
    失败 None。每次 LLM 尝试和验证错误都会写入 output_dir/naive_seed_debug，方便失败复盘。
    """
    prompt_dir = Path(config["prompt_dir"])
    system_tpl = _read(str(prompt_dir / "naive_seed_system.txt"))
    api_ref = _read(str(prompt_dir / "triton_api_reference.md"))
    system = _render(system_tpl, triton_api_ref=api_ref)

    user_tpl = _read(str(prompt_dir / "naive_seed_user.txt"))
    user = _render(user_tpl, kernel_code=problem.reference, op_type=op_type)

    model_cfg = config.get("model_frontend", config["model"])
    client = AsyncOpenAI(base_url=model_cfg["url"], api_key=model_cfg["api_key"])

    rel_tol = adaptive_rel_tol(problem)
    dbg = _debug_dir(config)
    _write_debug_json(dbg, "meta.json", {
        "problem": getattr(problem, "name", None),
        "op_type": op_type,
        "model": model_cfg.get("model"),
        "max_retries": max_retries,
        "attempts": max_retries + 1,
    })
    last_err = ""
    for rnd in range(max_retries + 1):
        _write_debug_text(dbg, f"attempt_{rnd}_user_prompt.txt", user)
        resp = await gen_one(client, model_cfg["model"], system, user,
                             temp=0.7 if rnd == 0 else 0.3)
        if resp is None:
            last_err = "llm request failed"
            _write_debug_text(dbg, f"attempt_{rnd}_error.txt", last_err)
            _write_debug_json(dbg, "failure_summary.json", {"last_round": rnd, "last_error": last_err})
            return None
        _write_debug_text(dbg, f"attempt_{rnd}_response.txt", resp)
        code = _extract_code(resp)
        if code is None:
            if "import triton" in resp or "def run" in resp:
                code = resp
            else:
                last_err = "no code block"
                _write_debug_text(dbg, f"attempt_{rnd}_error.txt", last_err)
                user = user + f"\n\n# 上次失败：{last_err}\n请用 ```python 代码块给出完整实现，重写。"
                continue
        _write_debug_text(dbg, f"attempt_{rnd}_candidate.py", code)

        # anti-pytorch
        ap = anti_pytorch_check(code)
        if ap:
            last_err = f"[anti_pytorch] {ap}"
            _write_debug_text(dbg, f"attempt_{rnd}_error.txt", last_err)
            user = user + f"\n\n# 上次失败：{last_err}\n请只用 triton/torch tensor 操作，重写。"
            continue

        pr = triton_profile(code, problem, timeout_seconds=180, rel_tol=rel_tol)
        _write_debug_json(dbg, f"attempt_{rnd}_profile.json", {
            "compiled": bool(pr.compiled),
            "correct": bool(pr.correct),
            "latency_ms": pr.latency_ms,
            "error": pr.error,
        })
        if pr.compiled and pr.correct and pr.latency_ms:
            _write_debug_text(dbg, "successful_seed.py", code)
            _write_debug_json(dbg, "success_summary.json", {
                "round": rnd,
                "latency_ms": pr.latency_ms,
                "naiveness": naiveness_score(code),
            })
            # 返回 shape 兼容 generator.generate_kernel（code/latency_ms/refinement_rounds）
            return {"code": code, "latency_ms": pr.latency_ms,
                    "refinement_rounds": rnd, "naiveness": naiveness_score(code)}
        last_err = pr.error or "verify failed"
        _write_debug_text(dbg, f"attempt_{rnd}_error.txt", last_err)
        # 重试时把错误反馈进去
        user = user + f"\n\n# 上次失败：{last_err[:500]}\n请修正重写。"

    _write_debug_json(dbg, "failure_summary.json", {"last_round": max_retries, "last_error": last_err})
    return None


async def main():
    import yaml
    config = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml")))

    out_root = os.path.join(PROJECT, "output/naive_seed_l1")
    os.makedirs(out_root, exist_ok=True)

    results = []
    for label, prob_name, op_type in PROBLEMS:
        prob_path = f"{DIR_PROBE}/problems/{prob_name}"
        print(f"\n{'='*60}\n[{label}] ({op_type}) L1 naive seed gen\n{'='*60}", flush=True)
        problem = load_problem(prob_path)
        t0 = time.time()
        r = await gen_seed(problem, op_type, config)
        dt = time.time() - t0

        if r["status"] == "ok":
            seed_path = os.path.join(out_root, f"{label}_naive.py")
            with open(seed_path, "w") as f:
                f.write(r["code"])
            nv = r["naiveness"]
            print(f"[{label}] ✅ ok latency={r['latency_ms']:.4f}ms "
                  f"naiveness={nv['score']:.2f}({nv['good']}/{nv['total']}) {dt:.0f}s")
            if nv["notes"]:
                print(f"   notes: {'; '.join(nv['notes'])}")
            results.append({"label": label, "op_type": op_type, "status": "ok",
                            "latency_ms": r["latency_ms"], "naiveness": nv, "elapsed_s": round(dt)})
        else:
            print(f"[{label}] ❌ {r['status']}: {str(r.get('error',''))[:120]} ({dt:.0f}s)")
            results.append({"label": label, "op_type": op_type, "status": r["status"],
                            "error": str(r.get("error", ""))[:300], "elapsed_s": round(dt)})
        torch.cuda.empty_cache()

    with open(os.path.join(out_root, "_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\nL1 NAIVE SEED GEN SUMMARY\n{'='*60}")
    print(f"{'label':14s} {'op':13s} {'status':>6s} {'lat_ms':>10s} {'naive':>6s} {'time':>6s}")
    for r in results:
        lat = f"{r['latency_ms']:.3f}" if r.get('latency_ms') else "N/A"
        nv = f"{r['naiveness']['score']:.2f}" if r.get('naiveness') else "N/A"
        print(f"{r['label']:14s} {r['op_type']:13s} {r['status']:>6s} {lat:>10s} {nv:>6s} {r.get('elapsed_s',0):>5}s")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n成功率 {ok}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
