"""blind_ablation.py — 盲区题对拍：naive_seed_gen vs AKG gen (v3/akg)。

确认 AKG fallback 在真盲区题上是否真无效。两道题：
  - 94_mseloss (reduction, 07-01 测得两边 gen+search 都失败)
  - 97_SDPA (attention, 07-01 测得两边 gen 失败)

每题两条路径：
  naive = naive_seed_gen.gen_seed (0 AKG, 默认路径)
  akg   = generator.generate_kernel(gen_mode 按 level：level1→v3)

只跑到出 seed，不进 search。记录 成败/compiled/correct/latency/naiveness。

用法：
    conda activate forge
    cd /home/wangyichen/DirecTune-MCTS
    python blind_ablation.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
PROJECT = str(Path(__file__).resolve().parent)
sys.path.insert(0, PROJECT)

import torch
from triton_backend import load_problem, profile as triton_profile, adaptive_rel_tol
from naive_seed_gen import gen_seed, naiveness_score

DIR_PROBE = os.environ.get("DT_DIR_PROBE", str(Path(__file__).resolve().parent.parent / "dir_probe"))
PROBLEMS = [
    ("94_mseloss", "94_mseloss.json", "reduction"),
    ("97_sdpa",    "97_sdpa.json",   "attention"),
]


async def gen_akg(problem, baseline_code, config):
    """调 DirecTune 原 generator.generate_kernel，gen_mode=v3（level1 路由）。"""
    from generator import generate_kernel as _gen
    gen_config = dict(config)
    gen_config["gen_mode"] = "v3"
    plan = "Generate a correct, high-performance Triton kernel for this computation"
    return await _gen(problem, baseline_code, plan, gen_config)


async def run_one(problem, config, op_type, method):
    t0 = time.time()
    try:
        if method == "naive":
            r = await gen_seed(problem, op_type, config)
        else:  # akg
            r = await gen_akg(problem, problem.reference, config)
    except Exception as e:
        return {"method": method, "status": "crash", "error": str(e)[:300],
                "elapsed_s": round(time.time() - t0)}
    dt = time.time() - t0

    if not r or not r.get("code"):
        err = str(r.get("error", ""))[:200] if r else "gen returned None"
        return {"method": method, "status": "gen_fail", "error": err,
                "elapsed_s": round(dt)}

    code = r["code"]
    rel_tol = adaptive_rel_tol(problem)
    pr = triton_profile(code, problem, timeout_seconds=180, rel_tol=rel_tol)
    nv = r.get("naiveness") or naiveness_score(code)
    return {
        "method": method, "status": "ok",
        "compiled": pr.compiled, "correct": pr.correct,
        "latency_ms": pr.latency_ms, "naiveness": nv,
        "refinement_rounds": r.get("refinement_rounds"),
        "code_chars": len(code), "elapsed_s": round(dt),
    }


async def main():
    import yaml
    config = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml")))
    out_root = os.path.join(PROJECT, "output/blind_ablation")
    os.makedirs(out_root, exist_ok=True)

    results = []
    for label, prob_name, op_type in PROBLEMS:
        prob_path = f"{DIR_PROBE}/problems/{prob_name}"
        problem = load_problem(prob_path)
        for method in ("naive", "akg"):
            print(f"\n{'='*60}\n[{label}] ({op_type}) method={method}\n{'='*60}", flush=True)
            r = await run_one(problem, config, op_type, method)
            r["label"] = label
            r["op_type"] = op_type
            results.append(r)
            if r["status"] == "ok":
                comp = "yes" if r["compiled"] else "NO"
                corr = "yes" if r["correct"] else "NO"
                lat = f"{r['latency_ms']:.4f}" if r["latency_ms"] else "None"
                nv = r["naiveness"]
                print(f"  ✅ {method}: compiled={comp} correct={corr} lat={lat} "
                      f"naive={nv['score']:.2f} {r['elapsed_s']}s")
            else:
                print(f"  ❌ {method}: {r['status']} {r.get('error','')[:80]} ({r['elapsed_s']}s)")
            torch.cuda.empty_cache()

    with open(os.path.join(out_root, "_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}\nBLIND ABLATION SUMMARY (naive vs AKG)\n{'='*60}")
    print(f"{'label':12s} {'op':10s} {'method':6s} {'status':9s} {'comp':>4s} {'corr':>4s} "
          f"{'lat_ms':>10s} {'naive':>6s} {'time':>6s}")
    for r in results:
        st = r["status"]
        if st == "ok":
            comp = "y" if r["compiled"] else "N"
            corr = "y" if r["correct"] else "N"
            lat = f"{r['latency_ms']:.3f}" if r.get("latency_ms") else "N/A"
            nv = f"{r['naiveness']['score']:.2f}"
        else:
            comp = corr = lat = nv = "-"
        print(f"{r['label']:12s} {r['op_type']:10s} {r['method']:6s} {st:9s} {comp:>4s} {corr:>4s} "
              f"{lat:>10s} {nv:>6s} {r.get('elapsed_s',0):>5}s")


if __name__ == "__main__":
    asyncio.run(main())
