"""naive_seed_l2.py — L2 naive 化验证：纯 LLM naive seed（0 AKG）+ .pt 权重处理。

验证目标（work-log「L2 naive 化」）：naive_seed_gen 加权重处理后，能否在 L2 融合题
（含 .pt frozen 权重）上产出正确 seed，不依赖 AKG。

用法：
    conda activate forge
    cd /home/wangyichen/DirecTune-MCTS
    python naive_seed_l2.py
"""
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

import torch
from triton_backend import load_problem, profile as triton_profile, adaptive_rel_tol
from naive_seed_gen import gen_seed, naiveness_score

DIR_PROBE = os.environ.get("DT_DIR_PROBE", str(Path(__file__).resolve().parent.parent / "dir_probe"))
# L2 题：含 .pt 权重。用自带 kb_level2（权重路径由 load_problem 重写为 JSON 同目录），不用 dir_probe（路径要重写）
DT = str(Path(__file__).resolve().parent / "problems" / "kb_level2")
PROBLEMS = [
    # (label, problem_json, op_type)  — 都含 .pt 权重的 L2 fusion
    ("l2_9",  "9_Matmul_Subtract_Multiply_ReLU.json",  "fusion"),
    ("l2_76", "76_Gemm_Add_ReLU.json",                 "fusion"),
    ("l2_30", "30_Gemm_GroupNorm_Hardtanh.json",       "fusion"),
]


async def main():
    import yaml
    config = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml")))
    out_root = os.path.join(PROJECT, "output/naive_seed_l2")
    os.makedirs(out_root, exist_ok=True)

    results = []
    for label, prob_name, op_type in PROBLEMS:
        prob_path = f"{DT}/{prob_name}"
        print(f"\n{'='*60}\n[{label}] ({op_type}) L2 naive seed gen (0 AKG, .pt weights)\n{'='*60}", flush=True)
        problem = load_problem(prob_path)
        t0 = time.time()
        r = await gen_seed(problem, op_type, config)
        dt = time.time() - t0

        if r and r.get("code"):
            seed_path = os.path.join(out_root, f"{label}_naive.py")
            with open(seed_path, "w") as f:
                f.write(r["code"])
            nv = r.get("naiveness", naiveness_score(r["code"]))
            print(f"[{label}] ✅ ok latency={r['latency_ms']:.4f}ms "
                  f"naiveness={nv['score']:.2f}({nv['good']}/{nv['total']}) {dt:.0f}s")
            if nv["notes"]:
                print(f"   notes: {'; '.join(nv['notes'])}")
            results.append({"label": label, "op_type": op_type, "status": "ok",
                            "latency_ms": r["latency_ms"], "naiveness": nv,
                            "refinement_rounds": r.get("refinement_rounds"), "elapsed_s": round(dt)})
        else:
            err = str(r.get("error", "")) if r else "gen returned None"
            print(f"[{label}] ❌ fail: {err[:120]} ({dt:.0f}s)")
            results.append({"label": label, "op_type": op_type, "status": "fail",
                            "error": err[:300], "elapsed_s": round(dt)})
        torch.cuda.empty_cache()

    with open(os.path.join(out_root, "_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\nL2 NAIVE SEED GEN SUMMARY (0 AKG)\n{'='*60}")
    print(f"{'label':8s} {'op':8s} {'status':>6s} {'lat_ms':>10s} {'naive':>6s} {'time':>6s}")
    for r in results:
        lat = f"{r['latency_ms']:.3f}" if r.get('latency_ms') else "N/A"
        nv = f"{r['naiveness']['score']:.2f}" if r.get('naiveness') else "N/A"
        print(f"{r['label']:8s} {r['op_type']:8s} {r['status']:>6s} {lat:>10s} {nv:>6s} {r.get('elapsed_s',0):>5}s")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n成功率 {ok}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
