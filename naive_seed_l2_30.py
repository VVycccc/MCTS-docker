"""naive_seed_l2_30.py — 批量确认 30 道 L2 题能否生成 naive seed（纯 LLM，0 AKG）。

题集：l2_selected_30_20260726.txt（贪心 max-coverage 选的 30 道，5 core × 6，33 token 全覆盖）。
每题成功后把 seed 落到 output/naive_seed_l2_30/{problem_name}_seed.py，
后续 search 可直接 --initial 该文件跳过生成阶段。

用法：
    conda activate forge
    cd /home/wangyichen/DirecTune-MCTS
    python naive_seed_l2_30.py                 # 跑全部 30
    python naive_seed_l2_30.py --only 1_Conv2D_ReLU_BiasAdd 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish   # 只跑指定题
    python naive_seed_l2_30.py --skip-existing   # 跳过已成功的（断点续跑）

产物：
    output/naive_seed_l2_30/{problem_name}_seed.py   成功 seed（search 直接 --initial 用）
    output/naive_seed_l2_30/{problem_name}_meta.json  latency/naiveness/refinement_rounds
    output/naive_seed_l2_30/_results.json            全量结果汇总
    output/naive_seed_l2_30/_manifest.json           {problem_name: seed_path} 供 runner 读
    output/naive_seed_l2_30/naive_seed_debug/{problem_name}/  每题 LLM 尝试/错误落盘
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

import torch
from triton_backend import load_problem, profile as triton_profile, adaptive_rel_tol
from naive_seed_gen import gen_seed, naiveness_score

# 题集来源
SELECTION_FILE = os.path.join(PROJECT, "l2_selected_30_20260726.txt")
PROBLEMS_DIR = os.path.join(PROJECT, "problems/kb_level2")
OUT_ROOT = os.path.join(PROJECT, "output/naive_seed_l2_30")


def load_selection() -> list[str]:
    names = []
    with open(SELECTION_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


def is_already_done(name: str) -> bool:
    """断点续跑判定：seed 文件存在且 meta 标 ok。"""
    seed = os.path.join(OUT_ROOT, f"{name}_seed.py")
    meta = os.path.join(OUT_ROOT, f"{name}_meta.json")
    if not (os.path.isfile(seed) and os.path.isfile(meta)):
        return False
    try:
        m = json.load(open(meta))
        return bool(m.get("status") == "ok" and m.get("latency_ms"))
    except Exception:
        return False


async def run_one(name: str, config: dict) -> dict:
    prob_path = os.path.join(PROBLEMS_DIR, f"{name}.json")
    out_dbg = os.path.join(OUT_ROOT, "naive_seed_debug", name)
    os.makedirs(out_dbg, exist_ok=True)

    # 把 debug 目录注入 config，让 gen_seed 的 _debug_dir 落到每题独立子目录
    cfg = dict(config)
    cfg["output_dir"] = out_dbg

    print(f"\n{'='*70}\n[{name}] L2 naive seed gen (0 AKG, .pt weights)\n{'='*70}", flush=True)
    problem = load_problem(prob_path)

    # op_type：L2 题都是 fusion，gen_seed 的 prompt 只用它做语义提示，不影响正确性
    op_type = "fusion"
    t0 = time.time()
    try:
        r = await gen_seed(problem, op_type, cfg)
    except Exception as e:
        r = None
        err = f"gen_seed raised: {e!r}"
    dt = time.time() - t0

    if r and r.get("code") and "@triton.jit" in r.get("code", ""):
        seed_path = os.path.join(OUT_ROOT, f"{name}_seed.py")
        with open(seed_path, "w") as f:
            f.write(r["code"])
        nv = r.get("naiveness") or naiveness_score(r["code"])
        meta = {
            "problem": name,
            "status": "ok",
            "latency_ms": r["latency_ms"],
            "naiveness": nv,
            "refinement_rounds": r.get("refinement_rounds"),
            "elapsed_s": round(dt),
            "seed_path": seed_path,
        }
        with open(os.path.join(OUT_ROOT, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[{name}] ✅ ok latency={r['latency_ms']:.4f}ms "
              f"naiveness={nv['score']:.2f}({nv['good']}/{nv['total']}) rounds={r.get('refinement_rounds')} {dt:.0f}s",
              flush=True)
        if nv["notes"]:
            print(f"   notes: {'; '.join(nv['notes'])}", flush=True)
        return meta
    else:
        err = (r.get("error") if r else None) or "gen returned None/no triton"
        # 若 gen_seed 抛异常，err 已在上面设过
        if 'err' in dir() and not r:
            pass
        meta = {
            "problem": name,
            "status": "fail",
            "error": str(err)[:500],
            "elapsed_s": round(dt),
        }
        with open(os.path.join(OUT_ROOT, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[{name}] ❌ fail: {str(err)[:150]} ({dt:.0f}s)", flush=True)
        return meta


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定题（空格分隔）")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已成功的题（断点续跑）")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(open(os.path.join(PROJECT, "config_full_l2_naive_triton.yaml")))
    os.makedirs(OUT_ROOT, exist_ok=True)

    all_names = load_selection()
    if args.only:
        want = set(args.only)
        all_names = [n for n in all_names if n in want]
        if not all_names:
            print(f"--only 没匹配到任何题。可用: {load_selection()[:5]}...")
            return

    todo = []
    skipped = []
    for name in all_names:
        if args.skip_existing and is_already_done(name):
            skipped.append(name)
            continue
        todo.append(name)

    print(f"题集 {len(all_names)} 道 | skip_existing={args.skip_existing} → "
          f"已成功跳过 {len(skipped)} | 待跑 {len(todo)}", flush=True)
    if skipped:
        print(f"跳过: {skipped}", flush=True)

    results = []
    for name in todo:
        meta = await run_one(name, config)
        results.append(meta)
        torch.cuda.empty_cache()

    # 汇总：合并已存在 meta + 本次结果
    all_results = []
    manifest = {}
    for name in all_names:
        mp = os.path.join(OUT_ROOT, f"{name}_meta.json")
        if os.path.isfile(mp):
            m = json.load(open(mp))
            all_results.append(m)
            if m.get("status") == "ok" and m.get("seed_path"):
                manifest[name] = m["seed_path"]

    with open(os.path.join(OUT_ROOT, "_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_ROOT, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 打印汇总表
    print(f"\n{'='*70}\nL2 NAIVE SEED GEN SUMMARY (30 题, 0 AKG)\n{'='*70}")
    print(f"{'problem':52s} {'status':>6s} {'lat_ms':>10s} {'naive':>6s} {'rnd':>4s} {'time':>6s}")
    for m in all_results:
        name = m.get("problem", "?")
        st = m.get("status", "?")
        lat = f"{m['latency_ms']:.3f}" if m.get("latency_ms") else "N/A"
        nv = f"{m['naiveness']['score']:.2f}" if m.get("naiveness") else "N/A"
        rnd = str(m.get("refinement_rounds", "-"))
        print(f"{name:52s} {st:>6s} {lat:>10s} {nv:>6s} {rnd:>4s} {m.get('elapsed_s',0):>5}s")
    ok = sum(1 for m in all_results if m.get("status") == "ok")
    print(f"\n成功率 {ok}/{len(all_results)} | manifest 写入 {len(manifest)} 个 seed 路径")
    print(f"产物目录: {OUT_ROOT}")


if __name__ == "__main__":
    asyncio.run(main())
