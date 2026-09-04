#!/usr/bin/env python3
"""对照试验 runner：等资源 MCTS vs AKG（方案 A，best-of-N）。

对题目清单里的每一题：
  1. mcts 臂：merge(config_ab_shared.yaml + config_ab_mcts.yaml) → 跑 main.py
     （gen_mode=naive 端到端，seed 生成计入资源）→ 读 final_results.json 的
     resource_usage（calls / total_tokens）作为本题资源包
  2. akg 臂：以该资源包为预算跑 scripts/run_akg_arm.py（best-of-N 独立生成，
     任一轴到限即停，取 harness 最优）

断点续跑：mcts 臂以 <out>/final_results.json 存在为准；akg 臂以 result.json
含 budget_done 标记为准。全程单 GPU 串行（CUDA_VISIBLE_DEVICES 由 --gpus 传）。

用法（forge env）：
  python scripts/run_ab_vs_akg.py --problems problems/ab_list_15.txt --repeats 1
  python scripts/run_ab_vs_akg.py --problems ... --arms akg      # 只补 akg 臂
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

MCTS_ROOT = Path(__file__).resolve().parent.parent
FORGE_PY = "/home/wangyichen/miniconda3/envs/forge/bin/python3"


def load_problems(path: Path):
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            p = Path(line if line.startswith("/") else MCTS_ROOT / line)
            if not p.exists():
                print(f"[warn] problem missing, skipped: {p}")
                continue
            out.append(p)
    return out


def merge_configs(base_path: Path, shared_path: Path, arm_path: Path) -> dict:
    """底座 = 仓库 config.yaml（保证 prompt_dir 等基础键齐全），
    依次覆盖 shared（资源口径）→ arm（search 策略）。"""
    cfg = yaml.safe_load(base_path.read_text()) or {}
    cfg.update(yaml.safe_load(shared_path.read_text()) or {})
    cfg.update(yaml.safe_load(arm_path.read_text()) or {})
    cfg.pop("budget", None)   # budget 块是 runner/akg 臂的概念，main.py 不认识
    return cfg


def run_mcts_arm(problem: Path, run_dir: Path, args) -> dict | None:
    final = run_dir / "final_results.json"
    if final.exists():
        print(f"  [mcts] skip (exists): {run_dir}")
        return json.loads(final.read_text())
    eff = merge_configs(MCTS_ROOT / "config.yaml",
                        MCTS_ROOT / args.shared_config, MCTS_ROOT / args.mcts_config)
    init = problem.parent / (problem.stem + "_initial.py")
    if init.exists():
        eff["initial_solution"] = str(init)   # 每题覆盖底座里的 stale 路径
    run_dir.mkdir(parents=True, exist_ok=True)
    eff_path = run_dir / "effective_config.yaml"
    eff_path.write_text(yaml.safe_dump(eff, allow_unicode=True, sort_keys=False))
    cmd = [FORGE_PY, "main.py", "--config", str(eff_path),
           "--problem", str(problem), "--output", str(run_dir)]
    print(f"  [mcts] run → {run_dir}")
    with (run_dir / "run.log").open("w") as log:
        subprocess.run(cmd, check=False, cwd=MCTS_ROOT, stdout=log,
                       stderr=subprocess.STDOUT)
    if not final.exists():
        print(f"  [mcts] FAILED — see {run_dir/'run.log'}")
        return None
    return json.loads(final.read_text())


def run_akg_arm(problem: Path, run_dir: Path, budget: dict, args) -> dict | None:
    res = run_dir / "result.json"
    if res.exists() and "budget_done" in res.read_text():
        print(f"  [akg] skip (done): {run_dir}")
        return json.loads(res.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [FORGE_PY, "scripts/run_akg_arm.py",
           "--problem", str(problem), "--output-dir", str(run_dir),
           "--config", str(MCTS_ROOT / args.shared_config),
           "--budget-calls", str(budget["calls"]),
           "--budget-tokens", str(budget["tokens"]),
           "--max-iters", str(args.akg_max_iters),
           "--gpus", args.gpus]
    print(f"  [akg] run (budget calls={budget['calls']} "
          f"tokens={budget['tokens']}) → {run_dir}")
    with (run_dir / "driver.log").open("w") as log:
        subprocess.run(cmd, check=False, cwd=MCTS_ROOT, stdout=log,
                       stderr=subprocess.STDOUT)
    if not res.exists():
        print(f"  [akg] FAILED — see {run_dir/'driver.log'}")
        return None
    data = json.loads(res.read_text())
    data["budget_done"] = True
    res.write_text(json.dumps(data, indent=1, default=str))
    return data


def write_metadata(out: Path, args, problems):
    """实验元数据：模型/资源/预算口径/配置快照/代码版本，不含 api_key。"""
    import platform
    import subprocess as sp

    def snap(p):
        d = yaml.safe_load((MCTS_ROOT / p).read_text()) or {}
        return {k: v for k, v in d.items()
                if not isinstance(v, dict) or "api_key" not in v}

    gpu = sp.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                  "--format=csv,noheader", "-i", args.gpus],
                 capture_output=True, text=True).stdout.strip()
    commit = sp.run(["git", "rev-parse", "HEAD"], cwd=MCTS_ROOT,
                    capture_output=True, text=True).stdout.strip()
    meta = {
        "experiment": "equal-resource MCTS vs AKG (best-of-N, 方案A)",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "gpu": gpu,
        "git_commit": commit,
        "model": (yaml.safe_load((MCTS_ROOT / args.shared_config).read_text())
                  .get("model_frontend", {})).get("model"),
        "budget_policy": ("每题先跑 mcts 臂，final_results.resource_usage 的 "
                          "llm_calls/total_tokens 回填为 akg 臂 best-of-N 预算；"
                          "上报延迟只认 profile_isolated 统一 harness"),
        "configs": {"shared": snap(args.shared_config), "mcts": snap(args.mcts_config)},
        "problems": [str(p) for p in problems],
        "repeats": args.repeats,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (out / "EXPERIMENT.md").write_text(
        "# 等资源对照实验（MCTS vs AKG）\n\n"
        f"- 启动：{meta['started']} @ {meta['host']}\n"
        f"- GPU：{gpu}\n- 模型：{meta['model']}\n- 代码：{commit}\n"
        f"- 预算口径：{meta['budget_policy']}\n"
        f"- 题目：{len(problems)} 题（trace 10 题集）× repeats {args.repeats}\n"
        "- champion 位置：`mcts/<题>_r<k>/champion.py`、"
        "`akg/<题>_r<k>/champion.py`\n")
    print(f"[meta] experiment metadata → {out/'metadata.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, help="题目清单（每行一个 problem json）")
    ap.add_argument("--arms", default="mcts,akg")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--shared-config", default="config_ab_shared.yaml")
    ap.add_argument("--mcts-config", default="config_ab_mcts.yaml")
    ap.add_argument("--out", default="output/ab_vs_akg")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--akg-max-iters", type=int, default=200,
                    help="akg 臂迭代数安全上限（透传 run_akg_arm.py）")
    args = ap.parse_args()

    problems = load_problems(Path(args.problems))
    arms = args.arms.split(",")
    print(f"{len(problems)} problems × arms={arms} × repeats={args.repeats}")
    write_metadata(Path(args.out), args, problems)

    for p in problems:
        pname = p.stem
        for r in range(1, args.repeats + 1):
            tag = f"{pname}_r{r}"
            print(f"== {tag} ==")
            ru = None
            if "mcts" in arms:
                m = run_mcts_arm(p, Path(args.out) / "mcts" / tag, args)
                ru = (m or {}).get("resource_usage") or {}
                if m and m.get("final_candidates"):
                    code = m["final_candidates"][0].get("code", "")
                    if code:
                        (Path(args.out) / "mcts" / tag / "champion.py").write_text(code)
                if not ru:
                    print("  [mcts] 无 resource_usage，akg 臂退化为自然收敛（不设 cap）")
            if "akg" in arms:
                if ru is None:   # --arms akg 单独跑时，读已恢复的 mcts 臂产物
                    mf = Path(args.out) / "mcts" / tag / "final_results.json"
                    if mf.exists():
                        ru = json.loads(mf.read_text()).get("resource_usage") or {}
                budget = {"calls": int(ru.get("llm_calls") or 0),
                          "tokens": int(ru.get("total_tokens") or 0)}
                if not budget["calls"] and not budget["tokens"]:
                    print("  [akg] SKIP — mcts 臂无 resource_usage，无法对齐预算")
                    continue
                run_akg_arm(p, Path(args.out) / "akg" / tag, budget, args)
    print("all done.")


if __name__ == "__main__":
    main()
