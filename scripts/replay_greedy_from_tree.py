#!/usr/bin/env python3
"""对一次真实 MCTS run 的带标签节点记录做反事实重放（2026-08-27）。

输入：output/<exp_name>/<problem>/mcts_tree.json（mcts.serialize_tree 落盘；
旧 run 没有此文件——记录功能是 2026-08-27 加的，需用新代码重跑）。

输出三件事（纯离线、零 GPU）：
  1. 祖先感知 accept-only 重放终点 vs MCTS 冠军 —— 论文 intro「贪心丢弃被拒
     分支」论断在同提议流下的量化分歧。贪心循环的 prompt 只含 current-best，
     被拒节点的后代根本不会被生成 ⇒ 重放时 parent 不在接受集则该提议不可见。
  2. champion_path 上第一条「贪心切断边」（vs_parent_x<1）→ mechanism 图数据。
  3. 冠军链的方向标签序列（每步收益归因到方向）。

用法：
  python scripts/replay_greedy_from_tree.py output/full_mcts/01_square_matrix_multiplication
"""

import json
import sys
from pathlib import Path


def load_tree(run_dir: Path) -> dict:
    for name in ("mcts_tree.json",):
        p = run_dir / name
        if p.exists():
            return json.loads(p.read_text())
    # 兜底：interim/final 的内嵌副本
    p = run_dir / "final_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("mcts_tree"):
            print("[info] mcts_tree.json 不存在，使用 final_results.json 内嵌副本")
            return d["mcts_tree"]
    sys.exit(f"[error] {run_dir} 下没有 mcts_tree.json。"
             f"该记录功能 2026-08-27 新增，请用当前代码重跑此题后再分析。")


def replay(records: list[dict], seed_latency: float) -> tuple[float, set[str]]:
    """祖先感知 accept-only：按 order_index 流处理，父不在接受集⇒不可见。"""
    kept = {"n0"}
    best = seed_latency
    for n in sorted((r for r in records if r["order_index"]),
                    key=lambda x: x["order_index"]):
        if n["parent_id"] not in kept or n["latency_ms"] is None:
            continue
        if n["latency_ms"] < best:
            best = n["latency_ms"]
            kept.add(n["node_id"])
    return best, kept


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rec = load_tree(run_dir)
    nodes = {n["node_id"]: n for n in rec["nodes"]}
    root = rec["root_node_id"]
    seed_lat = nodes[root]["latency_ms"]

    greedy_best, kept = replay(rec["nodes"], seed_lat)
    champ_id = rec["champion_node_id"]
    champ_lat = nodes[champ_id]["latency_ms"]

    print(f"run: {run_dir}")
    print(f"validated kernels: {len(nodes) - 1}   "
          f"(budget_counters={rec.get('budget_counters')})")
    print(f"\n[1] 同提议流反事实重放")
    print(f"    MCTS champion          : {champ_lat:.4f} ms ({champ_id})")
    print(f"    greedy accept-only 终点: {greedy_best:.4f} ms")
    if greedy_best and champ_lat:
        ratio = greedy_best / champ_lat
        print(f"    gap                    : {ratio:.2f}x"
              f"{' （贪心看不见冠军后代——机制成立）' if ratio > 1.01 else ''}")

    print(f"\n[2] champion_path 方向归因链")
    chain = []
    for step in rec["champion_path"]:
        lat = step["latency_ms"]
        vp = step.get("vs_parent_x")
        tag = ""
        if step["node_id"] == root:
            desc = f"seed {lat:.4f}ms"
        else:
            desc = f"{step['direction']} → {lat:.4f}ms"
            if vp is not None:
                tag = "  ⚡贪心切断边(vs_parent<1)" if vp < 1 else f"  (vs_parent {vp:.2f}x)"
        chain.append(f"    {step['node_id']:>4}  {desc}{tag}")
    print("\n".join(chain))

    sev = [s for s in rec["champion_path"]
           if s.get("vs_parent_x") is not None and s["vs_parent_x"] < 1]
    if sev:
        print(f"\n[结论] 冠军路径在第 "
              f"{[nodes[s['node_id']]['depth'] for s in sev][0]} 层存在贪心切断边"
              f"（{sev[0]['direction']}，{sev[0]['latency_ms']:.4f}ms）：accept-only 循环"
              f"在此永久丢弃了通往冠军的子树 —— intro 的 discard 论断在该 run 成立。")
    else:
        print(f"\n[结论] 该 run 冠军路径无回归边：本次胜利来自采样广度而非保留被拒分支，"
              f"叙事应落在 explore-exploit 预算分配上。")

    # direction 一览（全树，非仅冠军链）
    from collections import Counter
    c = Counter(n["direction"] for nid, n in nodes.items() if nid != root)
    print(f"\n[3] 全树方向分布: {dict(c)}")


if __name__ == "__main__":
    main()
