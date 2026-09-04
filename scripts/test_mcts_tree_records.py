#!/usr/bin/env python3
"""Mock 集成测试：MCTS 带标签节点记录（mcts_tree.json，2026-08-27 新增）。

不调 LLM、不动 GPU：把 agents.unified_editor 替换为脚本化的合成结果序列，
跑通 run_mcts 后断言三处产物（mcts_tree.json / checkpoint_iterN / interim
final_results）的标签一致性。

脚本化场景刻意复刻论文 intro 的机制结构——**首层全部回归，冠军路径穿过一条
「贪心会永久丢弃」的回归边**：
    n0 seed 10.0ms
    ├─ n1 mem_layout      11.9ms   （回归边）
    │   └─ n3 timing_overlap 6.4ms ← champion
    └─ n2 control_flow_spec 12.5ms （回归边）

附带反事实重放验证：按 order_index 取提议流，套**祖先感知** accept-only 规则
（贪心循环的 prompt 只含 current-best ⇒ 被拒节点的后代不会被生成）——重放永远停
在 seed 10.0ms，与树搜索的 6.4ms 形成分歧；champion_path 同时标出被切断的回归边。
这证明记录足以离线回答「贪心卡在哪」，而不需要重跑 GPU。
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mcts  # noqa: E402  (repo root)

SEED_LAT = 10.0


def ok(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def success(code_tag, branch_id, lat, baseline_code, mode="incremental"):
    return {
        "baseline_code": baseline_code,
        "plan_id": f"ue_{code_tag}",
        "branch_id": branch_id,
        "sample_id": "0",
        "code": f"# {code_tag}\ndef run():\n    return '{code_tag}'\n",
        "compiled": True,
        "correct": True,
        "runnable": True,
        "change_description": f"{branch_id} edit on {code_tag}",
        "edit_mode": mode,
        "match_levels": {"exact": 1},
        "latency_ms": lat,
    }


def failure(branch_id, baseline_code):
    return {
        "baseline_code": baseline_code,
        "plan_id": "ue_fail",
        "branch_id": branch_id,
        "sample_id": "0",
        "error": f"All 3 full-rewrite retries exhausted: parse fail ({branch_id})",
        "change_description": "",
        "edit_mode": "full_rewrite",
        "match_levels": {},
    }


class FakeEditor:
    """按父节点代码特征返回脚本化结果：模拟回归边 + 冠军从回归子树长出。"""

    def __init__(self):
        self.calls = 0

    async def __call__(self, candidates, experiences, problem, config,
                       applicable_directions=None, deadline=None):
        self.calls += 1
        base = candidates[0]["code"]
        tag = base.strip().splitlines()[0].lstrip("# ")
        if tag == "seed":
            # 首层：两个方向都回归 + 一个方向失败
            return [
                success("e1a", "dir_3_mem_layout", 11.9, base),
                success("e1b", "dir_8_control_flow_spec", 12.5, base),
                failure("dir_2_precision_tc", base),
            ]
        if tag.startswith("e1a"):
            # 回归节点内的改进——贪心永远走不到这里
            return [success("e2a", "dir_6_timing_overlap", 6.4, base)]
        # 其余（含 e2a 子树、e1b 子树）：全失败 → dead end
        return [failure("dir_2_precision_tc", base)]


async def main():
    out_dir = tempfile.mkdtemp(prefix="mcts_tree_test_")
    fake = FakeEditor()
    mcts.unified_editor = fake.__call__          # 打桩 LLM 扩展

    config = {
        "output_dir": out_dir,
        "mcts_rollouts": 4,
        "mcts_rollout_depth": 1,
        "mcts_cpuct": 1.0,
        "mcts_max_depth": 4,
        "search_time_budget": 0,
        "exp_n": 16,
        "topk": 8,
    }
    problem = type("P", (), {"name": "mock_problem"})()

    result = await mcts.run_mcts(
        initial_candidate={
            "code": "# seed\nprint('seed')\n",
            "solution_path": "seed.py",
            "latency_ms": SEED_LAT,
            "hw_metrics": None,
        },
        problem=problem,
        config=config,
        experiences=[],
        episode_id=0,
        episode_output_dir=out_dir,
        applicable_directions=[],   # 空 → 跳过分类器与 direction_store
    )

    print("\n== 断言产物 ==")
    tree_path = Path(out_dir) / "mcts_tree.json"
    ok(tree_path.exists(), "mcts_tree.json 落盘")
    ckpts = sorted(Path(out_dir).glob("checkpoint_iter*.json"))
    ok(len(ckpts) >= 1, f"checkpoint 存在（{len(ckpts)} 个）")
    interim = json.loads((Path(out_dir) / "final_results.json").read_text())
    ok("mcts_tree" in interim, "interim final_results 携带 mcts_tree")

    rec = json.loads(tree_path.read_text())
    nodes = rec["nodes"]
    by_id = {n["node_id"]: n for n in nodes}

    print("\n== 断言拓扑 ==")
    ok(rec["schema_version"] == 1, "schema_version")
    ok(rec["num_nodes"] == len(nodes) == 4, f"num_nodes=4 实际 {rec['num_nodes']}")
    ok(all(n["parent_id"] in by_id for n in nodes if n["parent_id"]),
       "所有 parent_id 可解析")
    orders = [n["order_index"] for n in nodes]
    ok(sorted(orders) == list(range(4)), f"order_index 连续 0..3：{orders}")
    ok([n["node_id"] for n in sorted(nodes, key=lambda x: x["order_index"])]
       == ["n0", "n1", "n2", "n3"], "创建序 = 预算序")

    print("\n== 断言方向标签 ==")
    ok(by_id["n1"]["direction"] == "mem_layout"
       and by_id["n2"]["direction"] == "control_flow_spec"
       and by_id["n3"]["direction"] == "timing_overlap", "三个节点的方向标签")
    ok(by_id["n2"]["vs_parent_x"] == round(SEED_LAT / 12.5, 6)
       and by_id["n2"]["vs_parent_x"] < 1, "回归边被 vs_parent_x<1 标出")
    ok(by_id["n3"]["edit_mode"] == "incremental", "edit_mode 标签透传")

    print("\n== 断言 champion 与归因链 ==")
    ok(rec["champion_node_id"] == "n3", "champion = 最低延迟节点 n3")
    path_dirs = [step["direction"] for step in rec["champion_path"]]
    ok(path_dirs == [None, "mem_layout", "timing_overlap"],
       f"champion_path 方向链 {path_dirs}")
    ok(result["best_latency_ms"] == 6.4
       and result["candidates"][0]["latency_ms"] == 6.4, "返回值 champion")

    print("\n== 断言 expansion 事件 ==")
    ev = rec["expansion_events"]
    ok(len(ev) == 3, f"3 次 expansion 有事件记录（实际 {len(ev)}）")
    ok(ev[0]["sampled"] == 3 and ev[0]["validated"] == 2
       and ev[0]["failures"][0]["branch_id"] == "dir_2_precision_tc",
       "首层 sampled/validated/失败明细")
    ok(rec["budget_counters"]["validated_nodes"] == 3
       and rec["budget_counters"]["expansions"] == 3, "预算计数器")
    ok(fake.calls == 3, f"fake editor 被调用 3 次（实际 {fake.calls}）")

    print("\n== 断言 checkpoint 一致性 ==")
    last = json.loads(ckpts[-1].read_text())
    ok(last["tree"]["champion_node_id"] == "n3", "checkpoint 带 champion_node_id")
    ok(last["tree"]["champion_path"][-1]["node_id"] == "n3",
       "checkpoint 带归因链")
    ok("labeled_records" in last["tree"], "checkpoint 指向 mcts_tree.json")

    print("\n== 反事实重放（祖先感知 accept-only，同提议流）==")
    # 贪心循环的 prompt 只含 current-best ⇒ 被拒节点的后代根本不会被生成。
    # 因此按 order_index 重放时：parent 不在接受集 ⇒ 该提议对贪心不可见。
    # （纯延迟过滤是不对的——6.4ms 绝对优于 seed 会被「看见」，但真实循环没这上下文。）
    kept = {"n0"}
    best = SEED_LAT
    for n in sorted(nodes, key=lambda x: x["order_index"])[1:]:
        if n["parent_id"] not in kept:
            continue          # 父链断掉：贪心循环内此提议不存在
        if n["latency_ms"] is not None and n["latency_ms"] < best:
            best = n["latency_ms"]
            kept.add(n["node_id"])
    ok(best == SEED_LAT == 10.0,
       f"贪心接受策略终点 {best}ms（回归子树全部不可见）")
    ok(best > 6.4, "贪心停在 seed，树搜索 6.4ms —— 分歧可由纯记录证明")

    print("\n== 冠军路径上的「被贪心切断的边」 ==")
    severed = [step for step in rec["champion_path"]
               if step.get("vs_parent_x") is not None and step["vs_parent_x"] < 1]
    ok(len(severed) >= 1 and severed[0]["node_id"] == "n1",
       "champion_path 标出第一条 vs_parent_x<1 的边（depth1 即被切）")

    print(f"\nALL PASS（产物目录 {out_dir}）")


if __name__ == "__main__":
    asyncio.run(main())
