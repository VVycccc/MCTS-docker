"""mcts.py — Monte Carlo Tree Search variant of DirecTune's kernel optimization.

理论依据（work-log 2026-07-11「搜索方法的总结分析理论」）：
  DirecTune 的"每轮单改"把空间压扁成 contextual bandit(#2)；MCTS(#8 收敛) 仅在
  **放弃单改归因、建模优化序列 A→B→C（patch 叠加）** 时才适用。本模块正是把空间
  形式化为树：每个节点 = 一个已验证正确的 kernel，子节点 = 在父节点上施加一个优化
  方向后产出的新 kernel，深度 = 叠加的优化数。这样 UCT 的 O(ln T) regret 收敛保证
  才挂得上，LLM 方向先验通过 P-UCT 的先验项 P 正式注入。

reward = log(seed_latency / node_latency)（work-log 指定对数处理 35× vs 1.2× 量级差）。

dir_probe（~/dir_probe/REPORT.md）调研结果的落地：
  - **结论 1+4**：结构方向（消中间张量 / algo_equiv / mem_layout）是 10-50× 第一梯队，
    参数方向（tile_config / autotune）是 ~2× 第二梯队且 autotune 已覆盖。→ 默认先验表
    按此分级（_DIR_PROBE_PRIOR），结构方向高 P、tile_config 低 P。
  - **结论 3**：方向有效性 op_type-dependent（split_two_stage memory-bound +50× /
    compute-bound −40%）。→ P 先验由 LLM 分类器（按算子语义判适用性+排序）提供，
    dir_probe 表只作 blend 兜底，不硬编码跨算子结论。
  - **结论 5**：memory-bound / GEMM-主导 fusion 是搜索死区。→ adaptive 深度策略：
    子节点全不超 stall_threshold 的分支提前终止（不继续加深），把搜索预算从死区挪走。
  - **结论 6**：真盲区是 seed 难写，不是 search 救不回。→ MCTS 不解决 gen，只优化已有
    seed；seed 质量决定树上限。

dynamic depth（用户要求可 sweep 的旋钮，全套都可 sweep）：
  深度本身：
  - mcts_max_depth          —— 树深硬上限（基础值，可 sweep 比较 shallow/deep）。
  - mcts_depth_scale        —— 运行时对 max_depth 的乘子；effective = round(max_depth*scale)。
                                两个叠加 = 把深度完全参数化，外层 sweep scale 即可探不同树深。
  - mcts_depth_strategy     —— "fixed"（恒定到 max_depth）| "adaptive"（默认，按信号自调）。
  adaptive 早停信号（也当独立旋钮 sweep）：
  - mcts_min_depth          —— adaptive 下允许早停的最小深度（避免在浅层误判死区）。
  - mcts_stall_threshold    —— 子节点 reward 低于此值视为"无改进"（默认 log(1.05)≈0.049）。
  - mcts_stall_patience     —— 连续 N 次扩展仍无改进才判死区（默认 1，保守可调大）。
  backprop 长度（rollout 旋钮，也当 sweep 维度）：
  - mcts_rollout_depth      —— 扩展后沿 P-UCT 下行几层再回传（=1 子节点即终端；=k 更贴近
                                MCTS 原型但每 rollout k× LLM 扩展。默认 1）。
  adaptive 下：高效分支（reward 持续 > threshold）自然被 UCT 反复选中、长到 max_depth；
  死区分支在 min_depth 后终止 → 实际树深由信号动态决定，每 rollout log 出来可观测。

复用：扩展（expansion）直接调 agents.unified_editor([parent], applicable_directions=…)，
子节点 = unified_editor 已验证正确的 result（同款 patch+verify+NCU+skill），不重写 GPU 路径。
champion = 全树最低延迟节点（root/seed 永在树中 → 自然 carry-forward，治 work-log TODO [A]）。
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import (
    Problem,
    _derive_op_type,
    determine_applicable_directions,
    unified_editor,
)
from direction_store import (
    format_op_stats,
    get_op_stats,
    load_stats,
    merge_run,
    record_iteration,
    save_stats,
)
from search import update_experiences


# ---------------------------------------------------------------------------
# dir_probe-calibrated default direction priors (REPORT.md 结论 1/4 分级)
# ---------------------------------------------------------------------------

# 结构方向 = 第一梯队（dir_probe: 10-50×）；参数方向 = 第二梯队（~2×，autotune 覆盖）。
# v6 8 方向先验表（dir_probe REPORT §4 分级）。这些是 LLM 分类器不可用时的兜底先验；
# 分类器可用时与其排序 blend（见 _compute_child_priors）。
_DIR_PROBE_PRIOR: dict[str, float] = {
    "algo_equiv": 0.28,        # reduction_tree 21× / precompute 11× / algorithmic 17×
    "reduction_struct": 0.24,  # reduction_tree_layout 21.2× / split_two_stage 50×(memory-bound) / online 1.33×
    "mem_layout": 0.22,        # data_layout 51× / dematerialization / coalesced_access
    "timing_overlap": 0.16,    # software_pipeline 中位 6.1×/max 9.6×；num_stages ns1→4=1.31×
    "precision_tc": 0.16,      # tensor_core 17× max / 2.5× median
    "fusion": 0.12,            # dematerialization 交集；GEMM-主导死区（结论 5）
    "control_flow_spec": 0.08, # mask_simplify/early_exit/constexpr 低杠杆，autotune 友好
    "tile_config": 0.10,       # median 1.94×, autotune 更可靠 → 低先验（仍高于 control_flow，因适用面广）
}


def _reward(node_latency: float | None, seed_latency: float | None) -> float:
    """log speedup vs seed，clamp 防极端值（35×→3.55；0.5×→-0.69）。"""
    if (node_latency is None or node_latency <= 0
            or seed_latency is None or seed_latency <= 0):
        return 0.0
    r = math.log(seed_latency / node_latency)
    return max(-2.0, min(5.0, r))


@dataclass
class MCTSNode:
    """一个 kernel 状态。root=seed；子节点=父节点施加某方向后的新 kernel。"""
    code: str
    latency_ms: float | None
    hw_metrics: dict | None
    depth: int
    parent: "MCTSNode | None" = None
    direction: str | None = None       # 产生本节点的方向名（root=None）
    children: list["MCTSNode"] = field(default_factory=list)
    expanded: bool = False
    terminal: bool = False
    terminal_reason: str = ""
    N: int = 0                         # 访问次数（backprop 次数）
    W: float = 0.0                     # 累计 reward
    P: float = 1.0                     # 先验（创建时按方向赋）
    result: dict | None = None         # unified_editor result（champion 提取用）

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def best_latency_in_subtree(self) -> float | None:
        best = self.latency_ms
        for c in self.children:
            cb = c.best_latency_in_subtree()
            if cb is not None and (best is None or cb < best):
                best = cb
        return best

    def best_node_in_subtree(self) -> "MCTSNode":
        best = self
        for c in self.children:
            cb = c.best_node_in_subtree()
            if (cb.latency_ms is not None
                    and (best.latency_ms is None or cb.latency_ms < best.latency_ms)):
                best = cb
        return best


# ---------------------------------------------------------------------------
# Prior computation: blend LLM classifier ordering with dir_probe payoff table
# ---------------------------------------------------------------------------

def _compute_child_priors(
    children_directions: list[str],
    applicable_directions: list[dict] | None,
    config: dict,
) -> list[float]:
    """给一组子节点（按 direction 名）算归一化先验 P。

    两个信号 blend：
      - rank_prior：LLM 分类器的收益排序（越靠前越高），无分类器时用 dir_probe 表序。
      - dir_probe_prior：_DIR_PROBE_PRIOR 的固定分级（结论 1/4）。
    blend_w 控制谁主导（默认 0.5）。free_explore 给小常数先验（兜底探索动作）。
    """
    blend_w = float(config.get("mcts_prior_blend", 0.5))
    rank_decay = float(config.get("mcts_rank_decay", 0.5))   # rank i → exp(-i*decay)
    free_prior = float(config.get("mcts_free_explore_prior", 0.05))

    # rank map: direction name -> rank index (from classifier ordering)
    rank_map: dict[str, int] = {}
    if applicable_directions:
        for i, d in enumerate(applicable_directions):
            rank_map[d["name"]] = i

    raws: list[float] = []
    for name in children_directions:
        dp = _DIR_PROBE_PRIOR.get(name, 0.10)
        if name == "free_explore":
            raws.append(free_prior)
            continue
        if name in rank_map:
            rank_p = math.exp(-rank_map[name] * rank_decay)
        else:
            rank_p = dp  # 无分类器排序时退化为 dir_probe 分级
        raws.append(blend_w * rank_p + (1.0 - blend_w) * dp)

    total = sum(raws)
    if total <= 0:
        n = len(raws)
        return [1.0 / n] * n if n else []
    return [r / total for r in raws]


# ---------------------------------------------------------------------------
# Selection (P-UCT) + backprop
# ---------------------------------------------------------------------------

def _uct_select(node: MCTSNode, c_puct: float) -> MCTSNode | None:
    """P-UCT: argmax_c [ Q(c) + c_puct * P(c) * sqrt(N_parent) / (1 + N(c)) ]."""
    if not node.children:
        return None
    sqrtN = math.sqrt(max(node.N, 1))
    best, best_val = None, -math.inf
    for ch in node.children:
        u = c_puct * ch.P * sqrtN / (1.0 + ch.N)
        val = ch.Q + u
        if val > best_val:
            best_val, best = val, ch
    return best


def _select_leaf(root: MCTSNode, c_puct: float) -> tuple[MCTSNode, list[MCTSNode]]:
    """从 root 沿 P-UCT 下行到首个可扩展叶（未扩展 / 终端）。返回 (leaf, path)。"""
    node = root
    path = [node]
    while node.expanded and not node.terminal and node.children:
        nxt = _uct_select(node, c_puct)
        if nxt is None:
            break
        node = nxt
        path.append(node)
    return node, path


def _backprop(path: list[MCTSNode], value: float) -> None:
    for n in path:
        n.N += 1
        n.W += value


# ---------------------------------------------------------------------------
# Experiences (mirrors main._build_experiences_from_results to avoid circular import)
# ---------------------------------------------------------------------------

def _build_experiences(results: list[dict], candidates: list[dict]) -> list[dict]:
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


def _direction_of_branch(branch_id: str | None) -> str:
    """branch_id 'dir_5_algo_equiv' → 'algo_equiv'；'free_explore' → 'free_explore'。"""
    if not branch_id:
        return "free_explore"
    if branch_id == "free_explore":
        return "free_explore"
    if branch_id.startswith("dir_"):
        parts = branch_id.split("_", 2)
        return parts[2] if len(parts) >= 3 else branch_id
    return branch_id


# ---------------------------------------------------------------------------
# Expansion: reuse unified_editor's directional patch+verify as the action set
# ---------------------------------------------------------------------------

async def _expand(
    node: MCTSNode,
    experiences: list[dict],
    problem: Problem,
    config: dict,
    applicable_directions: list[dict] | None,
    deadline: float,
    seed_latency: float | None,
) -> list[dict]:
    """扩展叶节点：调 unified_editor 对 node 采各方向 patch，成功的成为子节点。

    返回 unified_editor 原始 results（含成败），供 experiences / direction_store 复用。
    """
    # unified_editor 的 direction_mode 要求 config.direction_organized_frontier=True。
    # 用副本开启，不改调用方 config。
    mcts_config = dict(config)
    mcts_config["direction_organized_frontier"] = True
    # free_explore 作为额外探索动作（豁免方向截断），默认开。
    mcts_config.setdefault("direction_free_explore", True)

    parent_candidate = {
        "code": node.code,
        "solution_path": "",
        "latency_ms": node.latency_ms,
        "hw_metrics": node.hw_metrics,
    }
    results = await unified_editor(
        [parent_candidate], experiences, problem, mcts_config,
        applicable_directions=applicable_directions,
        deadline=deadline,
    )

    successes = [r for r in results
                 if r.get("latency_ms") is not None and not r.get("error")]

    # 给子节点方向名 + 算先验
    child_dirs = [_direction_of_branch(r.get("branch_id")) for r in successes]
    priors = _compute_child_priors(child_dirs, applicable_directions, config)

    for r, name, p in zip(successes, child_dirs, priors):
        child = MCTSNode(
            code=r["code"],
            latency_ms=r["latency_ms"],
            hw_metrics=r.get("hw_metrics"),
            depth=node.depth + 1,
            parent=node,
            direction=name,
            P=p,
            result=r,
        )
        node.children.append(child)

    node.expanded = True

    # —— 终端判定（动态深度的核心）——
    max_depth = int(config.get("mcts_max_depth", 4))
    min_depth = int(config.get("mcts_min_depth", 2))
    strategy = str(config.get("mcts_depth_strategy", "adaptive"))
    stall_threshold = float(config.get("mcts_stall_threshold", math.log(1.05)))
    patience = int(config.get("mcts_stall_patience", 1))

    if not node.children:
        node.terminal = True
        node.terminal_reason = "dead_end"
    else:
        # 子节点深度 >= max_depth → 终端（硬上限）
        for c in node.children:
            if c.depth >= max_depth:
                c.terminal = True
                c.terminal_reason = "max_depth"
        # adaptive：父节点子节点全无改进（best reward < stall_threshold）且过 min_depth
        # → 把子节点标终端，阻止继续加深（dir_probe 结论 5 死区早停）。
        if strategy == "adaptive":
            best_child_reward = max(_reward(c.latency_ms, seed_latency) for c in node.children)
            if best_child_reward < stall_threshold and (node.depth + 1) >= min_depth:
                # patience：记录父节点 stall 次数（这里 node 只扩展一次，patience>=1 即触发）
                for c in node.children:
                    if not c.terminal:
                        c.terminal = True
                        c.terminal_reason = "stall_deadzone"

    return results


# ---------------------------------------------------------------------------
# Main entry: run_mcts
# ---------------------------------------------------------------------------

async def run_mcts(
    initial_candidate: dict,
    problem: Problem,
    config: dict,
    experiences: list[dict] | None = None,
    episode_id: int = 0,
    episode_output_dir: str | None = None,
    applicable_directions: list[dict] | None = None,
) -> dict:
    """MCTS 搜索 episode。返回与 run_search_episode 同 shape：
    {candidates, experiences, results, best_latency_ms}。

    rollout 循环：select 叶 → expand（unified_editor 采各方向）→ 对每个新子节点
    backprop 其 reward。champion = 全树最低延迟节点。
    """
    if experiences is None:
        experiences = []

    output_dir = episode_output_dir or config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    seed_latency = initial_candidate.get("latency_ms")
    root = MCTSNode(
        code=initial_candidate["code"],
        latency_ms=seed_latency,
        hw_metrics=initial_candidate.get("hw_metrics"),
        depth=0,
    )

    # —— 方向（动作集 + 先验来源）——
    use_dirs = bool(config.get("mcts_use_directions", True))
    if applicable_directions is None and use_dirs:
        applicable_directions = await determine_applicable_directions(
            problem, config, seed_latency, initial_candidate.get("hw_metrics"),
        )
        print(f"[mcts] {len(applicable_directions)} applicable directions: "
              f"{[d['name'] for d in applicable_directions]}")
    elif not use_dirs:
        applicable_directions = None

    # —— direction_store 持久化（TODO-1 记账，跨 run 累积）——
    op_type = _derive_op_type(problem, config) if applicable_directions else None
    stats_db = load_stats(config.get("direction_stats_path", "direction_stats.json")) if op_type else None
    if stats_db is not None and op_type:
        print(format_op_stats(get_op_stats(stats_db, op_type)))
    run_outcomes: dict = {}

    # —— 预算 ——
    max_rollouts = int(config.get("mcts_rollouts", config.get("iters", 15)))
    c_puct = float(config.get("mcts_cpuct", 1.0))
    # rollout 旋钮：扩展后是否继续下行扩展再回传。
    #   =1 → 子节点即终端，回传子节点 reward（快，每 rollout 1 次扩展）。
    #   =k → 扩展后沿 P-UCT 下行 k 层（每层各扩展一次），回传最深叶 reward
    #        （更贴近 MCTS 原型，但每 rollout k 次 LLM 扩展，贵）。默认 1。
    rollout_depth = max(1, int(config.get("mcts_rollout_depth", 1)))
    time_budget = float(config.get("search_time_budget", 0))
    search_start = time.time()
    deadline = (search_start + time_budget) if time_budget else 0.0

    # 深度旋钮：effective_max_depth = round(max_depth * scale)
    base_max_depth = int(config.get("mcts_max_depth", 4))
    scale = float(config.get("mcts_depth_scale", 1.0))
    eff_max_depth = max(1, round(base_max_depth * scale))
    config = dict(config)  # 副本写入 effective，供 _expand 读
    config["mcts_max_depth"] = eff_max_depth
    print(f"[mcts] rollouts={max_rollouts} rollout_depth={rollout_depth} c_puct={c_puct} "
          f"max_depth={base_max_depth}×{scale}={eff_max_depth} "
          f"strategy={config.get('mcts_depth_strategy','adaptive')}")

    all_results: list[dict] = []
    expansions = 0
    max_depth_reached = 0

    for rollout in range(max_rollouts):
        if deadline and time.time() > deadline:
            print(f"[mcts] ⏰ time budget exceeded at rollout {rollout+1}, early stop")
            break

        # 1) selection：选首个未扩展/非终端叶
        leaf, path = _select_leaf(root, c_puct)

        # 终端叶（死路/到顶/死区/已扩展）→ 回传自身 reward，不扩展
        if leaf.terminal or leaf.expanded:
            v = _reward(leaf.latency_ms, seed_latency)
            _backprop(path, v)
            print(f"  [mcts] rollout {rollout+1}: terminal@depth{leaf.depth} "
                  f"({leaf.terminal_reason or 'revisited'}) backprop r={v:+.3f}")
        else:
            # 2) expansion + 沿 P-UCT 下行 rollout_depth 层，回传最深可扩展叶的 reward。
            # rollout_depth=1：只扩展 leaf 一次，子节点即终端，回传子节点 reward（快）。
            # rollout_depth=k：扩展 leaf 后选其最优子继续扩展，重复 k 次（贴近 MCTS 原型，贵 k×）。
            terminal_leaf = leaf
            terminal_path = path
            for _step in range(rollout_depth):
                if deadline and time.time() > deadline:
                    break
                if terminal_leaf.terminal or terminal_leaf.expanded:
                    break  # 已到终端/死区，停止下行
                results = await _expand(
                    terminal_leaf, experiences, problem, config,
                    applicable_directions, deadline, seed_latency,
                )
                expansions += 1

                # experiences + direction_store 记账
                new_exp = _build_experiences(results, [{
                    "code": terminal_leaf.code, "latency_ms": terminal_leaf.latency_ms,
                }])
                experiences = update_experiences(
                    experiences, new_exp,
                    capacity=config.get("exp_n", 16),
                    topk=config.get("topk", 8),
                )
                if applicable_directions:
                    # record_iteration(results, new_candidates, initial_latency):
                    # results = 本次扩展全部 patch（成败都算 sampled）；new_candidates =
                    # 通过验证且存活进树的子节点（child_results，带 branch_id）。方向名从
                    # branch_id 解析（dir_5_algo_equiv → algo_equiv；free_explore 单列）。
                    child_results = [c.result for c in terminal_leaf.children if c.result]
                    if child_results:
                        record_iteration(run_outcomes, child_results, child_results, seed_latency)

                if not terminal_leaf.children:
                    # 死路：回传小惩罚，告知祖先此分支无效
                    _backprop(terminal_path, -0.2)
                    print(f"  [mcts] rollout {rollout+1}: expanded@depth{terminal_leaf.depth} → dead end")
                    break

                # backprop 每个新子节点的 reward（让祖先即时更新统计）
                for c in terminal_leaf.children:
                    _backprop(path + [c] if _step == 0 else terminal_path + [c],
                              _reward(c.latency_ms, seed_latency))
                    if c.depth > max_depth_reached:
                        max_depth_reached = c.depth

                # 下行：选最优子继续扩展（rollout_depth>1 时）
                if rollout_depth > 1:
                    nxt = _uct_select(terminal_leaf, c_puct)
                    if nxt is None or nxt.terminal:
                        break
                    terminal_path = terminal_path + [nxt]
                    terminal_leaf = nxt

            # 本 rollout 概览 log
            best_child = min(terminal_leaf.children, key=lambda c: c.latency_ms or float('inf')) \
                if terminal_leaf.children else None
            if best_child and best_child.latency_ms is not None:
                print(f"  [mcts] rollout {rollout+1}: expanded to depth{terminal_leaf.depth+1} "
                      f"({expansions} expansions total), best child "
                      f"{best_child.latency_ms:.4f}ms "
                      f"(r={_reward(best_child.latency_ms, seed_latency):+.3f})")

        # champion = 全树最优
        champion_node = root.best_node_in_subtree()
        best_lat = champion_node.latency_ms

        # checkpoint（兼容现有 schema：episode/iteration/search_mode/.../candidates）
        ckpt = {
            "episode": episode_id,
            "iteration": rollout + 1,
            "search_mode": "mcts",
            "num_plans": expansions,
            "num_results": expansions,
            "num_experiences": len(experiences),
            "tree": {
                "total_nodes": _count_nodes(root),
                "max_depth_reached": max_depth_reached,
                "expansions": expansions,
                "best_latency_ms": best_lat,
            },
            "candidates": [{
                "solution_path": initial_candidate.get("solution_path", ""),
                "latency_ms": best_lat,
                "hw_metrics": champion_node.hw_metrics,
            }],
        }
        all_results.append(ckpt)
        ckpt_path = Path(output_dir) / f"checkpoint_iter{rollout + 1}.json"
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2, default=str)

        # 防丢失兜底：每 rollout 末用当前 champion 写一份 final_results.json。
        # search_time_budget 早停或 shell timeout 在 rollout 间隙/之间命中时，
        # main.py 的 finally 可能来不及写 → 至少有这份最新 champion 落盘（01 题实例：
        # rc=124 时 final_results 未写，champion 3.677ms 白丢）。增量覆盖，跑完再被
        # main.py 的最终版覆盖（内容一致）。
        _ch = root.best_node_in_subtree()
        _interim_final = {
            "search_mode": "mcts",
            "iterations": all_results,
            "final_candidates": [{
                "code": _ch.code,
                "solution_path": initial_candidate.get("solution_path", ""),
                "latency_ms": _ch.latency_ms,
                "hw_metrics": _ch.hw_metrics,
                "branch_id": f"mcts_depth{_ch.depth}_{_ch.direction or 'seed'}",
            }],
        }
        _interim_path = Path(output_dir) / "final_results.json"
        try:
            with open(_interim_path, "w") as f:
                json.dump(_interim_final, f, indent=2, default=str)
        except Exception as _e:
            print(f"[mcts] WARN: interim final_results write failed: {_e!r}")

    # —— champion 提取（全树最低延迟，seed 永在树中）——
    champion_node = root.best_node_in_subtree()
    champion = {
        "code": champion_node.code,
        "solution_path": initial_candidate.get("solution_path", ""),
        "latency_ms": champion_node.latency_ms,
        "hw_metrics": champion_node.hw_metrics,
        "branch_id": f"mcts_depth{champion_node.depth}_{champion_node.direction or 'seed'}",
    }

    # direction_store 合并落盘
    if applicable_directions and run_outcomes and op_type:
        stats_db = merge_run(stats_db or {}, op_type, run_outcomes)
        save_stats(stats_db, config.get("direction_stats_path", "direction_stats.json"))
        print(f"[mcts] direction-stats saved for op_type={op_type} "
              f"({len(run_outcomes)} dirs)")

    # 树概览 log
    print(f"\n[mcts] tree: {_count_nodes(root)} nodes, "
          f"max_depth_reached={max_depth_reached}, expansions={expansions}")
    _print_tree_summary(root, seed_latency)

    return {
        "candidates": [champion],
        "experiences": experiences,
        "results": all_results,
        "best_latency_ms": champion_node.latency_ms,
    }


def _count_nodes(root: MCTSNode) -> int:
    n = 1
    for c in root.children:
        n += _count_nodes(c)
    return n


def _print_tree_summary(root: MCTSNode, seed_latency: float | None) -> None:
    """按深度打印每层节点数 + 该层最优 latency/reward，让动态深度可观测。"""
    layers: dict[int, list[MCTSNode]] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        layers.setdefault(n.depth, []).append(n)
        stack.extend(n.children)
    print("[mcts] depth | nodes | best_latency | best_reward(vs seed)")
    for d in sorted(layers):
        nodes = layers[d]
        lats = [n.latency_ms for n in nodes if n.latency_ms is not None]
        best_lat = min(lats) if lats else None
        best_r = _reward(best_lat, seed_latency) if best_lat else None
        bl = f"{best_lat:.4f}ms" if best_lat else "N/A"
        br = f"{best_r:+.3f}" if best_r is not None else "N/A"
        term = sum(1 for n in nodes if n.terminal)
        print(f"         {d} | {len(nodes):5d} | {bl:>11} | {br}  (terminal={term})")
