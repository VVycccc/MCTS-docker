"""direction_store.py — per-op_type 方向结局统计持久化层（TODO-1）。

方向模式（direction_organized_frontier）默认关闭，本模块仅在开启时被调用。
此为最小可用实现：按 op_type 分桶记录各方向结局，跨 run 累积到 JSON。

数据结构：
  direction_stats.json = {
    "<op_type>": {
      "<direction_name>": {
        "sampled": int,        # 采样次数
        "passed_validation": int,
        "survived_selection": int,
        "best_speedup_vs_seed": float | None,
      },
      ...
    }
  }
"""
import json
from pathlib import Path


def load_stats(path: str = "direction_stats.json") -> dict:
    """加载全局方向统计 DB，文件不存在返回空 dict。"""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def get_op_stats(stats_db: dict, op_type: str | None) -> dict:
    """返回某 op_type 的方向统计；op_type 为 None 或不存在返回空 dict。"""
    if not op_type or not stats_db:
        return {}
    return stats_db.get(op_type, {})


def format_op_stats(stats: dict) -> str:
    """格式化某 op_type 的方向统计为可读字符串（run 开始时 log 已累积统计）。"""
    if not stats:
        return ""
    lines = ["[direction-stats] accumulated outcomes for this op_type:"]
    for name, s in stats.items():
        best = s.get("best_speedup_vs_seed")
        best_str = f"{best:.2f}x" if best is not None else "N/A"
        lines.append(
            f"  {name}: sampled={s.get('sampled', 0)} "
            f"passed={s.get('passed_validation', 0)} "
            f"survived={s.get('survived_selection', 0)} "
            f"best_speedup={best_str}"
        )
    return "\n".join(lines)


def _branch_direction(branch_id: str | None) -> str | None:
    """从 branch_id（如 'dir_5_algo_equiv'）提取方向名；非方向分支返回 None。"""
    if not branch_id:
        return None
    if not branch_id.startswith("dir_"):
        return None
    # dir_<priority>_<name>
    parts = branch_id.split("_", 2)
    if len(parts) < 3:
        return None
    return parts[2]


def record_iteration(
    run_outcomes: dict,
    results: list[dict],
    new_candidates: list[dict],
    initial_latency: float | None,
) -> None:
    """聚样本轮各方向结局到 run_outcomes（in-place）。

    按 result 的 branch_id 分组，统计 sampled/passed/survived + best_speedup_vs_seed。
    """
    survived_branches = set()
    for c in new_candidates:
        bid = c.get("branch_id")
        if bid:
            survived_branches.add(bid)

    for r in results:
        bid = r.get("branch_id")
        direction = _branch_direction(bid)
        if not direction:
            continue
        entry = run_outcomes.setdefault(direction, {
            "sampled": 0, "passed_validation": 0,
            "survived_selection": 0, "best_speedup_vs_seed": None,
        })
        entry["sampled"] += 1
        lat = r.get("latency_ms")
        if lat is not None and lat > 0:
            entry["passed_validation"] += 1
            if initial_latency and lat > 0:
                sp = initial_latency / lat
                cur = entry["best_speedup_vs_seed"]
                if cur is None or sp > cur:
                    entry["best_speedup_vs_seed"] = sp
        if bid in survived_branches:
            entry["survived_selection"] += 1


def merge_run(stats_db: dict, op_type: str | None, run_outcomes: dict) -> dict:
    """把本次 run 的方向结局合并进全局 DB（跨 run 累积）。"""
    if not op_type or not run_outcomes:
        return stats_db
    op_bucket = stats_db.setdefault(op_type, {})
    for direction, run_entry in run_outcomes.items():
        cur = op_bucket.setdefault(direction, {
            "sampled": 0, "passed_validation": 0,
            "survived_selection": 0, "best_speedup_vs_seed": None,
        })
        cur["sampled"] += run_entry.get("sampled", 0)
        cur["passed_validation"] += run_entry.get("passed_validation", 0)
        cur["survived_selection"] += run_entry.get("survived_selection", 0)
        run_best = run_entry.get("best_speedup_vs_seed")
        if run_best is not None:
            cur_best = cur["best_speedup_vs_seed"]
            if cur_best is None or run_best > cur_best:
                cur["best_speedup_vs_seed"] = run_best
    return stats_db


def save_stats(stats_db: dict, path: str = "direction_stats.json") -> None:
    """保存全局方向统计 DB 到 JSON。"""
    Path(path).write_text(json.dumps(stats_db, indent=2, ensure_ascii=False))
