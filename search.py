"""Beam search utilities — candidate selection and experience memory.

The main beam search loop lives in ``main.py:run_search_episode()``.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_candidates(
    executor_results: list[dict],
    k: int = 2,
) -> list[dict]:
    """Select top-k candidates by latency.

    For each (baseline, branch) group, picks the best representative,
    then returns the overall top-k by latency.

    The returned candidate records keep the core fields needed by later
    planner/summarizer iterations, including optional hardware metrics.
    """
    # Group by (baseline, branch_id)
    groups: dict[tuple, list[dict]] = {}
    for r in executor_results:
        if r.get("error") or r.get("latency_ms") is None:
            continue
        key = (r.get("baseline_code", ""), r.get("branch_id", ""))
        groups.setdefault(key, []).append(r)

    # Best per group
    best_per_group = []
    for items in groups.values():
        best = min(items, key=lambda x: x["latency_ms"])
        best_per_group.append({
            "code": best.get("code", ""),
            "solution_path": best.get("solution_path", ""),
            "latency_ms": best.get("latency_ms"),
            "speedup": best.get("speedup"),
            "hw_metrics": best.get("hw_metrics"),
            "compiled": best.get("compiled"),
            "correct": best.get("correct"),
            "runnable": best.get("runnable"),
            "plan_id": best.get("plan_id"),
            "branch_id": best.get("branch_id"),
            "baseline_code": best.get("baseline_code", ""),
        })

    # Top-k overall
    best_per_group.sort(key=lambda x: x["latency_ms"])
    return best_per_group[:k]


def select_candidates_by_direction(
    executor_results: list[dict],
    direction_max_width: int = 3,
    free_explore_label: str = "free_explore",
) -> list[dict]:
    """Select best patch per optimization direction (方向 0).

    Unlike ``select_candidates`` (pure latency top-k), this preserves semantic
    diversity: each returned candidate is the best patch for a *distinct*
    optimization direction, identified by ``branch_id`` (set to a direction
    label like ``dir_5_algo`` by the direction-aware ``unified_editor``).

    Directions are capped by **classifier priority** (first-seen order = the
    order ``unified_editor`` sampled them, which is the classifier's payoff
    ranking), NOT by latency — so a high-payoff-but-currently-slow direction
    (e.g. ⑤ algo-equiv, which may need multiple iterations to unlock) is not
    dropped just because its first patch is slow.

    The ``free_explore`` branch (unconstrained sampling) is exempt from the
    cap: if it produced a valid patch it is always kept as the escape hatch
    for cross-direction optimizations the classifier missed.

    Returns the same dict shape as ``select_candidates``, so the caller
    (main.py carry-forward + champion logic) is unchanged.
    """
    # Group by branch_id (direction label), preserving first-seen order.
    # First-seen order == classifier priority order, because unified_editor
    # samples directions in the order returned by determine_applicable_directions.
    groups: dict[str, list[dict]] = {}
    for r in executor_results:
        if r.get("error") or r.get("latency_ms") is None:
            continue
        label = r.get("branch_id", "")
        groups.setdefault(label, []).append(r)

    # Best latency per direction.
    best_per_direction: list[dict] = []
    for label, items in groups.items():
        best = min(items, key=lambda x: x["latency_ms"])
        best_per_direction.append({
            "code": best.get("code", ""),
            "solution_path": best.get("solution_path", ""),
            "latency_ms": best.get("latency_ms"),
            "speedup": best.get("speedup"),
            "hw_metrics": best.get("hw_metrics"),
            "compiled": best.get("compiled"),
            "correct": best.get("correct"),
            "runnable": best.get("runnable"),
            "plan_id": best.get("plan_id"),
            "branch_id": best.get("branch_id", ""),
            "baseline_code": best.get("baseline_code", ""),
        })

    # Separate free_explore (exempt from cap) from real directions.
    free = [c for c in best_per_direction if c["branch_id"] == free_explore_label]
    real = [c for c in best_per_direction if c["branch_id"] != free_explore_label]

    # Cap real directions by classifier priority (first-seen order), not latency.
    capped = real[:direction_max_width]
    return capped + free


# ---------------------------------------------------------------------------
# Experience memory
# ---------------------------------------------------------------------------

def update_experiences(
    old: list[dict],
    new: list[dict],
    capacity: int = 16,
    topk: int = 8,
) -> list[dict]:
    """Capped queue: combine old top-K experiences with new, trim to capacity."""
    # Select top-k positive + negative from new
    positive = [e for e in new if e.get("speedup", 0) > 1.0]
    negative = [e for e in new if e.get("speedup", 0) < 1.0]

    positive.sort(key=lambda x: x.get("speedup", 0), reverse=True)
    negative.sort(key=lambda x: x.get("speedup", 0))

    half = topk // 2
    selected = positive[:half] + negative[:half]

    # Combine with old, trim to capacity
    combined = old + selected
    if len(combined) > capacity:
        # Keep most recent (new items at the end)
        combined = combined[-capacity:]

    return combined

