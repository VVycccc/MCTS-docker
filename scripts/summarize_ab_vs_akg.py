#!/usr/bin/env python3
"""汇总等资源对照试验（MCTS vs AKG）：成功率 / geomean / 资源审计 / 配对对比。

用法：
  python scripts/summarize_ab_vs_akg.py --out output/ab_vs_akg
输出：stdout 表格 + <out>/summary.md
"""
import argparse
import json
import math
from pathlib import Path


def load_arm(out: Path, arm: str):
    rows = []
    for d in sorted((out / arm).glob("*_r*")):
        f = d / "final_results.json" if arm == "mcts" else d / "result.json"
        if not f.exists():
            rows.append({"problem": d.name, "done": False})
            continue
        d_ = json.loads(f.read_text())
        ru = d_.get("resource_usage") or d_.get("usage_total") or {}
        row = {
            "problem": d.name, "done": True,
            "passed": d_.get("passed", d_.get("status") == "success"
                             and d_.get("strict_triton", True)),
            # 统一口径：优先 isolated 复测值（mcts: champion_latency_isolated；
            # akg: harness.latency_ms 本就是 profile_isolated 产物）
            "latency_ms": (d_.get("champion_latency_isolated")
                           or d_.get("harness", {}).get("latency_ms")
                           or d_.get("champion_latency_ms")),
            "baseline_ms": d_.get("baseline_latency_ms"),
            "speedup": ((d_.get("baseline_latency_ms") / _lat)
                        if (d_.get("baseline_latency_ms") and
                            (_lat := (d_.get("champion_latency_isolated")
                                      or d_.get("harness", {}).get("latency_ms")
                                      or d_.get("champion_latency_ms"))))
                        else d_.get("speedup_vs_pytorch")),
            "calls": ru.get("llm_calls") or ru.get("calls"),
            "tokens": ru.get("total_tokens")
                      or (ru.get("prompt_tokens", 0) + ru.get("completion_tokens", 0)),
            "wall_s": d_.get("wall_seconds"),
            "iters": d_.get("iterations"),
        }
        rows.append(row)
    return rows


def geomean(vals):
    v = [x for x in vals if x]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ab_vs_akg")
    args = ap.parse_args()
    out = Path(args.out)

    data = {arm: load_arm(out, arm) for arm in ("mcts", "akg")}
    lines = ["# 等资源对照：DirecTune-MCTS vs AKG（best-of-N）", ""]

    # per-problem paired table
    lines += ["| 题目 | mcts passed | mcts ms | mcts speedup | akg passed | akg ms | "
              "akg speedup | 胜者 |", "|---|---|---|---|---|---|---|---|"]
    m_by_p = {r["problem"].rsplit("_r", 1)[0]: r for r in data["mcts"] if r.get("done")}
    a_by_p = {r["problem"].rsplit("_r", 1)[0]: r for r in data["akg"] if r.get("done")}
    for p in sorted(set(m_by_p) | set(a_by_p)):
        m, a = m_by_p.get(p, {}), a_by_p.get(p, {})
        winner = ""
        if m.get("latency_ms") and a.get("latency_ms"):
            winner = "mcts" if m["latency_ms"] < a["latency_ms"] else "akg"
        elif m.get("latency_ms"):
            winner = "mcts (akg 未通过)"
        elif a.get("latency_ms"):
            winner = "akg (mcts 未通过)"
        lines.append(f"| {p} | {fmt(m.get('passed', False))} | {fmt(m.get('latency_ms'))} "
                     f"| {fmt(m.get('speedup'), 2)} | {fmt(a.get('passed', False))} "
                     f"| {fmt(a.get('latency_ms'))} | {fmt(a.get('speedup'), 2)} | {winner} |")

    # aggregates
    lines += ["", "## 汇总", ""]
    for arm in ("mcts", "akg"):
        rows = [r for r in data[arm] if r.get("done")]
        n = len(rows)
        passed = sum(1 for r in rows if r.get("passed"))
        sp = geomean([r["speedup"] for r in rows if r.get("speedup")])
        calls = [r["calls"] for r in rows if r.get("calls")]
        toks = [r["tokens"] for r in rows if r.get("tokens")]
        lines.append(
            f"- **{arm}**: 完成 {n}，通过 {passed}（{100*passed/n if n else 0:.0f}%），"
            f"geomean speedup_vs_pytorch = {fmt(sp, 3)}，"
            f"calls mean = {fmt(sum(calls)/len(calls), 1) if calls else '—'}，"
            f"tokens mean = {fmt(sum(toks)/len(toks)/1000, 1) if toks else '—'}K")
    lines += ["", "## 资源对齐审计", "",
              "AKG 臂预算来自同题 MCTS 臂实际消耗；比值 ≈1 为对齐。", "",
              "| 题目 | mcts calls | akg calls | 比值 | mcts tokens | akg tokens | 比值 |",
              "|---|---|---|---|---|---|---|"]
    for p in sorted(set(m_by_p) & set(a_by_p)):
        m, a = m_by_p[p], a_by_p[p]
        mc, ac = m.get("calls"), a.get("calls")
        mt, at = m.get("tokens"), a.get("tokens")
        cr = fmt(ac / mc, 2) if mc and ac else "—"
        tr = fmt(at / mt, 2) if mt and at else "—"
        lines.append(f"| {p} | {fmt(mc, 0)} | {fmt(ac, 0)} | {cr} "
                     f"| {fmt(mt, 0)} | {fmt(at, 0)} | {tr} |")

    report = "\n".join(lines)
    print(report)
    (out / "summary.md").write_text(report + "\n")
    print(f"\nsaved → {out/'summary.md'}")


if __name__ == "__main__":
    main()
