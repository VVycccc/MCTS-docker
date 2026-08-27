#!/usr/bin/env python3
"""Summarize the thinking on/patch/alloff A/B runs under output/ab_thinking/.

Per arm x problem: champion speedup, patch/rewrite pass rates, expansions,
per-stage tokens, wall time. Quality gate for "thinking off 是否掉质量" and
"rewrite 是否可继续关".
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/wangyichen/DirecTune-MCTS/output/ab_thinking")
ARMS = ["on", "patch", "alloff"]


def parse_run(d: Path) -> dict | None:
    fr = d / "final_results.json"
    log = d / "run.log"
    if not fr.exists() or not log.exists():
        return None
    try:
        res = json.loads(fr.read_text())
    except json.JSONDecodeError:
        return None
    txt = log.read_text(errors="replace")

    r = {
        "speedup_vs_seed": res.get("speedup_vs_seed"),
        "seed_latency_ms": res.get("seed_latency_ms"),
        "champion_latency_ms": res.get("champion_latency_ms"),
        "inc_ok": len(re.findall(r"✓ incremental", txt)),
        "rw_ok": len(re.findall(r"✓ full_rewrite", txt)),
        "fail": len(re.findall(r"✗ failed", txt)),
        "expansions": None,
        "stages": {},
    }
    m = re.search(r"expansions=(\d+)", txt)
    if m:
        r["expansions"] = int(m.group(1))
    for stage in ["classifier", "patch", "rewrite", "seed"]:
        m = re.search(rf"\[{stage}\s*\] prompt=(\d+) completion=(\d+) calls=(\d+)", txt)
        if m:
            r["stages"][stage] = tuple(int(x) for x in m.groups())
    # wall: seed debug meta.json (first write) → final_results mtime
    meta = d / "naive_seed_debug" / "meta.json"
    if meta.exists():
        r["wall_s"] = int(fr.stat().st_mtime - meta.stat().st_mtime)
    return r


def main():
    print(f"{'arm':7s} {'problem':38s} {'spd':>7s} {'inc':>4s} {'rw':>4s} {'fail':>4s} "
          f"{'exp':>4s} {'wall_s':>7s} {'tok_p':>8s} {'tok_r':>8s} {'tok_s':>7s} {'tok_c':>7s} calls")
    for arm in ARMS:
        arm_dir = ROOT / arm
        if not arm_dir.exists():
            continue
        for d in sorted(arm_dir.iterdir()):
            if not d.is_dir():
                continue
            r = parse_run(d)
            if r is None:
                print(f"{arm:7s} {d.name:38s} (incomplete)")
                continue
            s = r["stages"]
            def tok(k, i=1):
                return s.get(k, (0, 0, 0))[i]
            spd = r["speedup_vs_seed"]
            spd_s = f"{spd:.2f}x" if spd else "N/A"
            print(f"{arm:7s} {d.name:38s} {spd_s:>7s} {r['inc_ok']:>4d} {r['rw_ok']:>4d} "
                  f"{r['fail']:>4d} {str(r['expansions']):>4s} {str(r.get('wall_s','?')):>7s} "
                  f"{tok('patch'):>8d} {tok('rewrite'):>8d} {tok('seed'):>7d} {tok('classifier'):>7d} "
                  f"{sum(s.get(k,(0,0,0))[2] for k in ['classifier','patch','rewrite','seed'])}")

    # per-arm aggregates
    print("\n=== arm aggregates ===")
    for arm in ARMS:
        arm_dir = ROOT / arm
        if not arm_dir.exists():
            continue
        rows = [r for r in (parse_run(d) for d in sorted(arm_dir.iterdir()) if d.is_dir()) if r]
        if not rows:
            continue
        spds = [r["speedup_vs_seed"] for r in rows if r["speedup_vs_seed"]]
        tok = sum(sum(v[1] for v in r["stages"].values()) for r in rows)
        wall = sum(r.get("wall_s", 0) for r in rows)
        inc = sum(r["inc_ok"] for r in rows); rw = sum(r["rw_ok"] for r in rows); fail = sum(r["fail"] for r in rows)
        geo = 1.0
        for s in spds:
            geo *= s
        geo = geo ** (1.0 / len(spds)) if spds else 0
        print(f"{arm:7s} n={len(rows)} geomean_spd={geo:.2f}x  pass: inc={inc} rw={rw} fail={fail} "
              f"({(inc+rw)}/{inc+rw+fail})  completion_tokens={tok}  wall={wall}s")


if __name__ == "__main__":
    sys.exit(main())
