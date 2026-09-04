#!/usr/bin/env python3
"""Export the champion kernel of every run into output/champions_export/.

Champion code lives in the `final_candidates[0].code` field of every run's
final_results.json. This script scans all layouts the pipeline has produced:

    output/<run>/final_results.json                        (single-problem run)
    output/<run>/mcts/final_results.json                   (trace runs: problem from config)
    output/<run>/<problem>/final_results.json              (per-problem sweep)
    output/<run>/<problem>_tryN/final_results.json         (re-runs)
    output/<run>/<variant>/<problem>/final_results.json    (ablations)

and writes `<run>__<problem>.py` (byte-identical champion code, never
rewritten once exported) plus `_manifest.{json,csv}`.

No config/API-key material is ever copied — only the champion code and
latency/speedup metadata.

Usage:
    python3 scripts/export_champions.py [--dry-run] [--output-dir output]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CSV_FIELDS = [
    "run", "problem", "latency_ms", "speedup_vs_pytorch", "speedup_vs_seed",
    "status", "strict_triton", "exported_file",
]
JSON_ONLY_FIELDS = ["solution_path", "source_json"]


def problem_name(run_dir: Path, fr_path: Path, data: dict) -> str | None:
    """Derive the problem identifier for an exported champion filename."""
    rel_dir = fr_path.parent.relative_to(run_dir)
    cfg = data.get("config") or {}
    cfg_problem = str(cfg.get("problem") or "").strip()
    cfg_base = cfg_problem.rsplit("/", 1)[-1].removesuffix(".json") if cfg_problem else ""
    if not rel_dir.parts:
        # root-level single-problem run: keep the historical run-name convention
        return run_dir.name
    if rel_dir == Path("mcts"):
        # trace run: problem lives in config
        return cfg_base or None
    if len(rel_dir.parts) == 1 and cfg_base == rel_dir.parts[0]:
        return rel_dir.parts[0]
    # nested (ablation variants, _tryN re-runs): keep full subpath for uniqueness
    return "_".join(rel_dir.parts)


def collect(output_dir: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    """Return (manifest_entries, skipped reasons) for every champion source."""
    entries: list[dict] = []
    skipped: list[tuple[str, str]] = []
    export_dir = output_dir / "champions_export"

    for fr_path in sorted(output_dir.glob("**/final_results.json")):
        if "champions_export" in fr_path.parts:
            continue
        rel = fr_path.relative_to(output_dir)
        run = rel.parts[0]
        run_dir = output_dir / run
        try:
            data = json.loads(fr_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append((str(rel), f"unreadable: {exc}"))
            continue

        problem = problem_name(run_dir, fr_path, data)
        if not problem:
            skipped.append((str(rel), "cannot derive problem name"))
            continue

        cands = data.get("final_candidates") or []
        code = cands[0].get("code") if cands else None
        if not code or not str(code).strip():
            skipped.append((str(rel), "no champion code in final_candidates"))
            continue

        exported = export_dir / f"{run}__{problem}.py"
        entry = {
            "run": run,
            "problem": problem,
            "latency_ms": cands[0].get("latency_ms"),
            "speedup_vs_pytorch": data.get("speedup_vs_pytorch", ""),
            "speedup_vs_seed": data.get("speedup_vs_seed", ""),
            "status": data.get("status", ""),
            "strict_triton": data.get("strict_triton", ""),
            "solution_path": cands[0].get("solution_path", ""),
            "source_json": str(rel),
            "exported_file": str(exported),
        }
        entries.append(entry)
    return entries, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report planned exports without writing anything")
    args = ap.parse_args()

    export_dir = args.output_dir / "champions_export"
    entries, skipped = collect(args.output_dir)

    new = [e for e in entries if not Path(e["exported_file"]).exists()]
    print(f"sources with champion code : {len(entries)}")
    print(f"already exported           : {len(entries) - len(new)}")
    print(f"to export now              : {len(new)}")
    print(f"skipped sources            : {len(skipped)}")
    for rel, why in skipped:
        print(f"  SKIP {rel}: {why}")
    for e in new:
        print(f"  NEW  {e['run']}__{e['problem']}.py  (latency {e['latency_ms']} ms)")

    if args.dry_run:
        return 0

    export_dir.mkdir(parents=True, exist_ok=True)
    for e in new:
        data = json.loads((args.output_dir / e["source_json"]).read_text())
        code = data["final_candidates"][0]["code"]
        Path(e["exported_file"]).write_text(code if code.endswith("\n") else code + "\n")
        print(f"wrote {e['exported_file']}")

    entries.sort(key=lambda e: (e["run"], e["problem"]))
    manifest_json = export_dir / "_manifest.json"
    manifest_csv = export_dir / "_manifest.csv"
    manifest_json.write_text(json.dumps(entries, indent=1) + "\n")
    with manifest_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    print(f"manifest: {manifest_json} / {manifest_csv} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
