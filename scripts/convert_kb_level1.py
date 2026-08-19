"""Convert KernelBench Level 1 definitions to DirecTune Problem format.

Reads the original flashinfer-bench JSON files and produces:
  1. problems/kb_level1/<name>.json  — simplified Problem with concrete shapes
  2. problems/kb_level1/<name>_initial.py — Python reference as initial solution
  3. problems/kb_level1/_batch.csv — index for batch experiments

Usage:
    python scripts/convert_kb_level1.py \
        --kb-defs ../AccelOpt/experiments/kernelbench/definitions/ \
        --kb-sols ../AccelOpt/experiments/kernelbench/solutions/ \
        --kb-workloads ../AccelOpt/experiments/kernelbench/workloads/ \
        --out problems/kb_level1/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def resolve_shape(shape: list | None, axes: dict) -> list[int] | None:
    """Replace axis names with their const values. Returns None for scalars/unresolved."""
    if shape is None:
        return None
    resolved = []
    for dim in shape:
        if isinstance(dim, str):
            axis = axes[dim]
            resolved.append(axis["value"])
        else:
            resolved.append(dim)
    return resolved


def infer_input_shapes_from_get_inputs(source: str, timeout_seconds: int = 60) -> list[list[int] | None] | None:
    """Execute get_inputs() in a subprocess and return tensor shapes, keeping scalars as None."""
    if "def get_inputs" not in source:
        return None
    runner = r'''
import json
import runpy
import sys
import torch

ns = runpy.run_path(sys.argv[1])
get_inputs = ns.get("get_inputs")
if get_inputs is None:
    print(json.dumps({"ok": False}))
    raise SystemExit(0)
torch.manual_seed(42)
with torch.no_grad():
    values = get_inputs()
if not isinstance(values, (list, tuple)):
    values = [values]
shapes = []
for value in values:
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        shapes.append(list(value.shape))
    else:
        shapes.append(None)
print(json.dumps({"ok": True, "shapes": shapes}))
'''
    with tempfile.TemporaryDirectory(prefix="DirecTune_convert_l1_") as tmpdir:
        tmp = Path(tmpdir)
        src_path = tmp / "source.py"
        runner_path = tmp / "infer_shapes.py"
        src_path.write_text(source)
        runner_path.write_text(runner)
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(src_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if proc.returncode != 0:
                return None
            lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if not lines:
                return None
            payload = json.loads(lines[-1])
            if payload.get("ok") and isinstance(payload.get("shapes"), list):
                return payload["shapes"]
        except Exception:
            return None
    return None


def convert_one(definition_path: str, solution_path: str, workload_path: str, out_dir: str) -> str | None:
    """Convert a single KernelBench problem to DirecTune format. Returns problem name."""
    with open(definition_path) as f:
        defn = json.load(f)

    name = defn["name"]
    axes = defn["axes"]

    # Resolve input shapes
    inputs = []
    for inp_name, inp_spec in defn["inputs"].items():
        shape = resolve_shape(inp_spec["shape"], axes)
        inputs.append({
            "name": inp_name,
            "shape": shape,
            "dtype": inp_spec["dtype"],
        })

    # Resolve output shapes
    outputs = []
    for out_name, out_spec in defn["outputs"].items():
        shape = resolve_shape(out_spec["shape"], axes)
        outputs.append({
            "name": out_name,
            "shape": shape,
            "dtype": out_spec["dtype"],
        })

    # Reference code from definition
    reference = defn.get("reference", "")

    # Some KernelBench definitions use shape:null for tensor inputs whose true
    # shape is only materialized by get_inputs().  Keep real scalars as null.
    if any(inp.get("shape") is None for inp in inputs):
        inferred_shapes = infer_input_shapes_from_get_inputs(reference)
        if inferred_shapes:
            for i, inferred in enumerate(inferred_shapes):
                if i < len(inputs) and inputs[i].get("shape") is None and inferred is not None:
                    inputs[i]["shape"] = inferred

    # Write problem JSON
    problem = {
        "name": name,
        "op_type": defn.get("op_type", ""),
        "inputs": inputs,
        "outputs": outputs,
        "reference": reference,
    }

    os.makedirs(out_dir, exist_ok=True)
    problem_path = os.path.join(out_dir, f"{name}.json")
    with open(problem_path, "w") as f:
        json.dump(problem, f, indent=2, ensure_ascii=False)

    # Extract and write initial solution from the python_reference
    with open(solution_path) as f:
        sol = json.load(f)

    for src in sol.get("sources", []):
        if src.get("path") == "main.py":
            code = src["content"]
            init_path = os.path.join(out_dir, f"{name}_initial.py")
            with open(init_path, "w") as f:
                f.write(code)
            break

    return name


def main():
    parser = argparse.ArgumentParser(description="Convert KernelBench Level 1 to DirecTune format")
    parser.add_argument("--kb-defs", type=str, required=True, help="Path to KernelBench definitions/ directory")
    parser.add_argument("--kb-sols", type=str, required=True, help="Path to KernelBench solutions/ directory")
    parser.add_argument("--kb-workloads", type=str, default="", help="Path to KernelBench workloads/ directory (optional)")
    parser.add_argument("--out", type=str, default="problems/kb_level1/", help="Output directory")
    parser.add_argument("--csv", type=str, default="", help="Path to profile_results.csv (optional, otherwise scan definitions/)")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    batch_entries = []

    if args.csv and os.path.exists(args.csv):
        # Read from CSV
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        # Scan definitions directory
        rows = []
        def_root = Path(args.kb_defs)
        for def_file in sorted(def_root.rglob("*.json")):
            rel = def_file.relative_to(def_root)
            op_type = rel.parent.name
            name = def_file.stem

            sol_file = Path(args.kb_sols) / op_type / name / "python_reference.json"
            wl_files = sorted(Path(args.kb_workloads).glob(f"{op_type}/{name}_*.json")) if args.kb_workloads else []

            rows.append({
                "definition_path": str(def_file),
                "solution_path": str(sol_file),
                "workload_path": str(wl_files[0]) if wl_files else "",
            })

    print(f"Found {len(rows)} problems to convert")

    converted = 0
    for row in rows:
        def_path = row["definition_path"]
        sol_path = row["solution_path"]

        if not os.path.exists(def_path):
            print(f"  SKIP {def_path}: definition not found")
            continue
        if not os.path.exists(sol_path):
            print(f"  SKIP {sol_path}: solution not found")
            continue

        try:
            name = convert_one(def_path, sol_path, row.get("workload_path", ""), out_dir)
            if name:
                batch_entries.append({
                    "name": name,
                    "problem": f"problems/kb_level1/{name}.json",
                    "initial_solution": f"problems/kb_level1/{name}_initial.py",
                })
                converted += 1
        except Exception as e:
            print(f"  ERROR converting {def_path}: {e}")

    # Write batch index
    batch_csv = os.path.join(out_dir, "_batch.csv")
    with open(batch_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "problem", "initial_solution"])
        writer.writeheader()
        writer.writerows(batch_entries)

    print(f"\nConverted {converted}/{len(rows)} problems")
    print(f"Output: {out_dir}")
    print(f"Batch index: {batch_csv}")


if __name__ == "__main__":
    main()
