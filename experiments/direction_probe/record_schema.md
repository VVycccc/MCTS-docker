# Direction Probe Record Schema

This schema is for the standalone kernel optimization direction-probing experiment. It is intentionally independent of DirecTune search internals.

## Two meanings of success

- **Record success**: a probe step completed and produced one valid JSON object matching this schema. Compile/correctness/performance may still fail.
- **Optimization success**: `compile_pass && correctness_pass && benchmark_pass && speedup >= 1.05`.

All attempted and skipped guided steps should be logged. Failed attempts are useful evidence for direction usability.

## `records.jsonl`

One JSON object per probe step.

Required fields:

```json
{
  "schema_version": "kernel_direction_probe_v1",
  "task_id": "kb_001",
  "task_type": "reduction",
  "run_id": "kb_001_guided_0",
  "step": 1,
  "selection_mode": "guided",
  "assigned_direction": "reduction_strategy",
  "direction": "reduction_strategy",
  "secondary_directions": ["autotune_parameters"],
  "applicability": "applicable",
  "attempted": true,
  "skip_reason": null,
  "direction_reason": "The task is dominated by a long reduction axis.",
  "change_summary": "Changed reduction tiling and adjusted num_warps.",
  "changed_items": ["Increased reduction block size", "Changed num_warps from 4 to 8"],
  "compile_pass": true,
  "correctness_pass": true,
  "benchmark_pass": true,
  "latency_before_ms": 0.123,
  "latency_after_ms": 0.098,
  "speedup": 1.255,
  "improved": true,
  "kept": true,
  "failure_type": null,
  "artifact": {
    "kernel_file": "output/direction_probe/.../candidate.py",
    "diff_file": "output/direction_probe/.../diff.patch",
    "benchmark_log": "output/direction_probe/.../benchmark.json",
    "correctness_log": "output/direction_probe/.../correctness.log"
  },
  "notes": ""
}
```

### Field rules

- `selection_mode`: `free` or `guided`.
- `assigned_direction`: `null` for free mode; one taxonomy direction for guided mode.
- `direction`: actual attempted direction. In guided mode this should normally equal `assigned_direction`.
- `applicability`: `applicable`, `not_applicable`, or `unknown`.
- `attempted`: `false` only for guided steps that were judged not applicable or for an agent failure before code generation.
- `improved`: true only when speedup passes the configured threshold, default 1.05.
- `kept`: true only when the candidate should become the starting point for the next step. Default rule: `kept == improved`.

### Failure types

Use one of:

```text
not_applicable
compile_error
runtime_error
correctness_mismatch
benchmark_error
performance_regression
no_measurable_improvement
timeout
invalid_output
agent_failed
other
```

Suggested mapping:

- `compile_pass == false` → `compile_error`
- `compile_pass == true && correctness_pass == false` → `correctness_mismatch`
- `correctness_pass == true && benchmark_pass == false` → `benchmark_error`
- `correctness_pass == true && benchmark_pass == true && speedup < 1.0` → `performance_regression`
- `correctness_pass == true && benchmark_pass == true && 1.0 <= speedup < 1.05` → `no_measurable_improvement`

## `run_summaries.jsonl`

One JSON object per free/guided run.

```json
{
  "schema_version": "kernel_direction_probe_run_summary_v1",
  "task_id": "kb_001",
  "task_type": "reduction",
  "run_id": "kb_001_free_0",
  "selection_mode": "free",
  "num_steps": 3,
  "num_attempted": 3,
  "num_applicable": 3,
  "num_compile_pass": 2,
  "num_correctness_pass": 2,
  "num_improved": 1,
  "directions_tried": ["reduction_strategy", "memory_access", "autotune_parameters"],
  "initial_latency_ms": 0.123,
  "best_latency_ms": 0.098,
  "best_speedup": 1.255,
  "best_step": 1,
  "best_direction": "reduction_strategy",
  "final_kernel_file": "output/direction_probe/.../best_kernel.py",
  "status": "completed"
}
```
