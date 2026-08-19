# Direction Probe Workflow

This is a Claude Code executable workflow for a lightweight kernel optimization direction-probing experiment. It is not a DirecTune search integration. Its purpose is to collect empirical evidence about optimization direction frequency and usability.

## Goal

For a representative subset of KernelBench tasks, run short free/guided direction probes and record:

1. which directions Claude Code naturally chooses;
2. which existing directions are applicable when assigned;
3. which directions compile, pass correctness, and improve performance;
4. which directions are over-used, under-used, or need more samples.

The output is a set of JSONL/CSV/markdown artifacts, not a modified DirecTune search policy.

## Do not modify

During this experiment, do not modify:

- `main.py`
- `agents.py`
- `search.py`
- `direction_store.py`
- `config.yaml`
- API/provider configuration files

Generated candidates and logs must stay under `output/direction_probe/...`.

## GPU constraint

All GPU commands must use:

```bash
CUDA_VISIBLE_DEVICES=0
```

## Direction taxonomy

Read the taxonomy from:

```text
experiments/direction_probe/direction_taxonomy.yaml
```

The initial directions are:

```text
tiling_blocking
parallelism_occupancy
memory_access
vectorization
reduction_strategy
fusion_or_fission
algorithmic_rewrite
data_layout_indexing
specialization_fast_path
precision_dtype
autotune_parameters
correctness_boundary_fix
other
```

Use `other` only when no existing direction fits, and explain why.

## Experiment scale

Pilot:

```text
tasks = 20
free_runs_per_task = 1
guided_runs_per_task = 1
steps_per_run = 3
expected_records ~= 120
```

Main:

```text
tasks = 60
free_runs_per_task = 1
guided_runs_per_task = 1
steps_per_run = 3
expected_records ~= 360
```

## Output layout

Each experiment writes:

```text
output/direction_probe/<run_id>/
  manifest.json
  selected_tasks.json
  run_schedule.json
  records.jsonl
  run_summaries.jsonl
  direction_stats.csv
  task_type_stats.csv
  failure_stats.csv
  direction_probe_report.md
  tasks/
    <task_id>/
      free_0/
        step_1/
          candidate.py
          diff.patch
          result.json
        step_2/
        step_3/
        summary.json
      guided_0/
        step_1/
        step_2/
        step_3/
        summary.json
```

## Free run

A free run lets Claude Code choose the direction.

For each step:

1. Read the current best candidate and the problem/reference.
2. Choose exactly one primary direction from the taxonomy.
3. Make one isolated optimization change.
4. Evaluate compile/correctness/benchmark.
5. Write one `records.jsonl` object.
6. Keep the candidate only if `speedup >= 1.05` and correctness passes; otherwise revert to previous best.

Free records are used to compute natural direction frequency.

## Guided run

A guided run assigns a direction to each step.

For each step:

1. Read `assigned_direction` from `run_schedule.json`.
2. Decide whether the direction is applicable.
3. If not applicable, write a skipped record with `attempted=false`, `applicability=not_applicable`, and `skip_reason`.
4. If applicable, make one isolated change under the assigned direction only.
5. Evaluate compile/correctness/benchmark.
6. Write one `records.jsonl` object.
7. Keep only improved candidates.

Guided records are used to compute applicability and effectiveness.

## Success definitions

Record success:

```text
A step produced one valid JSON object matching experiments/direction_probe/record_schema.md.
```

Optimization success:

```text
compile_pass == true
correctness_pass == true
benchmark_pass == true
speedup >= 1.05
```

Default keep rule:

```text
kept = improved
```

## Evaluation

Use the standalone evaluator:

```bash
cd /home/wangyichen/DirecTune
CUDA_VISIBLE_DEVICES=0 python experiments/direction_probe/evaluate_candidate.py \
  --problem problems/kb_level1/01_square_matrix_multiplication.json \
  --candidate output/direction_probe/<run_id>/tasks/<task_id>/free_0/step_1/candidate.py \
  --baseline problems/kb_level1/01_square_matrix_multiplication_initial.py \
  --out output/direction_probe/<run_id>/tasks/<task_id>/free_0/step_1/result.json
```

The evaluator reuses `triton_backend.py` for anti-PyTorch checks and profiling.

## Aggregation

After collecting records:

```bash
python experiments/direction_probe/aggregate_stats.py \
  --records output/direction_probe/<run_id>/records.jsonl \
  --out-dir output/direction_probe/<run_id>

python experiments/direction_probe/generate_report.py \
  --run-dir output/direction_probe/<run_id>
```

The report should include:

- natural frequency table from free records;
- guided applicability table;
- guided effectiveness table;
- failure distribution;
- task-type summary;
- `other` direction cases;
- recommended labels such as `high_priority`, `underused_but_effective`, or `needs_more_samples`.

## Pilot checklist

After the 20-task pilot, check:

- valid JSON rate >= 95%;
- guided direction-following rate >= 80% by sample inspection;
- `other` rate <= 10%;
- step 3 still provides useful records;
- no API keys or secrets appear in artifacts.

Only then scale to the main sample.
