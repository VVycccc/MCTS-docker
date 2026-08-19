# Direction Probe Experiment

Standalone workflow for probing kernel optimization directions with Claude Code.

This experiment is intentionally separate from DirecTune search. It uses KernelBench tasks and DirecTune validation/benchmark utilities to collect direction frequency and usability evidence.

## Files

- `direction_taxonomy.yaml` — direction labels and definitions.
- `record_schema.md` — JSONL schema for step and run records.
- `select_tasks.py` — select representative tasks from `_batch.csv` files.
- `schedule_runs.py` — create free/guided run schedules.
- `evaluate_candidate.py` — evaluate one generated candidate kernel.
- `aggregate_stats.py` — aggregate `records.jsonl` into CSVs.
- `generate_report.py` — generate a markdown report from aggregated CSVs.

Main workflow document:

```text
docs/direction_probe_workflow.md
```

## Default pilot

```bash
cd /home/wangyichen/DirecTune

python experiments/direction_probe/select_tasks.py \
  --csv problems/kb_level1/_batch.csv \
  --limit 20 \
  --out output/direction_probe/pilot/selected_tasks.json

python experiments/direction_probe/schedule_runs.py \
  --tasks output/direction_probe/pilot/selected_tasks.json \
  --taxonomy experiments/direction_probe/direction_taxonomy.yaml \
  --steps 3 \
  --out output/direction_probe/pilot/run_schedule.json
```

Candidate generation is performed by Claude Code following `docs/direction_probe_workflow.md`.

## Evaluate one candidate

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/direction_probe/evaluate_candidate.py \
  --problem problems/kb_level1/01_square_matrix_multiplication.json \
  --candidate output/direction_probe/pilot/tasks/01/free_0/step_1/candidate.py \
  --baseline problems/kb_level1/01_square_matrix_multiplication_initial.py \
  --out output/direction_probe/pilot/tasks/01/free_0/step_1/result.json
```

## Aggregate and report

```bash
python experiments/direction_probe/aggregate_stats.py \
  --records output/direction_probe/pilot/records.jsonl \
  --out-dir output/direction_probe/pilot

python experiments/direction_probe/generate_report.py \
  --run-dir output/direction_probe/pilot
```

## Constraints

- Use `CUDA_VISIBLE_DEVICES=0` for GPU evaluation.
- Do not write secrets or API keys into artifacts.
- Do not modify DirecTune main search files for this experiment.
