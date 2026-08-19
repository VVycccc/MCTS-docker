# DirecTune

DirecTune is a research prototype for generating and optimizing Triton GPU kernels from PyTorch reference implementations. The current code path is a unified single-path pipeline:

```text
PyTorch reference / initial solution
  → baseline profiling
  → one seed Triton kernel generation
  → DirecTune Search
  → fastest verified candidate
```

The core idea is to separate **getting a correct kernel** from **systematically optimizing an already-correct kernel**. The generator only needs to produce a valid seed; the search stage then improves it with verified incremental edits.

> This README is the public-facing overview. Detailed implementation notes and experiment history live in [`CLAUDE.md`](./CLAUDE.md) and `/home/wangyichen/work-log.md`.

## Current pipeline

| Stage | Purpose | Notes |
|---|---|---|
| **Baseline profiling** | Measure the PyTorch/reference implementation and optionally collect NCU metrics | Produces the latency/profile used by search prompts. |
| **Seed generation** | Generate one verified Triton seed kernel | Level 1 can use DirecTune's v3 generator; Level 2+ uses the vendored AKG frontend with a weight-loading shim. |
| **Search** | Optimize the verified seed | Default path is the v5 `unified_editor`; the v4 classic planner/executor/summarizer path is still available for ablation. |

Older Fuser/Dispatch/Compose and episodic outer-loop paths have been retired from the main flow. Search now starts from a single verified seed and directly calls `run_search_episode()`.

## Search stage

Search is the main DirecTune contribution. It takes a verified Triton kernel and repeatedly proposes small changes, validates them, and carries forward the fastest correct candidates.

### Default mode: unified editor

The default search mode is:

```yaml
search_mode: unified
direction_organized_frontier: false
```

In this mode, one LLM agent (`unified_editor`) handles both planning and editing:

```text
current candidate kernel
  + latency / hardware profile
  + short-term search experience
  + relevant Triton skills
  → LLM proposes search/replace modifications
  → patch is applied to the verified kernel
  → anti-PyTorch check, compile, correctness check, benchmark
  → successful candidates enter beam selection
```

This replaces the older v4 `planner → executor → summarizer` handoff. The goal is to reduce natural-language handoff loss and avoid full-file rewrites unless they are needed as a fallback.

### Incremental editing and safety

The LLM usually returns search/replace blocks rather than a full rewritten file. DirecTune applies them with a multi-level matcher:

1. exact match,
2. trimmed-line match,
3. whitespace-normalized match,
4. fuzzy match with confidence checks.

Every edited kernel goes through the same validation gate:

```text
anti-PyTorch scan → compile → numerical correctness → latency benchmark → optional NCU metrics
```

If incremental editing fails, DirecTune can fall back to a full rewrite for a bounded number of attempts. If all attempts fail, the original candidate is kept. This gives the search a **best-so-far / no-regression** property: a bad patch should not replace the current best verified kernel.

### Direction-organized frontier

DirecTune also has an opt-in direction mode:

```yaml
direction_organized_frontier: true
direction_max_width: 3
direction_free_explore: true
```

This mode addresses a common beam-search failure mode: multiple LLM samples often explore the same kind of change, such as only tuning block sizes. Direction mode makes the search budget cover distinct expert optimization ideas.

The current direction set (v6, eight directions) is organized by **transformation type, not in a hierarchy** — each direction is a flat bucket of related optimization moves, ordered below by expected payoff (highest first):

| Direction | Meaning | Typical payoff |
|---|---|---|
| `algo_equiv` | Algebraic / algorithmic equivalence that computes less work (GEMM+Sum→matvec, online softmax, im2col+GEMM, precomputation) | 3–8000×, rarest |
| `reduction_struct` | Restructure reductions and scans (axis blocking, split two-stage, online reduction, tree layout) | 1.3–50× |
| `mem_layout` | Memory access and data layout (coalescing, shared-memory tiling, bank conflicts, dematerialization, contiguous specialization) | 1.5–51× |
| `timing_overlap` | Overlap memory and compute (`num_stages` software pipelining, double buffering, async copy, compute/memory overlap) | 1.3–9.6× |
| `precision_tc` | Precision and Tensor Core (`allow_tf32`, fp16/bf16, `tl.dot`, tile ≥ 32) | 2–8× |
| `fusion` | Fuse adjacent ops to avoid intermediate writes (GEMM+bias+ReLU, epilogue fusion, dematerialization) | 1.3–2× |
| `control_flow_spec` | Control flow and specialization (mask simplification, early-exit, constexpr shapes, stride fast-path) | 1.0–1.5× |
| `tile_config` | Tile / parallel config (BLOCK_M/N/K, `num_warps`, GROUP_M, program mapping, persistent kernel, autotune) | 1.2–2×, lowest priority |

#### Design rationale

**Why these eight.** The set grew from five to eight after a standalone 48-direction probe (`/home/wangyichen/dir_probe/`, report in `dir_probe/REPORT.md`) measured the payoff of each fine-grained optimization move on a naive Triton seed. Three structural blind spots showed up as independent high-leverage dimensions that the original five directions either buried inside other buckets or had no home for:

- `timing_overlap` (`num_stages` / double buffering) — previously folded into `tile_config` / `mem_layout`, measured 6–9.6×,
- `reduction_struct` (tree / split-K / online reduction) — previously scattered across three buckets, measured up to 21×,
- `control_flow_spec` (mask / constexpr / stride specialization) — previously had no bucket at all.

The eight directions are a coarse clustering of those 48 fine-grained moves: each bucket maps to several specific moves in the probe (e.g. `reduction_struct` ← `reduction_axis_blocking` / `split_two_stage_reduction` / `online_reduction` / `reduction_tree_layout`). The full 48-direction table with measured per-direction payoffs lives in `dir_probe/REPORT.md`.

**Applicability is decided by operator semantics, not by the current bottleneck.** The classifier judges which directions apply from the operator's computation structure (does it contain a matmul? a reduction? a multi-tile loop? a boundary mask?). The NCU bottleneck only reorders priority — it does not gate applicability. This keeps the direction set stable within a search episode: it is computed once per episode, not recomputed every iteration.

**How the set is used.**

- In `unified` + direction mode: the classifier runs once per episode, returns the applicable directions ordered by expected payoff, `unified_editor` samples one direction-specific patch per direction, and frontier selection truncates by direction priority rather than by raw latency.
- In `mcts` mode: the same ordering is blended with the probe's measured payoff table (`_DIR_PROBE_PRIOR` in `mcts.py`) to form the P-UCT prior — structural directions get a higher prior, `tile_config` gets a low one — driving tree expansion.

When direction mode is enabled:

1. A direction classifier runs once per search episode and selects applicable directions for the current operator.
2. For each applicable direction, `unified_editor` sends a direction-specific prompt asking the LLM to focus on that optimization idea.
3. Each direction branch receives a label such as `dir_2_precision_tc` or `dir_5_algo_equiv`.
4. Candidate selection keeps the fastest valid candidate per direction, then truncates by direction priority instead of letting one short-term-fast direction occupy the whole beam.
5. An optional `free_explore` branch remains unconstrained as a fallback.

This turns `breadth` from "try several random patches" into "try several semantically different optimization moves".

### Direction statistics

`direction_store.py` records per-operator direction outcomes:

- patches sampled,
- patches that passed validation,
- candidates that survived selection,
- best speedup versus the seed,
- number of runs sampled.

This is currently a persistence and logging layer only. It does **not** yet change search decisions. The intended next step is to use these empirical statistics to calibrate future direction priorities.

### Classic mode

The earlier v4-style mode is still available:

```yaml
search_mode: classic
```

It uses:

```text
planner → executor → summarizer
```

- planner proposes natural-language optimization plans,
- executor generates full Triton implementations,
- summarizer extracts experience from slow/fast candidate pairs,
- beam selection keeps the fastest valid candidates.

This path is mainly useful for ablations and comparisons with the v5 unified editor.

## Quick start

```bash
conda activate forge
cd /home/wangyichen/DirecTune

CUDA_VISIBLE_DEVICES=0 python main.py --config config.yaml \
  --problem problems/kb_level1/01_square_matrix_multiplication.json \
  --initial problems/kb_level1/01_square_matrix_multiplication_initial.py \
  --rounds 3 --breadth 2 --num-samples 1
```

Direction-organized search can be run with a config that enables `direction_organized_frontier`, for example `config_direction.yaml` if present:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config config_direction.yaml \
  --problem problems/kb_level2/14_Gemm_Divide_Sum_Scaling.json \
  --initial problems/kb_level2/14_Gemm_Divide_Sum_Scaling_initial.py \
  --rounds 3 --breadth 2 --num-samples 1
```

GPU profiling note: NCU collection uses passwordless `sudo ncu` on this machine. When profiling with NCU, use the absolute Python interpreter path because `sudo` changes `PATH`.

## Configuration highlights

| Key | Purpose |
|---|---|
| `search_mode` | `unified` by default; set to `classic` for the v4-style three-agent path. |
| `rounds` / `iters` | Number of search iterations. |
| `breadth` | Number of patch samples in legacy unified mode; in direction mode, direction branches replace ordinary same-prompt sampling. |
| `num_samples` | Number of executor samples per plan in classic mode. |
| `topk_candidates` | Beam width for ordinary latency-based selection. |
| `unified_fail_threshold` | Maximum full-rewrite fallback attempts after an incremental edit fails. |
| `direction_organized_frontier` | Enables direction-specific prompting and direction-aware frontier selection. |
| `direction_max_width` | Maximum number of real direction branches carried forward. |
| `direction_free_explore` | Adds an unconstrained exploration branch in direction mode. |
| `direction_stats_path` | Path for persisted direction outcome statistics. |

## Dataset

DirecTune uses KernelBench-style tasks:

- `problems/kb_level1/`: single operators without weights,
- `problems/kb_level2/`: fused operators with `nn.Module` weights,
- `problems/kb_level3/`: larger model fragments.

Each task provides a PyTorch reference, input shapes, and expected output behavior. Generated kernels are rejected unless they compile and match the reference within the configured tolerance.

## Repository map

| File / directory | Role |
|---|---|
| `main.py` | Unified entry point and `run_search_episode()` loop. |
| `generator.py` | Seed kernel generation, including the vendored AKG frontend path. |
| `agents.py` | Unified editor, classic planner/executor/summarizer, direction classifier, validation glue. |
| `search.py` | Beam candidate selection, including direction-aware selection. |
| `direction_store.py` | Persistent direction outcome statistics. |
| `triton_backend.py` | Compilation, validation, benchmarking, and anti-PyTorch checks. |
| `hardware_profiler.py` | NCU metric collection and parsing. |
| `akg_frontend/` | Vendored AKG agent frontend used by the generator and skill system. |
| `prompts/` | Prompt templates. |
| `skills/` | Triton optimization skill material. |
| `problems/` | KernelBench problem definitions and frozen weights. |

## Documentation

- [`CLAUDE.md`](./CLAUDE.md): detailed architecture notes, implementation details, and current limitations.
- [`MCTS_28_SPEEDUPS.md`](./MCTS_28_SPEEDUPS.md): 28-problem (18 L1 + 10 L2) MCTS speedup summary from the full comparison experiment.
- [`figures/l1_speedup_chart.html`](./figures/l1_speedup_chart.html) / [`l1_speedup_chart.png`](./figures/l1_speedup_chart.png): 49 unique L1 speedups (vs PyTorch & vs seed), geomean 1.49×/3.10×.
- `/home/wangyichen/work-log.md`: chronological development notes and experiment logs.
