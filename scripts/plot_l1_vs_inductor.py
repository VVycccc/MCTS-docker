"""L1 champion speedup vs torch.compile (inductor) bar chart.

Mirrors the style of plot_l1_two_charts.py (vs PyTorch eager / vs seed):
same short labels, log-scale bars, parity line, geomean box, blue/red
coloring by >=1x or <1x. Sorted by vs-Inductor speedup descending.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "figures")
L1_PROBLEM_DIR = os.path.join(REPO, "problems", "kb_level1")

# short labels identical to plot_l1_two_charts.py
REP = {
    "square_matrix_multiplication": "matmul",
    "standard_matrix_multiplication": "matmul",
    "batched_matrix_multiplication": "bmm",
    "matrix_vector_multiplication": "matvec",
    "matrix_scalar_multiplication": "mat_scalar",
    "matmul_with_large_k_dimension": "matmul_largeK",
    "matmul_with_small_k_dimension": "matmul_smallK",
    "matmul_with_irregular_shapes": "matmul_irreg",
    "tall_skinny_matrix_multiplication": "tall_skinny",
    "3d_tensor_matrix_multiplication": "matmul_3d",
    "4d_tensor_matrix_multiplication": "matmul_4d",
    "matmul_with_diagonal_matrices": "matmul_diag",
    "matmul_for_symmetric_matrices": "matmul_sym",
    "matmul_for_upper_triangular_matrices": "matmul_upper",
    "matmul_for_lower_triangular_matrices": "matmul_lower",
    "matmul_with_transposed_a": "matmul_tA",
    "matmul_with_transposed_b": "matmul_tB",
    "matmul_with_transposed_both": "matmul_tAB",
    "max_pooling_1d": "maxpool_1d",
    "max_pooling_2d": "maxpool_2d",
    "max_pooling_3d": "maxpool_3d",
    "average_pooling_1d": "avgpool_1d",
    "average_pooling_2d": "avgpool_2d",
    "average_pooling_3d": "avgpool_3d",
    "sum_reduction_over_a_dimension": "sum_red",
    "mean_reduction_over_a_dimension": "mean_red",
    "max_reduction_over_a_dimension": "max_red",
    "min_reduction_over_a_dimension": "min_red",
    "argmax_over_a_dimension": "argmax",
    "argmin_over_a_dimension": "argmin",
    "conv_standard_2d_square_input_square_kernel": "conv2d",
    "conv_transposed_2d_square_input_square_kernel": "convT2d",
    "scaleddotproductattention": "sdpa",
    "mingptnewgelu": "newgelu",
}


def short(name: str) -> str:
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        idx, rest = parts
    else:
        idx, rest = "", name
    rest = REP.get(rest, rest)
    return f"{idx}_{rest}" if idx else rest


# ---- champion latencies (same aggregation as plot_l1_two_charts.py) ----
L1_NAMES = set(f[:-5] for f in os.listdir(L1_PROBLEM_DIR)
               if f.endswith(".json"))
best = {}
for snap in sorted(os.listdir(f"{REPO}/output")):
    sp = f"{REPO}/output/{snap}"
    if not os.path.isdir(sp):
        continue
    for d in sorted(os.listdir(sp)):
        if d not in L1_NAMES:
            continue
        fr = f"{sp}/{d}/final_results.json"
        if not os.path.isfile(fr):
            continue
        try:
            j = json.load(open(fr))
            fc = j.get("final_candidates", [{}])[0]
            champ = fc.get("latency_ms")
            has_t = "@triton.jit" in (fc.get("code") or "")
        except Exception:
            continue
        if not (champ and champ > 0):
            continue
        cur = best.get(d)
        if cur is None or (has_t and not cur[1]):
            best[d] = (champ, has_t)
strict = {d: c for d, (c, t) in best.items() if t}

ind = json.load(open(f"{REPO}/output/inductor_bench/l1_strict48.json"))

rows = []
for name, champ in strict.items():
    r = ind.get(name)
    if not r:
        continue
    d = r.get("default", {}).get("latency_ms")
    m = r.get("max-autotune", {}).get("latency_ms")
    bi = min([x for x in (d, m) if x], default=None)
    if not bi:
        continue
    rows.append((name, short(name), bi / champ))
rows.sort(key=lambda r: r[2], reverse=True)

labels = [r[1] for r in rows]
vals = [r[2] for r in rows]
n = len(rows)


def gm(xs):
    xs = [v for v in xs if v and v > 0]
    return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else float("nan")


gm_ind = gm(vals)

BAR_W = 0.62
COLOR_WIN = "#8172B3"    # purple >= 1x (distinct from eager-chart blue)
COLOR_LOSE = "#DD8452"   # orange < 1x (distinct from eager-chart red)

fig, ax = plt.subplots(figsize=(14, 6.5))
x = np.arange(n)
colors = [COLOR_WIN if v >= 1.0 else COLOR_LOSE for v in vals]
ax.bar(x, vals, BAR_W, color=colors, edgecolor="white", linewidth=0.4)
ax.set_yscale("log")
ax.set_ylim(0.4, max(vals) * 1.4)
for i, v in enumerate(vals):
    ax.text(x[i], v * 1.10, f"{v:.1f}×", ha="center", va="bottom",
            fontsize=6.0, rotation=90, color="#222")
ax.axhline(y=1.0, color="#2E7D32", linestyle=":", linewidth=1.0, alpha=0.75)
ax.text(n - 0.5, 1.06, "1× (parity)", ha="right", fontsize=7, color="#2E7D32")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6.8)
ax.set_xlabel("KernelBench Level-1 problem", fontsize=10)
ax.set_ylabel("Speedup vs torch.compile Inductor (×, log scale)", fontsize=10)
ax.set_title("DirecTune-MCTS Level-1: champion speedup vs torch.compile (Inductor, best mode)\n"
             f"(strict-Triton champions, n={n}; purple ≥1×, orange <1×)", fontsize=11)
ax.grid(axis="y", which="both", alpha=0.25, linestyle="-")
wins = sum(1 for v in vals if v >= 1.0)
ax.text(0.015, 0.97,
        f"Geomean vs Inductor-best (n={n}): {gm_ind:.2f}×\nchampion faster on {wins}/{n}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9))
plt.tight_layout()
for ext in ("pdf", "png"):
    out = f"{FIG_DIR}/l1_speedup_vs_inductor.{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
    print(f"saved: {out}")
plt.close(fig)
