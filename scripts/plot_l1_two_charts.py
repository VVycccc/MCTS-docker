"""Split the L1 speedup figure into two separate charts:
  (1) vs PyTorch eager   — champion speedup over the measured PyTorch baseline
  (2) vs pure LLM seed   — champion speedup over the naive LLM-generated Triton seed

Aggregates the best strict-Triton champion per unique Level-1 problem across all
output snapshots under DirecTune-MCTS/output/. Mirrors the data-scan logic of
output/full_mcts/plot_speedup.py but scoped to Level-1 problems and emitting two
figures into the paper's figures/ directory.
"""
import json, re, math, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "output"
FIG_DIR = Path(os.environ.get("DT_FIG_DIR", REPO / "figures"))

# canonical 100 L1 problem names
L1_NAMES = set(f[:-5] for f in os.listdir(REPO / "problems/kb_level1") if f.endswith(".json"))

# short display labels (keep the numeric prefix + a terse descriptor)
def short(name: str) -> str:
    # strip leading zero-padding of the index for compactness, keep rest
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        idx = parts[0]
        rest = parts[1]
    else:
        idx, rest = "", name
    # collapse long conv/pool names
    rep = {
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
    rest = rep.get(rest, rest)
    return f"{idx}_{rest}" if idx else rest


# ---- gather best strict-Triton champion per L1 problem across snapshots ----
best = {}  # name -> (champ, base, seed, triton, sk, src)
for snap in sorted(os.listdir(OUT_ROOT)):
    sp = OUT_ROOT / snap
    if not sp.is_dir():
        continue
    for d in sorted(os.listdir(sp)):
        if d not in L1_NAMES:
            continue
        fr = sp / d / "final_results.json"
        if not fr.is_file():
            continue
        try:
            j = json.load(open(fr))
            fc = j.get("final_candidates", [{}])[0]
            champ = fc.get("latency_ms")
            has_t = "@triton.jit" in fc.get("code", "")
        except Exception:
            champ, has_t = None, False
        if not (champ and champ > 0):
            continue
        txt = open(sp / d / "run.log", errors="replace").read()
        m = re.search(r"Baseline latency:\s*([\d.]+)", txt)
        base = float(m.group(1)) if m else None
        m = re.search(r"Generator \(naive\).*?latency=([\d.]+)", txt)
        if m:
            seed, sk = float(m.group(1)), "naive"
        elif "Generator (naive) failed" in txt:
            seed, sk = base, "ref"
        else:
            seed, sk = None, "?"
        rec = (champ, base, seed, has_t, sk, snap)

        def score(r):
            return (1 if r[3] else 0, 1 if (r[2] and r[2] > 0) else 0)

        cur = best.get(d)
        if cur is None or score(rec) > score(cur):
            best[d] = rec

# strict-Triton subset with valid champion
strict = {d: r for d, r in best.items() if r[3] and r[0] and r[0] > 0}
print(f"retained L1 (champ>0): {len(best)}  strict-triton: {len(strict)}")

# build rows, sort by vs-PyTorch speedup descending (shared ordering across both charts)
rows = []
for d, (champ, base, seed, has_t, sk, src) in strict.items():
    vp = (base / champ) if (base and base > 0) else None
    vs = (seed / champ) if (seed and seed > 0) else None
    rows.append((d, short(d), vp, vs, sk))
rows.sort(key=lambda r: (r[2] if r[2] else 0), reverse=True)

labels = [r[1] for r in rows]
vp = [r[2] for r in rows]
vs = [r[3] for r in rows]
sk = [r[4] for r in rows]


def gm(xs):
    xs = [v for v in xs if v and v > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


gm_pytorch = gm(vp)
gm_seed = gm([v for v, s in zip(vs, sk) if s == "naive"])  # vs pure-LLM seed only

# shared style constants
BAR_W = 0.62
COLOR_PYTORCH = "#4C72B0"   # blue
COLOR_SEED = "#55A868"      # green
COLOR_SLOWER = "#C0392B"   # red — slower than reference
COLOR_PARITY = "#999999"


def _annotate(ax, x, vals, rotate=True):
    for i, v in enumerate(vals):
        if v and v > 0:
            ax.text(x[i], v * 1.10, f"{v:.1f}×", ha="center", va="bottom",
                    fontsize=6.0, rotation=90,
                    color="#222")


def _finalize(ax, n, title, ylabel, gm_text):
    ax.axhline(y=1.0, color="#2E7D32", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.text(n - 0.5, 1.06, "1× (parity)", ha="right", fontsize=7, color="#2E7D32")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([r[1] for r in rows], rotation=80, ha="right", fontsize=6.8)
    ax.set_xlabel("KernelBench Level-1 problem", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", which="both", alpha=0.25, linestyle="-")
    ax.text(0.015, 0.97, gm_text, transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))


# ---- Chart 1: vs PyTorch eager ----
n = len(rows)
fig1, ax1 = plt.subplots(figsize=(14, 6.5))
x = np.arange(n)
vp_plot = [v if v else 0 for v in vp]
colors1 = [COLOR_PYTORCH if (v and v >= 1.0) else (COLOR_SLOWER if v else COLOR_PARITY) for v in vp]
ax1.bar(x, vp_plot, BAR_W, color=colors1, edgecolor="white", linewidth=0.4)
ax1.set_yscale("log")
ax1.set_ylim(0.4, max(12.0, (max([v for v in vp if v] + [1]) * 1.4)))
_annotate(ax1, x, vp)
_finalize(ax1, n,
          "DirecTune-MCTS Level-1: champion speedup vs PyTorch eager\n"
          f"(strict-Triton champions, n={n}; blue ≥1×, red <1×)",
          "Speedup vs PyTorch eager (×, log scale)",
          f"Geomean vs PyTorch (n={n}): {gm_pytorch:.2f}×")
plt.tight_layout()
out1_pdf = FIG_DIR / "l1_speedup_vs_pytorch.pdf"
out1_png = FIG_DIR / "l1_speedup_vs_pytorch.png"
plt.savefig(out1_pdf, bbox_inches="tight")
plt.savefig(out1_png, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"saved: {out1_pdf}")

# ---- Chart 2: vs pure LLM seed ----
fig2, ax2 = plt.subplots(figsize=(14, 6.5))
vs_plot = [v if v else 0 for v in vs]
colors2 = []
for v, s in zip(vs, sk):
    if s == "ref":
        colors2.append(COLOR_PARITY)  # degraded ref (no true LLM seed)
    elif v and v >= 1.0:
        colors2.append(COLOR_SEED)
    else:
        colors2.append(COLOR_SLOWER)
ax2.bar(x, vs_plot, BAR_W, color=colors2, edgecolor="white", linewidth=0.4)
ax2.set_yscale("log")
top = max([v for v in vs if v] + [1]) * 1.4
ax2.set_ylim(0.4, top)
_annotate(ax2, x, vs)
n_naive = sum(1 for s in sk if s == "naive")
_finalize(ax2, n,
          "DirecTune-MCTS Level-1: champion speedup vs pure-LLM seed\n"
          f"(strict-Triton champions, n={n}; green ≥1×, red <1×, grey = no valid LLM seed)",
          "Speedup vs pure-LLM seed (×, log scale)",
          f"Geomean vs pure-LLM seed (naive-seed only, n={n_naive}): {gm_seed:.2f}×")
plt.tight_layout()
out2_pdf = FIG_DIR / "l1_speedup_vs_seed.pdf"
out2_png = FIG_DIR / "l1_speedup_vs_seed.png"
plt.savefig(out2_pdf, bbox_inches="tight")
plt.savefig(out2_png, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out2_pdf}")

# ---- verification table ----
print(f"\n{'label':24} {'vs_pytorch':>10} {'vs_seed':>8} {'sk':>5}")
for r in rows:
    print(f"{r[1]:24} {r[2] if r[2] else 0:10.2f} {r[3] if r[3] else 0:8.2f} {r[4]:>5}")
print(f"\ngeomean vs pytorch: {gm_pytorch:.3f}x")
print(f"geomean vs seed (naive): {gm_seed:.3f}x")
