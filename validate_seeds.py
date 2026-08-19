"""validate_seeds.py — profile every candidate naive seed (compile+correct+latency)
to filter broken ones before the MCTS batch run. Prints a table.
"""
import os, sys, time, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from triton_backend import load_problem, profile as triton_profile

DIR_PROBE = os.environ.get("DT_DIR_PROBE", str(Path(__file__).resolve().parent.parent / "dir_probe"))

# (problem_json_basename, seed_basename, op_type, tensor_size_mb, dir_probe_headroom)
CANDIDATES = [
    ("01_square_matrix_multiplication.json", "01_matmul_naive.py",     "gemm",        134, "2.6×"),
    ("12_diag.json",                         "12_diag_naive.py",         "gemm",         67, "diag"),
    ("17_matmul_tb.json",                    "17_matmul_tb_naive.py",    "gemm",        201, "0.86×"),
    ("l2_9.json",                            "l2_9_fusion_naive.py",     "fusion",       34, "2.16×"),
    ("l2_30.json",                           "l2_30_fusion_naive.py",    "fusion",       34, "1.0×死区"),
    ("l2_76.json",                           "l2_76_fusion_naive.py",    "fusion",       34, "1.0×死区"),
    ("40_layernorm.json",                    "40_layernorm_naive.py",    "normalization",268, "1.18×"),
    ("23_softmax.json",                      "23_softmax_naive.py",      "softmax",    6442, "1.59×"),
    ("47_reduction.json",                    "47_reduction_naive.py",    "reduction",  8587, "21.2×"),
    ("49_maxred.json",                       "49_maxred_naive.py",       "reduction",  8587, "21.2×"),
    ("precompute_mean.json",                 "precompute_mean_naive.py", "precompute",  101, "9.3×"),
    ("precompute_test.json",                 "precompute_naive.py",      "precompute",  134, "11×"),
    ("scale_reassoc.json",                   "scale_reassoc_naive.py",  "scale",        84, "1.18×"),
    ("transpose.json",                       "transpose_naive.py",      "transpose",     17, "transpose"),
]

def main():
    print(f"{'problem':40s} {'op':12s} {'size':>7s} {'headroom':8s} | {'compiled':>8s} {'correct':>7s} {'lat_ms':>10s}  err")
    print("-" * 120)
    results = []
    for prob_name, seed_name, op, size_mb, headroom in CANDIDATES:
        prob_path = f"{DIR_PROBE}/problems/{prob_name}"
        seed_path = f"{DIR_PROBE}/seeds/{seed_name}"
        if not (os.path.exists(prob_path) and os.path.exists(seed_path)):
            print(f"{prob_name:40s} {op:12s} {size_mb:>6d}MB {headroom:8s} | MISSING FILE")
            continue
        try:
            prob = load_problem(prob_path)
            seed = open(seed_path).read()
            # 大张量题用宽松 rel_tol + 短超时只测能否跑通
            rel = 1e-2 if size_mb > 1000 else 1e-3
            t0 = time.time()
            pr = triton_profile(seed, prob, timeout_seconds=120, rel_tol=rel)
            dt = time.time() - t0
            lat = f"{pr.latency_ms:.3f}" if pr.latency_ms else "None"
            err = (pr.error or "")[:40].replace("\n"," ")
            print(f"{prob_name:40s} {op:12s} {size_mb:>6d}MB {headroom:8s} | "
                  f"{'yes' if pr.compiled else 'NO':>8s} {'yes' if pr.correct else 'NO':>7s} {lat:>10s}  {err}")
            results.append((prob_name, op, pr.compiled, pr.correct, pr.latency_ms, dt))
        except Exception as e:
            print(f"{prob_name:40s} {op:12s} {size_mb:>6d}MB {headroom:8s} | EXC {str(e)[:50]}")
            results.append((prob_name, op, False, False, None, 0))
        torch.cuda.empty_cache()

    print("\n=== VALID (compiled & correct & latency>0) ===")
    for prob_name, op, comp, corr, lat, dt in results:
        if comp and corr and lat:
            print(f"  {prob_name:40s} {op:12s} {lat:.3f}ms  (profile {dt:.0f}s)")

if __name__ == "__main__":
    main()
