#!/usr/bin/env python3
"""profile_isolated 验证三连：等价性 / 抗中毒性 / 开销（2026-08-27 抗毒修复配套）。

T1 等价性：真实 run 的 seed + champion 内核，进程内 profile() 与子进程
   profile_isolated() 各测多次，correctness 一致、latency 差在噪声带内。
T2 抗中毒：手工构造必然非法访存的坏 kernel —— 隔离路径应干净失败，且随后
   主进程内的好 kernel 照常验证成功（这在旧的进程内路径下必死无疑）。
T3 开销：报告每次隔离验证的 wall-clock（子进程 import + 验证）。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os_cwd = Path(__file__).resolve().parents[1]
import os
os.chdir(os_cwd)

import triton_backend as tb

PROBLEM_JSON = "problems/kb_level1/01_square_matrix_multiplication.json"
TREE_JSON = "output/full_mcts_treelog_20260827/01_square_matrix_multiplication_try2/mcts_tree.json"

BAD_KERNEL = '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _bad_kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # 每个程序越界 256MB 写 —— 必然 device-side illegal memory access
    offs = pid * 67108864 + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 0.0)


def run(A, B):
    x = torch.empty(64, device=A.device, dtype=A.dtype)
    _bad_kernel[(128,)](x, BLOCK=128)
    return A
'''


def ok(cond, msg):
    print(("  ok: " if cond else "FAIL: ") + msg)
    if not cond:
        sys.exit(1)


def main():
    problem = tb.load_problem(PROBLEM_JSON)
    tree = json.loads(Path(TREE_JSON).read_text())
    nodes = {n["node_id"]: n for n in tree["nodes"]}
    kernels = {
        "seed(n0)": nodes["n0"]["code"],
        "champion(n32)": nodes["n32"]["code"],
    }
    rel_tol = tb.adaptive_rel_tol(problem)

    print("== T1 等价性 ==")
    for name, src in kernels.items():
        iso_lats, proc_lats = [], []
        flags = []
        for i in range(2):
            t0 = time.time()
            r_iso = tb.profile_isolated(src, problem, rel_tol=rel_tol,
                                        config={"isolated_verify": True})
            t_iso = time.time() - t0
            r_proc = tb.profile(src, problem, rel_tol=rel_tol)
            iso_lats.append(r_iso.latency_ms)
            proc_lats.append(r_proc.latency_ms)
            flags.append((r_iso.compiled, r_iso.correct))
            if i == 0:
                print(f"  [{name}] isolated {t_iso:.1f}s | "
                      f"iso={r_iso.latency_ms:.4f}ms proc={r_proc.latency_ms:.4f}ms")
        ok(all(f == (True, True) for f in flags), f"{name} 两路径均 compiled+correct")
        lo, hi = min(proc_lats), max(proc_lats)
        mid = sum(iso_lats) / len(iso_lats)
        dev = abs(mid - sum(proc_lats) / len(proc_lats)) / (sum(proc_lats) / len(proc_lats))
        ok(dev < 0.05, f"{name} latency 偏差 {dev*100:.1f}% (iso~{mid:.3f} vs proc~{sum(proc_lats)/2:.3f}ms)")

    print("\n== T2 抗中毒（核心属性）==")
    r_bad = tb.profile_isolated(BAD_KERNEL, problem, rel_tol=rel_tol,
                                config={"isolated_verify": True})
    ok(not (r_bad.compiled and r_bad.correct),
       f"坏 kernel 隔离路径干净失败 ({str(r_bad.error)[:70]}...)")
    r_good_after_bad = tb.profile(kernels["seed(n0)"], problem, rel_tol=rel_tol)
    ok(r_good_after_bad.compiled and r_good_after_bad.correct
       and r_good_after_bad.latency_ms is not None,
       f"主进程随后验证好 kernel 依然成功 ({r_good_after_bad.latency_ms:.4f}ms)"
       " —— 上下文未被波及，这就是修复的意义")

    print(f"\nALL PASS")


if __name__ == "__main__":
    main()
