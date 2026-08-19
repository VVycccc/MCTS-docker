"""测三题 PyTorch baseline (expanded reference run()) 延迟，对比 generator 内核。"""
import json, time, importlib.util, sys
from pathlib import Path
import torch

PROBLEMS = ["76_Gemm_Add_ReLU", "12_Gemm_Multiply_LeakyReLU", "9_Matmul_Subtract_Multiply_ReLU"]
BASE = Path(__file__).resolve().parent / "problems" / "kb_level2"
OUT = Path(__file__).resolve().parent.parent / "DirecTune" / "output" / "l2val"

def load_run(problem):
    spec = importlib.util.spec_from_file_location(f"mod_{problem}", BASE / f"{problem}_initial.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def bench(fn, args, warmup=10, reps=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record(); fn(*args); ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / reps

print(f"{'problem':40} {'baseline_ms':>12} {'gen_ms':>10} {'speedup':>8}")
print("-" * 74)
for p in PROBLEMS:
    mod = load_run(p)
    x = torch.rand(1024, 8192, device="cuda", dtype=torch.float32)
    # warmup + correctness self-check
    _ = mod.run(x)
    base_ms = bench(mod.run, (x,))
    champ = json.load(open(OUT / p / "champion.json"))
    gen_ms = champ["latency_ms"]
    speedup = base_ms / gen_ms if gen_ms else float("nan")
    print(f"{p:40} {base_ms:12.4f} {gen_ms:10.4f} {speedup:8.2f}x")
    # save
    champ["baseline_latency_ms"] = base_ms
    champ["speedup_vs_pytorch"] = speedup
    json.dump(champ, open(OUT / p / "champion.json", "w"), indent=2)
torch.cuda.empty_cache()
