import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    x_ptr, y_ptr,
    N, C, HW: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * HW + c * HW
    num_iters = HW // BLOCK

    # Single pass: accumulate sum and sum of squares simultaneously
    sum_val = 0.0
    sum_sq = 0.0
    for k in range(0, num_iters):
        idx = k * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + idx)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    mean = sum_val / HW
    var = sum_sq / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and store
    for k in range(0, num_iters):
        idx = k * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + idx)
        y = (x - mean) * rstd
        tl.store(y_ptr + base + idx, y)


def run(x):
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    eps = 1e-5
    HW = H * W
    BLOCK = 4096
    grid = (N * C,)
    instance_norm_kernel[grid](x, y, N, C, HW, eps, BLOCK=BLOCK, num_warps=8, num_stages=2)
    return y
