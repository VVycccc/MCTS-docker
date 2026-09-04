import torch
import torch.nn as nn

import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, M, N, BLOCK_N: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(0)
    row = pid

    # Online softmax: single pass for running max and sum
    row_max = -float('inf')
    row_sum = 0.0
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        if EVEN_N:
            x = tl.load(x_ptr + row * N + offs)
        else:
            mask = offs < N
            x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # Write normalized output
    inv_sum = 1.0 / row_sum
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        if EVEN_N:
            x = tl.load(x_ptr + row * N + offs)
        else:
            mask = offs < N
            x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
        y = tl.exp(x - row_max) * inv_sum
        if EVEN_N:
            tl.store(y_ptr + row * N + offs, y)
        else:
            mask = offs < N
            tl.store(y_ptr + row * N + offs, y, mask=mask)


def run(x):
    y = torch.empty_like(x)
    M, N = x.shape
    grid = (M,)
    BLOCK_N = 4096
    softmax_kernel[grid](x, y, M, N, BLOCK_N=BLOCK_N, EVEN_N=N % BLOCK_N == 0, num_warps=8)
    return y
