import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsoftmax_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(0)

    x_row_ptr = x_ptr + pid * N
    y_row_ptr = y_ptr + pid * N

    # Fused pass 1+2: online softmax — compute max and sum in a single pass
    max_val = -float('inf')
    sum_val = 0.0
    for off in range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        if EVEN_N:
            x = tl.load(x_row_ptr + offs)
        else:
            mask = offs < N
            x = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        old_max = max_val
        max_val = tl.maximum(max_val, block_max)
        # Rescale previous sum and add new block sum
        sum_val = sum_val * tl.exp(old_max - max_val) + tl.sum(tl.exp(x - max_val), axis=0)

    log_sum = tl.log(sum_val)

    # Pass 2: write output y = x - max - log(sum)
    for off in range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        if EVEN_N:
            x = tl.load(x_row_ptr + offs)
            y = x - max_val - log_sum
            tl.store(y_row_ptr + offs, y)
        else:
            mask = offs < N
            x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
            y = x - max_val - log_sum
            tl.store(y_row_ptr + offs, y, mask=mask)

def run(x):
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    BLOCK_SIZE = 4096
    EVEN_N = N % BLOCK_SIZE == 0
    logsoftmax_kernel[grid](x, y, M, N, BLOCK_SIZE=BLOCK_SIZE, EVEN_N=EVEN_N, num_warps=8, num_stages=2)
    return y
