import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _add(a, b):
    return a + b

@triton.jit
def cumsum_kernel(x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    
    offs = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for k in range(0, N, BLOCK_SIZE):
        cols = k + offs
        
        if EVEN_N:
            x = tl.load(x_ptr + row * N + cols)
            y = tl.associative_scan(x, axis=0, combine_fn=_add)
            y = y + acc
            tl.store(out_ptr + row * N + cols, y)
        else:
            mask = cols < N
            x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
            y = tl.associative_scan(x, axis=0, combine_fn=_add)
            y = y + acc
            tl.store(out_ptr + row * N + cols, y, mask=mask)
        
        acc += tl.sum(x, axis=0)

def run(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    M, N = x.shape
    grid = (M,)
    cumsum_kernel[grid](x, out, M, N, BLOCK_SIZE=2048, EVEN_N=(N % 2048 == 0), num_warps=8, num_stages=2)
    return out
