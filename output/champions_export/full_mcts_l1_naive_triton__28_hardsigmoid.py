import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def hardsigmoid_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_N:
        x = tl.load(x_ptr + offs)
    else:
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # HardSigmoid: clamp(x/6 + 0.5, 0, 1)
    y = x * 0.16666667 + 0.5
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    if EVEN_N:
        tl.store(y_ptr + offs, y)
    else:
        tl.store(y_ptr + offs, y, mask=mask)

def run(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    EVEN_N = N % 8192 == 0
    hardsigmoid_kernel[grid](x, y, N, EVEN_N=EVEN_N)
    return y
