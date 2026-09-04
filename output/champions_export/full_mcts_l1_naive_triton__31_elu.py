import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n'],
)
@triton.jit
def elu_kernel(x_ptr, y_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    pos = x
    neg = alpha * (tl.exp(x) - 1.0)
    y = tl.where(x > 0.0, pos, neg)
    tl.store(y_ptr + offs, y)

def run(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    elu_kernel[grid](x, y, n, 1.0)
    return y
