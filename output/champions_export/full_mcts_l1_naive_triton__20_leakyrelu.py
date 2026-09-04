import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n'],
)
@triton.jit
def leaky_relu_kernel(x_ptr, y_ptr, n, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.where(x >= 0, x, x * negative_slope)
    tl.store(y_ptr + offs, y, mask=mask)

def run(x):
    x = x.contiguous()
    n = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    leaky_relu_kernel[grid](x, y, n, 0.01)
    return y
