import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    inner = 1.5957691216057308 * (x + 0.044715 * x * x * x)
    y = x * tl.sigmoid(inner)
    tl.store(y_ptr + offs, y)

def run(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    N = x.numel()
    gelu_kernel[(lambda meta: (N // meta['BLOCK_SIZE'],))](x, y, N)
    return y
