import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    y = tl.sigmoid(x)
    tl.store(y_ptr + offs, y)

def run(x):
    y = torch.empty_like(x)
    N = x.numel()
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x_flat, y_flat, N)
    return y
