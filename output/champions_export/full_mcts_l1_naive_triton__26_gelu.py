import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    EVEN_N: tl.constexpr = N % BLOCK_SIZE == 0
    if EVEN_N:
        x = tl.load(x_ptr + offs)
        inv_sqrt2 = 0.7071067811865476
        y = x * 0.5 * (1.0 + tl.erf(x * inv_sqrt2))
        tl.store(y_ptr + offs, y)
    else:
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        inv_sqrt2 = 0.7071067811865476
        y = x * 0.5 * (1.0 + tl.erf(x * inv_sqrt2))
        tl.store(y_ptr + offs, y, mask=mask)

def run(x):
    x_contig = x.contiguous()
    x_flat = x_contig.reshape(-1)
    y_flat = torch.empty_like(x_flat)
    N = x_flat.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x_flat, y_flat, N)
    return y_flat.reshape(x.shape)
