import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D1': 64, 'BLOCK_D2': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_D1': 128, 'BLOCK_D2': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_D1': 64, 'BLOCK_D2': 128}, num_stages=4, num_warps=4),
    ],
    key=['D1', 'D2'],
)
@triton.jit
def max_reduce_kernel(x_ptr, out_ptr, D1, D2, EVEN_D1: tl.constexpr, BLOCK_D1: tl.constexpr, BLOCK_D2: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d2 = tl.program_id(1)
    d2_offs = pid_d2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    d2_mask = d2_offs < D2
    max_val = tl.full((BLOCK_D2,), -float('inf'), tl.float32)
    for off in range(0, D1, BLOCK_D1):
        d1_offs = off + tl.arange(0, BLOCK_D1)
        x_ptrs = x_ptr + pid_b * D1 * D2 + d1_offs[:, None] * D2 + d2_offs[None, :]
        if EVEN_D1:
            x_vals = tl.load(x_ptrs, mask=d2_mask[None, :], other=-float('inf'))
        else:
            d1_mask = d1_offs < D1
            x_vals = tl.load(x_ptrs, mask=d1_mask[:, None] & d2_mask[None, :], other=-float('inf'))
        m = tl.max(x_vals, axis=0)
        max_val = tl.maximum(max_val, m)
    tl.store(out_ptr + pid_b * D2 + d2_offs, max_val, mask=d2_mask)

def run(x):
    B, D1, D2 = x.shape
    out = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    grid = lambda meta: (B, triton.cdiv(D2, meta['BLOCK_D2']))
    max_reduce_kernel[grid](x, out, D1, D2, EVEN_D1=(D1 % 128 == 0))
    return out
