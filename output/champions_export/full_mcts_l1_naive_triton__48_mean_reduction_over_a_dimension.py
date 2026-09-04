import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D1': 32, 'BLOCK_D2': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_D1': 64, 'BLOCK_D2': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_D1': 16, 'BLOCK_D2': 512}, num_warps=8, num_stages=4),
    ],
    key=['D1', 'D2'],
)
@triton.jit
def mean_reduce_kernel(x_ptr, out_ptr, B, D1, D2, BLOCK_D1: tl.constexpr, BLOCK_D2: tl.constexpr, EVEN_D1: tl.constexpr):
    pid = tl.program_id(0)
    num_d2_blocks = tl.cdiv(D2, BLOCK_D2)
    b = pid // num_d2_blocks
    d2_start = (pid % num_d2_blocks) * BLOCK_D2

    d2_offs = d2_start + tl.arange(0, BLOCK_D2)
    d2_mask = d2_offs < D2
    full_d2 = d2_start + BLOCK_D2 <= D2

    base = b * D1 * D2

    acc = tl.zeros((BLOCK_D2,), dtype=tl.float32)
    for d1_start in range(0, D1, BLOCK_D1):
        d1_offs = d1_start + tl.arange(0, BLOCK_D1)
        x_ptrs = x_ptr + base + d1_offs[:, None] * D2 + d2_offs[None, :]
        if EVEN_D1:
            if full_d2:
                x = tl.load(x_ptrs)
            else:
                x = tl.load(x_ptrs, mask=d2_mask[None, :], other=0.0)
        else:
            d1_mask = d1_offs < D1
            if full_d2:
                x = tl.load(x_ptrs, mask=d1_mask[:, None], other=0.0)
            else:
                x = tl.load(x_ptrs, mask=d1_mask[:, None] & d2_mask[None, :], other=0.0)
        acc += tl.sum(x, axis=0)

    if full_d2:
        tl.store(out_ptr + b * D2 + d2_offs, acc / D1)
    else:
        tl.store(out_ptr + b * D2 + d2_offs, acc / D1, mask=d2_mask)


def run(x):
    B, D1, D2 = x.shape
    out = torch.empty((B, D2), dtype=x.dtype, device=x.device)
    grid = lambda meta: (B * triton.cdiv(D2, meta['BLOCK_D2']),)
    mean_reduce_kernel[grid](x, out, B, D1, D2, EVEN_D1=True)
    return out
