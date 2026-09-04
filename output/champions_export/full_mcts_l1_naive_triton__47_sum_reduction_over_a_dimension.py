import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D1': 32, 'BLOCK_D2': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_D1': 64, 'BLOCK_D2': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_D1': 128, 'BLOCK_D2': 64}, num_stages=3, num_warps=8),
    ],
    key=['B', 'D1', 'D2'],
)
@triton.jit
def sum_reduce_stage1_kernel(x_ptr, partial_ptr, B, D1, D2, N_PARTS: tl.constexpr, D1_PART: tl.constexpr, BLOCK_D1: tl.constexpr, BLOCK_D2: tl.constexpr):
    pid = tl.program_id(0)
    num_d2_tiles = tl.cdiv(D2, BLOCK_D2)
    pid_d2 = pid % num_d2_tiles
    pid_rem = pid // num_d2_tiles
    part_id = pid_rem % N_PARTS
    b = pid_rem // N_PARTS
    d2_offs = pid_d2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    d2_mask = d2_offs < D2
    d1_chunk_start = part_id * D1_PART
    acc = tl.zeros((BLOCK_D2,), dtype=tl.float32)
    for d1_off in range(0, D1_PART, BLOCK_D1):
        d1_offs = d1_chunk_start + d1_off + tl.arange(0, BLOCK_D1)
        d1_mask = d1_offs < D1
        ptrs = x_ptr + b * D1 * D2 + d1_offs[:, None] * D2 + d2_offs[None, :]
        mask = d1_mask[:, None] & d2_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(partial_ptr + b * N_PARTS * D2 + part_id * D2 + d2_offs, acc, mask=d2_mask)

@triton.jit
def sum_reduce_stage2_kernel(partial_ptr, out_ptr, B, D2, N_PARTS: tl.constexpr, BLOCK_D2: tl.constexpr):
    pid = tl.program_id(0)
    num_d2_tiles = tl.cdiv(D2, BLOCK_D2)
    b = pid // num_d2_tiles
    pid_d2 = pid % num_d2_tiles
    d2_offs = pid_d2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    d2_mask = d2_offs < D2
    acc = tl.zeros((BLOCK_D2,), dtype=tl.float32)
    for part in range(N_PARTS):
        val = tl.load(partial_ptr + b * N_PARTS * D2 + part * D2 + d2_offs, mask=d2_mask, other=0.0)
        acc += val
    tl.store(out_ptr + b * D2 + d2_offs, acc, mask=d2_mask)

def run(x):
    B, D1, D2 = x.shape
    out = torch.empty((B, 1, D2), device=x.device, dtype=x.dtype)
    N_PARTS = 4
    D1_PART = D1 // N_PARTS
    partial = torch.empty((B, N_PARTS, D2), device=x.device, dtype=x.dtype)
    grid1 = lambda meta: (B * N_PARTS * triton.cdiv(D2, meta['BLOCK_D2']),)
    sum_reduce_stage1_kernel[grid1](x, partial, B, D1, D2, N_PARTS=N_PARTS, D1_PART=D1_PART)
    BLOCK_D2 = 128
    grid2 = (B * triton.cdiv(D2, BLOCK_D2),)
    sum_reduce_stage2_kernel[grid2](partial, out, B, D2, N_PARTS=N_PARTS, BLOCK_D2=BLOCK_D2)
    return out
