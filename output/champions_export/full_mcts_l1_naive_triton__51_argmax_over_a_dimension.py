import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def argmax_kernel(x_ptr, out_ptr, M, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # x: (M, N, K), argmax over dim=1 (the N dimension)
    # out: (M, K) int64
    # Each program handles one (m, k_tile) covering BLOCK_K k-values at once
    pid = tl.program_id(0)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    m = pid // num_k_tiles
    k_tile = pid % num_k_tiles

    offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    max_val = tl.full((BLOCK_K,), -float('inf'), tl.float32)
    max_idx = tl.zeros((BLOCK_K,), tl.int32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x = tl.load(
            x_ptr + m * N * K + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=-float('inf'),
        )
        block_max = tl.max(x, axis=0)
        block_argmax_local = tl.argmax(x, axis=0)
        is_better = block_max > max_val
        max_val = tl.where(is_better, block_max, max_val)
        max_idx = tl.where(is_better, n_start + block_argmax_local, max_idx)

    tl.store(out_ptr + m * K + offs_k, tl.cast(max_idx, tl.int64), mask=mask_k)

def run(x):
    x = x.contiguous()
    M, N, K = x.shape
    out = torch.empty((M, K), dtype=torch.int64, device=x.device)
    grid = lambda meta: (M * triton.cdiv(K, meta['BLOCK_K']),)
    argmax_kernel[grid](x, out, M, N, K)
    return out
