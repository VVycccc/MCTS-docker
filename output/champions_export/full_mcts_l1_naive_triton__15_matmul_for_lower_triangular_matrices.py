import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Skip blocks entirely in the upper triangle (tril will zero them anyway)
    if (pid_m + 1) * BLOCK_M <= pid_n * BLOCK_N:
        return

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(N, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * N + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        mask_k = offs_k < N
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)

    mask_m = offs_m < N
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)


def run(A, B):
    N = A.shape[0]
    C = torch.empty((N, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](A, B, C, N)
    return torch.tril(C)
