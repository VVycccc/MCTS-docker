import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_M < (pid_n + 1) * BLOCK_N:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * N + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k = tl.cdiv(N, BLOCK_K)
        for k in range(0, num_k):
            k_idx = k * BLOCK_K + offs_k
            a = tl.load(a_ptrs,
                        mask=(offs_m[:, None] < N) & (k_idx[None, :] < N),
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=(k_idx[:, None] < N) & (offs_n[None, :] < N),
                        other=0.0)
            acc = tl.dot(a, b, acc=acc, allow_tf32=True)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)


def run(A, B):
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((N, N), device=A.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](A, B, C, N)
    return torch.triu(C)
