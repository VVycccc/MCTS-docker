import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SPLIT_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Interleaved split-K: each pid_k handles every SPLIT_K-th block of K
    # This ensures all K elements are covered regardless of divisibility
    for k_start in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.atomic_add(c_ptrs, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    SPLIT_K = 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), SPLIT_K)
    matmul_kernel[grid](A, B, C, M, N, K,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                        SPLIT_K=SPLIT_K, num_stages=3, num_warps=4)
    return C
