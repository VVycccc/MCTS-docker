import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_start * BLOCK_K + offs_k

        if EVEN_K:
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
            b = tl.load(b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        else:
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B):
    M, K = A.shape
    N, _ = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    EVEN_M = M % BLOCK_M == 0
    EVEN_N = N % BLOCK_N == 0
    EVEN_K = K % BLOCK_K == 0
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        EVEN_M, EVEN_N, EVEN_K,
        num_stages=3,
        num_warps=8,
    )
    return C
