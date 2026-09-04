import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ak, stride_am,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A is (K, M): load tile as (BLOCK_M, BLOCK_K) so a[i,k] = A[k,i]
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is (K, N): load tile as (BLOCK_K, BLOCK_N) so b[k,j] = B[k,j]
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(B_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_idx = k * BLOCK_K + offs_k
            mask_k = k_idx < K
            a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(B_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask_mn)

def run(A, B):
    K, M = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    EVEN_K = (K % BLOCK_K == 0)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, EVEN_K=EVEN_K,
        num_warps=8, num_stages=3,
    )
    return C
