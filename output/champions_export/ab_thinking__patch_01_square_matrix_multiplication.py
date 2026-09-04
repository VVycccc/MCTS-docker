import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_rem = K - k * BLOCK_K
            mask_k = offs_k < k_rem
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def run(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # BLOCK_K=64 for higher arithmetic intensity per tl.dot (plan requirement)
    # BLOCK_N reduced 128->64 to fit shared memory with num_stages=3:
    # smem = (num_stages-1)*(BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*4B = 2*(128*64+64*64)*4 = 98304 < 101376
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
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
        num_stages=3, num_warps=8,
    )
    return C
