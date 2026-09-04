import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K + offs_k
        if EVEN_K:
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_a_m + k_off[None, :] * stride_a_k,
                mask=mask_m[:, None], other=0.0,
            )
            b = tl.load(
                b_ptr + k_off[:, None] * stride_b_k + offs_n[None, :] * stride_b_n,
                mask=mask_n[None, :], other=0.0,
            )
        else:
            k_mask = k_off < K
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_a_m + k_off[None, :] * stride_a_k,
                mask=mask_m[:, None] & k_mask[None, :], other=0.0,
            )
            b = tl.load(
                b_ptr + k_off[:, None] * stride_b_k + offs_n[None, :] * stride_b_n,
                mask=k_mask[:, None] & mask_n[None, :], other=0.0,
            )
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def run(A, B):
    K, M = A.shape
    N, K2 = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    EVEN_K = K % BLOCK_K == 0
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(1), A.stride(0),
        B.stride(1), B.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, EVEN_K=EVEN_K,
        num_stages=3, num_warps=8,
    )
    return C
