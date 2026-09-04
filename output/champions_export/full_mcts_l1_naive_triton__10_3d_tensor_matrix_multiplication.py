import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_L': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_L': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_L': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['NM', 'K', 'L'],
)
@triton.jit
def matmul_3d_kernel(
    A_ptr, B_ptr, C_ptr,
    NM, K, L,
    stride_a_nm, stride_a_k,
    stride_b_k, stride_b_l,
    stride_c_nm, stride_c_l,
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(NM, BLOCK_M)
    num_pid_l = tl.cdiv(L, BLOCK_L)

    num_pid_in_group = GROUP_M * num_pid_l
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_l = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    mask_m = offs_m < NM
    mask_l = offs_l < L

    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_a_nm + offs_k[None, :] * stride_a_k
        b_ptrs = B_ptr + offs_k[:, None] * stride_b_k + offs_l[None, :] * stride_b_l

        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_l[None, :], other=0.0)
        else:
            mask_k = offs_k < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_l[None, :], other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    c_ptrs = C_ptr + offs_m[:, None] * stride_c_nm + offs_l[None, :] * stride_c_l
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_l[None, :])

def run(A, B):
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2

    A_2d = A.contiguous().reshape(-1, K)
    B_cont = B.contiguous()
    C = torch.empty((N * M, L), device=A.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N * M, meta['BLOCK_M']) * triton.cdiv(L, meta['BLOCK_L']),)

    matmul_3d_kernel[grid](
        A_2d, B_cont, C,
        N * M, K, L,
        A_2d.stride(0), A_2d.stride(1),
        B_cont.stride(0), B_cont.stride(1),
        C.stride(0), C.stride(1),
        EVEN_K=True,
    )

    return C.view(N, M, L)
