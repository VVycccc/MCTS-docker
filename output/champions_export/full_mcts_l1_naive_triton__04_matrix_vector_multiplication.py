import torch
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        if EVEN_K:
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b = tl.load(b_ptr + offs_k * stride_bk)
        else:
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptr + offs_k * stride_bk, mask=k_mask, other=0.0)
        acc += tl.sum(a * b[None, :], axis=1)

    tl.store(c_ptr + offs_m, acc.to(c_ptr.dtype.element_ty), mask=mask_m)

def run(A, B):
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_M = 32
    BLOCK_K = 256
    grid = (triton.cdiv(M, BLOCK_M),)
    matvec_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        EVEN_K=K % BLOCK_K == 0,
        num_stages=4,
        num_warps=4,
    )
    return C
