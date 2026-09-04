import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

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

    a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_mask = k * BLOCK_K + offs_k < K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def run(A, B):
    B_b, M, K = A.shape
    B_b2, K2, N = B.shape
    assert B_b == B_b2 and K == K2
    
    C = torch.empty((B_b, M, N), device=A.device, dtype=A.dtype)

    # 转 fp16 走 FP16 Tensor Core 并减半访存带宽（fp32 累加保持精度）
    A_h = A.to(torch.float16)
    B_h = B.to(torch.float16)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n, B_b)
    
    bmm_kernel[grid](
        A_h, B_h, C,
        M, N, K,
        A_h.stride(0), A_h.stride(1), A_h.stride(2),
        B_h.stride(0), B_h.stride(1), B_h.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K,
        EVEN_K=K % BLOCK_K == 0,
        GROUP_M=4,
        num_stages=3,
        num_warps=8,
    )
    return C
