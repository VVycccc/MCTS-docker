import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_K': 32, 'num_warps': 16, 'num_stages': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = 8 * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * 8
    group_size_m = min(num_pid_m - first_pid_m, 8)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # masks for A
        a_mask_m = offs_m[:, None] < M
        a_mask_k = offs_k[None, :] < (K - k)
        a_mask = a_mask_m & a_mask_k
        # load A tile (fp16)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # masks for B
        b_mask_k = offs_k[:, None] < (K - k)
        b_mask_n = offs_n[None, :] < N
        b_mask = b_mask_k & b_mask_n
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

        # advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # store result (fp32)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(A, B):
    # ---- input preparation ----
    # Use the same dtype as reference for final output; kernel uses fp16 internally
    orig_dtype = A.dtype
    A_half = A.to(torch.float16)
    B_half = B.to(torch.float16)

    M, K = A_half.shape
    K_, N = B_half.shape
    assert K_ == K, "Inner dimensions must match"

    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # ---- kernel launch ----
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n,)

    matmul_kernel[grid](
        A_half, B_half, C,
        M, N, K,
        A_half.stride(0), A_half.stride(1),
        B_half.stride(0), B_half.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=128,
    )

    # ---- return in original dtype ----
    return C.to(orig_dtype)
