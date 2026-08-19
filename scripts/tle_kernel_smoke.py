"""验证 FlagTree TLE kernel 在 3090 (SM8.6) 上能编译 + 正确。
两个最小 kernel: GEMM (test_tle_gemm.py 抽取) + elementwise smem (test_tle_local_store.py 抽取)。
只验证「能跑通 + 数值对」,不测延迟(延迟留给后面的 stock vs tle 对比)。
"""
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


# ---------- 1. TLE GEMM (alloc + local_ptr + copy + dot) ----------
@triton.jit
def gemm_tle_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_N))
    col_ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (row_ids, col_ids))
    b_smem_ptrs = tle.gpu.local_ptr(b_smem, (row_ids, col_ids))
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        tle.gpu.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_N])
        tle.gpu.copy(b_ptrs, b_smem, [BLOCK_M, BLOCK_N])
        a_tile = tl.load(a_smem_ptrs)
        b_tile = tl.load(b_smem_ptrs)
        acc += tl.dot(a_tile, b_tile, input_precision="ieee")
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_gemm(A, B):
    M, K = A.shape; _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BM = BN = BK = 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    gemm_tle_kernel[grid](A, B, C, M, N, K,
                          A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                          C.stride(0), C.stride(1), BM, BN, BK, num_warps=4)
    return C


# ---------- 2. TLE elementwise add via smem (alloc + copy + local_ptr load/store) ----------
@triton.jit
def add_tle_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    idx = tl.arange(0, BLOCK)
    a_ptrs = tle.gpu.local_ptr(a_smem, (idx,))
    b_ptrs = tle.gpu.local_ptr(b_smem, (idx,))
    a_gptr = a_ptr + offs
    b_gptr = b_ptr + offs
    tle.gpu.copy(a_gptr, a_smem, [BLOCK])
    tle.gpu.copy(b_gptr, b_smem, [BLOCK])
    a_val = tl.load(a_ptrs)
    b_val = tl.load(b_ptrs)
    c_val = a_val + b_val
    tl.store(c_ptr + offs, c_val, mask=offs < n)


def run_add(A, B):
    n = A.numel()
    C = torch.empty_like(A)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_tle_kernel[grid](A, B, C, n, BLOCK=BLOCK, num_warps=4)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    # GEMM
    a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = run_gemm(a, b)
    ref = a @ b
    gemm_ok = torch.allclose(c, ref, atol=1e-4, rtol=1e-4)
    print(f"[GEMM]  max_err={ (c-ref).abs().max().item():.2e}  PASS={gemm_ok}")

    # elementwise add
    a2 = torch.randn(4096, device="cuda", dtype=torch.float32)
    b2 = torch.randn(4096, device="cuda", dtype=torch.float32)
    c2 = run_add(a2, b2)
    add_ok = torch.allclose(c2, a2 + b2, atol=1e-5, rtol=1e-5)
    print(f"[ADD]   max_err={ (c2-(a2+b2)).abs().max().item():.2e}  PASS={add_ok}")
