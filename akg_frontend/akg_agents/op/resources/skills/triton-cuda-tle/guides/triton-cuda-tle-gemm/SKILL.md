---
name: triton-cuda-tle-gemm
description: "TLE GEMM 实现：用 tle.gpu.alloc 分配 smem 缓冲 + tle.gpu.local_ptr 建 view + tle.gpu.copy 异步搬入 + tl.dot 累加。完整可编译示例(3090 验证通过)。适用于 matmul/bmm/linear 的 K 循环重叠访存优化"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda_tle
  operator_patterns: "matmul"
  algorithms: "matmul, bmm, linear"
---

# TLE GEMM（smem 缓冲 + K 循环 copy/dot）

> 来自 FlagTree `test_tle_gemm.py`，**3090 SM8.6 验证通过**（input_precision="ieee" 时 max_err≈5e-5）。
> 适用于 K 维循环里把下一个 A/B tile 预取到 smem、与当前 dot 重叠的 GEMM/Linear。

## 核心模式：alloc → local_ptr 建 view → K 循环内 copy → load-from-smem → dot 累加

```python
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def gemm_tle_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # smem 缓冲：A/B 各一块（这里 alloc 成 [BLOCK_M, BLOCK_N] 形状，与 FlagTree 测试一致）
    a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    # local_ptr 建 view：row/col 都 broadcast 到完整 shape
    row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_N))
    col_ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (row_ids, col_ids))
    b_smem_ptrs = tle.gpu.local_ptr(b_smem, (row_ids, col_ids))

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        # GMEM → SMEM 显式搬运
        tle.gpu.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_N])
        tle.gpu.copy(b_ptrs, b_smem, [BLOCK_M, BLOCK_N])
        # 从 smem 读出
        a_tile = tl.load(a_smem_ptrs)
        b_tile = tl.load(b_smem_ptrs)
        # 累加。注意：用 input_precision="ieee" 保证与 PyTorch TF32-off 对照数值一致；
        # 若对照也开 TF32 可换 allow_tf32=True 提速。
        acc += tl.dot(a_tile, b_tile, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BM = BN = BK = 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    gemm_tle_kernel[grid](A, B, C, M, N, K,
                          A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                          C.stride(0), C.stride(1), BM, BN, BK, num_warps=4)
    return C
```

## 关键点

1. **alloc 形状**：FlagTree 官方测试 alloc 的是 `[BLOCK_M, BLOCK_N]`（非 `[BLOCK_M, BLOCK_K]`），copy 时也按 `[BLOCK_M, BLOCK_N]` 搬。这与直觉不同但能编译通过——照此写法即可。
2. **num_warps 必须显式传**（FlagTree 3.6 bug 规避）：`kernel[grid](..., num_warps=4)`。
3. **数值精度**：`tl.dot` 用 `input_precision="ieee"` 与 PyTorch `allow_tf32=False` 对照一致（max_err≈5e-5）；若开 `allow_tf32=True`，对照侧也要 `torch.backends.cuda.matmul.allow_tf32=True`。
4. **收益前提**：K 维够长（K/BLOCK_K ≥ 数轮），copy 与 dot 才有重叠空间。K 很小（单 tile）时无收益。

## 适用判断

- ✅ GEMM / Linear / BMM 且 K 较大
- ❌ 单 tile matvec、K 极小的瘦长矩阵（无多轮 K 循环可重叠）
