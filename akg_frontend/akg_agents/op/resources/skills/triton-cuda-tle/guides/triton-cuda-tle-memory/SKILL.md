---
name: triton-cuda-tle-memory
description: "TLE 显式共享内存工作流：tle.gpu.alloc 分配 smem + tle.gpu.local_ptr 建 view + tle.gpu.copy 搬入 + tl.load/store 读写。含 elementwise 示例(3090 验证通过)。适用于 pool/conv/reduction/argmax 等需要显式 smem + 原子的算子"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda_tle
  operator_patterns: "elementwise, reduce, pool"
  algorithms: "shared-memory, atomic"
---

# TLE 显式共享内存工作流（3090 可用）

> 来自 FlagTree `test_tle_local_store.py`，**3090 SM8.6 验证通过**（elementwise add max_err=0）。
> 展示完整 TLE smem 流程：alloc → local_ptr 建 view → copy 搬入 → load 读 → store 写出。

## 完整模式：alloc → local_ptr → copy → load → compute → store

```python
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def add_tle_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # 1. 分配 smem buffer（返回 buffered_tensor，非普通 tensor）
    a_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)

    # 2. local_ptr 建 view：1D 用单元素 tuple (idx,)
    idx = tl.arange(0, BLOCK)
    a_ptrs = tle.gpu.local_ptr(a_smem, (idx,))
    b_ptrs = tle.gpu.local_ptr(b_smem, (idx,))

    # 3. GMEM → SMEM 显式搬运
    tle.gpu.copy(a_ptr + offs, a_smem, [BLOCK])
    tle.gpu.copy(b_ptr + offs, b_smem, [BLOCK])

    # 4. 从 smem 读出
    a_val = tl.load(a_ptrs)
    b_val = tl.load(b_ptrs)

    # 5. 计算 + 写回 GMEM
    tl.store(c_ptr + offs, a_val + b_val, mask=offs < n)


def run(A, B):
    n = A.numel()
    C = torch.empty_like(A)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_tle_kernel[grid](A, B, C, n, BLOCK=BLOCK, num_warps=4)
    return C
```

## local_ptr 的 load/store/atomic

`tle.gpu.local_ptr` 返回的指针可喂给：
- `tl.load(ptr)` / `tl.store(ptr, val)` —— 读写 smem
- `tl.atomic_add(ptr, val, sem="relaxed", scope="cta")` —— smem 上的 CTA 级原子（reduce/argmax/histogram 用）

## barrier（store-after-load hazard）

`TleInsertLocalPointerBarriers` 后端 pass 会自动在 smem 的 load-after-store 处插 barrier，一般无需手动同步。复杂序列（如 TopK 多阶段）可用 `tl.debug_barrier()` 显式插。

## 适用判断

- ✅ pool（窗口聚合到 smem 再 reduce）、conv（im2col tile 缓存到 smem）、reduction/argmax（smem 原子或树规约）
- ⚠️ 纯 elementwise 单次读写：smem 中转未必比直接 GMEM 快（多一次搬运）。示例只为演示 API，**真正 elementwise 不建议套 smem**
- ✅ 真正收益场景：同一数据被多次访问（窗口/多次 reduce），smem 缓存省重复 GMEM 读
