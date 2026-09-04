---
name: triton-cuda-tle-basics
description: "TLE (Triton Language Extensions, FlagTree fork) 基础。仅 forge_tle 后端可用。提供异步访存、显式共享内存控制、tile 级切片等原语。3090(Ampere)可用 API 速查与 import 写法"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda_tle
  operator_patterns: "all"
  algorithms: "tle, async-load, shared-memory"
---

# TLE 基础（仅 forge_tle 后端，3090 可用）

> TLE = Triton Language Extensions，来自 `flagos-ai/FlagTree`（triton fork 3.6.x）。stock Triton 无此扩展。

## import 写法（必须，不要写 `import tle`）

```python
import triton
import triton.language as tl
import triton.experimental.tle.language as tle  # 这一行
```

## 3090 (Ampere SM 8.6) 可用 API

| API | 用途 | 备注 |
|-----|------|------|
| `tle.load(ptr+offs, mask=, other=, is_async=True)` | 异步加载 hint | 语义同 `tl.load`，额外标异步 |
| `tle.gpu.alloc([shape], dtype=, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)` | 分配 smem buffer | 返回 buffered_tensor，非普通 tensor |
| `tle.gpu.local_ptr(buffer, (row_ids, col_ids))` | 在 smem buffer 上建指针 view | 喂给 `tl.load/store/atomic`，addrspace=3 |
| `tle.gpu.copy(gmem_ptrs, smem_buffer, [shape])` | GMEM→SMEM 显式搬运 | 单向 |
| `x.extract_tile(index=[i,j], tile_shape=[...])` / `x.insert_tile(sub, index=)` | register/smem 子 tile 切片 | 省手写指针 |
| `tle.cumsum(x, axis=0)` → `(exclusive_sum, total_sum)` | 独占 cumsum | TopK radix-select 用 |

## 3090 不可用 API（编译失败，勿用）

`tle.pipe` / `tle.pipe.writer/reader`、`tle.gpu.warp_specialize`、`tle.distributed_dot`、TMEM/TMA —— 依赖 Hopper+/Blackwell。

## local_ptr 索引规则（易错）

- `local_ptr(buf, indices)` 的 `indices` 必须是 **tuple/list**，元素数 == buffer 维度数。
- 1D buffer：`tle.gpu.local_ptr(buf, (arange,))`（单元素 tuple）。
- 2D buffer：`tle.gpu.local_ptr(buf, (row_ids, col_ids))`，两个都需 broadcast 到完整 shape。
- `indices=None` 返回覆盖整个 buffer 的 full-view 指针。

## num_warps 强制规则（FlagTree 3.6 专属）

FlagTree triton 3.6 连续编译不同 BLOCK 的 kernel 有编译器 bug（可能返回错误结果）。**所有 `@triton.jit` 启动必须显式 `num_warps`**：BLOCK≤1024→1 / ≤2048→2 / ≤4096→4 / ≤8192→8 / >8192→16。

## 何时收益 / 何时无收益

- **收益**：多 tile 循环且 load+compute 可重叠 —— GEMM(K循环)、attention(QK^T→softmax→AV)。
- **无收益甚至变慢**：单 block reduction（softmax/layernorm/sum-max reduction，load 与 compute 强依赖）、纯 elementwise、单 tile matvec。
