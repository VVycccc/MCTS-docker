# TLE (Triton Language Extensions) API 参考 — forge_tle 后端专用

> 仅在 `triton_backend: forge_tle` 时可用。TLE 扩展了 Triton，提供异步访存、显式共享内存控制、tile 级切片等原语，用于在多 tile 循环中重叠 load 与 compute 隐藏延迟。

## import 写法（必须，不要写 `import tle`）

```python
import triton.experimental.tle.language as tle
```

## 3090 (Ampere, SM 8.6) 可用 API

### 1. 异步加载 `tle.load`

`tl.load` 的扩展，增加 `is_async` 调度 hint。语义与 `tl.load` 完全一致，额外让编译器把该 load 标记为异步，便于与后续 compute 重叠。

```python
x = tle.load(ptr + offs, mask=mask, other=0.0, is_async=True)
```

适用：多 tile 循环里后续会被多次复用的全局内存读。边界 tile 仍需显式 `mask`/`other`。

### 2. 共享内存分配 `tle.gpu.alloc`

显式分配 shared / tensor memory buffer（返回 buffered_tensor，非普通 tensor）。

```python
smem = tle.gpu.alloc(
    [BLOCK_M, BLOCK_K],
    dtype=tl.float32,
    layout=None,
    scope=tle.gpu.smem,
    nv_mma_shared_layout=False,
)
```

- `scope=tle.gpu.smem`：shared memory（3090 每 block 上限 99 KB）
- `nv_mma_shared_layout=False`：非 MMA 布局（普通 smem 用）

### 3. 共享内存指针 `tle.gpu.local_ptr`

在 smem buffer 上构建任意形状指针 view，可喂给 `tl.load`/`tl.store`/`tl.atomic_*`。指针位于 shared memory 地址空间（LLVM addrspace=3）。

```python
rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
ptr = tle.gpu.local_ptr(smem, (rows, cols))
tile = tl.load(ptr)
```

- `indices=None` 时返回覆盖整个 buffer 的 full-view 指针。
- load-after-store hazard 由后端 pass `TleInsertLocalPointerBarriers` 自动插 barrier，一般无需手动加同步。

### 4. 内存拷贝 `tle.gpu.copy`

GMEM ↔ SMEM 显式搬运。

```python
tle.gpu.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_K])
```

### 5. Tile 级切片 `tle.extract_tile` / `tle.insert_tile`

在 register / smem 上切子 tile，做激活、量化、归一化等局部变换，省手写指针运算。

```python
sub = x.extract_tile(index=[1, 0], tile_shape=[2, 2])  # rows [2:4], cols [0:2]
sub = tl.maximum(sub, 0.0)
x = x.insert_tile(sub, index=[1, 0])
```

### 6. 独占 cumsum `tle.cumsum`

```python
exclusive_sum, total_sum = tle.cumsum(x, axis=0)
```

## 3090 不可用 API（无对应硬件，用了会编译失败）

| API | 不可用原因 |
|-----|-----------|
| `tle.pipe` / `tle.pipe.writer`/`reader` | 依赖 Hopper+ NVWS token / mbarrier |
| `tle.gpu.warp_specialize` | 依赖 Hopper warp-group + NVWS |
| `tle.distributed_dot` | 依赖 Hopper Thread Block Cluster + DSMEM |
| TMEM / TMA 相关 | Hopper+/Blackwell 才有 |

**不要使用上述 API。** 3090 上只能用上面的 1-6 项。

## 范例：GEMM K-loop 用 TLE 异步 smem 缓冲

参考 `FlagTree/python/test/tle/integration/test_tle_gemm.py`，核心模式：分配 smem buffer → local_ptr 建 view → K 循环内 copy/load → dot 累加。

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
    # smem 缓冲：把下一个 K-tile 的 A/B 预取到 shared memory
    a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.float32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    col_ids_k = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (row_ids, col_ids_k))

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        # 异步搬入 smem
        tle.gpu.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_ptrs, b_smem, [BLOCK_K, BLOCK_N])
        a_tile = tl.load(a_smem_ptrs)
        b_tile = tl.load(tle.gpu.local_ptr(b_smem, (
            tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N)),
            tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N)))))
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## 何时用 TLE（⑥ timing_overlap 的 forge_tle 实现）

TLE 异步访存是方向 ⑥ timing_overlap 在 forge_tle 后端（FlagTree triton 3.6）下的具体实现手段。stock triton（forge）下 ⑥ 退化为 `num_stages`/`double_buffer`。

- **适用**：含多 tile 循环且循环体内 load+compute 可重叠 —— GEMM/matmul（K 维循环）、attention（QK^T→softmax→AV）、conv im2col+GEMM。
- **不适用（实测无收益甚至变慢）**：单 block reduction（softmax level1、layernorm、sum/max reduction，load 与 compute 强依赖无法重叠）、纯 elementwise、单 tile matvec。

## num_warps 强制规则（forge_tle / FlagTree triton 3.6 专属）

FlagTree triton 3.6.0 在同进程连续编译不同 BLOCK 的 kernel 时存在编译器 bug（可能返回错误结果）。**所有 `@triton.jit` kernel 启动必须显式指定 `num_warps`**，按 BLOCK_SIZE 推断：

| BLOCK_SIZE | num_warps |
|-----------|-----------|
| ≤ 1024    | 1         |
| ≤ 2048    | 2         |
| ≤ 4096    | 4         |
| ≤ 8192    | 8         |
| > 8192    | 16        |

启动写法：`kernel[grid](...args..., BLOCK=block, num_warps=4)`。后端会在编译前 AST 检测，未指定 num_warps 的启动会自动注入默认值，但仍建议你显式写对。
