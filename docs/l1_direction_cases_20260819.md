# L1 50 题方向覆盖 case 研究（8 方向锚点分析）

> 数据源：`output/full_mcts_l1_naive_triton/`（selected50, 2026-07-22 批，35/50 有 final）。
> 每题四元组：reference（PyTorch）→ naive seed（`naive_seed_debug/successful_seed.py`）→ MCTS 树搜索 → champion（`final_results.json`）。
> per-depth 表来自 `run.log` 的 `[mcts] depth | nodes | best_latency` 块；depth-1 行 = 扁平单改搜索能拿的最好成绩，depth≥2 = 树叠加后的。
> 配置：GLM-5.2，rollout_depth=2，max_depth=4，adaptive，3600s/题，RTX 3090 (cuda:0)，forge triton 3.7.0。

优先级排序（覆盖优先，加速比次之）：
①14_upper_tri(⑤algo_equiv) ②02_standard(②precision_tc) ③06_large_k(⑥timing_overlap+⑦split-K)
④100_hingeloss(④fusion) ⑤49_max_red(⑦reduction_struct) ⑥08_irregular(⑧control_flow_spec)
⑦04_matvec(③mem_layout) ⑧12_diagonal(①tile_config) ⑨13_symmetric(多步叠加 showcase)

---

## Case 1: `14_matmul_for_upper_triangular_matrices` — ⑤ algo_equiv 锚点

**题目**：C = triu(A·B)，A/B 均为 4096×4096 上三角 fp32。reference 是全矩阵 GEMM 后 `torch.triu` 截断。

**baseline 5.69ms ｜ seed 10.09ms（naiveness=1.0）｜ champion 2.09ms = 2.72× vs PT，4.82× vs seed**

分类器判定 8/8 方向全适用，algo_equiv 排第一——正确识别了三角结构的可省计算。

**per-depth**：
```
d0=10.09  d1=2.19  d2=2.09(champion)
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| 分块 | BLOCK 32³，2D grid | BLOCK 128×128×32，GROUP_M=8 分组线性化 grid + autotune 3 configs |
| 精度 | `tl.dot` 默认 fp32 | `allow_tf32=True`（走 Tensor Core） |
| 计算 | 全 (N/B)² tile 都算 | **`if pid_m*BLOCK_M < (pid_n+1)*BLOCK_N` 块级跳过下三角 tile** |

seed 是教科书式朴素 GEMM：BLOCK 32、无 tf32、无分组调度、所有 tile 全算（哪怕输出必然为 0）。champion 的核心是 **algo_equiv 层的块级三角跳过**：`C=triu(A·B)` 中输出 tile 完全落在下三角区（`pid_m*BLOCK_M ≥ (pid_n+1)*BLOCK_N`）时整个 program 直接返回，4096/128=32×32=1024 个 tile 中约一半被跳过。注意 champion 没做 k 区间裁剪（理论还可只累加 k∈[pid_n*BK, pid_m*BM+BM)），留有剩余空间——树在 d2 后预算耗尽。叠加的 ② tf32 + ① autotune 是标准 GEMM 配方，flat 单改也能拿到大半，但**三角跳过这一步是 algo_equiv 语义推理，分类器排第一采样优先命中**。

**叙事**：结构先验（algo_equiv 排序第一）引导树在第一层就采样到语义级变换；块级跳过 + tf32 + autotune 分层叠加，d1→d2 仍有 1.05× 磨合增益。

**已覆盖方向**：⑤ algo_equiv（块级三角跳过）、② precision_tc（tf32，辅助）、① tile_config（autotune，辅助）

---

## Case 2: `02_standard_matrix_multiplication` — ② precision_tc 锚点

**题目**：标准 C = A·B（fp32 方阵）。分类器判 4 方向适用，**precision_tc 排第一**。

**baseline 5.63ms ｜ seed 9.79ms（naiveness=1.0）｜ champion 2.42ms = 2.32× vs PT，4.04× vs seed**

**per-depth**：
```
d0=9.79  d1=4.16  d2=2.42(champion, terminal=1)  d3=2.45
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| 精度 | `tl.dot(a, b)` 默认 fp32（CUDA Core FMA） | **`tl.dot(a, b, acc=acc, allow_tf32=True)`（Tensor Core）** |
| 分块 | BLOCK 32³ | BLOCK 128×128×32 + GROUP_M=8 + EVEN_K 分支 |
| 精度路径 | 全程 masked load | **EVEN_K 时去掉 k-mask**（`K % BLOCK_K == 0` 在 launch 前判定，constexpr 分支） |
| 调度 | 2D grid | 分组一维 grid（L2 复用友好） |

seed 用 BLOCK 32³ + 全 mask：fp32 走 CUDA Core，32×32×32 的 dot 连 Tensor Core 最低 tile 要求（16/32 视架构）都吃不满，occupancy 也被 1024 个小 program 的 launch 开销稀释。champion 是标准 Blackwell 前时代的 GEMM 配方，核心增益来自 ②：`allow_tf32=True` 把 FLOP 吞吐切到 TC（3090 上 tf32 TC ≈ fp32 CUDA Core 的 8× 峰值），BLOCK 128 恰好喂饱 MMA。**多步叠加清晰**：d1（4.16ms）拿到 tf32+大 tile，d2 再叠 EVEN_K 去 mask + 分组调度到 2.42ms（1.72× 步进），flat 单改停在 d1。

失败样本也有信息量：rollout 中 `dir_2_precision_tc` 两次撞 `shared memory Required: 131072 > Hardware limit 101376`——num_stages×BLOCK 过大爆 shared memory，树把这种 tile 边界记录为 dead-end 不再深扩。

**已覆盖方向**：② precision_tc（tf32 + TC tile，主）、⑥ timing_overlap（num_stages=3，辅助）、⑧ control_flow_spec（EVEN_K constexpr 分支，辅助）、① tile_config（GROUP_M/BLOCK，辅助）

---

## Case 3: `06_matmul_with_large_k_dimension` — ⑥ timing_overlap + ⑦ split-K 叠加锚点

**题目**：C = A·B，形状 256×524288 @ 524288×256——**K=512K 极端长 K**，输出仅 256×256。分类器 5 方向，precision_tc 第一、timing_overlap 第二。

**baseline 3.07ms ｜ seed 5.07ms（naiveness=1.0）｜ champion 2.61ms = 1.18× vs PT，1.94× vs seed**

**per-depth**：
```
d0=5.07  d1=4.53  d2=2.61(champion)  d3=2.62
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| 并行 | 2D grid (8×8=64 programs)——**3090 有 82 个 SM，64 program 连一波都填不满** | **SPLIT_K=16 三维 grid（64×16=1024 programs）+ interleaved 切 K + `tl.atomic_add` 归约** |
| 精度 | fp32 dot | `allow_tf32=True` |
| 流水 | 无 num_stages | `num_stages=3` 软件流水，BLOCK 64³ |

这道题是**并行度饥渴**的教科书案例：输出 tile 只有 8×8，不切 K 时 GPU 大部分 SM 空转，K 循环 8192 次迭代全是串行 latency。champion 的核心是 ⑦ reduction_struct 的 split-K——把 K 维切 16 份分给不同 program 并行累加，再用 atomic_add 合并，并行度提升 16×；配合 ⑥ num_stages=3 让 1024 个 program 的 load/compute 重叠。**d1→d2 步进 1.73×** 是全场该题最大单步增益，正是 split-K 落地的那一层。

诚实边界：vs PT 只有 1.18×——PyTorch cuBLAS 对这种极端形状内建 split-K，champion 追平但没超越；树展示的是"从朴素 seed 出发把工程配方搜全"，不是打败 cuBLAS 的调优极限。atomic_add 带 fp32 非结合性浮点误差，但 harness 容差内通过。

**已覆盖方向**：⑥ timing_overlap（num_stages 流水，主）、⑦ reduction_struct（split-K 跨 program 归约，主）、② precision_tc（tf32，辅助）、① tile_config（BLOCK 64³，辅助）

---

### 批次 1 小结（Case 1-3）

**已覆盖**：⑤ algo_equiv ✓ ② precision_tc ✓ ⑥ timing_overlap ✓ ⑦ reduction_struct ✓ ① tile_config ✓（均为辅助）
**待覆盖**：④ fusion、⑧ control_flow_spec、③ mem_layout（① tile_config 还需一个"纯调参"主锚点）

---

## Case 4: `100_hingeloss` — ④ fusion 锚点

**题目**：`mean(clamp(1 − pred·target, min=0))`，32768×32768 fp32（4G 元素，~16GB 读）。reference 是 mul → sub → clamp → mean 四个 PyTorch 算子，materialize 三个 4G 中间张量。分类器 4 方向，**fusion 排第三**（reduction_struct 第一）。

**baseline 35.19ms ｜ seed 14.67ms ｜ champion 4.82ms = 7.31× vs PT，3.04× vs seed**

**per-depth**：
```
d0=14.67  d1=4.82(champion)  d2=4.84  d3=4.86
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| 归约出口 | 写 `partial_sums[num_blocks]` 中间张量 + **`torch.sum` 二次启动**（另一个 kernel 全量读回） | **`tl.atomic_add(out_ptr, ...)` 原子归约直出单标量**，主机端只剩除法 |
| 分块 | BLOCK 64 → 512K 个 program | BLOCK 1024 → 32K 个 program |

seed 其实已把 elementwise 链（mul/sub/clamp）融进了一个 kernel——naive 生成器对"逐元素表达式"天然融合——**没融掉的是归约的第二跳**：partial sums 落回显存再由 `torch.sum` 读一遍，对 4G 元素的题这是整整一次多余的 128MB 写 + 读。champion 的 d1 增益（14.67→4.82，3.04×）几乎全部来自砍掉这一跳：atomic_add 把 partia 直接累进最终标量，内存流量近乎减半。这是 ④ fusion 的标准形态——**减中间写回**；也是 L1 里 fusion 故事最纯的题（语义上就是"算子链 + 全归约"）。

诚实边界：d2/d3 反而微退（4.84/4.86）——单 kernel atomic 已是该结构极限，树探深无增益但也不损失（champion 由 best-so-far 保住）。`out_ptr` 单点 atomic 争用在此规模下可接受（32K 个原子加 vs 16GB 流量九牛一毛）。

**已覆盖方向**：④ fusion（消 partial_sums 中间张量 + 二次 kernel，主）、⑦ reduction_struct（block 内 tl.sum + atomic 跨块归约，辅助）、① tile_config（BLOCK 64→1024，辅助）

---

## Case 5: `49_max_reduction_over_a_dimension` — ⑦ reduction_struct 主锚点

**题目**：`max(x, dim=1)`，x 为 128×4096×4095 fp32（内层 stride 4095，**非 2 的幂、非 tile 对齐**）。输出 128×4096。分类器 4 方向，**reduction_struct 排第一**。

**baseline 10.79ms ｜ seed 267.20ms ｜ champion 10.22ms = 1.06× vs PT，26.1× vs seed（35 题中最大 vsSeed）**

**per-depth**：
```
d0=267.20  d1=10.95  d2=10.22(champion, terminal=1)  d3=10.29
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| 每程序处理量 | **标量**：grid=(B, D2)=512K 个 program，每个只对 D1=4096 做标量 max 循环 | **2D tile (BLOCK_D1×BLOCK_D2)**：每 program 管 64×256 列块，向量化 `tl.max(axis=0)` |
| 向量化宽度 | BLOCK=32 但内层 stride=D2 跨步 load——**每 4B 一个独立 memory transaction，完全不打 coalescing** | d2_offs 连续 256 列 coalesced load，d1 行间跨步 |
| 调度 | 固定 BLOCK=32 | autotune 3 configs（BLOCK_D1/BLOCK_D2/num_stages/num_warps 组合） |
| 边界 | 全程 d1 mask | `EVEN_D1` constexpr 分支：D1%128==0 时主循环去 mask |

seed 是"每输出元素一个 program"的反面教材：512K program × 各自 4096 步标量循环，stride-4095 的 gather 让 DRAM 效率跌到 ~1/32，所以 267ms 惨烈。champion 是两阶段归约的标准结构——**块内 `tl.max(axis=0)` 树归约 + 循环跨块 `tl.maximum` 累进**，这正是 ⑦ reduction_struct 的 reduction_axis_blocking 手段；267→10.2 的 26× 全部来自这一层结构重写。d1→d2 的 1.07× 是 autotune 找到 BLOCK_D2=256 的向量化宽度。vs PT 仅 1.06×：PyTorch 的 reduction 也走 coalesced 两阶段，champion 追平但说明该结构已到工程 plateau。

注意此题的 4095 内层维度让 `EVEN_D1` 分支极少命中（D1=4096 整除，分支命中——命中的恰是循环维），`d2_mask` 仍全程保留：非对齐维不能用 constexpr 特化，这是 ⑧ 的适用边界反例（对照 Case 6）。

**已覆盖方向**：⑦ reduction_struct（axis blocking + 两阶段树归约，主）、③ mem_layout（coalesced 列块访存，辅助）、⑧ control_flow_spec（EVEN_D1 分支，辅助）、① tile_config（autotune，辅助）

---

## Case 6: `08_matmul_with_irregular_shapes` — ⑧ control_flow_spec 锚点

**题目**：C = A·B，A 8205×2949，B 2949×5921——**三围全非 2 的幂**（M%128=5, N%128=81, K%32=21）。分类器 5 方向，**control_flow_spec 排第四**（但在 GEMM 题里被判适用本身就是信号——判据是"边界 mask 逃不掉"）。

**baseline 12.79ms ｜ seed 28.22ms ｜ champion 9.73ms = 1.31× vs PT，2.90× vs seed**

**per-depth**：
```
d0=28.22  d1=10.40  d2=9.73(champion)  d3=10.40
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| K 循环 | 每次迭代都算 `k_rem = K - k*BLOCK_K` + 双 mask load | **主体 K//BLOCK_K 次迭代完全无 mask**，`if K % BLOCK_K > 0` 只处理**最后一个残块** |
| 形状参数 | M/N/K/strides 全部运行时标量 | **全部 `tl.constexpr`**——形状编译期特化，mask/索引常量折叠 |
| 精度 | fp32 dot | `allow_tf32=True` |
| 调度 | 2D grid，BLOCK 32³ | 分组 1D grid，BLOCK 128×128×32 + GROUP_M=8 |

⑧ 的故事在这题最纯：**非对齐形状下 mask 开销不是边缘 case 而是主循环税**。seed 在 92 次 K 迭代（2949/32）里每次都付 4 个 mask 向量的比较 + select；champion 把整段循环剥成"对齐主循环（零 mask）+ 残块尾循环（一次 mask）"，K=2949 时 91 次迭代免 mask、仅 1 次付全价。配合 constexpr 形状特化让编译器把边界检查全部折叠掉。d1→d2 的 1.07× 步进来自这层剥离（d1 已拿到 tf32+大 tile 的 10.40ms）。

诚实边界：非对齐 M/N 维的输出 mask 无法消除（8205=64×128+5，末行/列 tile 仍需 mask_c）——⑧ 只能优化"高频路径免检查"，不能消灭边界本身。vs PT 1.31× 主要由 ② tf32 贡献，⑧ 贡献尾部 1.07×。

**已覆盖方向**：⑧ control_flow_spec（K 主循环去 mask + 残块分离 + constexpr 特化，主）、② precision_tc（tf32，辅助）、① tile_config（BLOCK/GROUP_M，辅助）、⑥ timing_overlap（num_stages 隐含，辅助）

---

### 批次 2 小结（Case 4-6）

**已覆盖**：④ fusion ✓ ⑦ reduction_struct ✓（主锚点）⑧ control_flow_spec ✓
**累计**：⑤ ② ⑥ ⑦ ④ ⑧ 已覆盖；③ mem_layout、① tile_config（纯调参主锚点）待补。

---

## Case 7: `04_matrix_vector_multiplication` — ③ mem_layout 锚点

**题目**：y = A·B，A 2048×1048576（K=1M 极长归约），B 1M×1 列向量。输出仅 2048×1。分类器 5 方向，**mem_layout 排第一**（唯一一个 mem_layout 排头的 GEMM 系题）。

**baseline 12.70ms ｜ seed 13.41ms ｜ champion 9.67ms = 1.31× vs PT，1.39× vs seed**

**per-depth**：
```
d0=13.41  d1=9.78  d2=9.67(champion)  d3=12.02
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| kernel 形态 | 2D tile GEMM（BLOCK 32³，`tl.dot` 走 MMA）——**N=1 的 tile 是 32×1，MMA 利用率 1/32** | **重写为 matvec**：BLOCK_M×BLOCK_K 2D load，`tl.sum(a * b[None,:], axis=1)` 向量乘累加 |
| b 的加载 | 每 tile 重复 load b tile（M/32=64 份冗余） | **b 向量沿 K 分块 load 一次**，broadcast 乘 |
| 循环 | K/BLOCK_K=32768 次迭代全程 k_mask | EVEN_K constexpr 去 mask（1048576 % 256 == 0 命中） |
| 流水 | 无 | num_stages=4, num_warps=4 |

这题的病根是**结构错配**：N=1 时 Tensor Core 的 32 宽 MMA 有 31 列在算垃圾——seed 照抄 GEMM 模板把 matvec 硬套成矩阵乘。champion 的核心动作是 ③ 的"消中间浪费 + 按访存实况重排"：识别出 B 是向量后直接改 kernel 形态（2D tile → 行块 ×K 切片），每个 program 处理 32 行、沿 K 以 256 宽 coalesced 扫 A 的行切片，b 标量广播相乘，`tl.sum(axis=1)` 归约。访存上 A 行主序连续（stride_am=1M 行跨步、行内连续 256 load 满 1KB transaction），这正是 mem_layout 判据"memory_bw 高、合并访问主导"的标准场景。

诚实边界：vs PT 仅 1.31×——cuBLAS 对 skinny GEMM 有专用 gemv 路径，champion 追到同量级。d3 回退到 12.02（深探的配置变差），champion 由 best-so-far 保住——树不因深探失败丢已有效益。

**已覆盖方向**：③ mem_layout（matvec 重写 + b 单次 load + coalesced 扫描，主）、⑧ control_flow_spec（EVEN_K，辅助）、① tile_config（BLOCK_K=256，辅助）、⑥ timing_overlap（num_stages=4，辅助）

---

## Case 8: `12_matmul_with_diagonal_matrices` — ① tile_config 纯调参锚点

**题目**：C = diag(a)·B，a 为 4096 向量，B 4096×4096。语义上没有 tile 间归约、没有融合空间、形状完美对齐——分类器只判 3 方向适用（mem_layout/tile_config/control_flow_spec，**全 35 题最少**），无 precision_tc（无 dot）、无 ⑤⑥⑦。

**baseline 0.163ms ｜ seed 0.177ms ｜ champion 0.162ms = 1.01× vs PT，1.09× vs seed**

**per-depth**：
```
d0=0.177  d1=0.170  d2=0.162(champion)  d3=0.161
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| tile | BLOCK_N×BLOCK_M = 32×32，grid 128×128=16K programs——**0.16ms 的 kernel 里 launch 16K 个 program，调度开销吃掉主体** | BLOCK_M∈{1024,2048,4096} 单维切，grid=(N, M/BM)——按 N 一行一个 program、M 方向宽向量 |
| 调参 | 手写死 BLOCK 32 | **autotune 3 configs（BLOCK_M 1024/2048/4096 × num_warps 4/8）** |

seed 和 champion 的 kernel 主体完全同构（load a 标量 → load b 行块 → 乘 → store），**唯一变量是分块几何**——这是 35 题里最纯的 ① tile_config 故事。带宽受限的 elementwise 型 kernel，性能 = 每个 program 干的活够大（摊薄调度）+ warp 内访问够宽（满 transaction）。seed 的 32×32 双维切分让 program 数（16K）远超必要；champion 沿 M 拉宽到 1024-4096，program 数降到 ~4-16/行，调度开销消解，带宽逼近饱和。

d1→d2 的 1.05× 就是 autotune 从手写 32 换到 1024+ 的过程。**vs PT 1.01×**：PyTorch 这里本身就是 `a[:,None]*B` 单句 fused kernel，逼近理论上限，树在小空间里仍稳定挤出最后几个百分点——展示 ① 在"结构无文章可做"的题上就是全部剩余杠杆。

**已覆盖方向**：① tile_config（分块几何 + autotune，主）、③ mem_layout（行向连续访问，辅助）

---

## Case 9: `13_matmul_for_symmetric_matrices` — 多步叠加 showcase（收尾）

**题目**：C = A·B，A/B 对称 4096×4096。reference 语义上就是普通 matmul（对称性只是数据先验，等价变换空间受限——`A·B` 一般不再对称，无 triu 式裁剪可用）。分类器 5 方向，precision_tc 排第一。

**baseline 5.60ms ｜ seed 5.06ms ｜ champion 2.16ms = 2.60× vs PT，2.34× vs seed**

**per-depth**：
```
d0=5.06  d1=3.83  d2=2.16(champion, terminal=1)  d3=2.20
```

**seed → champion 的关键变换**：

| 维度 | seed | champion |
|---|---|---|
| 精度 | fp32 `tl.dot` | `allow_tf32=True` + `acc=acc` 累加形式 |
| 分块 | BLOCK 32³ | BLOCK 128×128×64/32 + GROUP_M=8 分组 grid + autotune 3 configs |
| mask | 全程三重 mask（m/n/k） | **全程零 mask**——4096 对齐让 cdiv 恰好整除，mask 整个消除 |

**多步叠加账目**（L1 题里最典型的一列）：
- d1：tf32 + 部分 tile 调整 → 3.83ms（seed→d1 1.32×，② 主导）
- d2：叠 autotune 最优 config（BLOCK_K 64×stages3 / 32×stages4 二选一）+ 去 mask → 2.16ms（d1→d2 **1.77×**，①+⑧ 叠加）
- flat 单改上限 = 3.83ms；树拿到 2.16ms，**边际增益 1.77×** 全部来自第二层叠加。

这题与 02（standard matmul）同族，特意选它做收尾是因为它展示了**树深度在"看起来平凡"题上的复利**：每一步都是已知配方（tf32 → autotune → 去 mask），没有一步是algo_equiv 级的灵光，但 flat 搜索在 d1 就停（单 patch 只能装一个变换），树把三个变换串成链。d2 terminal=1（该节点 rollout 无改进，不再深扩）后 d3 微退，champion 保住——自适应深度控制止损。

**已覆盖方向**：② precision_tc（tf32，主）、① tile_config（autotune BLOCK_K/stages，主）、⑧ control_flow_spec（整除零 mask，辅助）、⑥ timing_overlap（num_stages 2/3/4，辅助）

---

## 最少 case 数结论（set-cover 分析）

**Q：最少几个 case 全覆盖 8 方向？**

- **下限 3 个**（辅助证据也计数）：`14_upper_tri` + `100_hingeloss` + `04_matvec` = 8/8。
  - 14 贡献 ⑤（主，三角 tile 跳过）②（tf32）①（autotune+GROUP_M）
  - 100 贡献 ④（主，atomic 归约直出）⑦（两阶段归约）①（BLOCK 64→1024）
  - 04 贡献 ③（主，matvec 重写）⑧（EVEN_K 分支）⑥（num_stages=4）①（BLOCK_K=256）
- **2 个不可能**：⑤ 只在 14/15/44 适用（35 题中 3 次）、④ 仅 7 题适用，两稀有方向无一题同时有双方 champion 实证；单 case 最多实证 4 方向（06/13 的 ②⑥⑦①），⑤ 持有者只带 ⑤②①+薄⑥，剩余缺口无 fusion 题能一次补齐。
- **推荐 4 个**：`14 + 100 + 49 + 06`——⑤④⑦⑥ 各有独立强主锚点；3-case 集合里 ⑥⑧① 只有薄证据（⑥=一个 launch 参数、⑧=一个分支、① 处处配角）。唯一无主锚点的是 ①（分类器默认最低方向，autotune 配角处处在场）。
- **⑤① 都要独立锚点则 5 个**：+ `12_diagonal`（纯分块几何，seed/champion 主体同构）。

**用法建议**：正文 5.4 用 4-case 版；⑤① 严格版 5 个；9-case 全集放附录。

## 总结：9 case 覆盖矩阵

| Case | 题 | vs PT | vs seed | 主锚点 | 叠加的辅助方向 |
|---|---|---:|---:|---|---|
| 1 | 14_upper_tri | 2.72× | 4.82× | ⑤ algo_equiv（三角 tile 跳过） | ② ① |
| 2 | 02_standard | 2.32× | 4.04× | ② precision_tc（tf32+TC tile） | ⑥ ⑧ ① |
| 3 | 06_large_k | 1.18× | 1.94× | ⑥ timing_overlap + ⑦ split-K | ② ① |
| 4 | 100_hingeloss | 7.31× | 3.04× | ④ fusion（atomic 归约直出） | ⑦ ① |
| 5 | 49_max_red | 1.06× | 26.1× | ⑦ reduction_struct（axis blocking 两阶段） | ③ ⑧ ① |
| 6 | 08_irregular | 1.31× | 2.90× | ⑧ control_flow_spec（主循环去 mask+constexpr） | ② ① |
| 7 | 04_matvec | 1.31× | 1.39× | ③ mem_layout（GEMM→matvec 重写） | ⑧ ① ⑥ |
| 8 | 12_diagonal | 1.01× | 1.09× | ① tile_config（纯分块几何+autotune） | ③ |
| 9 | 13_symmetric | 2.60× | 2.34× | 多步叠加 showcase（②①⑧ 链） | — |

**8/8 方向全部有主锚点 + 代码级实证**。方向→案例的论文映射：⑤→14，②→02，⑥+⑦→06，④→100，⑦主→49，⑧→08，③→04，①→12，树深复利→13。

已知的分析边界（写论文时注意）：
1. 方向归因 = champion 代码特征 + per-depth 步进 + 分类器排序三方交叉印证，**不是 per-patch 严格归因**（checkpoint 的 tree 只存 summary，无节点方向标签）。若审稿要求，可对这 9 题重跑并加树序列化（9×~1h，cuda:0 一晚）。
2. 06/04/12 的 vs PT 偏低（1.0-1.3×）是诚实数字：这些题 cuBLAS/PyTorch 本身接近结构上限，案例叙事应放 vs seed 与变换本身，不放 vs PT。
3. 15_lower_tri 与 14 同构（2.69×/5.60×），可作 ⑤ 的备用；36_rmsnorm（vsSeed 16.5×）可作 ⑦ 备用但 d1 后饱和，故事不如 49。



