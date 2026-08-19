# DirecTune 项目方法论

> 用人类专家算子优化理论定位 DirecTune：它复现了什么、形式化了什么、差异化在哪、还差什么。用于对外讲述（汇报 / 论文 / 演讲）。

## 一句话定位

DirecTune 是把**人类专家的 profile-driven 算子优化闭环**用 LLM agent 自动化的系统：以 NCU 硬件反馈为信号，以"8 个优化方向"为招式库，以 beam search + 经验记忆替代人类专家的"直觉试错 + 个人经验"。

## 理论坐标：人类专家方法论的三层

| 层次 | 人类专家内容 | DirecTune 的位置 |
|---|---|---|
| **性能理论**（why slow/快） | Roofline model、算术强度、memory hierarchy、occupancy vs latency hiding —— 给出性能上限 | **部分复现**：NCU 指标 → bottleneck summary（`memory_bound` / `register_limited` / `tensor_core_underused` / `underutilized`）是 Roofline 分类的工程化。**缺**：算术强度解析预判、性能上限锚定 |
| **优化模式**（招式库） | fusion / tiling / coalescing / TC / reduction / pipelining / online algo / 等价变换 / autotune | **形式化为 5 方向**（见下），可定向采样 |
| **工程流程**（怎么做） | profile → roofline 定位 → 瓶颈归因 → 对症下药 → 验证 → 迭代 | **完整复现**（见下表逐步对应） |

## 8 个优化方向：人类专家招式库的形式化

人类专家脑中是隐性招式，DirecTune 把算子优化手段归纳为 5 个正交方向，覆盖优化空间：

| 方向 | 含义 | 典型手段 | NCU 信号 | 收益量级 |
|---|---|---|---|---|
| ① Tile & 配置 | 分块/并行配置 | BLOCK_M/N/K, num_warps, num_stages, GROUP_M, autotune | occupancy, stall | 1.2-2× |
| ② 精度 & Tensor Core | 用 TC | allow_tf32, fp16/bf16, dtype | tensor_core_util | 2-8×（compute-bound） |
| ③ 访存 & 布局 | 减带宽压力 | 合并访问, shared tiling, 转置, bank conflict, num_stages | memory_bw_util, l1/l2_hit, stall_memory | 1.5-3×（memory-bound） |
| ④ 算子融合 | 多 op 合一 | GEMM+激活+归约, 减中间写回 | memory_bw_util 降 | 1.3-2× |
| ⑤ 算法等价变换 | 算得更少 | GEMM+Sum→matvec, 预计算, online softmax, im2col | compute 量级降 | 3-8000× |

5 方向基本互斥（一个 patch 通常有一个主方向）。DirecTune 把它形式化为三件可计算的东西：

- **可判定的适用集**：方向分类器按算子语义 + NCU 瓶颈判定哪些方向适用（element-wise 跳 ②、memory-bound 跳 ②、compute-bound 跳 ③）
- **可定向采样的搜索分支**：每方向一个 beam 分支 + 1 个 free_explore 兜底
- **可去重的前沿**：同向 patch 去重，保证语义多样性，避免搜索同质化

这是相对人类专家的**结构化优势**：人容易陷在某一个招式里（反复调 tile），agent 的 beam 强制覆盖多个正交方向。

## 人类专家 7 步闭环 ↔ DirecTune 实现

| 人类专家步骤 | DirecTune 实现 | 代码位置 |
|---|---|---|
| 1. 基准测量 | baseline profile `def run` reference → baseline_latency + hw_metrics | `main.py` |
| 2. Profile（NCU） | 采集 `sm__cycles_active` / `dram throughput` / `l2_hit` / `stall_long_scoreboard` | `hardware_profiler.py` |
| 3. Roofline 定位 | bottleneck summary 归纳（memory / compute / latency bound） | `agents.py:_format_hw_profile` |
| 4. 瓶颈归因 | slow/fast profile 对比提炼"瓶颈 → 改动 → 指标变化 → 性能结果" | `summarizer` |
| 5. 对症下药 | `unified_editor` 一次推理内分析瓶颈 + 产 search/replace patch（按 5 方向定向采样） | `agents.py:unified_editor` |
| 6. 验证正确性 | `_verify_code` 共享关卡：anti_pytorch → compile → correctness → benchmark | `triton_backend.py` |
| 7. 迭代 | beam search 多轮迭代 + in-search experience 记忆（`update_experiences`） | `search.py` |

**关键论点**：流程层与人类专家**同构**——这是刻意复现，不是偶然。差异集中在"招式选择"和"经验积累"两处，用 LLM + 搜索替代人类直觉。

## agent 相对人类专家的差异化优势

1. **并行探索**：beam search 同时验证多个优化假设；人类专家串行试错
2. **经验可复现**：in-search experience 队列（`update_experiences`）跨 iter 累积优化经验、可复现；人类经验在脑子里、难迁移、易流失
3. **数学等价变换的系统性搜索**：14 题 GEMM+Sum→matvec（35.8×）——人类专家需灵光一现，agent 靠 ⑤ 方向定向采样 + 多轮搜索系统性覆盖
4. **不知疲倦的长尾投入**：多轮 beam search + `search_time_budget` 早停保 best-so-far，可在单题上持续迭代

## 当前 gap：人类专家会做、agent 还没做好

1. **算术强度预判**：人类专家先算"这题理论带宽下限、离屋顶多远"，对低天花板题（KernelBench L1 激活/pooling）直接放弃；agent 倾向盲目试 → 浪费算力。可加 pre-search 理论带宽估算模块跳过低天花板题
2. **性能上限锚定 / 防 reward hacking**：人类专家知道"这算子最多到 X"，看到异常高 speedup 会警觉；agent 看到 35× 就当成功。需要 NCU 指标作为 step-level 验证信号辅助判断
3. **⑤ 算法等价变换稳定性**：最高价值但靠 LLM 隐式知识，不稳定；人类专家有结构化"算子 → 等价高效算法"映射。缺结构化等价知识库（未来工作）
4. **working set 解析下界**：tiling 大小人类专家有解析约束（tile 须装进 shared mem），agent 靠 autotune 暴力搜

## 讲述用的核心数据（校准点）

| 题目 | 加速 | 主方向 | 说明 |
|---|---|---|---|
| 01_square_matrix_multiplication | 2.28× vs cuBLAS | ② 精度（allow_tf32） | 低垂果实，reference 默认没开 tf32 |
| 14_Gemm_Divide_Sum_Scaling | 35.8× vs generator 初始 | ⑤ 算法等价（GEMM+Sum→matvec） | "算得更少"非"算得更快"，计算量减 ~8000× |
| 55 | 0.86 → 1.66× | ①③ | 救活原本退化的慢 kernel |
| level2 geomean（10 题对拍） | 2.30× vs AKG 1.51×，胜 8/10 | ④ 融合为主 | 证明 search 阶段相对纯生成的增量价值 |

## 建议叙事线（讲述顺序）

1. **问题**：算子优化是人类专家的 profile-driven 闭环，依赖个人经验、难规模化、难复现
2. **理论坐标**：人类专家方法论三层（理论 / 模式 / 流程），DirecTune 复现流程层 + 形式化模式层
3. **做法**：NCU 反馈信号 + 5 方向招式库 + beam search + in-search 经验记忆
4. **差异化**：并行探索 / 经验可泛化 / 等价变换系统搜索 / 长尾投入
5. **结果**：上述校准数据（强调 ⑤ 的 35.8× 是"算得更少"范式，区别于常规 autotune 的"算得更快"）
6. **诚实 gap + 未来**：理论层预判、reward hacking 防御、⑤ 稳定性
