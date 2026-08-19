# Search 阶段改进 Ideas

> 来源：前沿 beam search 算法创新调研（2026-06-25），结合 DirecTune v5 Search 架构。
> 记录方向 0/2/3：方向 0（按优化方向组织 beam 前沿，是方向 1 多样性惩罚的语义版）、NCU-as-PRM、自适应 breadth。演化（原方向 4）已排除：改动大 + kernel crossover 难定义，暂不信任。方向 5（MCTS）未纳入。
>
> **现状基线**：v5 = `unified_editor`（agents.py:854）+ breadth 并行采样 + 串行验证 + champion 兜底。
> 本质是 **beam search（width=`topk_candidates`=2，`iters` 轮多步展开）+ 增量编辑 + NCU 事后反馈**。
> 每轮每个 candidate 采 `breadth` 个 patch → `select_candidates(k=topk_candidates)`（main.py:311, search.py:13）选 top-2 进下一轮。
>
> v5 相对 v4：planner+executor 合并为单 unified_editor、分支多样性从「planner 产 N 个不同 plan」变成「同 baseline 的 breadth 温度采样」、全量重写→增量编辑、移除每轮 summarizer。**beam 宽度（topk=2）与多步展开结构没变**——单 agent ≠ 单路径。

---

## 方向 2：把 NCU 做成 step-level verifier（PRM 思路）

### 动机

v5 仍是 beam（width=2，多轮展开），但 `select_candidates`（search.py:13）**只按 latency 选 top-2**，NCU 指标只在 patch 验证通过后采一次（`_build_unified_result` agents.py:760-767）、塞进下一轮 `unified_editor` 的 `profile_str`。这是**事后反馈**——NCU 影响下一轮 prompt，但不参与本轮 top-k 选择/剪枝。

前沿（OpenAI PRM800K / Math-Shepherd）的做法是 step-level beam：每步采样多个续写，用 Process Reward Model 给每步打分，**用打分引导 beam 剪枝**保留高分路径。

### 核心思路

把 NCU 从“事后反馈”升级成“beam 的 step-level 选择信号”（PRM），让 top-k 选择不只看 latency：

- patch 应用 + `_verify_code` 通过后采 NCU（已有）
- 用 NCU 瓶颈分类（`summarize_profile_metrics` 已产出 `bottleneck` + `confidence` + `suggested_actions`，见 `_format_hw_profile` agents.py:99）作为 patch 的**过程奖励**
- `select_candidates` 的 top-k 选择从纯 latency 改成 latency + NCU 瓶颈缓解度（如 memory_bw_util_pct 下降幅度）
- NCU 瓶颈明确（如 memory-bound 高置信）→ prompt 引导只往 `suggested_actions` 方向产 patch，剪掉无关探索

### 具体设计

- `select_candidates`（search.py:13）现按 latency 选 top-k。改成 beam：每轮保留 B 个 candidate，每个下一轮采 breadth patch → 验证 + NCU 打分 → 保留 top-B 进下一轮。这是 **v4 beam 的结构**。
- 关键区别于 v4：用 **NCU 做 step-level 剪枝**（PRM），而非 planner 自然语言方案（v4 的交接损耗根源）。
- 新增 `score_patch(pr, hw_metrics, baseline_hw_metrics)`：综合 latency 改进 + 瓶颈缓解度（如 memory_bw_util_pct 下降幅度）。
- prompt 侧：`unified_editor_system.txt` 的 `profile_str` 已有 bottleneck summary，强化"只针对当前瓶颈的 suggested_actions 产 patch"。

### 收益 / 风险

- **收益**：让 beam 剪枝从「只看 latency」升级到「latency + NCU 瓶颈缓解度」（PRM 式 step-level 评估），避免「latency 暂时没改善但瓶颈已缓解」的好 patch 被淘汰，或「latency 改善但瓶颈未变」的差 patch 被保留。
- **风险**：beam 展开 × NCU 采集（NCU 秒级 + sudo ncu 开销）放大验证成本。需 reps 自适应 + 子进程隔离扛。NCU 采集本身可能成瓶颈，可考虑只在"瓶颈不明确需深探"时采。

### 与现有架构关系

`_verify_code`（agents.py:662）和 NCU 采集链路直接复用，改动集中在 `select_candidates` 的选择函数（top-k 标准从纯 latency 改成 latency + NCU 瓶颈缓解度）。**v5 本就是 beam，方向 2 不改 beam 结构，只改 top-k 的选择标准**。

---

## 方向 3：自适应 breadth 匹配难度（test-time compute 思路）

### 动机

Snell et al. 2024《Scaling LLM Test-Time Compute Optimally》的核心结论：**搜索收益与问题难度强相关，compute-optimal 策略要匹配难度**。现在所有题统一 breadth（config 默认 4，实验常用 2），简单逐元素融合（work-log 里 1.6× 那类）和难题（Conv/归一化/Softmax）用同样搜索预算——简单题浪费 token，难题搜索不足。

### 核心思路

按难度信号自适应 breadth（及 `unified_fail_threshold`、rounds）：

**难度信号候选**：
- baseline NCU 瓶颈置信度（`summarize_profile_metrics.confidence`）
- 首轮 patch 成功率（验证通过数 / breadth）
- 算子类型经验（elementwise 易 / Conv·归一化难，来自 work-log 累积规律）
- Generator 初始 Triton vs PyTorch baseline 的 speedup（<1 说明生成质量差，是难题）

**策略**：
- 简单题（高置信瓶颈 + 首轮成功率高）→ breadth=1-2，少轮
- 难题（低置信 + 高失败率）→ breadth=6-8，多轮，必要时换更强模型

### 具体设计

- 第一轮用默认 breadth 探测，统计 patch 成功率 + NCU 瓶颈置信度 → 算 difficulty score
- 第二轮起按 difficulty 调 `breadth` / `unified_fail_threshold` / `rounds`
- `config.yaml` 加 `adaptive_breadth: true` 开关
- 现有 `unified_fail_threshold`（06-21 改 1）和早停 `search_time_budget: 1200`（06-21）是"省"的视角，自适应 breadth 补"难则多搜"的视角——两者合起来才是完整的 test-time compute scaling。

### 收益 / 风险

- **收益**：简单题省 token（你 work-log 06-21 关注的 token 成本），难题多搜提升成功率。固定总 token 预算下整体成功率/加速比提升。
- **风险**：难度信号噪声大（首轮 patch 失败可能是 LLM 随机，非题难）。需平滑/置信区间，或综合多信号加权。自适应逻辑本身增加复杂度和调参负担。

### 与现有架构关系

跟 `fail_threshold` / 早停 / `reps 自适应`（triton_backend.py）同族，是 test-time compute scaling 的直接落地，改动最小。

---

## 方向 0：按算子优化方向组织 beam 前沿（多样性原则，统领方向 1/3）

### 动机

v5 的 beam 前沿（`select_candidates` topk=2）按 latency 截断，分支多样性靠 `breadth` 温度采样（unified_temperature=0.7）——**没有语义依据，可能同质**（N 个 patch 都改 tile）。beam width 是任意选的 2，不对应任何优化语义。

### 算子优化的 5 个常见方向

| 方向 | 含义 | 典型手段 | NCU 信号 | 收益量级 |
|---|---|---|---|---|
| ① Tile & 配置 | 分块/并行配置 | BLOCK_M/N/K, num_warps, num_stages, GROUP_M, autotune | occupancy, stall | 1.2-2× |
| ② 精度 & Tensor Core | 用 TC | allow_tf32, fp16/bf16, dtype | tensor_core_util | 2-8×（compute-bound） |
| ③ 访存 & 布局 | 减带宽压力 | 合并访问, shared tiling, 转置, bank conflict, num_stages | memory_bw_util, l1/l2_hit, stall_memory | 1.5-3×（memory-bound） |
| ④ 算子融合 | 多 op 合一 | GEMM+激活+归约, 减中间写回 | memory_bw_util 降 | 1.3-2× |
| ⑤ 算法等价变换 | 算得更少 | GEMM+Sum→matvec, 预计算, online softmax, im2col | compute 量级降 | 3-8000×（14题） |

8 个方向基本互斥（一个 patch 通常有一个主方向），覆盖优化空间。

### 核心想法：beam width = 该算子适用的方向数

每个前沿分支锚定一个主方向，不再是固定 topk=2 或盲目 breadth 采样：

- **动态 width**（按算子类型 + NCU 瓶颈筛适用方向）：
  - 大 GEMM：①②③④⑤ 都适用 → width up to 5
  - elementwise：② 不适用（无 TC）、① 弱 → 主要 ③④⑤ → width ~3
  - softmax：①③④⑤（online softmax 属 ⑤）→ width ~4
  - memory-bound 跳过 ②，compute-bound 跳过 ③ → 瓶颈驱动剪枝
- **分类实现：LLM 语义判断（非 op_type 正则匹配）**：方向选择由 LLM 判断——prompt 给 ①-⑧ 定义 + 适用性指南（含矩阵乘→②适用、归约/softmax→⑤可能、memory-bound→去②等）+ 算子 reference + NCU 瓶颈，LLM 输出适用方向 + 理由。**不做测试集字母匹配**（如 `if 'matmul' in op_type`），保证对新测试集/新命名的鲁棒性——LLM 理解算子语义（如"GEMM+Sum 可 ⑤ 融合成 matvec"），比正则匹配 op_type 通用。patch diff → 方向 的去重同样由 LLM 判（或规则辅助）。验证：①-⑧ 已确认覆盖 KernelBench 全量 201 题（0 无法归类），且 ② 精度只 76%（含矩阵乘题）天然提供 width 剪枝信号。
- **每分支锚定一个主方向**：`unified_editor` 采样时 prompt 指定"本次 patch 往 ④ 融合方向改"，替代温度 0.7 盲目采样 → **语义多样性**，比 KL 散度惩罚可解释。
- **价值倾斜**：⑤ 等价变换收益最高但最稀有 → 多给采样 budget；① tile 收益中等且 autotune 更可靠 → 移出 beam 交给 `triton.autotune`，beam 专注 ②③④⑤。
- **预算约束**：`width = min(适用方向数, 算力预算上限)`，受 `search_time_budget` 约束。
- **方向分类器**：从 patch 修改模式 / `change_description` 推断主方向（改 `BLOCK_`=①, 加 `allow_tf32`=②, 改 load 模式=③, 合并 op=④, 改算法结构=⑤），用于**去重同向 patch**，保证前沿覆盖不同方向。

### 收益 / 风险

- **收益**：beam width 有了语义依据（覆盖优化方向数，非任意值）；前沿天然多样（每分支不同方向，治 v5 同质问题）；可解释（每分支 = 一个优化假设）；向高收益方向（⑤）倾斜 budget。
- **风险**：方向分类器准确性（误分类导致去重错误）；定向采样可能限制探索（LLM 被方向约束，漏掉跨方向优化——需保留 1 个"自由探索"分支兜底）；适用方向判定需按算子类型维护规则表。

### 与方向 1/3 的关系

这是 IDEAS **方向 1（多样性惩罚）的语义版**，并给**方向 3（自适应 breadth）提供理论依据**：
- 替代方向 1 的 KL 散度多样性惩罚——用"优化方向"做多样性，更可解释、更可控
- 具体化方向 3 的自适应 breadth——breadth 按适用方向数定，而非按难度信号拍脑袋
- 把 1 和 3 统一成一个原则：**beam 前沿按优化方向组织，而非按候选个数截断**

---

## 优先级建议

按"性价比 / 风险"排：

| 顺序 | 方向 | 成本 | 理由 |
|------|------|------|------|
| 1 | **方向 0**（按优化方向组织前沿） | 中低 | 治 v5 breadth 同质 + 给 beam width 语义依据，是方向 3 的前提；加方向分类 + 定向采样即可 |
| 2 | **方向 3**（自适应 breadth） | 低 | 配合 0 做预算分配，跟 fail_threshold/早停同族，落地 test-time compute |
| 3 | **方向 2**（NCU-as-PRM） | 中 | 让 beam 剪枝从纯 latency 升级到 latency + NCU 瓶颈缓解，复用现有 NCU 信号 |

建议路径：先 0（治同质 + beam 语义化）→ 配 3（预算按方向/难度分配）→ 再 2（NCU 引导剪枝）。所有方向都复用 v5 的 unified_editor + _verify_code + episode_store，增量演进而非推倒重来。
