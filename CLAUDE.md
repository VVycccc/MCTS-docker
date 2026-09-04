# DirecTune-MCTS (v6)

DirecTune 的新版本项目（mcts 搜索 + 8 优化方向），独立目录用于与老 `/home/wangyichen/DirecTune`（v5: unified+5方向）对照。baseline → 1× seed 生成 → DirecTune search（Fuser 已退役）

## 版本演进

| 版本 | git tag | 内容 |
|------|---------|------|
| v4 | `v4` (78a520a) | Search 三角色 beam search（planner→executor→summarizer）+ executor 全量静态 skill（72KB）+ 全量重写重试。Generator 为 AKG v5 全量前端。 |
| v5 | `v5` (7a14472) | Search 统一编辑 agent：融合 planner+executor 为单 agent、动态 skill 检索（72KB→~20KB）、FixCodeGen 增量编辑（默认 search/replace，失败回退全量重写）、移除每轮 summarizer（in-search experience 靠 `update_experiences` 跨 iter 累积）。`search_mode` 开关保留 classic。 |
| **v6** | (本项目) | **search_mode 默认 mcts**（树+P-UCT+8方向，`mcts.py`）+ **方向分类 5→8**（补 ⑥ timing_overlap / ⑦ reduction_struct / ⑧ control_flow_spec，work-log 2026-07-09 §极简重构）。⑥ tle_async_smem 合进 ⑥ timing_overlap（forge=num_stages/double_buffer，forge_tle 增补 TLE 异步）。老 DirecTune 保持 unified+5方向作对照基线。 |

> 注：此处 v4/v5/v6 指 **Search 架构版本**；Generator 内部版本号（v3/v5 AKG 前端）独立。

## Quick start


```bash
conda activate forge
cd /home/wangyichen/DirecTune

# 一次性优化（one-shot search）
python main.py --config config.yaml \
    --problem problems/kb_level1/01_square_matrix_multiplication.json \
    --initial problems/kb_level1/01_square_matrix_multiplication_initial.py \
    --rounds 3 --breadth 2 --num-samples 1
```

API 配置在 `config.yaml` 里，当前使用 DS4Pro (`api.deepseek.com/v1`, model `deepseek-v4-pro`)。

## 规范化输出（实验记录要求）

所有实验运行必须产出结构化日志。默认包含 1+3，明确要求「完整 trace」时才包含 2。

### 1. 流程走到哪一步（默认必须）

输出文件：`output/{exp_name}/{problem_name}/pipeline_status.json`，记录每个阶段的状态、耗时、关键指标。格式同上节。

### 2. 大模型每次返回的结果（仅在明确要求时启用）

按 agent 类型分目录存档每次 LLM 调用的 system/user/response/耗时，目录结构同上一版。

### 3. 错误原因（默认必须）

每个失败点记录结构化错误信息：

```json
{
  "stage": "executor",
  "agent": "executor",
  "plan_id": "plan_0_0",
  "retry": 2,
  "error_type": "validation",
  "error_message": "name 'x' is not defined",
  "root_cause": "LLM dropped helper function get_init_inputs() during rewrite",
  "baseline_has_function": true,
  "generated_has_function": false
}
```

错误分类：`compile_error` | `validation_error` | `llm_error` | `pipeline_error` | `oom_error`

### 4. MCTS 带标签节点记录（2026-08-27 新增，默认必须）

输出文件：`output/{exp_name}/{problem_name}/mcts_tree.json`，每 rollout 刷新（崩溃安全），由 `mcts.serialize_tree()` 生成。背景：checkpoint 原本只存 summary（total_nodes/best_latency_ms），无法回答「冠军路径经过哪些方向、是否穿过被贪心丢弃的回归节点」——论文 intro 的机制论断和 feedback-loop 反事实重放都需要这份数据。

- **nodes**（BFS 全集，含完整 code）：方向标签 `direction`/`branch_id`/`edit_mode`、拓扑 `node_id`/`parent_id`/`depth`、预算序 `order_index`（创建序 = validated-kernel 消耗序）、性能 `latency_ms`/`reward_vs_seed`/**`vs_parent_x`**（<1 即「贪心 accept-only 会切断的回归边」）、P/N/W/Q 搜索统计、终态原因
- **champion_path**：root→champion 的 (direction, latency, vs_parent_x) 链，冠军收益逐步归因到方向
- **expansion_events**：每次 expansion 的 sampled/validated/failures 明细（LLM 调用口径，树里只有成功者）
- **budget_counters**：checkpoints/expansions/validated_nodes

配套：`checkpoint_iterN.json` 的 `tree` 块加 `champion_node_id`/`champion_path`；`final_results.json` 顶层嵌 `mcts_tree`。离线分析零 GPU：

```bash
# 反事实重放 + 冠军归因链 + 贪心切断边定位
python scripts/replay_greedy_from_tree.py output/full_mcts/<problem>
```

重放语义注意：贪心循环的 prompt 只含 current-best ⇒ 被拒节点的后代根本不会被生成。所以祖先感知重放（按 order_index 流、parent 不在接受集则跳过）才是忠实模拟，纯延迟过滤会高估贪心。mock 测试：`scripts/test_mcts_tree_records.py`（forge env 运行，不占 GPU）。

### 实现方式

在 `main.py` 和 `agents.py` 的关键节点插入 `_save_trace()` 调用，统一写入上述结构。实验启动时通过 `config.yaml` 中的 `exp_name` 字段指定实验名称。

## NCU Profiling

已配置 passwordless sudo，无需密码即可采集 GPU 性能计数器：

```bash
# 验证: 无密码提示
sudo -n ncu --version

# 采集 Triton 内核指标（注意: 必须用绝对路径）
sudo -n ncu --csv --target-processes all \
    --log-file=output/ncu_metrics.csv \
    --metrics="sm__cycles_active.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" \
    /home/wangyichen/miniconda3/envs/forge/bin/python3 script.py
```

关键限制: `sudo ncu` 会修改 PATH，必须使用 Python 解释器的绝对路径（`/home/wangyichen/miniconda3/envs/forge/bin/python3`），不能依赖 `python3` 或 `conda run`。

tileforge 的 `run_ncu.py` 提供了完整的 NCU 采集/解析/prompt 生成工具函数，可直接复用。

### 当前 DirecTune 中的 NCU 闭环

- `executor()` 会对成功候选采集 `hw_metrics`，并把结果写回 candidate/result。
- `main.py` 会在 baseline profiling 成功后，额外尝试做一次 baseline NCU 采集。
- `search.select_candidates()` 会保留 `hw_metrics`，使下一轮 planner 能继续看到同一个候选的硬件反馈。
- `agents.py` 会先把原始指标归纳成 **bottleneck summary**（如 `memory_bound` / `register_limited` / `tensor_core_underused` / `underutilized`），再连同原始指标一起喂给 planner 和 summarizer。
- `summarizer` 现在比较 slow/fast kernel 时，也会同时比较 slow/fast profile，从而提炼“瓶颈 → 改动 → 指标变化 → 性能结果”的经验。

### Baseline profiling 兼容性说明

baseline 源码不一定总是直接包含可被正则提取的 `@triton.jit def ...` kernel。当前 `hardware_profiler.py` 已增强为：

- 如果能提取到 kernel 名，就继续按 kernel-name filter 精准采集；
- 如果提取不到，就退化为 **profile 所有实际启动的 kernel**，再优先选 `sm__cycles_active.avg` 最大的热点行做汇总；
- 因此 baseline 即使不是最标准的内联 Triton 写法，也更有机会拿到可用的 NCU 指标，而不是直接跳过。

### null-shape 输入解析（2026-07-23）

KernelBench 转换出的部分 JSON 会把输入 `shape` 写成 `null`：有些是真标量（如 matrix-scalar 的 `s`），有些只是转换器没落出具体 tensor shape（如 L1 14/15 的三角矩阵 matmul，真实 `get_inputs()` 是 4096×4096）。`triton_backend.resolve_problem_inputs()` 现在会在运行时遇到 null 输入时优先隔离执行 reference `get_inputs()` 取元数据：非标量 tensor 补成具体 shape，0D tensor 仍保留 `None` 标量。`check_correctness` / `profile` / `run_isolated_validation` / `hardware_profiler` / fallback `generator.py` benchmark 均应复用该解析逻辑，避免 baseline profiling 把未知 tensor 错当 `torch.randn(())`。

## 架构：统一单路径入口（baseline → 1× seed 生成 → search）

```
main.py（统一入口，单路径；Fuser 已退役）
 ├─ baseline profile def run（PyTorch reference）→ baseline_latency + hw_metrics
 ├─ 1× seed 生成（v6 默认 gen_mode=naive，纯 LLM naive seed，0 AKG 依赖）
 │   ├─ naive → naive_seed_gen.gen_seed()（默认路径；system/user prompt + triton_api_reference.md）
 │   └─ fallback only: generator.py:generate_kernel()
 │       ├─ v3 → _generate_kernel_v2(enhanced=True)
 │       └─ akg → _generate_kernel_akg（vendored AKG LangGraph + shim 适配 .pt frozen 权重）
 └─ search.py: run_search_episode(initial_candidate=seed) 按 config.search_mode 分支:
     ├─ mcts (v6 默认): mcts.run_mcts() 树搜索 + P-UCT + 8 方向
     ├─ unified: agents.unified_editor() 单 agent 编辑 + 验证环
     └─ classic: planner→executor→summarizer 三角色 beam search（保留消融）
```

> 2026-06-27 统一：以前按 `"nn.Module" in initial_code` 分 Fuser/direct 两路，Model 式 level1（91/100，有 nn.Module 但无 .pt 权重）误走 Fuser → Dispatch v3 + Compose 丢 @triton.jit + in-episode 二次 v3 生成（2×）→ GLM-5.2 慢下超时。现统一成单路径（baseline → 1× seed → search），Fuser/Dispatch/Compose 从 main.py 移除。`fuser.py` / `prompts/compose.txt` / `prompts/extract_subgraphs.txt` / `_build_dispatch_plan` 已于 2026-06-28 删除（Fuser 完全退役）；`optimize_loop.py` + `episode_store.py`（episodic 外循环）于 2026-06-29 全删，main.py 直连 `run_search_episode`。
>
> **v6 seed 生成口径（2026-07-21 纠偏）**：MCTS 侧默认必须是 `gen_mode: "naive"`，即纯 LLM naive seed，0 AKG 依赖；AKG/v3 仅作为显式 fallback/盲区对照，不能混入正式 naive+MCTS 实验目录。Triton strict 指的是最终 `final_results.json` 的 champion code 必须含 `@triton.jit`，它与 seed 来源正交，不等于 `gen_mode: "akg"`。

**Generator（v6 默认：naive 纯 LLM seed；AKG 仅 fallback）**:
- `gen_mode: "naive"`（默认）：`naive_seed_gen.py` 单次/少轮纯 LLM 生成朴素 Triton seed，只读 `prompts/naive_seed_{system,user}.txt` + `triton_api_reference.md`，不 import `akg_frontend`。默认最多 3 次 LLM 尝试（初次 + 2 次错误反馈重试）；每轮响应、候选代码、anti-pytorch/compile/verify 错误会落盘到 `output_dir/naive_seed_debug/`，用于分析 seed 生成失败原因。
- `gen_mode: "v3"` / `"akg"`（显式 fallback）：`generator.py` 才会启用旧 v3 或 AKG 前端；AKG 路径会打印 `[Generator AKG]`，正式 naive 实验中不应出现。
- L2/全量 MCTS naive 实验必须检查日志含 `Generating Triton seed (gen_mode=naive)`；若出现 `gen_mode=akg` 或 `[Generator AKG]`，该 run 只能归为 AKG-seed 对照，不能计入 naive+MCTS 结果。

**Search 阶段（v5 统一编辑 agent，`search_mode: unified`）**:
- `unified_editor()` 合并 planner+executor 为单 agent：一次推理内分析瓶颈 + 产 search/replace patch，消除自然语言交接损耗
- 多次采样（`breadth`）产生前沿多样性，LLM 调用并行、验证串行（GPU 安全）
- 默认 FixCodeGen 增量编辑（`parse_modifications` + `DiffApplier`），patch 失败或连续验证失败回退全量重写（规则化简，无 Conductor LLM，`unified_fail_threshold` 控制回退次数）
- 动态 skill 检索：`_load_skill_context_dynamic()` 按 op_type 选 skill（72KB→~20KB），替代全量静态；`_derive_op_type` 重读 problem JSON 拿真 op_type
- 移除每轮 summarizer：`_build_experiences_from_results()` 从 change_description 产 experience，in-search 短期记忆靠 `update_experiences` 跨 iter 累积
- `search_mode: classic` 保留 v4 三角色路径供消融
- `search_mode: mcts` 见下「MCTS 搜索模式」节

#### MCTS 搜索模式（`search_mode: mcts`，`mcts.py`）

理论依据 work-log 2026-07-11：unified 的"每轮单改"把空间压成 contextual bandit(#2)；MCTS(#8 UCT 收敛) 仅在**建模优化序列（patch 叠加 A→B→C）** 时才适用。`mcts.py` 正是把空间形式化为树——节点=已验证正确的 kernel，子节点=父节点上施加一优化方向后的新 kernel，深度=叠加优化数——让 UCT 的 O(ln T) regret 收敛挂上，LLM 方向先验经 P-UCT 先验项 P 注入。champion=全树最低延迟节点，seed 永在树中 → 自然 carry-forward（治 work-log TODO [A]）。

- **复用**：扩展（`_expand`）直接调 `agents.unified_editor([parent], applicable_directions=…)`，子节点=unified_editor 已验证正确的 result（同款 patch+verify+NCU+skill），不重写 GPU 路径。`_verify_code`/`triton_profile` 是 sync 且用 `signal.alarm`（主线程），必须在主事件循环直接调用、不可 offload 到线程池。
- **reward** = `log(seed_latency / node_latency)`（clamp [-2,5]），对数处理 35× vs 1.2× 量级差（work-log 指定）。
- **P-UCT**：`Q + c_puct·P·√(N_parent)/(1+N)`；先验 P 由 `_compute_child_priors` blend 两信号——LLM 分类器收益排序（`mcts_prior_blend` 权重）+ dir_probe 固定分级表 `_DIR_PROBE_PRIOR`（结论 1/4：结构方向高、tile_config 低）。
- **dir_probe 落地**：结论 3（方向 op_type-dependent）→ P 先验由分类器按语义判、dir_probe 表只 blend 兜底，不硬编码跨算子结论；结论 5（memory-bound/GEMM-主导死区）→ adaptive 深度策略在子节点全无改进时早停，把预算从死区挪走；结论 6（真盲区是 seed 难写）→ MCTS 只优化已有 seed，不解决 gen。
- **动态深度旋钮（全套可 sweep）**：`mcts_max_depth`（基础硬上限）× `mcts_depth_scale`（运行时乘子，effective=round(max_depth·scale)，外层 sweep scale 探不同树深）；`mcts_depth_strategy: adaptive|fixed`；adaptive 早停信号 `mcts_min_depth`/`mcts_stall_threshold`(≈log1.05)/`mcts_stall_patience` 各自可 sweep；backprop 长度 `mcts_rollout_depth`（=1 子节点即终端快，=k 下行 k 层更贴近 MCTS 原型但 k× LLM 调用）。
- **main.py 集成**：`run_search_episode` 开头 `if search_mode=="mcts":` 委托 `mcts.run_mcts`，整树搜索后返回同 shape `{candidates, experiences, results, best_latency_ms}`，与 unified/classic 分流（rollout 循环在 run_mcts 内部，不共享下方 iter 循环）。
- **验证**（2026-07-14 smoke）：01_square_matmul (4096² fp32, naive BLOCK=32 seed @10.01ms)，1 rollout / depth1 / max_depth=2 / free_explore off → 树 4 节点 depth1，champion=precision_tc 方向（BLOCK 128/128 + GROUP_M + num_stages3/num_warps8 + allow_tf32）@3.927ms = **2.55× vs seed**。树深 log + champion.py + summary.json + direction_stats 均正确落盘。详见 work-log 2026-07-14。
- **L1/L2 v6 批量验证**（2026-07-15，`run_l1l2_mcts.py`）：13 题 0 崩溃 0 回归。L1 十道：改进 9（01_matmul 2.58× / 40_layernorm **9.33×** / precompute_test **6.49×** / 47_reduction **27.22×** / 49_maxred 21.25× / scale_reassoc 1.28× / 12_diag 1.01× / precompute_mean 1.05× / transpose 1.02×）+ 超时 1（17_matmul_tb 预算不够非 bug）。L2 三道 fusion：l2_9 ✅2.20×、l2_76 rollout1 已出 champion 4.18ms=2.19×（1800s 超时没写 final_results，checkpoint 有）、l2_30 是 dir_probe seed 自身 GroupNorm 数值不稳（1e-2 仍 mismatch）非系统问题。对比老 5 方向 batch，8 方向在 40_layernorm(2.80→9.33×)、precompute_test(1.00→6.49×)、scale_reassoc(1.005→1.28×) 明显更优——新方向集确实解锁了老版搜不到的优化。

#### 方向 0：按优化方向组织 beam 前沿（opt-in，`direction_organized_frontier`）

治 v5 breadth 同质 + 给 beam width 语义依据。默认关，开启后仅 unified 模式生效（classic no-op）。三件事，全 gated on `direction_organized_frontier`：

1. **方向分类器**（`agents.py:determine_applicable_directions`，每 run 1 次 LLM）：按**算子语义**判定 ① Tile/② precision-TC/③ mem-layout/④ fusion/⑤ algo-equiv 哪些适用 + 每方向一句采样指令，按预期收益排序。NCU 瓶颈仅调优先级不作适用性闸门 → noop profiler 下也能判。fallback=全 5 方向（② 含 matmul guard）。prompt=`prompts/direction_classifier_system.txt`。
2. **定向采样**（`unified_editor` 加 `applicable_directions` param）：方向模式每 candidate 采「每适用方向 1 patch + 1 free_explore 兜底」，`branch_id` 带方向标签（`dir_5_algo_equiv`），`asyncio.gather` 并行采（非同质 `_gather_llm`）；legacy 模式原样 `_gather_llm(n=breadth)`。
3. **按方向选前沿**（`search.py:select_candidates_by_direction`，不改共享 `select_candidates`）：每方向留最快 patch，按**分类器优先级**（非 latency）截断到 `direction_max_width`，free_explore 豁免。返回 shape 与 `select_candidates` 一致 → main.py 下游零改动。

**main.py 集成**：`run_search_episode` iter 循环前算一次方向（仅 unified+开关开），传入 `unified_editor`，选择分支 `if applicable_directions:` 切 `select_candidates_by_direction`。`applicable_directions` 仅非 None（即 enabled）时走方向选择，否则原 `select_candidates` —— classic/默认零回归。

**关键决策**：不按 latency 截断方向（早期 latency 是差的假设价值代理，会误杀需多轮解锁的 ⑤）；每 run 算一次（算子语义稳定，per-iter 瓶颈已由 `_format_hw_profile` 独立喂 unified_editor）。

**验证**（2026-06-26）：14_Gemm_Divide_Sum_Scaling e2e，分类器判 ⑤ algo_equiv 适用且排第一，iter 1 系统性搜到 GEMM+Sum→matvec（champion=`fused_matvec_kernel`，0.1259ms，35.80× vs generator 初始 4.5ms，纯 Triton），比对拍时 breadth 撞上的 33.71× 还快。详见 work-log 2026-06-26。

**持久化层（TODO-1，2026-06-29）**：direction 模式下 `run_search_episode` 每轮按 `branch_id` 聚样各方向结局（`patches_sampled`/`passed_validation`/`survived_selection`/`best_speedup_vs_seed`），run 末合并进全局 `direction_stats.json`（按 `op_type` 分桶，跨 run 累积；`direction_store.py`）；run 开始 log 已累积的该 op_type 统计。**仅记账 + 持久化 + log，不改搜索**——消费（TODO-2 经验排序遍历 / TODO-3 注入分类器 prompt）是独立下一步。classic/legacy 模式跳过。

**已知限制**（MVP）：mid-iteration 超时靠 `search_time_budget` 兜底；方向截断不轮转靠 free_explore 兜底。

#### 增量编辑执行流程（`agents.py unified_editor` → `_try_incremental` + `_rewrite_fallback`）

输入 `candidates` 均为已验证正确的 baseline 内核。准备两套 prompt：增量用 `system`（triton_api_ref + skill_context + 硬件指标解读，`_build_unified_system`），回退用 `full_rewrite_system`（skill + executor_system.txt）。

```
for each candidate (baseline 内核):
├─ 阶段1 并行采样: async.gather 发各方向 patch LLM 调用 (temp=0.7, stage="patch")
│              每个产 JSON {modifications:[{old_string,new_string,replace_all,anchor}]}
├─ 阶段2a 串行增量尝试: _try_incremental（GPU 绑定，CUDA 不并发安全）
│    ├─ 解析 parse_modifications(resp)
│    ├─ 应用 DiffApplier.apply_modifications (4级匹配→替换)
│    │    diff.success? ──否──→ 记入 fails（fail_ctx 携带 last_error/last_code）
│    ├─ 验证 _verify_code (anti_pytorch + compile + correctness + benchmark)
│    │    通过 → edit_mode="incremental" (+NCU hw_metrics)
│    └─ 失败 → 记入 fails
└─ 阶段2b 全量重写回退: _rewrite_fallback（LLM 绑定 → 并行）
     第 1 轮: 所有 fails 的 rewrite 调用 async.gather 并行发起（互不依赖），
              返回后逐个串行 _verify_code（GPU 仍串行）
     第 2..unified_fail_threshold 轮（罕见）: 串行带新错误反馈重试（同旧语义）
     切 executor_system prompt (产完整代码, 非patch) + temp=0.3
     user 拼入失败代码 + 错误 (truncate_error_log)
     _verify_code 共享关卡 → 通过返回 edit_mode="full_rewrite"
     全失败 → 返回 error (baseline 字节不动)
│
select_candidates: 过滤 error/latency=None → 按(baseline,branch)分组选最快 → top-k   search.py:13
```

#### LLM 调用效率（2026-08-27：thinking 分阶段开关 + rewrite 并行化）

**问题定位**（trace10 实测）：GLM-5.2 / deepseek-v4-flash / kimi-k3 均为推理模型，`reasoning_content` 计入 completion_tokens 且占大头（单次调用 completion 5.6-8.8K tokens，可见输出仅 1-2K），但流水线从不消费推理内容 → 单次调用 80-125s（即 work-log"GLM-5.2 每次~107s"的构成），3600s 预算只够 ~4 次 expansion。

**三项优化**（`agents.py`）：

1. **thinking 按阶段开关**：`_chat()` 加 `stage` 参数（patch/classifier/rewrite/seed），经 `config.llm_thinking.<stage>` 控制，默认 patch/classifier **off**（search/replace JSON 与方向判定不需要深推理；实测 autodl 端点 thinking off：GLM-5.2 4.7s→2.6s、ds4flash 4.4s→1.7s，completion token 省 ~70%）、rewrite/seed on（质量依赖）。实现走 `extra_body={"thinking":{"type":"disabled"}}`；端点返回 400 时自动整进程降级（`_thinking_off_unsupported`），不影响正确性。
2. **rewrite 并行化**：见上阶段 2b——旧版每个失败 patch 的 full-rewrite 是 patch 循环内串行 await（N 个失败 × ~100s），现在第 1 轮全并行，验证仍串行（`_verify_code` 用 signal.alarm 必须主线程串行的约束不变）。
3. **patch 输出预算右移**：`patch_max_tokens=8000`（原 20000 全量重写预算）、`patch_timeout=300`（原 600s 会放任单次调用卡死吃掉时间预算——deadline 检查只在 patch 之间生效）。

**配套**：`AsyncOpenAI` client 按 (url,api_key) 进程内缓存（`_get_client`，免每次 expansion 重建 TLS 连接）；`TOKEN_USAGE` 按阶段拆分记账（`record_usage(resp, stage=)`），run 末打印各阶段 prompt/completion/calls，用于 A/B 验证节省量。**注意 autodl 端点无前缀缓存**（同 prompt 两次调用 `cached_tokens=0`），prompt 侧没有免费复利，token 节省主要来自 thinking off。

**A/B 判决（2026-08-27，3 题 × 3 臂，`output/ab_thinking/`，`run_ab_thinking.sh` + `summarize_ab_thinking.py`）**：

| 臂 | 01 matmul | 40 layernorm | 30 L2 融合 | 通过率 | completion |
|---|---|---|---|---|---|
| on（全推理） | 2.35× | 30.55× | seed 三连挂* | 12/12 | 268K |
| patch/classifier 关 | 2.75× | 55.35× | 3.88×† | 43/55 | 341K |
| + rewrite 关（现行默认） | **5.16×** | 34.25×† | **13.27×**‡ | 72/112 | **71K** |

*GLM-5.2 在 30 题 seed 是抽卡（开推理也挂）；†35min 撞 shell timeout 截断；‡跑 10min 被停。

- **rewrite 关推理判可通过**：失败率 22%→36%（细节幻觉，如路径写成 `wangchenyi`），但单次调用快 5-10×，MCTS 靠多试补偿——champion 不降反升（01: 5.16× vs 2.75×；30: 13× vs 3.9×），token 341K→71K。错误 patch 由 `_verify_code` 关卡兜底，失败只浪费一次便宜调用。已落 `config.yaml` 默认。
- **seed 保推理不可砍**：关推理 3/3 把 `normalized_shape=(64,256,256)` 错归约成最后一维 256（教科书 LayerNorm 模板套题，l2_rel≈0.076 数值错）；开推理样本在推理链里做了读题核对。kernel body 都写得对，挂的全是"归约多宽"这类读题语义判断——正是推理链兜底的环节。
- alloff 高失败率的补偿机制依赖"失败便宜 + 验证关卡"，若未来切到**按 token 计费敏感**或**验证很贵**的场景需重评。

#### CodeMatcher 4 级匹配（`fix_code_gen.py:67`）

LLM 凭记忆写的 old_string 常与 baseline 有空白/缩进差异，4 级 fallback 提高匹配率（避免频繁回退全量重写）：
- L1 exact（`str.find`）/ L2 trimmed（逐行 strip 滑窗）/ L3 whitespace_normalized（连续空白压单空格）/ L4 fuzzy（SequenceMatcher ≥0.8 **且** best−second ≥0.1 confidence gap 防歧义）
- 辅助：`anchor`（多处相似只改一处时先定位锚点）、`replace_all`、`_detect_conflicts`（old_string 互相包含时警告）
- `match_levels` 记录用了哪级匹配，进结果供调试 + experience（实例：`ue_0_1 ✓ incremental (match={'exact': 2})`）

#### 动态 skill 检索（`agents.py:366 _load_skill_context_dynamic`）

复用 AKG `OperatorSkillCatalog.filter_by_context`（`akg_frontend/.../operator_skill_catalog.py`），7 个 filter AND 过滤：backend / dsl / framework / hardware / operator_type / category_include / category_exclude。**全 fail-open**：context 或 skill 无该字段即放行，只有两边都有值且不匹配才拒绝。
- `_derive_op_type`（`agents.py:340`）重读 problem JSON 拿 op_type（绕过 Problem 丢弃 op_type 的 bug）+ 词表映射（gemm→matmul / reduction→reduce，对齐 AKG 的 operator_patterns）
- 实例：op_type=matmul 时，operator_patterns 标了 attention/elementwise/reduce 的 3 个 skill 被过滤，13→10（basics 标 "all" 放行，未标 operator_patterns 的 7 个 fail-open 放行）
- ⚠️ 隐患：`render_as_markdown` 的 by_cat 只认 {fundamental,reference,guide,example,case}，而 13 个 skill 的 category 是 fundamental/implementation/example/method/fix——implementation/method/fix 类的 content 可能被 render 丢弃。实际注入 prompt 的 content 或少于 coarse_filter 数量，建议跑一次打印 render 输出长度确认（可能解释"search 改进有限"）。

**AKG 归属**（Search 阶段对 AKG 的依赖比 Generator 轻得多）:

| 组件 | 归属 | 耦合方式 |
|------|------|---------|
| skill 内容（13 个 SKILL.md）+ 检索机制（Catalog + 7 filter + render） | AKG | **直接 import** akg_frontend/，运行时依赖 |
| 增量编辑工具（`fix_code_gen.py` CodeMatcher/DiffApplier） | AKG diff_utils.py **移植** | 已脱离 AKG 包，版权 DirecTune |
| unified_editor prompt + NCU 反馈 + op_type 适配 | DirecTune 自己 | — |

对比：Generator 阶段 Coder/Conductor/Verifier/FixCodeGen/Skill 全是 vendored AKG 直接 import；Search 只复用 skill + 增量编辑两样，prompt 和反馈环自己写（因要对接 NCU + RTX 3090，AKG 没有）。去 AKG 依赖时 `fix_code_gen` 不用动，skill 检索需自重写 catalog。

#### 正确率保证机制

增量编辑不"保证改对"，保证"不退化"：
- baseline 已验证正确，增量只替换 old_string，**未改部分天然正确**（接口/helper/imports/索引原封保留）
- 每次改动过 `_verify_code` **共享关卡**（增量与全量重写同一套：anti_pytorch → compile → correctness → benchmark），不过即丢弃
- **best-so-far 兜底**：champion = 最快 correct 候选（seed 一直在候选里），某轮全失败保留 seed（不退化）
- 缩小改动表面积 = 降低弱模型改坏对的代码的概率（治 v4 "全量重写改坏原本对的代码"）
- 局限：保证"不比 baseline 差"，不保证"一定改进"；old_string 4 级匹配全失败 → 回退全量重写

## 打包部署（2026-08-19，2026-09-04 加 champion 导出）

详见 `DEPLOY.md`。要点：`./build.sh` 构建 `directune-mcts:latest` Docker 镜像（自动解引用 `akg_frontend`/`problems` 两个指向老 DirecTune 的 symlink 成自包含上下文）；镜像不含任何 API key。**2026-09-04 起**镜像内置历史 champion 导出：`scripts/export_champions.py` 扫全部 run 的 `final_candidates[0].code`（三种布局：run 根 / mcts 子目录 / 嵌套 ablation+tryN）生成 `output/champions_export/<run>__<problem>.py` + `_manifest.{json,csv}`（已导出文件永不重写，只增量补），build.sh 把它拷进 staging、Dockerfile 烘焙到 **`/app/champions/`**（独立路径，防 `-v output:/app/output` 挂载遮蔽）。配套可移植化改动：`triton_backend.load_problem()` 会把 reference 里写死的 `_weights_path` 重写为按 problem JSON 所在目录解析（L2 JSON 原含 `/home/wangyichen/...` 绝对路径，迁移即失效）；`hardware_profiler` NCU 解释器默认 `sys.executable`（`DT_NCU_PYTHON` / config 可覆盖）；散落脚本（bench_inductor/naive_seed_*/plot_* 等）的绝对路径全部改为相对或环境变量（`DT_DIR_PROBE` / `KB_L1` / `KB_L2`）。

## 模块职责

| 文件 | 职责 | 行数 |
|------|------|------|
| `main.py` | 入口，自动路由 Level 1/2/3，包含可复用 `run_search_episode()` | ~570 |
| `fuser.py` | **[已删除]** 子图分解（1次 LLM 调用）；2026-06-27 统一入口后不再调用，2026-06-28 删除 | — |
| `generator.py` | generate_kernel 统一入口 + v3/akg 分发 + AKG LangGraphTask 全量前端 + shim | ~1350 |
| `agents.py` | classic: Planner/Executor/Summarizer；**v5: unified_editor + 动态 skill 检索 + FixCodeGen 增量编辑** | ~750 |
| `search.py` | 纯 Beam Search + 候选选择 + 经验管理 | ~150 |
| `mcts.py` | **MCTS 搜索**：MCTSNode + P-UCT 选择 + `_expand`（复用 unified_editor）+ dir_probe 先验表 + adaptive 动态深度（全旋钮可 sweep）。`search_mode: mcts` 时由 `run_mcts` 接管 | ~430 |
| `triton_backend.py` | 编译/计时/正确性/反PyTorch检测 | ~370 |
| `direction_store.py` | **持久化层**：per-op_type 方向结局统计（sampled/passed/survived/best_speedup_vs_seed），跨 run 累积到 `direction_stats.json`（TODO-1，仅记账） | ~140 |
| `akg_frontend/` | **Vendored AKG akg_agents 包** (350 py, 6.9M) | — |

**prompts 可变部分**：每个 prompt 独立一个 .txt，`{var}` 占位符。改 prompt 不需要改代码。

## 配置 (config.yaml)

```yaml
model: {url, model, api_key}    # LLM 配置
rounds, breadth, num_samples     # DirecTune 超参
gen_workers, gen_refinement_rounds  # Generator 超参
problem, initial_solution       # 路径

# 方向 0：按优化方向组织 beam 前沿（opt-in，仅 unified 模式生效）
direction_organized_frontier: false   # true 启用 LLM 方向分类 + 定向采样
direction_max_width: 3                # 前沿 carry-forward 上限（按分类器优先级截断）
direction_free_explore: true          # 1 个无约束兜底分支，豁免截断
direction_classifier_temp: 0.3        # 方向分类器 LLM 采样温度
```

批量实验：复制 config.yaml 改参数，循环 `python main.py --config configs/xxx.yaml`。

## 关键设计决策

### Generator v5（默认）：全量 AKG 前端移植

v5 将 AKG 的完整 LangGraph 生成前端 vendor 到 DirecTune（`akg_frontend/akg_agents/`），移除对外部 AKG 仓库的依赖。

| 组件 | 来源 | 功能 |
|------|------|------|
| **LangGraphTask** | `akg_frontend/akg_agents/op/langgraph_op/task.py` | LangGraph 编排容器：构建图、准备状态、执行 ainvoke |
| **Coder** | `akg_frontend/akg_agents/core/agent/coder.py` | RAG + local examples → Triton 代码生成 |
| **CodeChecker** | `akg_frontend/akg_agents/op/utils/code_checker.py` | AST/compile/import 静态检查 |
| **KernelVerifier** | `akg_frontend/akg_agents/op/verifier/kernel_verifier.py` | 模板生成验证脚本 → subprocess 隔离运行 |
| **KernelConductor** | `akg_frontend/akg_agents/op/agents/kernel_conductor.py` | Skill 系统诊断 → 路由决策（Coder/FixCodeGen/finish） |
| **FixCodeGen** | AKG 原生（nodes.py 内建） | Search/Replace 增量修复 |
| **Skill 系统** | `akg_frontend/akg_agents/op/skill/` + `core_v2/skill/` | 分层 skill 选择（fundamental→guide→example→case→fix） |
| **Worker 系统** | `akg_frontend/akg_agents/core/worker/` | LocalWorker + DevicePool GPU 管理 |

**工作流（coder_only_workflow）**：
```
Coder → codegen_router → Conductor / CodeChecker
CodeChecker → verifier / FixCodeGen / Coder
FixCodeGen → Verifier
Verifier → END / Conductor
Conductor → Coder / FixCodeGen / END
```

**API 桥接**：`generator.py:_generate_kernel_akg()` 在调用 LangGraphTask 前将 `config.yaml` 的 `model_frontend` 映射到 `AIKG_*` + `AKG_AGENTS_*` 环境变量。

**与旧 pipeline 的关键差异**：
- **自包含**：不依赖外部 AKG 仓库的 sys.path hack
- **完整 skill 系统**：13 个 triton-cuda SKILL.md + 分层选择器
- **AKG 原生 Verifier**：使用 AKG 的模板渲染验证脚本，而非 DirecTune 的 `run_isolated_validation`
- **架构**：agent 代码全量 vendor，LLM 调用走 AKG 的 `create_llm_client()` 而非 DirecTune 的 `_chat()`

### Generator 正确性
- **反 PyTorch 检测** (`triton_backend.py:anti_pytorch_check`): 22 个正则扫描生成的代码，发现 `torch.relu/matmul/nn.Module` 等即拒绝
- **精炼循环**: v1/v2 完整重写，v3 含 FixCodeGen 增量修复路径
- **Level 2+ 权重可见性**: 转换时提取 Model state_dict，expanded reference 中以具名变量暴露权重形状

### Level 2+ 的 Expanded Reference
`scripts/convert_kb_level2.py` 生成的 reference 包含：
1. 所有权重以 `_weights['key']` 加载，注释标注形状
2. 原始 forward body 展开为显式 `_torch.nn.functional.linear/relu/...` 调用
3. 常量（scaler factor, clamp_min 等）以内联字面量显示

### 容差策略
- Level 1: `rel_tol=1e-3`
- Level 2+: `rel_tol=1e-2`（TF32 TensorCore 精度损失 + 大 K 累积误差）
- `check_correctness` 使用 l2norm 相对 + 绝对组合检查

### GPU 内存
- asyncio Lock 串行化 CUDA 操作（`generator.py:_gpu_lock`）
- 每轮/每次 correctness check 后 `torch.cuda.empty_cache()`
- gen_workers > 1 在大张量问题上可能 OOM

### CUDA 上下文中毒防护（2026-08-27）
一个候选 kernel 的非法访存会**永久毒化主进程 CUDA context**——之后所有验证连带报同一错误且不可恢复，剩余预算静默作废（历史 6/252 run 中招、零恢复；treelog try1 整场报废实例）。防护：`config["isolated_verify"]=true` 时 `agents._verify_code` 走 `triton_backend.profile_isolated()`（run_isolated_validation 加 reps/warmup 参数扩展出的子进程全链验证），坏 kernel 只死子进程。
- **代价**：每候选 +2~3s 子进程 import；latency 与进程内路径偏差 <3%（scripts/test_isolated_verify.py 三断言：等价性/抗毒性/开销）
- **注意**：子进程崩溃类错误前缀为 `[isolated-crash]`（区分不了语法错与 device abort，stderr 关键字粗分 [compile]）；main.py 的 seed/baseline profiling 仍走进程内 `triton_profile`（seed 生成期中毒概率低、有 sys.exit 兜底），如需全覆盖再迁
- 批量实验建议在 config 里开 true

## KernelBench 数据集

### Level 1 (100 题)
`problems/kb_level1/` — 纯数学算子（matmul, relu, softmax 等）
- 无 nn.Module，无权重
- Generator 直接生成 Triton
- 通过率 ~80%，加速 1.1x~1.7x

### Level 2 (100 题)
`problems/kb_level2/` — 融合算子（含 nn.Module 权重）
- Fuser 分解 → Dispatch 每个子图 → Compose 拼接
- Matmul/Gemm 类工作良好（1.3x~90x）
- Conv2d 类：~~LLM 写不出 Triton~~（07-01 推翻，GLM-5.2 能 gen 全类 conv）；Level 2 conv 融合题受限于权重传递链路，生成质量取决于模型能力

### Level 3 (50 题)
`problems/kb_level3/` — 完整模型（MLP, ResNet, ViT 等）
- 多层 Linear/Conv 权重可达 2.6GB
- Fuser + Dispatch 可拆解，但大规模 Triton 生成受限

## 已知限制

1. **大权重 (2GB+)**: 受 24GB GPU 限制
2. **Fuser dispatch 失败**: 子图 Generator 全部失败时，DirecTune 回退到直接优化原始 reference
3. **Gen workers > 1**: 串行化 GPU 访问，实际并发度受限于 asyncio Lock
4. **Conv 生成**: ~~LLM 写不出正确的 Triton conv~~（07-01 推翻：GLM-5.2 能 gen 全类 conv，详见「Conv 生成」节）。Conv 生成质量取决于模型能力（GLM-5.2 足够，DS4Pro 不够）。Level 2 conv 融合题受限于权重传递链路，已验证可走通。
5. **当前真盲区（07-01 实测）**: reduction（94_mseloss）+ attention（97_SDPA）—— GLM/DS4Pro 都 gen+search 失败，详见 work-log 2026-07-01 节。

---

## 全量对比实验配置（MCTS vs 老 DirecTune beamsearch，2026-07-16 定）

> 目的：严肃对比 MCTS 树搜索与老 DirecTune 扁平 beam search，验证"建模优化序列"是否带来差异化增益。
> **公平性核心**：控制变量是 **LLM 调用预算（墙上时间）**，不是 rollout 数 vs iter 数——两边都撞时间墙，对比的可比单位是"定长时间内能做的扩展次数"。故两边同步给等量时间预算，各自跑满名义配置。

### MCTS 侧（`config_full.yaml`，基于 config.yaml 改）

```yaml
search_mode: "mcts"
search_time_budget: 3600        # 60min/题（原 1200s → 3600s，让名义 rollout 能跑完而非被时间截断）
mcts_rollouts: 12               # 配合 3600s，实际跑 6-8 个（rollout_depth=2 翻倍调用）
mcts_rollout_depth: 2           # ★关键★：评估两步终局再回传，让多步叠加协同被回传（MCTS 价值释放点；=1 退化为带先验贪心单改，与 unified 无差异）
mcts_max_depth: 4               # 不变
mcts_depth_strategy: "adaptive" # 死区早停不变
mcts_cpuct: 1.0
mcts_use_directions: true
gen_mode: "naive"               # 0 AKG 默认路径
gen_refinement_rounds: 2
# 其余 mcts_* 旋钮（prior_blend/rank_decay/stall_* 等）同 config.yaml 默认
```

### 老 DirecTune 对照侧（`/home/wangyichen/DirecTune/config_full.yaml`）

```yaml
search_mode: "unified"          # 扁平 beam search（v5 unified editor）
search_time_budget: 3600        # 等量时间预算
rounds: 12                      # 配合 3600s（原 15，以时间墙为准）
breadth: 4
num_samples: 2
direction_organized_frontier: true   # 同样开 8 方向分类，公平
direction_max_width: 4
gen_mode: "akg"                 # 老 DirecTune 用 AKG 生成 seed（各用各自原状生成路径，不强制统一）
gen_refinement_rounds: 2
```

> **生成路径不强制统一**：MCTS 侧用 naive seed（0 AKG 默认路径），老 DirecTune beamsearch 侧用 akg 生成——各用各自的生成原状。这本身也是对比维度之一（naive 纯生成是否足以支撑搜索，见 work-log 2026-07-16 消融：L1/L2 naive 5/5+3/3 成功 naiveness 1.0，正确性反超手写 seed）。搜索算法对比的是"给定各自 seed 后的搜索增益"。

### 为什么是 rollout_depth=2（核心论证）

- **=1（当前默认）**：每次只扩展一步拿子节点实测延迟当 reward 回传，不评估序列——MCTS 退化成带先验的贪心单改，和 unified 拉不开差距。当前 1200s 配置即此态，是"摘低垂果实"模式。
- **=2（推荐）**：回传的 reward 至少反映一次"叠加优化"效果，能检验"MCTS 树结构比扁平搜索强"这个核心命题。是最小可证伪配置。
- **=3+**：GLM-5.2 太慢（每次 rollout 3× 调用 × ~2min × 6 方向 ≈ 单 rollout 半小时），3600s 只够 2 个 rollout，覆盖反而更差。
- **=2 是"树价值能体现"与"预算内跑得动"的最佳折中。**

### 题集（严肃对比用，缩量保深度）

- **L1**：20 道，按 op_type 覆盖（matmul/softmax/layernorm/reduction/conv2d/pooling/elementwise/scan/loss/attention 各 2 道，attention 题标记盲区）
- **L2**：10 道融合题（含 .pt 权重，覆盖 GEMM-fusion / norm-fusion / conv-fusion）
- 合计 30 题，单 GPU 串行：L1 60min×20 + L2 90min×10 = **~35 GPU·小时/方法**
- 两方法对比 ×2 = **~70 GPU·小时**（单 3090 串行约 3 天；双卡并行约 1.5 天）

### 评测指标

- **主指标**：加速比（champion latency / seed latency），每题分别报 + 全集 geomean
- **公平性校验**：两方法 LLM 调用次数 × 平均调用时长 ≈ 等量（都以时间墙 3600s 为准，记录实际调用数）
- **MCTS 特有**：树深度分布、各 depth 最优 reward（看多步叠加是否解锁更高收益）、死区早停命中率
- **诚实边界**：rollout_depth=2 仍非完整 MCTS（完整需 rollout 走到 max_depth 终局），是"最小可证伪"配置；若 MCTS 在此配置下不显著优于 unified，说明树结构价值有限，而非配置不足。

### 跑法

```bash
# MCTS 侧（DirecTune-MCTS）
conda activate forge
cd /home/wangyichen/DirecTune-MCTS
python main.py --config config_full.yaml --problem <P.json> --initial <P_initial.py> \
    --output output/full_mcts/<P> --rounds 12 --breadth 4 --num-samples 1

# 对照侧（老 DirecTune unified beamsearch）
cd /home/wangyichen/DirecTune
python main.py --config config_full.yaml --problem <P.json> --initial <P_initial.py> \
    --output output/full_unified/<P> --rounds 12 --breadth 4 --num-samples 1
```
shell 套 `timeout 3900`（3600s 搜索 + 生成/验证余量）。

---

## 2026-06-11 修复记录：Conv 流水线 Bug + 前后端 API 分离

### API 分离

`config.yaml` 新增 `model_frontend` / `model_backend`，允许 Fuser/Generator/Compose 和 Beam Search 使用不同 API：

```yaml
model_frontend:  # Fuser + Generator + Compose → AutoDL gpt-5.4
model_backend:   # Planner + Executor + Summarizer → DS4Pro
```

不配置时 fallback 到 `model`，向后兼容。

改动文件：`config.yaml`, `fuser.py:137`, `generator.py:465`, `main.py:357`, `optimize_loop.py:505`（文件已删除）, `agents.py:217/295/495`

### Conv 流水线 7 项修复

| # | 问题 | 类型 | 文件 | 修复 |
|---|------|------|------|------|
| 1 | `torch.randn(*shape)` 未归一化，符号化 shape 崩溃 | 项目逻辑 | `triton_backend.py:profile()`, `generator.py:_worker_v2/_worker`, `hardware_profiler.py:_build_ncu_bench_wrapper` | 4 处加 `_normalize_shape` 校验 |
| 2 | Fuser prompt 要求符号化 shape `["N","C","H","W"]` | 项目逻辑 | `prompts/extract_subgraphs.txt` | 改为要求具体整数值 |
| 3 | Fuser 返回空数组时崩溃 | 项目逻辑 | `main.py` | 加 fallback：跳过 dispatch/compose，直接 baseline profiling |
| 4 | Fuser dispatch 用全模型 Problem，子图 shape 不匹配 | 项目逻辑 | `main.py` dispatch 段 | 为每个子图构造独立 `Problem(sub_inputs, sub_output, ref_code)` |
| 5 | 验证环境无权重 → `'NoneType' object is not subscriptable` | 项目逻辑 | `triton_backend.py:run_isolated_validation`, `_validation_script`, `generator.py`, `main.py` | 加载 .pt → 按 shape 映射 key → 传入子进程 → 传给 ref_fn 和 kernel_fn |
| 6 | reference_code 缺少 `import torch.nn.functional as F` | 项目逻辑 | `triton_backend.py:run_isolated_validation` | 写入 reference 前自动注入标准 imports |
| 7 | `latency_ms=None` 时 format 崩溃 | 项目逻辑 | `main.py` | `{None:.4f}` → 判断后打 `N/A` |

### 验证结果 (61_ConvTranspose3d_ReLU_GroupNorm)

改前：`randn() crash` → 未进入 Generator
改后：Fuser 提取 2 子图（具体 shape）→ 权重加载+映射成功 → Generator Designer→Coder→Conductor 正常精炼 → 验证通过 → 进入 Search 阶段

剩余问题属于 **LLM 生成质量**（写不出正确的 conv_transpose3d Triton、GroupNorm 数值偏差），非流水线 bug。

---

## 待解决：Conv 生成（⚠️ 现状已更新 2026-07-01）

### 现状（2026-07-01 更新）
Level 1 含 37 道 conv 题（50-87）、Level 2 含 10 道 conv 融合题。

**早期结论（06-02，deepseek + 旧代码）已过时**：当时判"全部无法通过 Generator——LLM 从零写不出正确的 Triton 卷积"。

**当前现状（GLM-5.2 + 统一入口）**：GLM-5.2 能 gen 全类 conv，**conv 生成无盲区**。07-01 level1 覆盖测试实测：
- 50_conv_standard_2d：gen ✅，best 3.45ms（1.34×）
- 57_conv_transposed_2d（06-29 判的"唯一真盲区"）：gen ✅，best 49.11ms（0.58×，质量偏低但能写）
- 历史测试（06-29）：82 depthwise✅ / 64 transposed-1d✅ / 61 transposed-3d✅ / 74 transposed-1d-dilated✅

**模型差异**：DS4Pro 在 50/57 都 gen 失败（07-01 测得），GLM-5.2 都成功。conv gen 能力依赖模型，GLM-5.2 足够。

**当前真盲区（非 conv）**：reduction（94_mseloss）+ attention（97_SDPA）—— GLM/DS4Pro 都 gen+search 失败，详见 work-log 2026-07-01 节。

### 历史根因（保留作背景）
卷积需要正确的 stride/padding/dilation 索引计算：
```python
h_in = oh * stride_h + kh * dilation_h - pad_h
w_in = ow * stride_w + kw * dilation_w - pad_w
```
早期 deepseek 把索引关系搞错（越界崩溃 / 数值错误）。GLM-5.2 下 v3 自己会用了 im2col+GEMM（50 题 14.61ms，06-29 测得），不一定要硬塞模板。

### 为什么 KernelAgent 能跑？
KernelAgent 并非有特殊 conv 处理——同样是 LLM 纯生成。差异在于：

| 维度 | KernelAgent | DirecTune |
|------|-------------|-----------------|
| 模型 | **GPT-5 / Claude Sonnet 4** | GLM-5.2（07-01 验证足够）/ deepseek-v4-pro（早期不够）|
| 并行 worker | **4 个**竞速 | 1 个 |
| 精炼轮数 | **10 轮** | 2 轮 |
| 反 PyTorch 检测 | 同样禁止 torch.conv* | 同样禁止 |
| Conv 模板/分解 | **无** | **无**（07-01 验证不需要）|

更强的模型 + 更多重试 = 概率上覆盖更多题目，但并非从根本上解决 conv 问题。

### 方案：im2col + GEMM 模板回退（**未实施，07-01 判大概率不需要**）

卷积可以等价转换为矩阵乘法：

```
原始: input[N,C,H,W] ⊗ weight[F,C,KH,KW] → output[N,F,OH,OW]

等价:
  im2col: input[N,C,H,W] → matrix[N*OH*OW, C*KH*KW]  (仅索引重排)
  GEMM:   matrix[N*OH*OW, C*KH*KW] × weight[C*KH*KW, F] → output[N*OH*OW, F]
  reshape: output[N*OH*OW, F] → output[N,F,OH,OW]
```

**LLM 写不对索引，但我们可以提供正确的索引计算。** im2col 那步不需要计算，只需正确的索引公式。GEMM 那步用 `tl.dot` 走 TensorCore 加速。分两步拆开，每步都简单正确。

可复用资产：`/home/wangyichen/tilelang/examples/convolution/example_convolution.py` 有完整、已验证的 im2col+GEMM 实现，需翻译为 `@triton.jit`。

> **07-01 状态**：GLM-5.2 已能 gen 全类 conv（含 transposed-2d），此模板方案暂不需要。若未来切到 gen 能力更弱的模型（如 DS4Pro，07-01 测得 conv gen 全失败）可重新评估。当前优先攻关 reduction/attention 真盲区。

**实施路径（若未来需要）：**
1. 新建 `conv_templates.py`：翻译 tilelang im2col+GEMM → `@triton.jit`，覆盖 Conv1d/2d/3d + stride/padding/dilation/groups
2. 修复 `scripts/convert_kb_level2.py`：`nn.Conv2d` 被错误展开为 `F.linear`（4D 张量不能传 linear），应保持 `F.conv2d`
3. 在 `generator.py` 添加回退：LLM 全失败 + op_type 是 conv → 自动使用模板
4. Level 2 expanded reference 修复后才能正确验证 conv 融合题

## 与原 AccelOpt 的关系

本项目和 `/home/wangyichen/AccelOpt` 相互独立：
- 去掉了 flashinfer-bench 依赖，直接使用 Triton
- 新增 Fuser + Generator 阶段
- 所有 prompt 从嵌套目录扁平化为独立 .txt
- 配置从分散 JSON 集中到单一 YAML

---

## Generator 接口规范（可替换）

`generate_kernel()` 是 Generator 和 Search 之间的唯一接口，替换时只需保持此签名。

### 函数签名

```python
async def generate_kernel(
    problem: Problem,        # 问题定义（输入/输出形状、dtype、reference）
    baseline_code: str,      # 参考实现源码（PyTorch，供 LLM 理解计算逻辑）
    optimization_plan: str,  # 实现方案（操作链、形状、权重、Triton 实现要点）
    config: dict,            # config.yaml 全量字典
) -> dict | None:
```

### 返回值

```python
# 成功：返回包含正确 Triton 内核的字典
{
    "worker_id": 0,              # int: worker 编号（单 worker 时为 0）
    "code": "...",               # str: 完整 Triton Python 源码
    "latency_ms": 0.405,         # float: 基准测试延迟（毫秒）
    "refinement_rounds": 2,      # int: 精炼轮数（0 = 首轮直接通过）
}

# 失败：返回 None
None
```

### 内部流程（当前实现）

```
_worker()
  ├─ 1. LLM 生成 Triton 代码
  │     model=config["model"]["model"]
  │     system=prompts/executor_system.txt (含 Triton API 参考、硬件参数)
  │     user=prompts/executor_user.txt  (注入 {problem_code}, {kernel_code}, {optimization_plan})
  │     温度=config["gen_temperature"] (首次) / 0.3 (精炼)
  │     max_tokens=20000
  │
  ├─ 2. 反 PyTorch 检测 (anti_pytorch_check)
  │     22 个正则，拒绝 torch.matmul/nn.Module/F.relu 等
  │
  ├─ 3. 编译 (compile_triton)
  │     导入源码 → 查找 def run(*args): 函数
  │     返回 (callable, error_message)
  │
  ├─ 4. 隔离验证 (run_isolated_validation)
  │     子进程中运行 kernel + reference → 对比输出
  │     失败返回 ValidationResult (failure_kind: compile_error/runtime/numerical_mismatch/...)
  │
  ├─ 5. 精炼循环 (最多 config["gen_refinement_rounds"] 轮)
  │     失败反馈 → LLM 修正 → 回到步骤 2
  │
  └─ 6. 基准测试 (benchmark)
        warmup=10, reps=100, CUDA event 测时
```

### 依赖的内部函数

| 函数 | 位置 | 签名 | 用途 |
|------|------|------|------|
| `anti_pytorch_check` | `triton_backend.py:55` | `(code: str) -> str \| None` | 扫描生成代码中的 PyTorch 调用 |
| `compile_triton` | `triton_backend.py:215` | `(source: str) -> tuple[Callable \| None, str \| None]` | 从源码加载 `run()` 函数 |
| `run_isolated_validation` | `triton_backend.py:365` | `(kernel_source, reference_source, problem, rel_tol, abs_tol, ...) -> ValidationResult` | 子进程隔离验证 |
| `benchmark` | `triton_backend.py:224` | `(fn: Callable, *args, warmup, reps, device) -> float` | CUDA event 测延迟 |
| `validate_problem_shapes` | `triton_backend.py:134` | `(problem: Problem) -> tuple[bool, str \| None]` | 验证输入形状合法性 |

### 两个调用点的区别

| | Fuser dispatch (main.py) | Search 阶段 (agents.py) |
|---|---|---|
| `optimization_plan` | `_build_dispatch_plan(sg)` 生成的结构化方案 | Planner LLM 基于当前候选+经验生成的优化方案 |
| `baseline_code` | 子图的 reference_code（PyTorch） | 当前最佳候选的 code（可能是 Triton） |
| 目的 | **从零生成** Triton kernel | **优化现有** Triton kernel |
| 失败处理 | 跳过该子图，最终可能回退到 PyTorch baseline | 保留上一轮候选 |

### 替换要求

替换 `generate_kernel()` 只需保证：
1. **函数签名不变** — `(problem, baseline_code, optimization_plan, config) -> dict | None`
2. **返回值结构不变** — 至少包含 `{"code": str, "latency_ms": float}`
3. **正确性保证** — 返回的 code 必须是已经过验证的正确 Triton 内核
4. **延迟测量** — latency_ms 必须通过 CUDA event 等可靠方式测出
5. **异步** — 保持 `async`，内部 LLM 调用使用 `asyncio.sleep` 做指数退避重试

---

## v5 vs 原始 AKG 对拍工作流（2026-06-25）

### 工作流
- `akg_compare_test.py`：AKG 侧，`LangGraphTask(task_type="profile")` 让 verifier 节点跑 `run_profile`，接住 `coder_code` + `profile_res`（speedup）。`bridge_model_config`（config.yaml → AIKG_* env，绕过 .akg/settings.json 的 deepseek 余额耗尽）+ PATH 修复（forge bin 加到 PATH，解决 subprocess 调 `python` 失败导致 speedup=0.0）。
- `compare_report.py`：主驱动，subprocess 跑 AKG + v5（main.py --rounds 2 --breadth 1），`bench_baseline` 测统一 PyTorch baseline，断点续跑。
- 12 题（8 能生成 + 4 难题 28/30/37/99）。产物 `output/compare/report.md`。

### 结果
- **成功率**：AKG 11/12，v5 **8/12**（v5 不及 AKG）——难题组 0/4 生成出 Triton
- **加速比**（能生成组 8 题）：v5 明显提升 4 题（14/45/55/76）、持平 2 题、略输 2 题
  - 14 题 33×（search 发现 GEMM+Sum→matvec 等价融合，计算量减 8000×，"算得更少"非"算得更快"）
- **v5_passed 判定 bug**（已修）：原 `latency>0` 误判退化为 reference 的题为 passed；改为检查 best candidate code 含 `@triton.jit`

### 关键发现：level2 用 v3 是接口妥协
v5 level2 generator 用 **v3**（非 AKG 前端），因 AKG 前端与 level2 `.pt` 权重接口不兼容：
- AKG 前端产出 `ModelNew(manual_seed0)`，v5 用 `.pt` frozen 权重（seed42）+ `run_isolated_validation` → 权重不一致 → 验证失败
- → `main.py:235/475` 强制 level2 `gen_mode=v3`
- v3 在 GroupNorm/Softmax 上生成弱 → 难题组退化 reference
- `akg_compare_test` 证明 AKG 前端**能**生成 level2 正确 Triton（给 AKG Model class，AKG 用 seed0 自对齐 Model↔ModelNew，绕过 .pt）

### shim 已实现（2026-06-26）：v5 level2 用 AKG 前端，成功率追平 AKG
`generator.py:_generate_kernel_akg` 加 shim：注入 task_desc（Model class + get_init_inputs）+ `load_state_dict(.pt)` 到 Model 后**按参数顺序+形状 zip copy** 到 ModelNew（绕过 ModelNew 改层名导致 `load_state_dict` key 不匹配）+ `def run(x, *args)` 缓存 `_MODEL`（修复 `_patch_device_in_source` 对 `def run(*args)` 注入 `_dev=x.device` 的 NameError）。`main.py:234` level2 二次生成 gen_mode `v3`→`akg`。
**结果**：v5 成功率 8/12→**11/12（追平 AKG）**，难题组 3/4 生成 Triton（30 追平 1.67×、37 反超 1.92×、99 追平 1.72×，28 多输入接口 bug 除外），加速比 v5 geomean 2.30× vs AKG 1.51×（提升 1.53×，v5 胜 8/10）。
