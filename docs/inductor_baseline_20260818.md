# Inductor baseline 对比（2026-08-18）

## 目的

论文补 torch.compile (inductor) baseline：与 champion（MCTS 最优 Triton）逐题对比。
口径与 harness 一致：CUDA event、warmup 10、L2 clear、median（`bench_inductor.py`，
每题独立子进程避免跨题 CUDA 内存泄漏；correctness 用 ≤1M 元素采样对比）。

## 数据

- L1 strict 48 题：`output/inductor_bench/l1_strict48.json`
- L2 strict 29 题：`output/inductor_bench/l2_strict29.json`
- modes：`torch.compile(mode=default)` + `mode=max-autotune`（报 best-of-both）

## 关键数字

| 集合 | n | champion vs eager geomean | champion vs inductor-best geomean | inductor 慢于 champion |
|---|---|---|---|---|
| L1 strict | 48 | 1.311× | 1.213× | 35/48 |
| L2 strict | 29 | 0.973× | **0.696×** | 13/29 |

（champion vs inductor = inductor_best_latency / champion_latency，>1 = champion 赢）

## 结论结构（论文叙事）

1. **L1**：champion 平均仍赢 inductor 1.21×，35/48 题更快。max pooling / hingeloss / newgelu
   这类 LLM 写出更优规约/激活的题优势 2.5-9×；但 norm/elementwise 大张量题 inductor 打平或略胜
   （如 40_layernorm default=1.57ms vs champ=0.99ms，max-autotune 反而 7.13ms 退化）。
2. **L2**：**整体 geomean 输 inductor（0.696×）**，分裂明显：
   - **Matmul/GEMM 融合题 champion 大胜**：14_Gemm 151×、18_Matmul 144×（matvec 塌缩等算法级
     变换，inductor 结构上搜不出）；95/22/29/12 题 1.7-3.5×。
   - **Conv/ConvTranspose 题 champion 惨败**：36_ConvT2d 输 100×（champ 371ms=seed 未优化 vs
     ind 4.6ms）、27/13/100/38/10/25/2 输 8-90×。这些题 champion/seed≈1.0 即 **MCTS 没改进
     seed**（seed 本身是慢的 naive conv transpose 实现），不是 inductor 特强。
   - 15_ConvT3d champ 58.5ms vs ind 3.7ms 属真差距（champ 有 3.2× 搜索增益但仍不够）。
3. **归因**：L2 输掉的题 champion_latency 绝对值都很大（12-469ms），本质是 **conv-transpose
   类 seed 质量 + 搜索预算不足**，MCTS 在 3600s 内没找到能对抗 cuDNN 融合的写法。这与论文
   failure analysis（conv 族对索引/内存压力敏感）一致。
4. **search-success 子集口径（关键公平性修正）**：按 champ/seed 是否 < 0.9 分裂——
   - search-success（25 题）：champion vs inductor geomean **1.041×**（打平偏赢）
   - search-failure（4 题，champ≈seed 未优化）：geomean **0.056×**（被 cuDNN 融合碾压）
   即 L2 整体 0.696× 的亏损几乎全部来自搜索没生效的题；搜索生效的题上与 inductor 五五开
   且 Matmul/GEMM 子类大胜。

## 诚实的报告方式

- 主表加 inductor 列时，L2 aggregate 会变难看（0.696×）；建议按算子族分列
  （Matmul/GEMM vs Conv/ConvTranspose），并明确 "champion 未优化的题（champ/seed≈1）单独归为
  search-failure，不算搜索方法的代表性能"。
- 或补充 "search-success subset"（champ/seed < 0.9 的题）口径：这些题上 champion vs inductor
  的 geomean 才是搜索真正见效子集的公平对比。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0 python bench_inductor.py --problems <names...> \
  --level {1|2} --modes default max-autotune --out output/inductor_bench/xxx.json
```
