# DeepSeek-V4-Flash Profile 资料总索引

更新日期：2026-08-19（UTC）

本文是后续会话读取 Profile 资料的唯一总入口。所有百分比必须先确认分母：Torch Profile 的 kernel 占比以单个代表 Rank 的 GPU kernel duration 求和为分母，多 stream 可以重叠，不能直接当作墙钟时间或端到端收益。

## 建议阅读顺序

| 顺序 | 目录 | 用途 | 当前状态 |
|---:|---|---|---|
| 1 | `20260819_ep8_e2e_torch_profile/` | 确认 SM120 上 Profiler 的安全采集边界与 Xid 13 规避方式 | Profiler 方法论依据 |
| 2 | `20260819_dsv4_ep8_dspark_c8/` | 查看 TP8/Attention DP8/EP8 + DSpark 的正式 Prefill/Decode kernel 构成 | 当前正式阶段 Profile |
| 3 | `20260819_dsv4_split_ab/` | 查看 PR #32320 对 Decode page-split kernel 的严格 ON/OFF 影响 | kernel 层 A/B 依据 |
| 4 | `20260819_dsv4_split_e2e_ab/` | 查看 PR #32320 的完整请求生命周期和无 Profiler 端到端收益 | 当前最终性能结论 |
| 5 | `20260819_dsv4_ep8_dspark_formal/` | 早期 warmup 10/active 5 分阶段采集，保留用于追溯 | 已被 C8 正式采集替代 |

## 当前可引用结论

1. TP8/Attention DP8/EP8 + DSpark 的 Profile 中，通信和小 batch GEMM 是主要开销；不同采集窗口的百分比不能直接横比。
2. PR #32320 将每个 Decode step 的行为从“拆分整个 SWA KV pool”改为“只拆分本轮 touched pages”。严格 Decode A/B 中，关闭优化后 `_page_split_kernel` 从 7.576 ms 增至 57.219 ms，TP0 GPU kernel duration 求和增加 10.81%。
3. 完整端到端 Profile 中，page mark + split 的净开销从 344.85 ms 降至 72.23 ms，降低 79.06%；这是 kernel 证据，不是端到端吞吐百分比。
4. 带 DSpark 的无 Profiler 三轮 A/B 方差较大，Output TPS 配对中位收益为 4.50%。关闭 DSpark 后，三组配对测试的 Output TPS 中位收益为 14.35%，完成时间中位降低 12.54%；后者样本仍只有三组，不应引用单轮最大收益 26.76%。
5. Split ON 中 `_page_split_kernel` 只占约 0.8%，是优化已经生效后的结果；Split OFF 中占 6.77%，且它位于每轮 Attention 前的串行关键路径，不能用 ON 占比反推优化无价值。

## Profiler 采集规则

- 不再使用 `bench_serving.py --profile` 对整场 benchmark 做无边界 CPU+GPU 采集。
- 使用 `/start_profile`，设置 `num_steps`，让 Profiler 自动停止和落盘。
- Prefill 与 Decode 优先分开采集；默认先 warmup 10 steps，再采 active 5 steps。
- 正式对比必须保留相同模型、并行配置、CUDA Graph、请求形状、并发、请求数和 seed。
- 判断端到端收益时使用无 Profiler benchmark；Trace 只用于解释收益来自哪些 kernel/同步路径。

## Git 归档范围

Git 目录为 `/sgl-workspace/workspace/overall/artifacts/profile/`。大型 8-rank 原始 Trace 没有全部提交；已提交代表 Rank 的轻量 Perfetto Trace、CSV、分析报告、采集参数与原始 benchmark 日志。未提交 Trace 的原始路径、大小和 SHA256 见 `20260819_RAW_TRACE_MANIFEST.tsv` 以及各实验的 `RAW_TRACE_MANIFEST.tsv`。

每个实验根目录的 `SESSION_HANDOFF.md` 说明该实验的目的、入口、结论与限制。后续会话应先读本文件，再读目标实验的交接说明。
