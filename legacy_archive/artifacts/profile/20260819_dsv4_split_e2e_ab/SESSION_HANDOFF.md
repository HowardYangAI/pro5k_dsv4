# 会话交接：PR #32320 完整端到端 A/B

## 这是最终结论目录

本目录把 `8192-token Prefill → 首 token → Decode → 请求完成` 放在同一请求生命周期中，同时保留 Torch Profile 的 kernel 证据和无 Profiler 的真实性能数据。后续讨论 PR #32320 收益时优先从这里开始。

## 证据分层

### 1. 完整生命周期 Torch Profile

- C4、warmup 40、正式 4 requests、8192→64。
- TP8/Attention DP8/EP8 + DSpark 6 draft tokens，Decode CUDA Graph 开启。
- Split ON/OFF 除 page-split 代码外保持一致。
- page mark + split：ON 72.23 ms，OFF 344.85 ms，ON 降低 79.06%。
- TP0 GPU kernel duration 求和：ON 4632.52 ms，OFF 5091.22 ms，OFF 增加 9.90%。

这部分用于解释 kernel 原因；单轮 Profile 的 Output TPS 和 TPOT 受 Profiler 与 DSpark accept length 干扰，不能作为最终收益。

### 2. 带 DSpark、无 Profiler三轮 A/B

- C8、40 requests、8192→64，三个相同 seed。
- Output TPS 均值：ON 38.36，OFF 35.32 tok/s。
- 逐 seed 收益：1.27%、21.48%、4.50%；配对中位数 4.50%。
- 结论：收益存在但 DSpark 接受率和运行波动较大，正式表述为“几个百分点”。

### 3. 关闭 DSpark、无 Profiler三轮 A/B

- 其他负载相同，只移除 speculative/draft/verify。
- Output TPS：ON 75.60±2.10，OFF 65.16±4.49 tok/s。
- Output TPS 配对中位收益 14.35%；完成时间中位降低 12.54%。
- Split OFF 的 seed 1787128003 两次均复现约 4.8 秒长 ITL，说明该停顿不是 DSpark accept length 导致。
- 样本只有三组，不能把单轮最大收益 26.76% 作为稳定结论。

## 文件入口

- `README.md`：完整主结论。
- `split_on_c4_8k64_e2e/`、`split_off_c4_8k64_e2e/`：时间序列、聚合表、三表分析和 Perfetto Trace。
- `05_e2e_split_on_off_kernel_comparison_tp0.csv`：逐 kernel 对比。
- `no_profiler_e2e_ab/`：带 DSpark 的三轮 benchmark、CSV 和原始日志。
- `no_dspark_e2e_ab/`：关闭 DSpark 的三轮 benchmark、复测、CSV 和原始日志。
- `RAW_TRACE_MANIFEST.tsv`：未提交大型 Trace 的路径、大小和 SHA256。

## 关键解释

Split ON 中 `_page_split_kernel` 只占 0.80%，是优化生效后的结果；Split OFF 中为 6.77%。原实现每个 Decode step 复制整个 SWA KV pool，且该操作位于 Attention 前的串行关键路径，会延后后续 kernel 和跨 rank collective。40 个请求会把每 step 成本累计放大，因此“优化后算子占比低”与“端到端有收益”并不矛盾。
