# 会话交接：C8 正式分阶段 Profile

## 定位

这是 TP8/Attention DP8/EP8 + DSpark 6 draft tokens 的当前正式阶段 Profile，用于回答 Prefill 和 Decode 分别由哪些计算与通信 kernel 主导。

## 采集口径

- Prefill：C8、8 requests、8192→1，active 6 scheduler steps。
- Decode：C8、8 requests、1→128，active 8 scheduler steps。
- Decode CUDA Graph 开启；CPU+GPU 有界自动采集。
- 代表 Rank：TP0/DP0/EP0。

## 主要结论

- Prefill 中 NCCL AllReduce + Reduce 是主要通信开销，随后是 SM120 CUTLASS GEMM 和 sparse MLA。
- Decode 中 AllReduce、Reduce、AllGather、小 N GEMM 与 SM120 CUTLASS GEMM 占主要部分。
- `_page_split_kernel` 在 Split ON 的这组采集中占比不足 1%；它是优化后结果，不能用来判断优化前开销。
- 这组 Profile 用于解释 kernel 构成，不用于给出 PR #32320 的端到端收益；A/B 请读取 `20260819_dsv4_split_ab/` 和 `20260819_dsv4_split_e2e_ab/`。

## 文件入口

1. `delivery/README.md`：采集配置、Top kernel 和 Perfetto 路径。
2. `delivery/01_operator_timeline_tp0.csv`：Prefill/Decode 合并时间序列。
3. `delivery/02_prefill_kernel_aggregate_tp0.csv`、`03_decode_kernel_aggregate_tp0.csv`：聚合表。
4. `delivery/03_skill_triage_report.md`：kernel、overlap、fuse 三表分析。
5. `delivery/perfetto_lite/`：Git 可归档的代表 Rank Trace。

## 限制

- 单 Rank duration 可能包含等待慢 rank 的通信时间。
- 多 stream duration 可以重叠，类别占比不能与端到端延迟逐项相加。
- 完整 8-rank Trace 体积较大，Git 只保存代表 Rank 轻量 Trace；清单见上级 `20260819_RAW_TRACE_MANIFEST.tsv`。
