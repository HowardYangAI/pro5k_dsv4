# 会话交接：早期正式 Prefill/Decode Profile

## 定位

这是第一次按 Prefill 与 Decode 分开、自动停止的正式采集：输入 8192→1 和 1→128，warmup 10、active 5 scheduler steps，TP8/Attention DP8/EP8 + DSpark 6 draft tokens。它验证了分阶段采集流程和 kernel 到 Python 路径映射。

## 当前状态

该结果已被后续 `20260819_dsv4_ep8_dspark_c8/` 的 C8 正式采集替代。后续分析优先引用 C8 目录；本目录仅用于追溯早期结果和源码映射，不用于与最终 C8 百分比直接比较。

## 主要观察

- 早期 Prefill 窗口中 NCCL AllReduce 占比异常高，说明窗口长度和 rank 同步等待会显著影响 duration 占比。
- Decode 中 AllReduce、小 N GEMM、NCCL Reduce/AllGather 和 MoE GEMM 是主要项目。
- 百分比是 TP0 的 GPU kernel duration 求和，不是墙钟时间；多 stream 可能重叠。

## 文件入口

- `delivery/README.md`：完整 Top kernel 表和采集口径。
- `delivery/01_*_timeline_tp0.csv`：代表 Rank 时间序列。
- `delivery/02_*_kernel_aggregate_tp0.csv`：Prefill/Decode 聚合表。
- `delivery/03_skill_triage_report.md`：kernel、overlap、fuse 三表分析。
- `prefill/`、`decode/`：本机完整 8-rank Trace。

## 使用限制

- 不要把这组 AllReduce 占比和后续 C8 结果做绝对性能对比；采集窗口和工作量不同。
- 不要从 Profile duration 推导 Output TPS；端到端性能以无 Profiler benchmark 为准。
