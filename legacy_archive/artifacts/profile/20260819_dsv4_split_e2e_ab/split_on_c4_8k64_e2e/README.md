# DeepSeek-V4-Flash 完整端到端 Torch Profile

本目录不是 Prefill/Decode 分段拼接结果。同一个 Profile 窗口完整覆盖：

```text
8192-token Prefill → 首 token → DSpark Decode → 64-token 请求完成
```

## 采集配置

- 并行：TP8 / Attention DP8 / EP8
- 推测解码：DSpark，6 draft tokens
- CUDA Graph：Decode target/draft 均开启；DeepSeek-V4 Prefill CUDA Graph 按服务规则关闭
- 并发与请求：C4，正式 4 requests
- Warmup：采集前完成 40 个同形状 `8192→64` 请求，warmup 不进入 Trace
- Profiler：GPU only，`with_stack=false`，`record_shapes=false`，`profile_by_stage=false`
- 代表 Rank：TP0 / DP0 / EP0；原始目录保留全部 8-rank Trace

## 正式请求结果

| 指标 | 结果 |
| --- | ---: |
| Successful requests | 4 |
| Benchmark duration | 5.09 s |
| Input tokens | 32,768 |
| Output tokens | 256 |
| Output throughput | 50.34 tok/s |
| Mean E2E latency | 4,929.35 ms |
| Mean TTFT | 3,793.42 ms |
| Mean TPOT | 18.03 ms |
| Accept length | 2.90 |

## 文件

- `01_e2e_operator_timeline_tp0.csv`：完整端到端 GPU kernel 时间顺序表。
- `02_e2e_kernel_aggregate_tp0.csv`：完整窗口算子聚合表。
- `03_e2e_8k_to_64_tp0_perfetto_lite.trace.json.gz`：可直接拖入 Perfetto UI 的轻量 Trace。
- `04_skill_triage_report.md`：统一 LLM Torch Profiler 三表分析。
- `profile_request.json`：Profiler 请求参数。

完整端到端 TP0 GPU kernel 时间中，NCCL AllReduce 占 20.80%，NCCL Reduce 占 19.88%，两者合计 40.68%；随后是 SM120 CUTLASS MoE GEMM（10.64%）和 small-N GEMM（7.25%）。

更大的 `C8 × 8/16 requests × 8192→128` 完整窗口在当前 SM120 驱动环境均触发 Xid 13，未形成有效 Trace；因此本目录使用能够稳定落盘且仍覆盖完整请求生命周期的 C4 基线。
