# Split OFF 完整端到端 Profile

同一个 Profile 窗口覆盖 `8192-token Prefill → 首 token → DSpark Decode → 64-token 请求完成`。配置为 TP8 / Attention DP8 / EP8、DSpark 6 draft tokens、Decode CUDA Graph、C4、warmup 40、正式 4 requests。

正式结果：4/4 请求成功，Benchmark duration 5.45 s，Output throughput 46.98 tok/s，Mean E2E 5,328.44 ms，Mean TTFT 3,922.02 ms，Mean TPOT 22.32 ms，Accept length 3.13。

- `01_e2e_operator_timeline_tp0.csv`：完整端到端时间顺序表。
- `02_e2e_kernel_aggregate_tp0.csv`：完整窗口 kernel 聚合表。
- `03_e2e_8k_to_64_tp0_perfetto_lite.trace.json.gz`：可直接拖入 Perfetto UI。
- `04_skill_triage_report.md`：统一三表分析。
