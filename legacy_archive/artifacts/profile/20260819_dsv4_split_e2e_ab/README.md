# DeepSeek-V4-Flash PR #32320 完整端到端 A/B Profile

两组均在同一个 Profile 窗口中完整覆盖：

```text
8192-token Prefill → 首 token → DSpark Decode → 64-token 请求完成
```

统一配置为 TP8 / Attention DP8 / EP8、DSpark 6 draft tokens、Decode CUDA Graph、C4、warmup 40、正式 4 requests。Split OFF 使用 PR #32320 的父提交，并补回相同的 DSpark+EP batch 元数据修复；A/B 的 page-split 运行时代码之外保持一致。

## Profile 窗口内的端到端请求结果

| 指标 | Split ON | Split OFF | ON 相对 OFF |
| --- | ---: | ---: | ---: |
| Benchmark duration | 5.09 s | 5.45 s | -6.61% |
| Output throughput | 50.34 tok/s | 46.98 tok/s | +7.15% |
| Mean E2E latency | 4,929.35 ms | 5,328.44 ms | -7.49% |
| Mean TTFT | 3,793.42 ms | 3,922.02 ms | -3.28% |
| Mean TPOT | 18.03 ms | 22.32 ms | -19.22% |

该表每组只有一轮，并受 DSpark accept length（ON 2.90、OFF 3.13）和 Profiler 扰动影响，只作为方向性观测，不能将 TPOT 的 19.22% 差异归因于 PR #32320。

## 无 Profiler 三轮完整端到端结果

相同 `C8、40 requests、8192→64` 负载下，Split ON/OFF 使用相同三组 seed：

| 指标（三轮均值） | Split ON | Split OFF | ON 改善 |
| --- | ---: | ---: | ---: |
| Benchmark duration | 66.80 s | 72.72 s | 8.15% |
| Output TPS | 38.36 tok/s | 35.32 tok/s | 8.61% |
| Mean E2E | 13,208.47 ms | 14,427.23 ms | 8.45% |
| Mean TPOT | 150.12 ms | 169.91 ms | 11.65% |

逐 seed Output TPS 收益为 1.27%、21.48%、4.50%，配对中位数为 4.50%，说明收益存在但方差较大。更稳妥的表述是“端到端收益量级约为几个百分点”，而不是 19.22%。完整日志和统计见 `no_profiler_e2e_ab/`。

## TP0 GPU kernel 结果

| 指标 | Split ON | Split OFF | 变化 |
| --- | ---: | ---: | ---: |
| GPU kernel duration 求和 | 4,632.52 ms | 5,091.22 ms | OFF 增加 9.90% |
| `_page_split_kernel` | 37.18 ms | 344.85 ms | OFF 增加 827.41% |
| `_page_mark_kernel` | 35.04 ms | 0 | ON 新增 touched-page 标记 |
| page mark + split 净开销 | 72.23 ms | 344.85 ms | ON 降低 79.06% |

结论：PR #32320 在完整端到端窗口中仍有明确收益。它新增约 35.04 ms 的 touched-page 标记，但将 page split 从 344.85 ms 降到 37.18 ms，净节省 272.63 ms；TP0 全部 GPU kernel duration 求和下降约 9.0%（反向口径为 OFF 比 ON 高 9.90%）。

## 文件

- `split_on_c4_8k64_e2e/`、`split_off_c4_8k64_e2e/`：两组时间顺序表、聚合表、三表分析和可直接打开的 Perfetto Trace。
- `05_e2e_split_on_off_kernel_comparison_tp0.csv`：逐 kernel A/B 对比。

更大的 `C8 × 8/16 requests × 8192→128` 无边界完整采集在当前 SM120 驱动环境触发 Xid 13，未形成有效 Trace；C4 版本保留了完整请求生命周期，同时可稳定落盘。
