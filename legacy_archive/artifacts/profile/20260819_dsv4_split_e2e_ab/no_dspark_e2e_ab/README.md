# PR #32320：关闭 DSpark 的完整端到端 A/B

## 测试目的

关闭 DSpark 后复测 Split ON/OFF，排除 draft/verify、accept length 和投机接受率波动，判断 PR #32320 的端到端收益是否仍存在。

测试负载为 C8、40 requests、8192 input、64 output；TP8 / Attention DP8 / EP8；Decode CUDA Graph；无 Profiler。ON/OFF 使用三个相同 seed 做配对比较。

## 结果

| 指标 | Split ON，均值 ± 标准差 | Split OFF，均值 ± 标准差 | 配对中位收益 |
|---|---:|---:|---:|
| Output TPS | 75.60 ± 2.10 tok/s | 65.16 ± 4.49 tok/s | ON +14.35% |
| 完成时间 | 33.88 ± 0.95 s | 39.42 ± 2.83 s | ON -12.54% |
| Mean E2E | 6762.02 ± 188.26 ms | 7806.65 ± 467.95 ms | ON -12.44% |
| Mean TTFT | 5199.36 ± 177.21 ms | 5305.71 ± 154.98 ms | ON -1.10% |
| Mean TPOT | 24.80 ± 1.38 ms | 39.70 ± 9.88 ms | ON -31.59% |

关闭 DSpark 后，Split ON 三轮 Output TPS 为 73.33 / 77.46 / 76.02，变异系数约 2.8%，结果稳定。Split OFF 前两轮约 67.75 tok/s；seed 1787128003 两次均出现约 4.8 秒的长 ITL，Output TPS 分别为 59.97 和 60.24，说明该现象可重复，不是 DSpark 接受率导致的随机波动。

## 为什么“算子占比看起来低”，端到端仍有明显收益

不能用 Split ON 优化后的 0.80% 占比反推优化前影响很小。已有同口径 Profile 显示：

- Split OFF 的 `_page_split_kernel` 为 344.85 ms，占 TP0 GPU kernel 时间 6.77%。
- Split ON 降至 37.18 ms，占 0.80%；新增 touched-page 标记约 35.04 ms。
- 单次 C4、4-request Profile 的净节省为约 272.63 ms；40-request 压测会重复更多轮，累计影响会放大。

原实现每个 Decode step 都重写整个 SWA KV pool。它是显存带宽型、位于 Attention 前的串行关键路径；后续 Attention 不能提前执行，各 DP/EP rank 的 collective 也会等待慢 rank。因此它不仅消耗自己的 kernel duration，还会延后后续 kernel、增加同步等待。PR #32320 改为只标记并拆分本轮实际访问的 SWA pages，工作量从“整个池”降到“本轮 touched pages”。

所以准确结论是：算子在 **ON 结果中** 占比低，正是优化已经生效；在 OFF 中它并不低，而且位于每轮重复执行的关键路径。当前关闭 DSpark 的 A/B 支持 #32320 存在明确端到端收益，但三组样本仍较少，建议正式宣称使用配对中位数（Output TPS +14.35%），不要用单轮最大值 +26.76%。

## 文件

- `01_runs.csv`：全部原始指标，包含 seed 1787128003 的复测。
- `02_summary.csv`：三组主配对样本的均值和样本标准差。
- `03_paired_comparison.csv`：逐 seed 的 ON/OFF 收益。
- `raw_logs/`：bench_serving 原始输出。
