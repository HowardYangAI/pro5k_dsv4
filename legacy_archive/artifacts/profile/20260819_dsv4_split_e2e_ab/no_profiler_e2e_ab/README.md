# PR #32320 无 Profiler 完整端到端 A/B

负载：C8、40 requests、8192 input tokens、64 output tokens、TP8/Attention DP8/EP8、DSpark 6 draft tokens、Decode CUDA Graph。Split ON/OFF 使用相同的 3 组 seed，每轮均成功 40/40。

| 指标（三轮均值） | Split ON | Split OFF | ON 相对 OFF |
| --- | ---: | ---: | ---: |
| Benchmark duration | 66.80 s | 72.72 s | +8.15% |
| Output TPS | 38.36 tok/s | 35.32 tok/s | +8.61% |
| Mean E2E | 13208.47 ms | 14427.23 ms | +8.45% |
| Mean TTFT | 3750.86 ms | 3722.80 ms | -0.75% |
| Mean TPOT | 150.12 ms | 169.91 ms | +11.65% |
| Accept length | 2.66 | 2.84 | -6.33% |

逐 seed 的 Output TPS 收益分别为：1.27%, 21.48%, 4.50%；配对中位数为 4.50%。

结论：三轮均值下 Split ON 的 Output TPS 高约 8.61%，完成时间低约 8.15%，Mean TPOT 低约 11.65%。但逐轮收益波动较大，且 ON 的平均 Accept length 更低。结合配对中位数和此前 Decode-only 结果，更稳妥的结论是“优化有端到端收益，量级约为几个百分点”；不使用单轮 Profile 得到的 19.22% 作为确定收益。
