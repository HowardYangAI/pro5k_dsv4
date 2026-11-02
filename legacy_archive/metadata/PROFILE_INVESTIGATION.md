# 端到端 Torch Profile 定位记录

## 已确认现象

- 相同的端到端 TP8 + Attention DP8 + EP8 + DSpark 服务，不开启 Profile 时 C4 和 C8 均成功。
- `bench_serving.py --profile` 的 C4 压测曾触发 GPU 2 的 Xid 13：`Graphics Exception: EXTRA_INLINE_DATA`，随后出现 `CUDA error: unspecified launch failure` 并导致服务退出。
- 回移 PR #32320 前后均出现过该现象，因此目前没有证据表明 page-split PR 是根因。

## 代码层发现

普通 `--profile` 会调用 `/start_profile`，而当前 SGLang 默认同时启用：

- CPU + GPU activities
- `with_stack=true`
- `record_shapes=true`
- 8 个 TP/DP/EP rank 同时采集

这比“只采 GPU kernel”明显更重，会改变同步、内存与 trace 行为。当前最可能的方向是 SM120/PyTorch profiler 与 CUDA Graph 或某个异步 GPU kernel 的组合问题，但尚未完成逐项证伪，不能作为最终根因。

## 回收前最后一次实验

已改为手工调用 `/start_profile`：

```json
{
  "activities": ["GPU"],
  "with_stack": false,
  "record_shapes": false
}
```

并发 C4、请求数 4、输入 8192、输出 64。结果为 4/4 请求成功，服务保持健康，并正常生成 8-rank trace：

- Benchmark duration：4.97 s
- Output token throughput：51.51 tok/s
- Mean TTFT：3810.44 ms
- Mean TPOT：14.38 ms

- `artifacts/profile/20260819_ep8_e2e_torch_profile/`

因此已确认：GPU-only、关闭 stack/shape 的最小 Torch Profile 可以正常工作。问题并非由 Profile 开关本身必然触发，更可能来自默认 CPU+GPU、stack、shape 的组合，或较长采集窗口。仍需在新机器完成下述矩阵才能锁定具体触发项。

## 后续建议矩阵

均先使用 C4 × 4，发生 Xid 后重启服务：

1. GPU only，stack=false，shapes=false。
2. CPU+GPU，stack=false，shapes=false。
3. CPU+GPU，stack=true，shapes=false。
4. CPU+GPU，stack=false，shapes=true。
5. 默认 CPU+GPU，stack=true，shapes=true。

若第 1 组仍失败，使用 `CUDA_LAUNCH_BLOCKING=1` 重启服务，以定位真正失败的异步 kernel。

## 后续正式采集结果

已使用有界 Profile 完成正式 C8 采集：

- Prefill：C8、8 requests、8192→1、6 scheduler steps。
- Decode：C8、warmup 40 requests、正式 80 requests、1→128、12 scheduler steps。
- Profile：CPU+GPU、`with_stack=false`，Decode 使用 `record_shapes=false`，由 `num_steps` 自动停止。
- Decode CUDA Graph 与 DSpark target/draft graph 均保持开启。

该方式成功生成 8-rank Trace，服务未触发 Xid 13。说明此前风险主要来自无边界/重型采集与收尾压力，而不是 Torch Profiler 或 CUDA Graph 必然不可用。

## PR #32320 Split ON/OFF

相同 C8、warmup 40、正式请求 80、1→128、12-step Profile 下：

| 指标 | Split ON | Split OFF |
| --- | ---: | ---: |
| `_page_split_kernel` 调用次数 | 503 | 503 |
| `_page_split_kernel` 总时间 | 7.576 ms | 57.219 ms |
| `_page_split_kernel` 平均时间 | 0.0151 ms | 0.1138 ms |
| TP0 GPU kernel duration 求和 | 673.767 ms | 746.606 ms |

Split ON 额外 `_page_mark_kernel` 总开销仅 0.584 ms。关闭优化后，page-split 路径净增加约 49.06 ms，TP0 kernel duration 求和增加 10.81%，因此应保留 PR #32320。
