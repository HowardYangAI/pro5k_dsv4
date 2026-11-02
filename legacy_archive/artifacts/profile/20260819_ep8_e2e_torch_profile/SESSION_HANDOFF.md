# 会话交接：Profiler 稳定性与采集边界

## 这组实验解决什么问题

定位为什么普通 C4/C8 推理正常，而网页或 `bench_serving.py --profile` 的完整 CPU+GPU Profile 会导致服务失去响应并出现 NVIDIA Xid 13。

## 配置与实验范围

- 模型：DeepSeek-V4-Flash-0731
- 并行：TP8 / Attention DP8 / EP8 + DSpark 6 draft tokens
- Decode CUDA Graph：`full`
- 代表有效 Trace：`traces/cpu_gpu_by_stage_5_8k/`
- 有效负载：输入 8192、输出 64、C4×4，Prefill/Decode 各采 5 个 scheduler step

## 结论

- 普通推理、GPU-only、CPU-only、有界 CPU+GPU Profile 均可成功。
- 故障集中在无 `num_steps` 的长时间 CPU+GPU 采集及手工 stop/flush；stack/shape 会增大开销，但不是唯一触发条件。
- 安全做法是手工调用 `/start_profile`，设置 `num_steps` 并让 Profiler 自动停止；benchmark 不再添加 `--profile`。
- `Xid 13: EXTRA_INLINE_DATA` 表示 GPU 上下文发生非法或异常执行，Python 看到的报错位置可能只是异步错误随后暴露的位置，不一定是真正根因。

## 从哪里开始读

1. `README.md`：故障矩阵、推荐命令和诊断结论。
2. `PROFILE_SUMMARY.md`：代表 Rank 的 Prefill/Decode kernel 摘要。
3. `diagnostics/`：无边界采集失败时的驱动与进程现场。
4. `logs/`：各轮请求和服务日志。

## 限制

- 关闭 CUDA Graph 的对照在 DSpark eager warmup 阶段失败，因此没有形成有效的 Graph ON/OFF Profile。
- 已确定触发边界和规避方案，但未定位到 PyTorch、Kineto、CUPTI 或 NVIDIA driver 中的具体代码缺陷。
- 本目录用于说明“怎样安全采集”，不是 PR #32320 的性能结论来源。
