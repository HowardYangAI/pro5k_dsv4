# MF=0.55 自然 retraction 压力测试归档

- 时间：2026-08-27 05:35–05:45 UTC
- Decode MF：0.55
- 请求：2048 prompts，max concurrency 1024
- 长度：input 8192，output 8192
- 触发方式：自然 KV 压力，未使用 `SGLANG_TEST_RETRACT`
- 测试由客户端提前终止。

## 归档时链路计数

- `PD_REBOOTSTRAP_PREPARE`: 356
- `PD_REBOOTSTRAP_ENQUEUE`: 356
- `PD_REBOOTSTRAP_DISPATCH`: 356
- `PD_REBOOTSTRAP_HTTP_OK`: 324
- `PD_REBOOTSTRAP_COMMIT`: 324
- 最终 `Finish` 且 `num_retractions > 0`: 32

这说明已有 324 个请求完成 Prefill 重算响应并在 Decode 侧 commit；客户端终止测试后，服务端继续排空队列，其中 32 个经过 retraction 的请求最终生成完成，端到端 rebootstrap 链路验证成功。

本轮负载过重，原始 PD KV 传输出现大量 `KVPoll.WaitingForInput` 300 秒超时，不适合作为稳定性结论。下一轮建议使用单波请求：`num-prompts=640`、`max-concurrency=640`、`input=8192`、`output=4096`。

## 文件

- `benchmark_output.txt`：用户提供的压测输出快照
- `router.log`：Router 日志快照
- `prefill.log`：Prefill 日志快照
- `decode_mf055_trace.log`：Decode 日志快照，包含临时 rebootstrap 链路日志
- `decode_mf055_trace_post_drain.log`：客户端终止后服务端排空队列的最终 Decode 日志，包含完整成功证据
