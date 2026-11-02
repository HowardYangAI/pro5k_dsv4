# DeepSeek-V4-Flash PD KV 压力测试记录

归档时间：2026-08-30T10:25:46Z

## 测试配置

- 代码目录：`/sgl-workspace/sglang-swa-evict-fixed`
- Router：`10.207.193.156:30000`
- Prefill：`10.207.193.156:31001`，`mem-fraction-static=0.70`
- Decode：`10.207.193.166:30002`，`mem-fraction-static=0.55`
- PD 模式：真实 Prefill/Decode 分离，非 fake transfer
- 输入长度：8192
- 输出长度：8192
- 并发：1024
- 请求数：4096
- 本轮由用户提前终止；已保存的客户端输出结束于 560/4096。

## 验证目标

验证以下修改组合：

1. KV Pool 不支持同步 CPU copy 时，只 abort 被撤回请求，Scheduler 继续服务。
2. Full/SWA Decode 预分配预算按物理 page 对齐，避免预算允许接纳、实际分配失败。
3. DSpark DP/EP CUDA Graph 使用初始化后的真实非 padding token 数。

## 结果

Decode 日志统计：

```text
KV cache pool is full:                   30
retract_decode: aborted request:         30
NotImplementedError:                      0
KV page budget AssertionError:            0
Scheduler hit an exception:               0
```

首次触发证据：

```text
[2026-08-30 10:15:12 DP1 TP1 EP0] retract_decode: aborted request chatcmpl-1mFSeffyFSNzIZAZnIV2gzED-10.207.193.156 because the KV cache does not support synchronous CPU save and restore
[2026-08-30 10:15:12 DP1 TP1 EP0] KV cache pool is full. Retract requests. #retracted_reqs: 0, #aborted_reqs: 1, #new_tokens_gained: 8448, #new_token_ratio: 0.0294 -> 0.3921
```

最后一次触发证据：

```text
[2026-08-30 10:24:36 DP1 TP1 EP0] retract_decode: aborted request chatcmpl-ALhsMCotclF7YsYoy20W4xPr-10.207.193.156 because the KV cache does not support synchronous CPU save and restore
[2026-08-30 10:24:36 DP1 TP1 EP0] KV cache pool is full. Retract requests. #retracted_reqs: 0, #aborted_reqs: 1, #new_tokens_gained: 8448, #new_token_ratio: 0.4690 -> 0.4803
```

压测停止后 Decode 已回落：

```text
[2026-08-30 10:25:31 DP2 TP2 EP1] Decode batch, #running-req: 1, #full token: 15104, full token usage: 0.02, #swa token: 1280, swa token usage: 0.02, #prealloc-req: 0, #transfer-req: 0, #retracted-req: 0
```

归档时服务状态：

```text
Prefill: 200
Decode:  200
Router:  200
```

结论：本轮成功触发 KV 内存压力和请求级 abort；服务未因不支持 CPU copy 或 page 预算误差而崩溃，符合 abort-only 修复预期。

## 日志文件

- `benchmark_client.txt`：本轮客户端输出
- `decode.log`：从 Decode 机器复制的本轮完整日志快照
- `prefill.log`：Prefill 日志（服务仍运行，文件可能继续增长）
- `router.log`：Router 日志（服务仍运行，文件可能继续增长）
- `health_watch.log`：三端健康检查记录（监控仍运行，文件可能继续增长）
