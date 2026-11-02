# Full KV Abort-only 修复端到端验证

## 测试版本

- Base: `d07dd66fb75e9382e8455d1e40adc6e06a4feb89`（包含 CR 6624145）
- Worktree: `/sgl-workspace/sglang-swa-evict-fixed`
- 额外修改：KV Pool CPU-copy 能力判断；不支持 CPU copy 时仅 abort 被撤回请求，不进行 Prefill rebootstrap。
- 修复文件已同步到 Decode 主机，关键文件 SHA-256 与本机一致。

## 服务配置

| 角色 | 地址 | MF | tmux |
|---|---|---:|---|
| Router | `10.207.193.156:30000` | - | `abort-router` |
| Prefill | `10.207.193.156:31001` | 0.70 | `abort-prefill` |
| Decode | `10.207.193.166:30002` | 0.55 | `abort-decode`（远端） |
| 健康监控 | 每约 4 秒记录三端状态 | - | `abort-health` |

公共配置：TP8 / DP8 / EP4、DSpark 6 draft tokens、context length 131072、Mooncake TCP、radix cache disabled。Decode 未设置 `SGLANG_TEST_RETRACT`，本轮只验证自然 KV 压力。

## 启动状态

```text
2026-08-30T05:32:29Z router=200 prefill=200 decode=200
```

启动阶段未发现 `Traceback`、`NotImplementedError`、`Scheduler hit an exception`、CUDA OOM 或子进程退出。

## 日志

- `prefill.log`
- `router.log`
- `health_watch.log`
- Decode 原始日志位于远端同目录的 `decode.log`，测试结束后复制回本目录。

## 验收证据

1. Decode 日志出现 `full token usage: 1.00`；
2. 出现 `retract_decode: aborted request ... does not support synchronous CPU save and restore`；
3. 汇总日志出现 `#aborted_reqs > 0` 和 `#new_tokens_gained > 0`；
4. 上述日志之后仍持续产生 Decode batch；
5. `health_watch.log` 中三端持续为 200；
6. 不出现 `NotImplementedError`、scheduler crash 或 PD rebootstrap；
7. 压力结束后补发短请求并成功完成。
