# DeepSeek-V4-Flash PD Rebootstrap 双机测试

面向评审和转发的完整问题、修复及验证报告见：
[`PD_DECODE_RETRACTION_FIX_REPORT.md`](PD_DECODE_RETRACTION_FIX_REPORT.md)。

## 1. 测试目标

验证内部 DCL CR 6565881 在真实 PD 分离环境中的完整恢复链路：

```text
Decode KV 空间不足/测试开关触发 retraction
  -> DeepSeek-V4 KV pool 不支持同步 CPU copy
  -> 释放 Decode 侧 KV
  -> 请求返回 Prefill 重算历史 KV
  -> Mooncake TCP 将 KV 重新传给 Decode
  -> 请求恢复 Decode，且不重复或遗漏 boundary token
```

本次不测试准备提交官方的独立分支，也不使用 fake transfer。

## 2. 测试代码

```text
DCL CR: 6565881
Base: 48886803b94596884d737132a381bd5124ea1cfd
Patch: deepseek-v4-flash/source/kv-only-staged.patch.gz
Patch scope: 24 files, 889 insertions, 55 deletions
Test worktree: /sgl-workspace/sglang-dcl-6565881-pd-test
```

现有 `/sgl-workspace/sglang` 不做修改。

## 3. 双机拓扑

| 角色 | 主机 | IP | HTTP | Bootstrap | GPU |
| --- | --- | --- | ---: | ---: | --- |
| Prefill | ml-pro5000-ser187.qgz02 | 10.207.193.156 | 30001 | 8998 | 8 x RTX PRO 5000 72GB |
| Decode | ml-pro5000-ser237.qgz02 | 10.207.193.166 | 30002 | 8998 | 8 x RTX PRO 5000 72GB |

业务请求入口为同一台 Prefill 主机上的 PD Router：

```text
http://10.207.193.156:30000
```

不要把普通 OpenAI 请求直接发送到 Prefill `:30001` 或 Decode `:30002`。

KV 传输使用 `mooncake_tcp`。两台机器没有发现 RDMA HCA，因此不使用 RDMA。

## 4. 软件和模型

```text
flashinfer-python: 0.6.14（SGLang 固定版本，不升级）
mooncake-transfer-engine: 0.3.11.post1
nixl: 1.3.2（本次不使用）
model: /data/modelscope_models/deepseek-ai/DeepSeek-V4-Flash-0731
model size: 约 156GB，48 个 safetensors 分片
```

## 5. 服务配置

```text
TP8 / Attention DP8 / EP4
DSpark: 6 draft tokens
Decode CUDA Graph: full
context length: 131072
chunked prefill: 2048
max prefill tokens: 8192
Prefill mem-fraction-static: 0.7
Decode mem-fraction-static: 0.5（自然 KV 压力阶段）
max-running-requests: 1024
radix cache: disabled
```

首次真实 PD 冒烟使用 Decode `MF=0.7`。自然 KV 压力阶段已将 Decode
调整为 `MF=0.5`，其每 DP rank 的 `max_total_num_tokens` 从 2,075,136
降为 439,040；未启用强制 retraction。

## 6. 测试阶段

### 阶段 A：真实 PD 冒烟

Decode 不设置 `SGLANG_TEST_RETRACT`。验证正常 Prefill、Mooncake KV 传输和 Decode 生成。

### 阶段 B：确定性 rebootstrap

仅 Decode 设置：

```bash
SGLANG_TEST_RETRACT=true
SGLANG_TEST_RETRACT_INTERVAL=20
```

该开关只强制触发 retraction；Prefill 重算、Mooncake KV 传输和 Decode 恢复仍走真实链路。

### 阶段 C：自然 KV 压力

关闭测试开关，使用长输入/长输出和高并发自然触发 `check_decode_mem()` 失败。建议逐级增加并发，最终复测 8192 input / 2048 output / c512 或更高。

## 7. 验收标准

- Prefill 与 Decode `/health` 均正常；
- 日志不存在 `get_cpu_copy()` / `NotImplementedError`；
- 日志能够证明请求进入 rebootstrap、Prefill recompute、KV transfer 和 Decode resume；
- Scheduler/API Server 不退出；
- 无重复 token、漏 token或重复结束事件；
- 无 `Received output ... state was deleted` 告警；
- 压力结束后新请求仍可完成。

## 8. 日志

```text
logs/model_sync.log
logs/code_prepare.log
logs/unit_tests.log
logs/prefill.log
logs/decode.log
logs/router.log
logs/service_status.log
```

远端 Decode 原始日志先写入远端相同日期目录，检查和归档时同步回本目录。

## 9. 请求侧

服务启动和日志由本测试负责；线上路由及请求发送由用户负责。可用服务地址：

```text
业务入口（Router）: http://10.207.193.156:30000
Prefill 内部地址:    http://10.207.193.156:30001
Decode 内部地址:     http://10.207.193.166:30002
```

## 10. 当前状态（2026-08-27 04:17 UTC）

```text
Router:  tmux pd-router，/health = 200
Prefill: tmux pd-prefill，/health = 200
Decode:  tmux pd-decode，/health = 200
阶段: A（真实 PD 冒烟）
强制 retraction: 未启用
```

Prefill 首次启动时 `--enable-metrics` 自动选择的 `metrics_port=30238`
与自身 DP 内部端口冲突，进程按预期退出；随后移除该可选参数重新启动成功。
该失败日志保留在 `logs/prefill.log`，与 rebootstrap 逻辑无关。

## 11. Benchmark 请求方式

`llm-benchmark-online/bench_serving.py` 必须访问 Router 的 `30000` 端口：

```text
--host 10.207.193.156
--port 30000
--pd-separated
```

其中，真正完成 PD 路由的是 `--port 30000`。当前脚本中的
`--pd-separated` 仅作为模式标记传入 `benchmark()`，本身不改写请求地址。

不要添加 `--decode-server-port 30002`：当前实现假定 Decode 与 Router 使用
同一个 `--host`，但本次 Decode 位于另一台机器 `10.207.193.166`。不传该参数时，
脚本会通过 Router 的 `/get_server_info` 获取服务信息。

第一次误将请求发至 `:30001` 时，Prefill 收到一个缺少 `bootstrap_room` 的
warmup 请求并按防御性断言退出；正式的 2048 个请求没有发出。Prefill 已重启，
Router、Prefill、Decode 当前均健康。

## 12. 最终对照结果（2026-08-27）

- 修复版真实 PD：356 次 retraction prepare/enqueue，324 次 Prefill HTTP
  重算成功并在 Decode commit；端到端 rebootstrap 链路通过。
- 修复版 fake：343 个被撤回请求安全 abort，196 个请求正常完成，Decode
  始终健康；fake 请求没有错误进入 rebootstrap dispatch。
- 未修复基线：KV 使用率约 99% 时触发 retraction，无条件调用
  `get_cpu_copy()`，命中 `NotImplementedError`，Decode scheduler 崩溃，Router
  随后返回大量 503。
- 未修复 fake：直接向 Decode 发送 fake 压力请求后，同样在 retraction 中命中
  `get_cpu_copy()` 的 `NotImplementedError`，scheduler 以 `-3` 退出，客户端连接
  被拒绝。

详细证据分别位于 `archives/20260827_0535_mf055_c1024_n2048_8k_8k`、
`archives/20260827_0551_fake_decode_mf055` 和
`archives/20260827_0655_unfixed_retraction_crash`；未修复 fake 对照位于
`archives/20260827_0713_unfixed_fake_retraction_crash`。
