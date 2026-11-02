# DeepSeek-V4-Flash SWA 公共维护与 OOM 前强制回收测试

## 测试版本

- 代码：`/sgl-workspace/sglang-swa-evict-fixed`
- 基础提交：`2984ac56eb`
- 未提交测试修改：参照 bytedance-iaas/sglang#739，将 Spec-v2 SWA 维护移到公共入口，并在 Decode 内存检查失败时强制回收一次窗口外 SWA 页。
- `sglang-kernel`：两台机器均为 `0.4.5+cu129`

## 验证目标

1. DFLASH、DSPARK、EAGLE 通过公共 `spec_prepare_for_decode()` 每轮只更新一次 `decode_batch_idx` 并执行 SWA 维护入口。
2. `check_decode_mem()` 第一次失败后先执行 `maybe_evict_swa(force=True)`，避免延迟维护造成假 OOM。
3. 真实 Full KV 压力仍无法解除时，KV Pool 不支持完整 CPU copy 的请求被单独 abort，Scheduler 保持存活。
4. 不出现 `NotImplementedError`、KV page budget assertion 或 Scheduler 子进程崩溃。

## 配置

- Router：`10.207.193.156:30000`
- Prefill：`10.207.193.156:31001`，MF 0.70
- Decode：`10.207.193.166:30002`，MF 0.55
- TP8 / DP8 / EP4
- DSPARK，draft tokens 6
- Context length 131072，page size 256
- Mooncake TCP，关闭 radix cache
- 最大运行请求数与 Router 并发上限：1024

## 已完成的本地回归

```text
新增 SWA/CPU-copy/ownership 测试：16 passed，另 5 subtests passed
原 CR Decode/DSpark/Tokenizer 回归：38 passed
pre-commit：passed
git diff --check：passed
```

## E2E 结果

### 真实 PD 压测

压测参数：input 8192、output 8192、concurrency 1024、prompts 2048。

```text
Decode batches:                  35463
Full KV max usage:                1.00
SWA KV max usage:                 0.78
Full KV usage >= 0.99:            3689
SWA KV usage >= 0.99:                0
KV cache pool is full:              23
retract_decode aborted request:      23
NotImplementedError:                  0
Scheduler hit an exception:           0
AssertionError:                        0
```

首次 Full KV retraction：

```text
[2026-08-30 13:28:03 DP5 TP5 EP2] KV cache pool is full. Retract requests. #retracted_reqs: 0, #aborted_reqs: 1, #new_tokens_gained: 8448, #new_token_ratio: 0.0294 -> 0.3618
```

最后一次 retraction 后仍观察到 803 个 Decode batch；三端健康恢复并持续为 200。客户端快照未出现 503、Traceback 或 Request failed。

结论：公共 Spec-v2 SWA 维护生效，压力期间 SWA 未错误涨满；Full KV 真正满载时只 abort 被撤回请求，Scheduler 和剩余请求继续运行。

### 归档文件

- `benchmark_real_snapshot.txt`
- `decode_real_snapshot.log`
- `prefill_real_snapshot.log`
- `router_real_snapshot.log`
- `health_real_snapshot.log`
- `decode_fake.log`：远端持续写入的 fake Decode 日志
- `benchmark_fake_snapshot.txt`
- `decode_fake_snapshot.log`

### Fake Decode 压测

压测直接请求 `10.207.193.166:30002`，参数为 input 8192、output 8192、concurrency 1024、prompts 2048，并启用 `--enable-test-pd-without-transfer`。

```text
HTTP POST:                       2049（1 warmup + 2048 requests）
Decode batches:                48816
Full KV max usage:              1.00
SWA KV max usage:               1.00
Full KV usage >= 0.99:         22742
SWA KV usage >= 0.99:             52
KV cache pool is full:            711
retract_decode aborted request:    711
NotImplementedError:                 0
Scheduler hit an exception:          0
AssertionError:                       0
```

Fake 流量绕过真实 PD admission，一次性向 Decode 注入大量请求，因此部分 DP rank 的 Full/SWA Pool 会短暂同时满载。SWA 满载只占 52/48816 个 batch；同一 DP rank 随后从 1.00 回落到约 0.43，说明 SWA 回收仍在工作。

最后一次 retraction 后仍继续执行 15617 个 Decode batch，最终 batch 回落到 `running-req=1, full usage=0.00, swa usage=0.00`。Decode 健康持续为 200，说明 711 个请求级 abort 未导致 Scheduler 退出。
