# PD Decode KV 满载与 DSpark 稳定性修复

CR：<https://kunpeng.xiaojukeji.com/view/revision/6647271>

## 1. 问题说明

DeepSeek-V4-Flash 在 PD 分离 Decode 场景中存在四个相互关联的问题：

1. DSPARK/DFLASH 等 Spec-v2 路径没有统一执行 SWA 窗口维护，窗口外的 SWA 页可能未及时回收，从而错误触发 KV 内存不足和请求撤回。
2. 当 Full KV 确实耗尽并进入 `retract_decode()` 时，原逻辑默认通过 `get_cpu_copy()` 将被撤回请求的 KV 同步保存到 CPU；但 DSV4 当前使用的 KV Pool 不支持完整的 CPU copy，最终抛出 `NotImplementedError` 并导致 Decode Scheduler 退出。
3. Decode 预分配的容量检查使用未对齐的 token 数，而底层 allocator 按整页分配；`page_size=256` 时可能出现预算认为空间足够、实际分配却失败的情况。
4. DSpark 手工构造的 `ForwardBatch` 绕过 `ForwardBatch.init_new()`，没有填写 `num_token_non_padded`；开启 DP/EP CUDA Graph 后，warmup 阶段会读取到 `None` 并崩溃。

修复前的真实 PD 压测中，Full KV 达到 100%，随后 Scheduler 崩溃：

```text
[2026-08-30 03:41:54 DP2 TP2 EP1] Decode batch,
#full token: 848128, full token usage: 1.00,
#swa token: 7424, swa token usage: 0.09

retract_decode
  -> release_req
  -> offload_kv_cache
  -> get_cpu_copy
  -> raise NotImplementedError()

[2026-08-30 03:41:55 DP2 TP2 EP1] Scheduler hit an exception
Subprocess scheduler_0 ... crashed with exit code -3
```

该问题会把一次可局部处理的 KV 内存压力放大为整个 Decode 服务不可用。

页预算不一致时曾出现以下预分配失败特征：

```text
KV cache is full! Bug in memory estimation.
available=256, required_alloc=8448, page_size=256
```

DSpark + EP 的修复前错误如下：

```text
File "forward_batch_info.py", line 235,
  in compute_local_num_token_non_padded
  global_num_token_non_padded - tokens_per_rank * attn_tp_rank
TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'

Subprocess scheduler_0 ... crashed with exit code -3
```

## 2. 解决方案

本次修改包含六层处理：

1. **统一 SWA 维护入口**：将 `maybe_evict_swa()` 和 `decode_batch_idx += 1` 移到公共 `spec_prepare_for_decode()`，使 DSPARK、DFLASH、EAGLE 每轮 Decode 都执行相同的窗口推进和过期页回收。
2. **OOM 前强制回收**：`check_decode_mem()` 首次判断空间不足时，先通过 `maybe_evict_swa(force=True)` 立即回收所有可释放的窗口外 SWA 页；重新检查后仍不足才进入 retraction。
3. **按 KV Pool 能力撤回**：增加 `supports_cpu_copy()` 能力判断。支持完整 CPU round-trip 的 Pool 保留原 KV offload/恢复逻辑；不支持的 Pool 不再调用 `get_cpu_copy()`，而是直接释放被撤回请求的 KV 并只 abort 该请求。
4. **请求级异常收尾**：完善 abort 状态、时间统计、残留 offload 状态清理，并丢弃 abort 后已进入 Tokenizer 流水线的迟到 token。Scheduler 和 batch 中的其他请求继续运行，不走 Prefill rebootstrap。
5. **预分配预算按页对齐**：Full/SWA 的 KV 需求先通过 `ceil_align(length, page_size)` 向上对齐，再参与容量判断和预算扣减，使检查口径与 allocator 的实际整页分配保持一致。
6. **补齐 DSpark DP/EP 元数据**：手工构造 draft `ForwardBatch` 后，根据 `input_ids.numel()` 写入设备端 `num_token_non_padded` 和 CPU 端 `num_token_non_padded_cpu`，供 DP/EP CUDA Graph 正确计算各 rank 的真实 token 数。

核心逻辑可简化为：

```python
def spec_prepare_for_decode(batch):
    batch.maybe_evict_swa()
    for req in batch.reqs:
        req.decode_batch_idx += 1
    prepare_by_spec_algorithm(batch)

def check_decode_mem(batch):
    if enough_memory(batch):
        return True
    batch.maybe_evict_swa(force=True)
    return enough_memory(batch)

full_len = ceil_align(full_len, page_size)
swa_len = ceil_align(swa_len, page_size)

num_tokens = forward_batch.input_ids.numel()
forward_batch.num_token_non_padded = torch.tensor(
    num_tokens, dtype=torch.int32, device=device
)
forward_batch.num_token_non_padded_cpu = num_tokens

if allocator.supports_cpu_copy():
    try:
        release_req(req, offload_kv=True)
        retracted_reqs.append(req)
    except NotImplementedError:
        release_and_abort(req)
else:
    release_req(req, offload_kv=False)
    abort_only_this_request(req)
```

## 3. 测试结果

### 3.1 本地回归

```text
新增 SWA/CPU-copy/资源所有权测试：16 passed，另 5 subtests passed
原 CR Decode/DSpark/Tokenizer 回归：38 passed
合计：54 passed，另 5 subtests passed
pre-commit：passed
git diff --check：passed
```

### 3.2 DSpark + EP 与页对齐专项验证

修复前，DSpark + DP8/EP4 Decode 在 CUDA Graph warmup 阶段即因 `num_token_non_padded=None` 退出；补齐元数据后，正常 PD 和 fake Decode 两组服务均能完成 warmup、启动并进入压力测试。对应单元测试同时验证了设备端值、类型及 CPU 镜像值：

```text
forward_batch.num_token_non_padded.item() == 12
forward_batch.num_token_non_padded.dtype == torch.int32
forward_batch.num_token_non_padded_cpu == 12
```

`page_size=256` 的专项压力回归结果：

```text
真实 PD：
KV cache pool is full:                   30
retract_decode: aborted request:         30
KV page budget AssertionError:            0
Scheduler hit an exception:               0

Fake Decode：
KV cache pool is full:                  920
retract_decode: aborted request:        974
Bug in memory estimation:                 0
Scheduler hit an exception:               0
Decode health:                          200
```

结论：页对齐预算与 allocator 的实际分配口径一致，Full/SWA 高压力下未再出现预分配估算断言。

### 3.3 正常双机 PD 请求

测试配置：TP8 / DP8 / EP4、DSPARK draft tokens 6、Prefill MF 0.70、Decode MF 0.55；请求参数为 `input=8192`、`output=8192`、`concurrency=1024`、`prompts=2048`。

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

Full KV 满载后，日志显示仅撤回并 abort 当前请求：

```text
[2026-08-30 13:28:03 DP5 TP5 EP2]
retract_decode: aborted request ... because the KV cache does not
support synchronous CPU save and restore

[2026-08-30 13:28:03 DP5 TP5 EP2]
KV cache pool is full. Retract requests.
#retracted_reqs: 0, #aborted_reqs: 1, #new_tokens_gained: 8448,
#new_token_ratio: 0.0294 -> 0.3618
```

最后一次 retraction 后仍继续执行 **803 个 Decode batch**，Router、Prefill、Decode 健康检查持续返回 `200`：

```text
2026-08-30T13:39:32Z router=200 prefill=200 decode=200
2026-08-30T13:39:36Z router=200 prefill=200 decode=200
2026-08-30T13:39:40Z router=200 prefill=200 decode=200
```

结论：真实 PD 链路下 SWA 没有错误涨满；Full KV 确实满载时，仅相关请求失败，Scheduler 和其他请求继续运行。

### 3.4 Fake Decode 请求

直接向 Decode `10.207.193.166:30002` 注入 fake 请求并启用 `--enable-test-pd-without-transfer`；请求参数同样为 `input=8192`、`output=8192`、`concurrency=1024`、`prompts=2048`。

```text
HTTP POST:                       2049
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

典型请求级撤回日志：

```text
[2026-08-30 13:55:37 DP3 TP3 EP1]
retract_decode: aborted request ... because the KV cache does not
support synchronous CPU save and restore

[2026-08-30 13:55:37 DP3 TP3 EP1]
KV cache pool is full. Retract requests.
#retracted_reqs: 0, #aborted_reqs: 1, #new_tokens_gained: 8960
```

Fake 流量绕过真实 PD admission，因此 Full/SWA Pool 会短暂同时达到 100%。SWA 满载仅出现在 `52/48816` 个 batch，随后由 `1.00` 回落到约 `0.43`，最终回落至 `0.00`，说明窗口外 SWA 页能够正常回收，并非持续泄漏。

最后一次 retraction 后仍继续执行 **15617 个 Decode batch**，最终状态为：

```text
running-req=1, full usage=0.00, swa usage=0.00
GET /health HTTP/1.1 200 OK
```

结论：极端 fake 压力下共发生 711 次请求级 abort，但未出现 `NotImplementedError`、断言或 Scheduler 崩溃，Decode 服务持续存活。

## 4. 结果汇总

### 4.1 修复前后行为对照

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 真实 PD | Full KV 满载后，`get_cpu_copy()` 抛出 `NotImplementedError`，Scheduler 崩溃，随后 Router 503 | 不支持 CPU copy 时只 abort 被撤回请求；其他请求继续 Decode，Router、Prefill、Decode 保持存活 |
| Fake Decode | `get_cpu_copy()` 抛异常，Scheduler 崩溃 | 只 abort 被撤回请求；不进入 Prefill/rebootstrap，Decode 服务存活 |
| Page-aligned 预分配 | 容量预算按未对齐 token 数计算，allocator 按整页收费，可能出现 `available=256, required_alloc=8448` 的预分配断言 | Full/SWA 需求均按 `page_size` 向上对齐后再检查和扣减；压力测试未再出现预算断言 |
| DSpark + EP | `num_token_non_padded=None`，DP/EP CUDA Graph warmup 触发 `TypeError` 并退出 | 补齐设备端和 CPU 端真实 token 数，服务正常完成 warmup 并启动 |
| Spec-v2 SWA 回收 | 部分算法路径未统一推进 `decode_batch_idx` 和回收窗口外 SWA 页，可能造成假 OOM | 公共入口统一维护；内存检查失败时再强制回收一次，仍不足才 retraction |

> 当前 CR 的 OOM 策略是：真实 PD 与 fake 请求均只 abort 被撤回请求，不执行 Prefill KV 重算。此前测试过的 PD rebootstrap 属于另一版方案，不在本 CR 中。

### 4.2 KV 压力测试数据

| 场景 | Full KV 峰值 | SWA KV 峰值 | 请求级 Abort | `NotImplementedError` | Scheduler 崩溃 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 修复前真实 PD | 1.00 | 0.09 | 0 | 1 | 1 | Full KV 满载后服务崩溃 |
| 修复后真实 PD | 1.00 | 0.78 | 23 | 0 | 0 | 仅相关请求失败，三端服务存活 |
| 修复后 Fake Decode | 1.00 | 1.00（短暂） | 711 | 0 | 0 | 服务存活，SWA 最终回落至 0 |

> 修复前后的压测负载不同，因此上表用于验证故障处理语义和服务稳定性，不用于吞吐性能对比。

## 5. 日志归档

- 修复前基线：`../20260830_full_kv_retraction_repro/decode.log`
- 修复前 DSpark + EP 启动错误：`../20260827_pd_rebootstrap_e2e/archives/20260827_0604_unfixed_baseline_startup_crash/decode_unfixed_startup.log`
- 页对齐专项回归：`../20260830_full_kv_abort_only_pagealigned_c1024/`、`../20260830_full_kv_abort_only_pagealigned_fake_c1024/`
- 修复后真实 PD：`decode_real_snapshot.log`、`prefill_real_snapshot.log`、`router_real_snapshot.log`、`health_real_snapshot.log`、`benchmark_real_snapshot.txt`
- 修复后 Fake Decode：`decode_fake_snapshot.log`、`benchmark_fake_snapshot.txt`
