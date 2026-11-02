# DeepSeek-V4-Flash PD Decode Retraction 稳定性修复

CR：[#6625071 - fix: make PD decode retraction KV-pool aware](https://kunpeng.xiaojukeji.com/view/revision/6625071)

## 1. 问题

DeepSeek-V4-Flash 在 PD 分离 Decode 节点发生 KV cache 压力时会触发
`retract_decode()`。原实现默认把被撤回请求的 KV 同步复制到 CPU：

```text
retract_decode
-> release_req
-> offload_kv_cache
-> allocator.get_cpu_copy
```

但 DeepSeek-V4-Flash 使用的分页 SWA/特殊 KV Pool 不支持完整的 CPU
save/restore。调用 `get_cpu_copy()` 会抛出 `NotImplementedError`，并导致整个
Decode scheduler 退出，而不只是单个请求失败。

修复前真实 PD 压测日志：

```text
[2026-08-27 06:55:47 DP6 TP6 EP3] Decode batch,
#full token: 836608, full token usage: 0.99,
#swa token: 83968, swa token usage: 0.99

[2026-08-27 06:55:48 DP6 TP6 EP3] Scheduler hit an exception:
  File "schedule_batch.py", line 2625, in retract_decode
    self.release_req(...)
  File "schedule_batch.py", line 1733, in release_req
    req.offload_kv_cache(...)
  File "schedule_batch.py", line 1561, in offload_kv_cache
    self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(...)
  File "allocator/swa.py", line 402, in get_cpu_copy
    return self._kvcache.get_cpu_copy(...)
  File "memory_pool.py", line 1656, in get_cpu_copy
    raise NotImplementedError()
NotImplementedError

Subprocess scheduler_0 ... crashed with exit code -3.
```

Decode 崩溃后，Router 返回：

```text
503 Service Unavailable - No available servers
```

fake 请求也能复现同样的问题：

```text
[2026-08-27 07:13:05 DP3 TP3 EP1] Scheduler hit an exception:
  ...
  File "memory_pool.py", line 1656, in get_cpu_copy
    raise NotImplementedError()
NotImplementedError

[2026-08-27 07:13:06] Subprocess scheduler_0 ... crashed with exit code -3.
```

此外，测试过程中还发现一个启动阻塞问题：DSpark 手工构造的
`ForwardBatch` 绕过 `ForwardBatch.init_new()`，没有填写
`num_token_non_padded`。开启 DP/EP CUDA Graph 后会发生：

```text
File "forward_batch_info.py", line 235,
  in compute_local_num_token_non_padded
  global_num_token_non_padded - tokens_per_rank * attn_tp_rank
TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
```

## 2. 解决思路

### 2.1 按 KV Pool 能力选择恢复方式

为 allocator/KV Pool 增加 `supports_cpu_copy()`，明确判断当前布局、K/V 和
scale buffer 是否支持完整的 CPU save/restore，然后将 retraction 分成三条路径：

1. KV Pool 支持 CPU copy：沿用原来的 CPU 保存/恢复；
2. 不支持 CPU copy，但请求可以仅通过 token 重算：释放 Decode KV，返回
   Prefill 重算，再把 KV 传回 Decode；
3. 不支持 CPU copy，且请求不能安全重算（例如 fake、multimodal、LoRA、
   session）：只 abort 当前请求，Decode 服务继续运行。

### 2.2 接通 PD rebootstrap

普通纯文本请求被撤回后：

```text
释放 Decode KV
-> 保存 decode 边界 token
-> 加入 DecodePreallocQueue
-> Decode 重新预分配 KV 接收页
-> Prefill 使用 origin_input_ids + output_ids 重算前缀
-> KV 传回 Decode
-> commit 时恢复边界 token
-> 请求继续 decode
```

保存并重放边界 token，是为了避免重复输出 token 或少计算一个 token 的 KV。

### 2.3 修复相关分配和运行问题

- Full/SWA 预分配预算按照 `page_size` 对齐，与 allocator 的实际页分配保持一致；
- SWA 没有 radix eviction，直接扣除 page-aligned SWA requirement；
- fake 请求禁止提交真实 Prefill recompute，只对受影响请求 graceful abort；
- abort 使用统一完成流程，避免晚到 token 和重复资源释放；
- DSpark 手工 `ForwardBatch` 补齐 `num_token_non_padded` 及其 CPU 值。

## 3. 核心伪代码

```python
def retract_decode(req):
    if kv_allocator.supports_cpu_copy():
        release_req(req, offload_kv=True)
        cpu_retracted_reqs.append(req)

    elif req.supports_pd_rebootstrap(server_args):
        release_req(req, offload_kv=False)
        rebootstrap_reqs.append(req)

    else:
        release_req(req, offload_kv=False)
        req.to_finish = FINISH_ABORT(...)
        reqs_to_abort.append(req)


for req in cpu_retracted_reqs:
    add_to_normal_resume_queue(req)

for req in rebootstrap_reqs:
    req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
    req.pd_rebootstrap_in_progress = True
    decode_prealloc_queue.add(req, is_rebootstrap=True)


def on_decode_kv_preallocated(req):
    if req.is_rebootstrap:
        submit_prefill_recompute(
            input_ids=req.origin_input_ids + req.output_ids,
            decode_kv_destination=req.preallocated_kv_pages,
        )


def on_recomputed_kv_arrived(req):
    req.output_ids.append(req.pd_rebootstrap_forced_output_id)
    req.pd_rebootstrap_in_progress = False
    waiting_queue.append(req)
```

DSpark 补丁的核心逻辑：

```python
num_tokens = forward_batch.input_ids.numel()
forward_batch.num_token_non_padded = torch.tensor(
    num_tokens, dtype=torch.int32, device=device
)
forward_batch.num_token_non_padded_cpu = num_tokens
```

## 4. 修复结果

### 4.1 真实 PD：Prefill 重算后继续生成

测试配置：Decode MF=0.55、concurrency=1024、input/output=8192/8192。

同一个请求的完整日志链路：

```text
PD_REBOOTSTRAP_PREPARE
rid=chatcmpl-iyIq2Uc5iIiNdg7lVYYZtt78-10.207.193.156
origin_input_len=7986 output_len_before=135
recompute_prefix_len=8120 boundary_token=680

PD_REBOOTSTRAP_ENQUEUE
rid=chatcmpl-iyIq2Uc5iIiNdg7lVYYZtt78-10.207.193.156
recompute_prefix_len=8120

PD_REBOOTSTRAP_DISPATCH
rid=chatcmpl-iyIq2Uc5iIiNdg7lVYYZtt78-10.207.193.156
prefill_url=http://10.207.193.156:30001 recompute_prefix_len=8120

PD_REBOOTSTRAP_HTTP_OK
rid=chatcmpl-iyIq2Uc5iIiNdg7lVYYZtt78-10.207.193.156 status=200

PD_REBOOTSTRAP_COMMIT
rid=chatcmpl-iyIq2Uc5iIiNdg7lVYYZtt78-10.207.193.156
replayed_boundary=True committed_output_id=680 output_len=135
```

汇总结果：

```text
PD_REBOOTSTRAP_PREPARE  = 356
PD_REBOOTSTRAP_ENQUEUE  = 356
PD_REBOOTSTRAP_DISPATCH = 356
PD_REBOOTSTRAP_HTTP_OK  = 324
PD_REBOOTSTRAP_COMMIT   = 324
最终完成且 num_retractions > 0 的请求 = 32
```

已经收到 Prefill HTTP 200 的 324 个请求全部在 Decode 侧完成 commit，证明链路
`retraction -> Prefill recompute -> KV transfer -> Decode commit` 已经打通。

### 4.2 fake 请求：安全终止请求，服务不崩溃

```text
安全 abort 的 fake 请求             = 343
正常完成的 fake 请求                 = 196
进入 PD_REBOOTSTRAP_DISPATCH 的 fake = 0
测试后 Decode /health                = 200
```

日志：

```text
retract_decode: aborted request ...
(KV cache does not support synchronous CPU save and restore)
```

修复后 fake 请求不会进入不支持的 CPU copy，也不会被错误提交给 Prefill；只终止
被撤回的请求，Decode scheduler 保持运行。

### 4.3 修复前后对照

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 真实 PD | `get_cpu_copy()` 抛异常，scheduler 崩溃，Router 503 | Prefill 重算 KV，请求继续执行，服务存活 |
| fake 请求 | `get_cpu_copy()` 抛异常，scheduler 崩溃 | 只 abort 被撤回请求，服务存活 |
| DSpark + EP | `num_token_non_padded=None`，CUDA Graph warmup 崩溃 | 补齐真实 token 数，服务正常启动 |

## 5. 测试结果

```text
44 passed, 17 warnings, 13 subtests passed
15 passed, 15 warnings
```

完整日志归档位于当前目录的 `archives/` 下。
