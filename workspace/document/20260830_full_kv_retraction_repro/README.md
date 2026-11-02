# Full KV 满载与撤回异常：基线记录及 Abort 修复计划

## 基线结论

- 版本：`d07dd66fb75e9382e8455d1e40adc6e06a4feb89`，包含 CR `6624145`，不包含 PD rebootstrap 修复。
- 压测：端到端 PD 分离，`input=65536`、`output=32768`、`concurrency=1024`、`num_prompts=4096`。
- 结果：请求成功进入 Prefill、KV transfer 和 Decode。SWA KV 使用率保持在约 `0.09`，但 Full KV 使用率达到 `1.00`。
- 异常：Full KV 满后触发 `retract_decode`，代码尝试同步 CPU copy；当前 SWA KV Pool 的底层池不支持 `get_cpu_copy`，抛出 `NotImplementedError`，导致 Decode scheduler 退出。随后 Router 因完整 PD 链路不可用返回 503。

关键证据（`decode.log:1089` 起）：

```text
[2026-08-30 03:41:54 DP2 TP2 EP1] Decode batch,
#full token: 848128, full token usage: 1.00,
#swa token: 7424, swa token usage: 0.09

retract_decode
  -> release_req
  -> offload_kv_cache
  -> get_cpu_copy
  -> raise NotImplementedError()

Subprocess scheduler_0 ... crashed with exit code -3
```

原始日志：

- `prefill.log`
- `decode.log`
- `router.log`

## 本轮修复目标

当 Decode KV 不足且当前 KV Pool 不支持完整 CPU copy 时：

1. 释放被撤回请求占用的资源；
2. 只 abort 被撤回的请求，并向客户端返回明确错误；
3. 不把该请求送回 Prefill，不进行 KV 重算；
4. 未被撤回的请求继续执行，Decode scheduler、Prefill 和 Router 保持存活；
5. fake transfer 与真实 PD 请求使用相同的失败语义。

注意：被 abort 的请求不会继续生成；“请求继续执行”指同一服务中的其他请求继续执行。

## 实施与验证计划

| 阶段 | 工作内容 | 预期结果 / 验收标准 |
|---|---|---|
| 1. 固化基线 | 保留本轮三端日志、版本、启动参数和压测参数 | 能从日志证明 Full KV `1.00` 后因 `get_cpu_copy` 抛异常并导致 scheduler 退出 |
| 2. 收敛修改 | 以 `d07dd66fb7` 为基线，只移植 KV Pool 能力判断、资源安全释放、撤回请求 abort 和迟到输出清理；移除 rebootstrap/Prefill 重算路径 | 不新增 rebootstrap 状态、队列、metadata 或 forced token 逻辑 |
| 3. 撤回策略 | `supports_cpu_copy=True` 时保留原 CPU offload/恢复；不支持时释放该请求 KV 并加入 `reqs_to_abort`，禁止调用 `get_cpu_copy` | 不支持 CPU copy 的 Pool 不再进入 `get_cpu_copy` |
| 4. Abort 收尾 | 统一完成 `finished_reason`、时间统计、残留资源清理，并向 Tokenizer/客户端发送 `AbortReq` | 被撤回请求收到明确失败，且不会遗留 GPU/Host KV、请求槽位或 offload 状态 |
| 5. 防竞态 | Tokenizer 记录有限数量的 recently-aborted RID，丢弃 abort 后已在流水线中的迟到 token；新请求复用 RID 时清除旧标记 | 无“state was deleted”错误刷屏，不误伤复用 RID 的新请求 |
| 6. 单元测试 | 覆盖支持/不支持 CPU copy、只撤回部分请求、最后一个请求仍无法容纳、重复资源释放、迟到输出 | 相关测试、格式检查及 pre-commit 通过 |
| 7. fake 验证 | 直接向 Decode 发送 fake 请求，强制 Full KV 满载 | 仅被撤回请求 abort；scheduler 和服务存活；后续请求仍可处理 |
| 8. 真实 PD 验证 | 使用与基线相同的双机配置和压测参数复现 Full KV `1.00` | 无 `NotImplementedError`、无 scheduler crash、无 rebootstrap；日志出现被撤回请求 abort，其他请求继续完成 |
| 9. 对比归档 | 保存修复前后 Prefill/Decode/Router/benchmark 日志及关键片段 | 可直接展示“修复前服务崩溃，修复后仅局部请求失败且服务存活” |

## 核心预期逻辑

```python
if kv_pool.supports_cpu_copy():
    release_req(req, offload_kv=True)
    retracted_reqs.append(req)
else:
    release_req(req, offload_kv=False)
    req.to_finish = FINISH_ABORT("retracted due to KV memory pressure")
    reqs_to_abort.append(req)

for req in reqs_to_abort:
    finalize_retraction_abort(req)

# 不提交 Prefill recompute；batch 中剩余请求继续运行。
```

## 实现状态（2026-08-30）

代码已在以下 worktree 完成，尚未提交 Git：

```text
/sgl-workspace/sglang-swa-evict-fixed
```

本轮实现仅修改运行中 KV 满载触发的 `retract_decode` 路径：

- 为 KV Pool/allocator 增加显式 `supports_cpu_copy()` 能力判断；
- 支持完整 CPU round-trip 的 Pool 保留原 offload/恢复行为；
- 不支持的 Pool 使用 `offload_kv=False` 立即释放被撤回请求资源并 abort；
- Scheduler 继续执行 batch 中剩余请求，不向 Prefill 提交 recompute；
- Abort 完成状态、时间统计和残留 offload 状态清理；
- Tokenizer 丢弃 abort 后已经进入 detokenizer 流水线的迟到输出；
- 日志分别记录 `#retracted_reqs` 和 `#aborted_reqs`。

现有 upstream 管理员暂停/恢复功能使用的 rebootstrap 代码没有删除；本轮 OOM 撤回路径不会进入该流程。

### 已完成的定向验证

```bash
PYTHONPATH=/sgl-workspace/sglang-swa-evict-fixed/python \
python3 -m pytest -q \
  test/registered/unit/managers/test_retraction_cpu_copy.py \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
  test/registered/disaggregation/test_specv2_kvcache_offloading.py
```

结果：

```text
40 passed, 5 subtests passed
```

静态检查：

```text
git diff --check: passed
ruff --select=F401,F821,UP037: passed
```

待完成：使用与基线相同的双机 PD 配置再次将 Full KV 压到 `1.00`，确认日志出现 `#aborted_reqs > 0`，无 `NotImplementedError` 和 scheduler crash，并验证后续请求仍可处理。
