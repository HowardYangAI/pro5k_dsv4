# DeepSeek-V4 PD Decode KV Cache 回收崩溃修复计划

## 1. 文档信息

- 创建日期：2026-08-11
- 问题场景：DeepSeek-V4-Flash、PD Decode、Fake KV Transfer、DSpark、高并发
- 当前代码基线：SGLang v0.5.16 + PR #33407
- 当前状态：已完成问题复现和官方仓库调研，尚未实施代码修复
- 目标：KV Cache 容量不足触发请求回收时，服务不能因 `NotImplementedError` 整体退出

## 2. 问题摘要

在 Decode-only 压测中，并发提高到 c512 后，DeepSeek-V4 的 SWA KV pool 接近耗尽。Scheduler 为释放 KV 空间会 retract 部分运行中的请求，但 PD Decode 的回收路径无条件尝试把请求 KV Cache 复制到 CPU。

`DeepSeekV4TokenToKVPool` 没有实现完整的 `get_cpu_copy()` 和 `load_cpu_copy()`，最终调用基类未实现接口并抛出 `NotImplementedError`，导致 Scheduler 进程退出，继而整个服务退出。

KV Cache 满是正常的容量压力状态；因为未实现接口而让整个服务崩溃属于服务缺陷。

## 3. 复现环境

### 3.1 模型及服务配置

```text
模型：DeepSeek-V4-Flash-0731
模式：Decode-only
PD：--disaggregation-mode decode
KV 传输：--disaggregation-transfer-backend fake
并行：TP8 + Attention DP8 + EP8
推测解码：DSpark，6 verify tokens
输入：8192 tokens
输出：2048 tokens
mem-fraction-static：0.6
max-running-requests：512
```

### 3.2 相关容量

```text
SWA pool：140032 tokens / DP rank
c512：512 / 8 = 64 requests / DP rank
仅输出预算：64 × 2048 = 131072 tokens / DP rank
```

此外还需要考虑：

- 每个请求保留的 SWA 输入尾部；
- DSpark draft/verify 额外槽位；
- KV page 对齐；
- 调度与 overlap 期间的临时保留空间。

因此 c512 会逼近或超过实际安全容量，而 c384 约为 48 requests/DP rank，没有触发相同回收路径。

## 4. 现场证据

原始服务日志：

```text
/sgl-workspace/workspace/stress test/20260807_092330/server.log
```

关键调用链：

```text
ScheduleBatch.retract_decode()
  -> release_req()
  -> Req.offload_kv_cache()
  -> SWATokenToKVPoolAllocator.get_cpu_copy()
  -> DeepSeekV4TokenToKVPool
  -> KVCache.get_cpu_copy()
  -> NotImplementedError
```

关键结论：

1. 不是普通 CUDA OOM。
2. 不是 DSpark kernel 直接崩溃。
3. 不是 Fake KV Transfer 本身传输失败。
4. 直接原因是 KV-full retraction 进入了不受该 KV pool 支持的 CPU copy 路径。

## 5. 官方仓库调研

### 5.1 直接相关 Issue/PR

#### Issue #33385

- 链接：https://github.com/sgl-project/sglang/issues/33385
- 状态：Open
- 结论：与本次现场堆栈一致，明确指出 `DeepSeekV4TokenToKVPool` 缺少 CPU copy 实现，并且 PD Decode retraction 无条件触发 offload。

#### PR #31905

- 链接：https://github.com/sgl-project/sglang/pull/31905
- 状态：Closed，未合并
- 方案：根据 `disaggregation_decode_enable_offload_kvcache` 判断是否执行 CPU copy；无法恢复时终止被回收请求。
- 问题：维护者认为 offload 配置开关不是正确的能力判断条件。是否能恢复取决于 allocator 是否支持完整 CPU save/restore round trip。

#### PR #32413

- 链接：https://github.com/sgl-project/sglang/pull/32413
- 标题：`[PD] Handle unsupported decode KV retraction`
- 状态：Open，尚未合并，当前 CI 仍有失败
- 方案方向：
  - 支持 CPU round trip 的 pool 继续保存并恢复；
  - 不支持 CPU copy、但满足条件的真实 PD 请求通过 Prefill rebootstrap 重算 KV；
  - 两种方式都不安全时，仅终止被回收请求，服务继续运行。
- 评价：这是当前最接近正确修复的官方方案，应作为实现基础，但不能直接视为已稳定发布的解决方案。

### 5.2 其他相关但不直接解决本问题的 PR

- PR #31713：SWA 尾部被淘汰后的局部 recompute，面向 Radix Cache 命中复用。
- PR #30762：DeepSeek-V4 hybrid KV 的 HiCache/HostPoolGroup 支持。
- PR #33218：HiSparse CPU KV buffer 与 Radix Cache L1 集成。
- PR #33118：修复 decode retraction 期间健康检查误报 503。
- PR #33805：修复 hybrid-SWA speculative 单请求被提前终止。

目前没有发现已合并且能够保证以下行为的完整方案：

```text
DeepSeek-V4 + Fake decode-only + SWA pool 满
-> 请求排队
-> 所有请求最终完成
-> 服务持续存活
```

## 6. 修复目标与非目标

### 6.1 第一阶段目标

1. KV pool 满时 Scheduler 不崩溃。
2. 不支持 CPU copy 的 pool 绝不能调用未实现接口。
3. 不能恢复的请求必须获得明确错误，不能污染其他请求。
4. 高压结束后服务仍可以接受新请求。
5. 支持 CPU round trip 的现有 KV pool 行为保持不变。

### 6.2 第一阶段非目标

1. 不保证 Fake decode-only 的 c512 请求全部成功。
2. 不在第一阶段实现完整 DeepSeek-V4 CPU KV copy。
3. 不通过无限提高 `mem-fraction-static` 掩盖问题。
4. 不将 `disaggregation_decode_enable_offload_kvcache` 当作 allocator 能力标识。

## 7. 推荐实现方案

### 7.1 引入明确的 CPU round-trip 能力接口

在 KV Cache 或 allocator 基类增加明确能力，例如：

```python
def supports_cpu_copy(self) -> bool:
    return False
```

只有经过验证、同时支持以下操作的实现才返回 `True`：

```text
get_cpu_copy(): GPU KV -> CPU
load_cpu_copy(): CPU KV -> 新分配的 GPU slot
```

不要采用以下判断方式：

- 依据模型名称硬编码；
- 依据是否存在同名方法；
- 调用后捕获 `NotImplementedError`；
- 依据主动 KV offload 开关；
- 只检查 save、不检查 restore。

### 7.2 明确区分三种回收动作

建议定义清晰的策略枚举或等价结构：

```python
class RetractionAction(Enum):
    CPU_SAVE_AND_RESUME = ...
    PREFILL_REBOOTSTRAP = ...
    ABORT_REQUEST = ...
```

决策逻辑：

```text
需要 retract
    |
    +-- allocator 支持完整 CPU round trip
    |     -> CPU_SAVE_AND_RESUME
    |
    +-- 不支持 CPU round trip，但是真实 PD 且请求可安全重算
    |     -> PREFILL_REBOOTSTRAP
    |
    +-- Fake transfer、无可用 Prefill、请求类型不支持重算等
          -> ABORT_REQUEST
```

### 7.3 CPU 保存恢复路径

对于 `CPU_SAVE_AND_RESUME`：

1. 调用 `get_cpu_copy()`；
2. 确认保存成功后释放 GPU KV；
3. 把请求加入 `retracted_queue`；
4. GPU KV 有空间时重新分配 slot；
5. 调用 `load_cpu_copy()`；
6. 删除 CPU 临时副本并恢复生成。

要求：保存失败时不能继续释放并假装请求可恢复。

### 7.4 Prefill rebootstrap 路径

对于 `PREFILL_REBOOTSTRAP`：

1. 释放旧的 Decode KV；
2. 保留已经向客户端输出的 token 状态；
3. 将请求交给已有 PD true-retraction rebootstrap 流程；
4. Prefill 使用原输入和已生成 token 重算 KV；
5. KV 重新传给 Decode；
6. 正确处理 boundary token，不能重复输出或漏 token；
7. 恢复流式输出和采样状态。

Fake KV Transfer 场景通常没有真实 Prefill 计算节点，不应假设它可以完成 rebootstrap。

### 7.5 无恢复路径时的 graceful abort

对于 `ABORT_REQUEST`：

1. 释放该请求占用的 GPU KV；
2. 不把它加入普通 `retracted_queue`；
3. 不调用 `load_kv_cache()`；
4. 通过现有输出通道返回单请求错误；
5. Scheduler 继续处理其他请求。

错误对象需包含完整字段，例如：

```python
FINISH_ABORT(
    message=(
        "Request was retracted because KV cache is full, but this KV "
        "pool does not support CPU save/restore and cannot rebootstrap."
    ),
    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
    err_type="InternalServerError",
)
```

必须验证 TokenizerManager 中 rid 的创建、输出、删除顺序，避免出现：

```text
Received output for rid=... but the state was deleted in TokenizerManager
```

### 7.6 调整 `release_req()` 职责

`release_req()` 不应根据 server flag 自行猜测是否 copy，而应由上层传入已经确定的策略，例如：

```python
release_req(..., retraction_action=RetractionAction.CPU_SAVE_AND_RESUME)
```

或者至少显式传入：

```python
release_req(..., save_cpu_kv=True/False)
```

调用方必须保证：

- `save_cpu_kv=True` 时 allocator 支持完整 round trip；
- `save_cpu_kv=False` 时请求不会进入依赖 `kv_cache_cpu` 的恢复路径。

## 8. 第二阶段：实现容量不足时等待

第一阶段保证服务稳定，但 Fake decode-only 中被 retract 的请求仍可能失败。如果目标是 c512 全部等待并最终完成，需要增加 SWA-aware admission control。

### 8.1 预算模型

每个请求的安全 SWA 预算至少应包括：

```text
SWA 输入尾部
+ max_new_tokens
+ speculative draft/verify reserve
+ KV page alignment
+ overlap/in-flight reserve
+ 固定安全余量
```

### 8.2 调度行为

```text
新请求到达
    |
    +-- 剩余 SWA budget 足够覆盖完整安全预算
    |     -> 加入 running batch
    |
    +-- 预算不足
          -> 保留在 waiting queue
          -> 已运行请求完成并释放 KV 后再进入
```

该方案符合“容量满后等待”的预期，但可能降低激进超卖带来的峰值吞吐，需要单独做性能回归。

## 9. 不建议优先实现的方案

### 9.1 直接为 DeepSeekV4TokenToKVPool 补 CPU copy

完整实现至少需要正确保存和恢复：

- SWA KV ring；
- C4 compressed KV；
- C128 compressed KV；
- C4 indexer K/scale；
- C4/C128 compressor state；
- full-to-SWA 映射；
- request-scoped ring state；
- DSpark speculative state；
- Unified KV 与非 Unified KV 两种布局；
- HiSparse 分支。

遗漏任意状态可能不立即崩溃，却会导致静默输出错误。建议将其作为独立的后续功能，不与第一阶段稳定性修复混在同一个 PR。

### 9.2 单纯提高 mem-fraction-static

提高 MF 会扩大部分 KV pool，使问题更晚出现，但：

- 不修复错误的 retraction 路径；
- 更高 MF 会减少 CUDA graph、临时 workspace 和运行时显存余量；
- 仍可能在更高并发或更长输出下再次触发。

### 9.3 仅降低 max-running-requests

这是当前有效的压测规避方式，但不是框架修复。它可以作为修复完成前的运行策略。

## 10. 测试计划

### 10.1 单元测试

#### 支持 CPU copy 的 KV pool

- 强制触发 retraction；
- 验证调用 `get_cpu_copy()`；
- 验证请求进入 `retracted_queue`；
- 验证恢复时调用 `load_cpu_copy()`；
- 验证生成结果连续。

#### 不支持 CPU copy 的 KV pool

- 使用 mock pool，使 `get_cpu_copy()` 明确不可用；
- 验证不会调用 `get_cpu_copy()`；
- 验证不会调用 `load_cpu_copy()`；
- 验证请求不会错误进入 CPU-resume 队列；
- 验证请求 rebootstrap 或 graceful abort；
- 验证 Scheduler 仍存活。

#### Abort 生命周期

- 验证 `FINISH_ABORT` 包含 `status_code` 和 `err_type`；
- 验证流式请求只收到一次结束事件；
- 验证 TokenizerManager rid 状态正确删除；
- 验证其他并发请求不受影响。

#### 边界测试

- 只剩一个请求且仍无法满足 KV 分配；
- 多次连续 retraction；
- speculative decoding 开启；
- page size 大于 1；
- overlap schedule 开启；
- 请求已输出部分 token；
- 请求恰好完成时发生 retraction；
- abort 与 retraction 同时发生。

### 10.2 集成测试

构造比真实 c512 更快触发的测试环境，可通过缩小测试 KV pool 或 `max_total_tokens` 实现。

至少覆盖：

1. 普通 PD Decode + 支持 CPU copy；
2. 真实 PD Decode + 不支持 CPU copy + 可 rebootstrap；
3. Fake Decode-only + 不支持 CPU copy；
4. DeepSeek-V4 hybrid SWA；
5. DeepSeek-V4 + DSpark；
6. streaming OpenAI API；
7. 非 streaming `/generate` API。

### 10.3 现场回归

使用原始压测规格：

```text
input=8192
output=2048
concurrency=384 -> 512
request-rate=INF
ignore_eos=true
```

检查：

- 服务进程是否持续存活；
- 是否还有 `NotImplementedError`；
- 是否发生 request-level abort；
- 高压结束后新请求是否成功；
- SWA pool 使用率；
- retracted/aborted/rebootstrap 请求计数；
- DSpark accept length/rate；
- Output TPS 和 TPOT 是否明显回退。

## 11. 验收标准

### 11.1 第一阶段验收

- KV pool 满时 API Server 和 Scheduler 不退出；
- 日志中没有来自 KV CPU copy 的 `NotImplementedError`；
- unsupported pool 不执行 CPU copy；
- 无法恢复时仅终止必要请求；
- 服务随后仍可完成低并发请求；
- 支持 CPU copy 的模型保持原有 save/resume 行为；
- c384 性能无明显回退。

### 11.2 第二阶段验收

- Fake decode-only c512 不因容量压力产生请求失败；
- 超出安全并行容量的请求进入 waiting queue；
- 所有请求最终完成；
- 无死锁、无无限 retraction、无重复输出；
- 吞吐和尾延迟变化有完整对比数据。

## 12. 建议实施顺序

1. 拉取并审阅 PR #32413 最新提交，不直接使用已关闭的 PR #31905。
2. 将 allocator CPU round-trip 能力判断移植到当前 v0.5.16 + #33407 分支。
3. 完成 `CPU_SAVE / REBOOTSTRAP / ABORT` 三路生命周期。
4. 修复并补齐 abort 输出状态和单元测试。
5. 使用缩小 KV pool 的集成测试快速验证服务不崩。
6. 在测试机复现 8K/2K c512。
7. 第一阶段稳定后，再设计并实现 SWA-aware admission control。
8. 如确有需求，最后单独评估 DeepSeek-V4 完整 CPU KV round-trip。

## 13. 当前压测规避策略

修复合入前：

```text
建议最高先测到 c384。
c512 前必须检查每个 DP Rank 的 SWA pool 余量。
不要依赖提高 MF 作为最终解决方案。
```

如果需要继续探索吞吐极限，应逐级增加并发，并确保服务日志、GPU dmon 和系统监控持续保存。

## 14. 2026-08-11 第一阶段实施结果

已完成能力驱动的安全降级修复：

- KV pool/allocator 显式声明是否支持 CPU KV save/restore round trip；
- DeepSeek-V4 hybrid SWA pool 判定为不支持；
- Decode retraction 不再进入未实现的 `get_cpu_copy()`；
- 仅对必须释放的请求执行 request-level abort，Scheduler 保持运行；
- TokenizerManager 识别并静默丢弃主动 abort 后跨通道迟到的旧输出；
- 支持 CPU copy 的 pool 保留原 save/resume 路径。

验证结果：

```text
配置：TP8 + Attention DP8 + EP4 + DSpark + MF 0.6
负载：512 concurrent，8192 input IDs，2048 output tokens，random-ids
结果：504 × HTTP 200/length，8 × HTTP 500/graceful abort
NotImplementedError：0
Scheduler hit an exception：0
TokenizerManager deleted-state warning：0
压力后短请求：HTTP 200，正常生成 16 tokens
```

日志目录：

```text
/sgl-workspace/workspace/stress test/20260811_tp8_dp8_ep4_c512_fix
```

结论：第一阶段“容量耗尽不导致服务崩溃”已验收。Fake Transfer 没有真实 Prefill
节点可用于 KV 重算，因此当前超载语义是少量请求 graceful abort，不承诺 C512 下
512/512 全部成功；若需要全部排队完成，仍需第二阶段 admission control/backpressure。
