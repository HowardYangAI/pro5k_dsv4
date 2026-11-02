# DeepSeek-V4-Flash Decode 压测与稳定性修复讲稿

大家好，这部分我介绍的是 DeepSeek-V4-Flash Decode 节点的性能压测和高并发稳定性修复。

这项工作的目标主要有两个：第一，探索单机和双机 Decode 节点的并行配置与性能上限；第二，在高并发场景下验证服务稳定性，并解决压测过程中暴露的问题。

## 一、Decode 节点压测

首先看压测部分。

测试模型是 DeepSeek-V4-Flash，开启 DSpark 推测解码，固定输入 8K、输出 2K，并逐步提高并发。硬件是 NVIDIA RTX PRO 5000 72GB，分别测试单机 8 卡和双机 16 卡。

在并行方案方面，DSpark 当前暂不支持 Decode CP 和 Decode PP，同时 Attention DP 要求 DP 等于 TP。因此，在保留 DSpark 的情况下，可调整空间比较有限，主要扫描了不同的 EP 配置。

单机 8 卡中，TP8、DP8、EP4 在 C384 时，Output TPS 达到 10,754 tok/s，Mean TPOT 为 33.17 ms。随着并发提高，吞吐仍然能够持续增长，说明单机内部的 PCIe 和 NUMA 通信开销总体可接受。

双机 16 卡使用 TP16、DP16、EP16，同样在 C384 下，Output TPS 只有 1,385 tok/s，Mean TPOT 上升到 259.40 ms。

与单机结果相比，双机 Output TPS 下降约 87.1%，TPOT 增加到 7.8 倍。因此，增加 GPU 数量并没有带来性能提升，反而出现明显退化。

结合日志分析，主要原因是当前跨节点通信使用 NCCL Socket/TCP，并且跨节点进程组无法使用 SGLang 的 CustomAllreduce。模型每轮 Decode 都需要等待跨节点 collective，网络通信延迟进入关键路径，最终成为双机方案的主要瓶颈。

所以这一部分的结论是：当前机器和网络条件下，单机 8 卡更适合作为 Decode 基线；双机方案如果仍然使用 TCP，则不适合通过增加 TP、DP 和 EP 规模来提升 Decode 性能。后续若继续探索双机，需要优先解决 RDMA 或其他高性能跨节点通信问题。

## 二、高并发下的 SWA KV Cache 崩溃

第二部分是在高并发压测中发现的稳定性问题。

当并发提高到 C640 和 C1024 时，SWA KV Cache 会逐渐达到 100%。Scheduler 每轮会通过 `check_decode_mem()` 检查剩余空间；空间不足时，进入 `retract_decode()`，回收部分正在运行的请求，为其余请求释放 KV。

原来的请求回收流程默认 KV Pool 支持 CPU copy。它会把被回收请求的 KV Cache 拷贝到 CPU，然后把请求重新放回等待队列，后续再恢复执行。

但 DeepSeek-V4-Flash 使用的 KV Pool 没有实现 CPU copy。原代码没有先判断能力，直接调用未实现的接口，因此触发 `NotImplementedError`，导致整个 Decode 服务退出。

这里的问题不是 KV Cache 满本身。KV 满载是高并发下的正常边界状态，真正的问题是回收逻辑错误地假设所有 KV Pool 都支持 CPU copy。

针对这个问题，我增加了 `supports_cpu_copy()` 能力判断，并把恢复逻辑拆成三条路径。

第一，如果 KV Pool 支持 CPU copy，就继续使用原来的 offload、等待和恢复流程。

第二，如果不支持 CPU copy，但处于真实 PD 环境，就释放 Decode 侧 KV，并让请求返回 Prefill 节点进行 rebootstrap，由 Prefill 重新计算并传输 KV，然后继续 Decode。

第三，如果使用 Fake PD，由于没有真实 Prefill 节点，无法重算和重新传输 KV，因此只 graceful abort 少量超载请求并回收它们的 KV，保证其他请求和服务继续运行。

修复后重新测试 C640 和 C1024。SWA KV Pool 都能够达到 100%，但服务不再崩溃；Fake PD 下的超载请求按预期退出，其余请求可以继续完成。

## 三、Paged SWA 容量估算偏差

在验证请求回收逻辑时，还发现了另一个独立问题：Scheduler 的容量估算与 KV Pool 的真实分配方式不一致。

原来的准入检查直接按 token 数计算。例如，一个请求需要 257 个 token，检查逻辑只判断剩余空间是否大于等于 257。

但 KV Cache 实际按固定 Page 分配。假设 Page Size 是 256，那么 257 个 token 需要向上取整为两个 Page，真实占用是 512，而不是 257。

这就会出现“调度器判断放得下，但分配器实际分配失败”的情况。

修复方式是参考并回移官方 PR #33794，对 Full KV Pool 和 SWA KV Pool 分别按 Page Size 向上对齐。修复后，准入检查和真实 Page 分配使用相同的容量口径，257 会按 512 判断，不会再错误接纳请求。

## 四、DSpark + EP 的 ForwardBatch 元数据问题

压测过程中还遇到了 DSpark 开启 EP 后的报错：EP1 可以正常运行，但 EP8 会在 draft forward 阶段出现 `None - int`。

普通请求通过 `ForwardBatch.init_new()` 创建 batch，这个过程会根据输入初始化 `num_token_non_padded` 等调度元数据。

但 DSpark 的 draft model 路径是手工调用 `ForwardBatch(...)` 创建 batch，没有经过 `init_new()`，因此 `num_token_non_padded` 以及对应的 CPU 字段没有初始化，值一直是 `None`。

EP 等于 1 时，不会进入这段非 Padding token 统计逻辑，所以问题没有暴露。EP 大于 1 后，会开启 `enable_num_token_non_padded`，后续在计算 Attention 和 EP rank 对应的真实 token 区间时读取这个字段，于是执行了 `None - int` 并报错。

修复方式是在创建 DSpark draft batch 后，使用 `input_ids.numel()` 得到当前 batch 的真实 token 数，并同时设置 `num_token_non_padded` 和 `num_token_non_padded_cpu`。

这项修改修复的是 batch 描述元数据，不会改变模型的数学计算。补齐后，TP8、DP8、EP8 场景可以正常执行，EP collective 和 CUDA Graph 能够获得一致的 tensor shape。

## 五、验证与总结

最后是验证结果。

相关修复通过了 28 个测试和 13 个子测试，并完成 C640、C1024 高并发服务验证。目前补丁已经提交内部 CR 6565881。

这项工作最终形成了三方面产出：

第一，完成了 DeepSeek-V4-Flash Decode 单机 8 卡和双机 16 卡的性能矩阵，明确了 TCP 跨节点 collective 是当前双机 Decode 的主要瓶颈。

第二，修复了 SWA KV 满载时服务整体崩溃的问题，把单一 CPU copy 假设改成了按 KV Pool 能力选择恢复方式。

第三，修复了 Paged SWA 容量估算以及 DSpark EP batch 元数据两个边界问题，使高并发准入、请求回收和 EP 执行流程更加稳定。

我的介绍到这里，谢谢大家。

## 可能的追问

### 为什么双机 16 卡反而比单机 8 卡慢？

Decode 每一步计算量较小，但同步频率高。双机 TP16、DP16、EP16 把 collective 扩展到跨节点，而当前网络走 Socket/TCP，通信延迟无法被计算掩盖，所以增加 GPU 后反而变慢。

### 为什么不继续测试 CP 和 PP？

当前 DSpark 暂不支持 Decode CP 和 Decode PP，Attention DP 又要求 DP 等于 TP。为了保持相同推测解码能力，只能主要扫描 EP。强行关闭 DSpark 再比较会引入新的变量，不适合作为同一基线直接对比。

### Fake PD 为什么只能 abort？

Fake PD 没有真实 Prefill 节点，也没有 Mooncake/NIXL KV 传输。Decode 释放 KV 后，没有节点可以重新计算并传回 KV，因此只能终止部分超载请求。真实 PD 可以返回 Prefill rebootstrap。

### `num_token_non_padded` 是模型权重或计算结果吗？

不是。它是 batch 调度元数据，用于区分有效 token 和 Padding，并帮助 Attention、EP collective 与 CUDA Graph 计算一致的 tensor shape。
