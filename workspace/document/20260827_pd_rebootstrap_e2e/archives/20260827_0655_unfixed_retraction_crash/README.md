# 未修复版本 retraction 崩溃对照

## 测试条件

- 版本：未修复基线（仅补入 DSpark + EP 启动修复，以便隔离 retraction 问题）
- PD 分离真实请求
- Decode memory fraction：0.55
- max concurrency：1024
- prompts：1024
- input/output：8192/8192

## 结果

Decode KV 使用率达到约 99% 后触发 retraction。未修复逻辑无条件尝试同步 CPU KV copy：

```text
retract_decode
-> release_req
-> offload_kv_cache
-> SWA allocator.get_cpu_copy
-> HiSparse/DSv4 KV pool.get_cpu_copy
-> NotImplementedError
-> Decode scheduler crash
-> Router 返回大量 503
```

这准确复现了本次修复针对的原始问题。

## 与修复版对比

- 未修复：KV 满 -> 无条件 CPU copy -> `NotImplementedError` -> scheduler 崩溃。
- 修复后真实传输：检测 `supports_cpu_copy()`；不支持时走 PD rebootstrap，返回 Prefill 重算并继续请求。
- 修复后 fake 传输：无法安全重算时只 graceful abort 被撤回的请求，服务保持存活。

## 文件

- `decode_unfixed_retraction_mf055.log`：Decode 完整服务日志。
- `benchmark_output.txt`：客户端压测输出。
