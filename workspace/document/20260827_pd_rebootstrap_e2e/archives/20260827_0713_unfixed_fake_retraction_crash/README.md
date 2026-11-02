# 未修复版本 fake retraction 崩溃对照

## 测试条件

- 版本：未修复基线（仅保留 DSpark + EP 启动修复）
- 请求入口：Decode `10.207.193.166:30002`
- fake 标记：`--enable-test-pd-without-transfer`（`bootstrap_host=2.2.2.2`）
- Decode memory fraction：0.55
- max concurrency：1024
- prompts：4096
- input/output：8192/8192

## 结果

fake 压力触发 retraction 后，未修复逻辑无条件尝试同步 CPU KV copy：

```text
retract_decode
-> release_req
-> offload_kv_cache
-> SWA allocator.get_cpu_copy
-> KV pool.get_cpu_copy
-> NotImplementedError
-> Decode scheduler crash (exit code -3)
-> 客户端访问 /get_server_info 时 Connection refused
```

Decode 测试后 `/health=000`，tmux 服务会话已经退出。这与原始缺陷及本轮
未修复对照的预期完全一致。

## 与修复版 fake 对比

- 未修复：fake retraction -> 不支持的 CPU copy -> scheduler 崩溃。
- 修复后：识别 fake 请求不能执行 PD rebootstrap，只 graceful abort 被撤回请求；
  其他请求继续，Decode 保持健康。

## 文件

- `decode_unfixed_fake_mf055.log`：Decode 完整服务日志。
- `benchmark_output.txt`：客户端压测输出。
