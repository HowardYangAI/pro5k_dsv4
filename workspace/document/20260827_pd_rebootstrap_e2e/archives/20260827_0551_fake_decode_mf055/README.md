# Fake Decode retraction 防御性测试

- 时间：2026-08-27 05:51–06:01 UTC
- Decode MF：0.55
- 请求入口：Decode `10.207.193.166:30002`
- fake 标记：`bootstrap_host=2.2.2.2`
- 不经过 Router/Prefill，不执行真实 KV 传输

## 结果

- 因 OOM retraction 被安全 abort 的 fake 请求：343
- 正常完成的 fake 请求：196
- fake 请求进入 `PD_REBOOTSTRAP_DISPATCH`：0
- 测试后 Decode `/health`：200
- Decode 进程未崩溃

典型预期错误为：

```text
retract_decode: aborted request ...
(KV cache does not support synchronous CPU save and restore)
```

对应请求返回 `FINISH_ABORT`/HTTP 500；其他请求继续执行。该结果验证了修复后的防御行为：fake 请求无法访问真实 Prefill，不能 PD rebootstrap，因此只终止被撤回请求，不使调度器或服务进程崩溃。

## 文件

- `decode_fake_full.log`：包含本轮 fake 测试及此前同一服务进程日志的完整 Decode 日志快照；本轮记录从约 05:51 UTC 开始。
