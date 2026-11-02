# DeepSeek-V4-Flash Decode fake 压力测试记录

## 测试配置

- Decode：`10.207.193.166:30002`
- 输入/输出长度：8192/8192
- 并发：1024
- 请求数：4096
- 参数：`--enable-test-pd-without-transfer`
- `mem-fraction-static=0.55`
- Full/SWA page size：256

## 结果快照

```text
retract_decode: aborted request:       974
KV cache pool is full:                 920
NotImplementedError:                     0
Bug in memory estimation:                0
Scheduler hit an exception:              0
crashed with exit:                        0
Decode health:                          200
```

压测中 Full KV 使用率达到约 0.94～0.97，SWA KV 使用率达到 1.00。KV 满后请求级 abort 持续生效；补齐 page-aligned 预算后，没有再次出现 `available=256, required_alloc=8448` 的预分配断言，Scheduler 保持存活。

## 文件

- `benchmark_client.txt`：客户端输出快照
- `decode.log`：Decode 日志快照
