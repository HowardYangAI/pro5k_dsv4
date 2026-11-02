# 完全未修复基线启动失败

- 基线提交：`48886803b94596884d737132a381bd5124ea1cfd`
- Decode MF：0.55
- 配置：DeepSeek-V4-Flash，DP8/EP4，DSpark，Decode CUDA Graph
- 结果：服务在 warmup 阶段崩溃，尚未接收 fake 压力请求

## 原始错误

```text
compute_local_num_token_non_padded(...)
TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
Subprocess scheduler_0 ... crashed with exit code -3
```

原因是手工构造的 DSpark `ForwardBatch` 没有填写 `num_token_non_padded`，EP/DP CUDA Graph 路径读取到 `None`。这验证了本 CR 中独立的 DSpark 元数据修复确有必要。

为了继续隔离验证 retraction 修复，后续对照服务只补回该 DSpark 启动修复，仍移除 CPU-copy capability、PD rebootstrap 和 graceful-abort 等 retraction 修改。

## 文件

- `decode_unfixed_startup.log`：完整启动及异常栈
