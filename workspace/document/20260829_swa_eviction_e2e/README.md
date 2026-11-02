# DSpark SWA eviction（CR 6624145）端到端测试

- Commit: `d07dd66fb75e9382e8455d1e40adc6e06a4feb89`
- 内容：`fix swa not eviction when dspark enabled`
- 不包含 PD rebootstrap 修复
- 为通过 DSpark + DP/EP CUDA Graph warmup，仅额外补入 10 行
  `num_token_non_padded` metadata 修复；该补丁与 SWA eviction/rebootstrap 无关。
- Prefill: `10.207.193.156:30001`
- Decode: `10.207.193.166:30002`
- Router: `10.207.193.156:30000`
- Decode MF: `0.55`
- 目标：验证高并发长输入/输出下，DSpark 的过期 SWA KV 是否得到淘汰，
  SWA 使用量是否从单调增长变为有界波动/排空后回落。

日志：

- `prefill_fixed.log`
- `decode_fixed.log`（远端 Decode 机器）
- `router_fixed.log`
