# DeepSeek-V4 FlashMLA SWA page-split A/B Profile

## 口径

- 配置：TP8 / Attention DP8 / EP8 / DSpark 6 draft tokens
- Decode CUDA Graph：开启，backend=`full`
- 负载：C8、输入 1 token、输出 128 tokens
- 每组 warmup：40 requests
- 每组正式请求：80 requests
- Profile：正式负载中自动采集 12 scheduler steps
- 代表 Rank：TP0 / DP0 / EP0；完整 8-rank Trace 保留在 `split_on/decode` 和 `split_off/decode`

## A/B 定义

- Split ON：包含 PR #32320，只拆分/复制本轮访问的 SWA pages。
- Split OFF：回退 PR #32320，每轮拆分/复制整个 SWA KV pool。

## 主要结果

- `_page_split_kernel`：7.576 ms → 57.219 ms，关闭优化后增加 655.23%。
- Split ON 额外 `_page_mark_kernel`：0.584 ms。
- 两者合计：Split ON 8.161 ms，Split OFF 57.219 ms，净增加 49.058 ms。
- TP0 GPU kernel duration 求和：673.767 ms → 746.606 ms，关闭优化后增加 10.81%。

百分比基于所选 Rank 的 GPU kernel duration 求和；多 stream 可能重叠，不等同于端到端墙钟时间。
