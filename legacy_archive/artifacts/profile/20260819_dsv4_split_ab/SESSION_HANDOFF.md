# 会话交接：PR #32320 Decode-only Split A/B

## 要验证的问题

PR #32320 是否真正减少 SM120 FlashMLA 的 SWA page-split 工作量，以及 kernel 层净收益是多少。

## A/B 唯一变量

- Split ON：提交 `3cf4ce7db`，只标记并拆分本轮访问的 SWA pages。
- Split OFF：父提交 `f5868c99b`，每轮拆分整个 SWA KV pool。
- 共同配置：TP8/Attention DP8/EP8、DSpark 6 draft tokens、Decode CUDA Graph `full`。
- 负载：C8、1→128；每组 warmup 40、正式 80 requests，自动采集 12 scheduler steps。

## Kernel 层结果

- `_page_split_kernel`：ON 7.576 ms，OFF 57.219 ms；OFF 增加 655.23%。
- ON 新增 `_page_mark_kernel` 0.584 ms。
- page mark + split：ON 8.161 ms，OFF 57.219 ms，净节省 49.058 ms。
- TP0 全部 GPU kernel duration 求和：ON 673.767 ms，OFF 746.606 ms；OFF 增加 10.81%。

## 文件入口

- `delivery/README.md`：A/B 配置和核心结果。
- `delivery/01_split_on_decode_timeline_tp0.csv`、`01_split_off_decode_timeline_tp0.csv`：两组时间序列。
- `delivery/02_split_on_decode_kernel_aggregate_tp0.csv`、`02_split_off_decode_kernel_aggregate_tp0.csv`：两组聚合表。
- `delivery/04_skill_triage_report.md`：三表分析。
- `delivery/03_split_on_off_kernel_comparison_tp0.csv`：逐 kernel 对比。
- `delivery/perfetto_lite/`：代表 Rank 可视化 Trace。

## 如何解释

该实验只能证明 kernel 层收益，不能直接把 10.81% 当作端到端吞吐提升。端到端结论必须读取 `20260819_dsv4_split_e2e_ab/` 的无 Profiler A/B。
