# DeepSeek-V4-Flash Decode 工作备份

归档时间：2026-08-19（UTC）

本目录汇总了本轮 DeepSeek-V4-Flash Decode 工作中产生的实验数据、Profile、设计文档、会话记录以及可恢复的源码状态。

## 目录

- `artifacts/stress test/`：单机 8 卡、双机 16 卡、不同 EP 配置及 C512/C640/C1024 稳定性压测。
- `artifacts/profile/`：NSYS、Torch Profile 资料及 2026-08-19 Profile 定位实验。
- `artifacts/legacy-profiles/`：原 `/sgl-workspace/profiles` 中已生成的 8-rank Torch traces。
- `artifacts/plan/`：技术方案、答辩图、Excalidraw 源文件与讲解稿。
- `session/all-sessions/`：Codex 原始会话 JSONL。
- `session/primary-session.jsonl`：本轮跨 2026-08-06 至 2026-08-19 的主会话快照。
- `session/attachments/`：会话中上传/粘贴的压测结果与附件。
- `source/`：SGLang 和压测工具的 Git bundle，以及各 worktree 的未提交 patch。
- `metadata/`：源码状态、Profile 初步结论及文件校验信息。

## 主要工作内容

1. DeepSeek-V4-Flash Decode 单机 8 卡与双机 16 卡性能压测。
2. TP/Attention DP/EP 并行配置及 TCP 跨节点通信影响分析。
3. 高并发 SWA KV Cache 耗尽时的 retraction、PD rebootstrap 与 graceful abort 修复。
4. KV Cache 分页容量估算修复及 C640/C1024 稳定性验证。
5. DSpark + EP 手工构造 `ForwardBatch` 时缺失有效 token 元数据的修复。
6. SM120 FlashMLA page split PR #32320 的回移与 Profile 对比。

## 补充归档

- `artifacts/early-logs/`：最初的 DSpark smoke 与监控日志。
- `source/cupti-kernel-monitoring.bundle`：CUPTI Profile 辅助工具完整仓库。
- `source/old-experiments-stash.patch.gz`：`stash@{0}` 中 KV-only CR 之前的旧实验修改。
- `metadata/runtime-versions.txt`、`pip-freeze.txt`、`constraints.txt`：运行环境与 Python 依赖快照。
- CR 地址：`https://kunpeng.xiaojukeji.com/view/revision/6565881`。

## 2026-08-19 Profile 定位补充

已保存 CPU+GPU 按阶段、每阶段 5 step 的推荐 trace，以及服务日志和 Xid 现场：

- `artifacts/profile/20260819_ep8_e2e_torch_profile/README.md`
- `artifacts/profile/20260819_ep8_e2e_torch_profile/PROFILE_SUMMARY.md`
- `artifacts/profile/20260819_ep8_e2e_torch_profile/traces/cpu_gpu_by_stage_5_8k/`

CPU-only、`num_steps=1/10` 等对照实验的数百 MiB 原始 trace 未重复提交到 Git；对应日志、测试结论均已保存，完整原始 trace 仍位于机器的 `/sgl-workspace/workspace/profile/20260819_ep8_e2e_torch_profile/`。

## 2026-08-19 正式 Profile 与 PR #32320 A/B

- `artifacts/profile/PROFILE_INDEX.md`：全部 Profile 实验的统一入口、阅读顺序、可引用结论、采集规则和 Git 归档范围。后续会话应先读该文件，再读每组目录中的 `SESSION_HANDOFF.md`。
- `artifacts/profile/20260819_dsv4_ep8_dspark_c8/`：TP8/Attention DP8/EP8 + DSpark 的 C8 Prefill/Decode 正式 Profile，包含时间序列表、聚合表、skill 报告和 Perfetto 轻量 Trace。
- `artifacts/profile/20260819_dsv4_split_ab/`：PR #32320 Split ON/OFF 严格 A/B；每组 C8、warmup 40、正式请求 80、Decode 1→128、自动采集 12 scheduler steps。
- A/B 结论：关闭 #32320 后 `_page_split_kernel` 总时间从 7.576 ms 增至 57.219 ms；TP0 GPU kernel duration 求和增加 10.81%。
- `artifacts/profile/20260819_RAW_TRACE_MANIFEST.tsv`：未重复提交到 Git 的 32 份完整 8-rank Trace 的大小、原始路径和 SHA256。
- `artifacts/profile/20260819_dsv4_split_e2e_ab/`：PR #32320 Split ON/OFF 的完整端到端 Profile 和无 Profiler A/B。Profile 证明 page mark + split 净开销降低 79.06%；无 Profiler 的 `C8、40 requests、8192→64` 三轮测试中，ON 的 Output TPS 均值高 8.61%，配对中位数收益 4.50%，完成时间均值降低 8.15%。逐轮方差较大，因此最终表述为“端到端收益量级约为几个百分点”，不采用单轮 Profile 的 19.22% TPOT 差异作为性能结论。目录包含原始 benchmark 日志、逐轮/汇总 CSV、时间序列表、逐 kernel A/B、skill 报告和可直接打开的 Perfetto Trace。
- `artifacts/profile/20260819_dsv4_split_e2e_ab/no_dspark_e2e_ab/`：关闭 DSpark 后的补充配对测试。Split ON 三轮 Output TPS 为 73.33/77.46/76.02，配对中位收益为 +14.35%；Split OFF 的 seed 1787128003 两次均复现约 4.8 秒长 ITL。该结果说明波动不是 DSpark accept length 引入，同时验证全池 page split 位于 Decode 串行关键路径。鉴于仅三组主样本，报告保留全部原始数据，不把单轮 +26.76% 当作稳定收益。

完整 8-rank 原始 Trace 共约 1.47 GiB，未放入代码仓库；交付目录中的轻量 `.trace.json.gz` 可直接使用 Perfetto 打开，CSV 和分析报告保留了正式结论。

## 注意事项

- 压测副本中超过 20 MiB 的文本日志已使用 gzip 压缩，原始工作目录未被修改。
- `source/sglang-all-refs.bundle` 因单文件超过常见 Git 托管平台 100 MiB 限制，被拆分为若干 `sglang-all-refs.bundle.part-*`；恢复方式见 `source/RESTORE.md`。
- Torch trace 可使用 Chrome `chrome://tracing`、Perfetto 或 PyTorch TensorBoard Profiler 打开。
