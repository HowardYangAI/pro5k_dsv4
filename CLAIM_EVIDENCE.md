# 两项成果的归档证据索引

2026-11-02 核对：支撑以下两项成果的现存代码、补丁、测试、命令、压测日志、统计表和图文资料已同步。本页把证据按成果对应起来，同时记录版本归因和历史缺失文件。本次只核对已有材料，没有重新运行 GPU 实验。

首批归档提交：`8036a44341b67f1caebb449750a7156689c709e7`。对话快照截至 2026-11-02 17:55:04（北京时间），包括 22 个线程、4067 条可读消息、41 个附件；后续核对不递归更新对话快照。

## 1. PRO5000 / SGLang / DSV4-FLASH 高并发稳定性

对应问题：SWA KV Cache 耗尽、不支持 CPU save/restore 的 KV Pool 进入 offload 路径导致 PD Decode 崩溃，以及容量预算与物理 Page 分配不一致。

| 内容 | 已归档证据 | 阅读要点 |
| --- | --- | --- |
| 根因、KV Pool 能力判断、CPU 保存 / Prefill 重算 / 请求级降级策略 | [修复方案](legacy_archive/artifacts/plan/deepseek-v4-pd-decode-kv-retraction-fix-plan.md) · [实现与回归报告](workspace/document/20260827_pd_rebootstrap_e2e/PD_DECODE_RETRACTION_FIX_REPORT.md) | 方案前半部分是历史规划；第 14 节记录第一阶段落地。后续 rebootstrap 和 abort-only 属于不同版本，不能合并成一次实验。 |
| 实际修复代码和提交历史 | [dsv4-fixed 工作树](source/worktrees/sglang-dsv4-fixed/) · [swa-evict-fixed 工作树](source/worktrees/sglang-swa-evict-fixed/) · [早期 KV-only 补丁](legacy_archive/source/kv-only-staged.patch.gz) | 工作树内有 `metadata.json`、`committed.patch`、`committed_history.mbox`、`files/`；可追踪能力接口、Decode 队列、Page 预算和请求清理。 |
| 单元与回归测试 | [CPU copy / retraction 测试](source/worktrees/sglang-swa-evict-fixed/files/test/registered/unit/managers/test_retraction_cpu_copy.py) · [SWA 回收测试](source/worktrees/sglang-swa-evict-fixed/files/test/registered/unit/managers/test_schedule_batch_swa_retraction.py) · [迟到输出清理](source/worktrees/sglang-swa-evict-fixed/files/test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py) · [Decode 队列清理](source/worktrees/sglang-swa-evict-fixed/files/test/registered/unit/disaggregation/test_decode_queue_cleanup.py) | 归档的是原测试及历史运行记录；本次归档核对没有重新执行 GPU 回归。 |
| C640 Page 对齐修复前后 | [修复前崩溃日志](<legacy_archive/artifacts/stress test/20260811_three_path_c512/server-c640-pre-page-align-crash.log>) · [修复后日志](<legacy_archive/artifacts/stress test/20260811_three_path_c512/server-c640-page-align-passed.log.gz>) | 修复前出现 `available=256, required_alloc=8448` 及 Scheduler 异常；修复后日志记录 1937 次 KV full、2406 次请求级 abort、零 Scheduler 异常。 |
| C1024 真实 PD、Page 对齐后的容量压力 | [2026-08-30 报告及完整日志目录](workspace/document/20260830_full_kv_abort_only_pagealigned_c1024/README.md) | 8192→8192，计划 4096 请求；用户提前停止，客户端记录到 560/4096。触发 30 次请求级 abort，未出现 NotImplementedError、Page 预算断言或 Scheduler 异常，三端健康检查均为 200。 |
| C1024 FakePD 的 KV 满载 | [2026-08-30 FakePD 报告及日志](workspace/document/20260830_full_kv_abort_only_pagealigned_fake_c1024/README.md) | SWA 使用率达到 1.00；974 次请求级 abort、920 次 KV full 记录，未出现 CPU copy 未实现、预算错误或 Scheduler 崩溃，Decode health=200。 |
| 真实 PD 重算与未修复对照 | [rebootstrap 端到端实验](workspace/document/20260827_pd_rebootstrap_e2e/README.md) · [未修复崩溃现场](workspace/document/20260827_pd_rebootstrap_e2e/archives/20260827_0655_unfixed_retraction_crash/README.md) | 与后续 abort-only 实验分开阅读。 |
| DSpark 下 SWA 回收及后续修复 | [SWA eviction 实验](workspace/document/20260829_swa_eviction_e2e/README.md) · [SWA common force 修复描述](workspace/document/20260830_swa_common_force_e2e/CR_DESCRIPTION.md) | 包含其他触发条件和排障记录。 |
| 架构、问题与修复图 | [KV 回收图](legacy_archive/artifacts/plan/02-swa-kv-retraction-fix.png) · [Page 与 DSpark 修复图](legacy_archive/artifacts/plan/03-page-and-dspark-ep-fixes.png) · [真实 PD 请求流程](legacy_archive/artifacts/plan/request-to-real-pd-decode-flow.png) | 同目录保留 Excalidraw 源文件。 |

这里的“稳定”指在容量压力下避免 Scheduler/服务级崩溃，并通过重算或请求级 abort 释放资源、恢复健康。它不等于所有过载请求均成功，也不能由提前停止的真实 PD C1024 实验推出长时间满载全量完成。

早期 C512 测试明确记录 504 个 HTTP 200、8 个 graceful abort。2026-08-13 C1024 的 `server.log` 实际是端口冲突的启动失败日志，不能用作成功压测证据；当时运行中服务的统计还保存在[主会话](conversations/threads/019fd6e4-10ac-7611-b6e0-32143580ab3d.md)中，但未留存独立的完整运行服务日志。上述 2026-08-30 目录则包含实际保存在磁盘上的 C1024 压力日志。

## 2. SM120 KV Page Mask 与 E2E +8.61%

对应问题：Decode 为每层切分整个 KV Page 池，搬运了当前批次未访问的 Page。实测优化以 Page Mask 标记本轮访问的 Page，只切分被标记的 Page。

| 内容 | 已归档证据 |
| --- | --- |
| 实测 Split ON 完整源码 | [experiment_split_on.py](source/claims/page_split/experiment_split_on.py)，提交 `3cf4ce7dbe461c9b86447851ca4acef2caab1f37` |
| 实测 Split OFF 完整源码 | [experiment_split_off.py](source/claims/page_split/experiment_split_off.py)，父提交 `f5868c99b434c65aa3905cdf09518c3cfb477d89` |
| 两版本差异及来源 | [原始单文件 diff](source/claims/page_split/experiment_split_on_vs_off.patch) · [源码来源与 SHA-256](source/claims/page_split/provenance.json) · [Git bundle 恢复说明](legacy_archive/source/RESTORE.md) |
| 无 Profiler 的三轮 E2E A/B | [实验报告](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_profiler_e2e_ab/README.md) · [命令和参数](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_profiler_e2e_ab/BENCHMARK_COMMAND.md) |
| 原始数据与计算 | [六份 benchmark 原始日志](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_profiler_e2e_ab/raw_logs/) · [逐轮数据](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_profiler_e2e_ab/01_runs.csv) · [均值及方差](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_profiler_e2e_ab/02_summary.csv) · [配对对比](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_profiler_e2e_ab/03_paired_comparison.csv) · [统计脚本](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/summarize_no_profiler.py) |
| Profile 机制分析 | [报告](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/README.md) · [逐 kernel 对照](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/05_e2e_split_on_off_kernel_comparison_tp0.csv) · [ON 轻量 Trace 和时间线](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/split_on_c4_8k64_e2e/) · [OFF 轻量 Trace 和时间线](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/split_off_c4_8k64_e2e/) |
| 关闭 DSpark 的补充对照 | [独立实验目录](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/no_dspark_e2e_ab/) |
| 问题、方案、Kernel 图示 | [问题](workspace/graph/deepseek-v4-swa-page-split-optimization-01-problem.png) · [方案](workspace/graph/deepseek-v4-swa-page-split-optimization-02-solution.png) · [Kernel](workspace/graph/deepseek-v4-swa-page-split-optimization-03-kernel.png)，旁有 Excalidraw 源文件；涉及后续实现的细节应按下面的版本说明区分。 |

### +8.61% 的精确口径

负载是 TP8 / Attention DP8 / EP8、DSpark 6 draft tokens、Decode CUDA Graph、C8、40 请求、8192 输入→64 输出。ON/OFF 使用相同三个 seed：`1787127001`、`1787127002`、`1787127003`，六轮均为 40/40 完成。

Output TPS 三轮均值为 ON **38.36 tok/s**、OFF **35.32 tok/s**，相对差值 `(38.36 / 35.32 - 1) × 100 = 8.61%`。逐对收益为 **1.27%、21.48%、4.50%**，配对中位数 **4.50%**。只有三对样本，波动和 accept length 差异均已保存，不应外推为所有负载下稳定提升 8.61%。

另一次带 Profiler 的实验显示，OFF 的 Page split 344.85 ms 降为 ON 的 split 37.18 ms + mark 35.04 ms，净开销降低约 79.06%。这是机制证据，与上面的无 Profiler 吞吐统计是不同实验。

### 批量标记、uint64 搬运的版本归因

实测 ON 版本为**每个 Triton program 标记一个索引、以 uint8 搬运**，没有 1024 索引的批量标记和 uint64 搬运。因此不能把 +8.61% 归因于“Page Mask + 批量标记 + uint64”这个组合，也不能把后续社区实现写成该次本地实验的代码。

批量标记、uint64 搬运确实出现在 2026-09-08 会话抓取的社区 `main` 中，相关材料也已归档：

- [社区源码片段](source/claims/page_split/community_main_20260908_excerpt.py.txt)：保留原文件第 240–620 行的行号；其中包含 uint64 copy、批量 Page mark、`_MARK_BLOCK=1024`。
- [提取来源](source/claims/page_split/provenance.json)：记录原始事件路径、记录行号、抓取时间、URL、SHA-256。该抓取未固定 commit，只是完整文件的一段，不是可独立运行的完整模块。
- [对应可读会话](conversations/threads/01a0466f-37e1-7b11-83d6-bb8ea61ae9d5.md)：保留当时分析和后续表述。历史会话曾把版本合并描述，本页按实际源码纠正归因。

## 3. 保存范围和现有缺口

1. **48 份全量多 Rank 原始 Trace 未保存**：两个历史 manifest 共列出 1,880,427,160 字节；当时明确没有提交，这次检查原路径及补查存储也未找到。保存的是路径、大小、SHA-256、CSV、分析报告及轻量 Trace。见 [逐项缺失清单](metadata/claims/missing_full_traces.json)、[原 manifest 之一](legacy_archive/artifacts/profile/20260819_RAW_TRACE_MANIFEST.tsv)、[E2E manifest](legacy_archive/artifacts/profile/20260819_dsv4_split_e2e_ab/RAW_TRACE_MANIFEST.tsv)。此外，旧 README 提及但未逐项列出的 CPU-only / num_steps=1/10 探索性 Trace 也不能宣称齐全。
2. **部分运行日志只存在于历史工具输出或远端引用**：对话仍保留，但不能当作本地完整原始日志。早期 C1024 的上述边界以及后续实存日志见第 1 节。
3. **离线复跑输入不齐全**：模型权重没有归档；历史 `/tmp/ShareGPT_V3_unfiltered_cleaned_split.json` 已不存在。这一版本 `bench_serving.py` 的 `--dataset-name random` 仍会借助 ShareGPT 采样数据，复跑需重新取得模型和公开数据，且历史数据原件哈希未记录。其余临时文件缺失见[原清单](metadata/artifacts/missing_artifacts.json)。
4. **代码、测试和文档可按版本恢复**：原始工作树路径映射在 [offline_file_index.json.gz](metadata/workspace/offline_file_index.json.gz)，恢复命令见[仓库 README](README.md)。本页的实测 ON/OFF 源码从已归档 Git bundle 精确提取，字节与原 Git 对象一致；两个完整 Python 文件已做 AST 检查。

独立核对记录：[产物覆盖与缺口](metadata/claims/artifact_coverage.json) · [稳定性覆盖](metadata/claims/stability_coverage.json) · [Page split / 对话覆盖](metadata/claims/page_split_coverage.json)。

## 可据现有证据使用的成果表述

- **高并发稳定性：** 在 PRO5000 上以 SGLang 部署 DSV4-FLASH，修复 SWA KV Cache 耗尽、未实现 CPU Offload 导致的 PD Decode 崩溃及 Page 容量估算偏差；补齐 KV Pool 能力判断、重算与请求级降级回收，在 C640/C1024 容量压力场景验证服务存活与健康恢复。
- **KV Page 切分：** 针对 SM120 Decode 每层全量 Page 切分的无效搬运，以 Page Mask 仅切分本轮访问的 Page；本地三轮 C8 E2E A/B 的输出吞吐均值提升 8.61%（配对中位数 4.50%）。批量标记与 uint64 搬运另列为后续社区实现分析。
