# 同步状态与最后补齐材料

2026-11-02：本机现存相关资料未发现新遗漏；进一步完整核对可访问的历史 Decode 节点实验目录后，又补入 31 份内容不同的完整日志。原本的早期快照继续保留，新增版本按远端节点单独存放。

## 本轮补齐

- 31 份日志原件共 553,106,070 字节，涵盖 8/27 PD rebootstrap、8/30 KV/SWA 修复及 9/1–9/5 性能和精度调查。逐份下载原件与远端 SHA-256 一致，压缩后哈希见[清单](metadata/recovery/remote_document_manifest.json)。
- [早期 CPU copy capability 回归测试](source/claims/stability_history/README.md)从历史工具输出还原为可读源码；原始对话已有其内容，本轮另存便于审查，已做 AST 检查，没有重新运行历史测试。
- 之前已找回的三份远端日志继续保留在[原实验目录](RELOCATION_AUDIT.md)。本轮新增版本见下表。

| 原远端实验路径（相对 document） | 原件字节 | 完整日志 |
| --- | ---: | --- |
| `20260827_pd_rebootstrap_e2e/logs/decode.log` | 8,600,630 | [gzip](workspace/remote_document/ser237/20260827_pd_rebootstrap_e2e/logs/decode.log.gz) |
| `20260827_pd_rebootstrap_e2e/logs/decode_mf05.log` | 5,128,010 | [gzip](workspace/remote_document/ser237/20260827_pd_rebootstrap_e2e/logs/decode_mf05.log.gz) |
| `20260827_pd_rebootstrap_e2e/logs/decode_mf055_trace.log` | 6,389,939 | [gzip](workspace/remote_document/ser237/20260827_pd_rebootstrap_e2e/logs/decode_mf055_trace.log.gz) |
| `20260830_full_kv_abort_only_e2e/decode.log` | 34,754,625 | [gzip](workspace/remote_document/ser237/20260830_full_kv_abort_only_e2e/decode.log.gz) |
| `20260830_full_kv_abort_only_pagealigned_c1024/decode.log` | 16,671,496 | [gzip](workspace/remote_document/ser237/20260830_full_kv_abort_only_pagealigned_c1024/decode.log.gz) |
| `20260830_full_kv_abort_only_pagealigned_fake_c1024/decode.log` | 11,107,233 | [gzip](workspace/remote_document/ser237/20260830_full_kv_abort_only_pagealigned_fake_c1024/decode.log.gz) |
| `20260830_official_pr36185_e2e/baseline_decode.log` | 2,698 | [gzip](workspace/remote_document/ser237/20260830_official_pr36185_e2e/baseline_decode.log.gz) |
| `20260830_swa_common_force_e2e/decode.log` | 15,271,745 | [gzip](workspace/remote_document/ser237/20260830_swa_common_force_e2e/decode.log.gz) |
| `20260830_swa_common_force_e2e/decode_fake.log` | 17,196,604 | [gzip](workspace/remote_document/ser237/20260830_swa_common_force_e2e/decode_fake.log.gz) |
| `20260901_cr6647271_perf_accuracy/baseline_without_fix/decode.log` | 130,394 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/baseline_without_fix/decode.log.gz) |
| `20260901_cr6647271_perf_accuracy/baseline_without_fix/decode_restart_20260902.log` | 108,132,484 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/baseline_without_fix/decode_restart_20260902.log.gz) |
| `20260901_cr6647271_perf_accuracy/baseline_without_fix/decode_runnable.log` | 127,774 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/baseline_without_fix/decode_runnable.log.gz) |
| `20260901_cr6647271_perf_accuracy/fixed/decode.log` | 106,665,391 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/fixed/decode.log.gz) |
| `20260901_cr6647271_perf_accuracy/fixed/decode_startup_misconfigured.log` | 101,345 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/fixed/decode_startup_misconfigured.log.gz) |
| `20260901_cr6647271_perf_accuracy/fixed/decode_startup_unpinned_defaults.log` | 81,851 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/fixed/decode_startup_unpinned_defaults.log.gz) |
| `20260901_cr6647271_perf_accuracy/fixed_with_parsers/decode.log` | 32,733,135 | [gzip](workspace/remote_document/ser237/20260901_cr6647271_perf_accuracy/fixed_with_parsers/decode.log.gz) |
| `20260902_accuracy_investigation/baseline_no_ep_no_dspark_no_fix/decode.log` | 40,450,361 | [gzip](workspace/remote_document/ser237/20260902_accuracy_investigation/baseline_no_ep_no_dspark_no_fix/decode.log.gz) |
| `20260902_cr6647271_perf_accuracy/baseline_no_ep_no_fix/decode.log` | 46,555 | [gzip](workspace/remote_document/ser237/20260902_cr6647271_perf_accuracy/baseline_no_ep_no_fix/decode.log.gz) |
| `20260903_accuracy_investigation/baseline_dspark_ep4_compat_with_parsers/decode.log` | 764,244 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/baseline_dspark_ep4_compat_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/baseline_dspark_ep4_with_parsers/decode.log` | 125,287 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/baseline_dspark_ep4_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/baseline_dspark_no_ep_with_parsers/decode.log` | 672,336 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/baseline_dspark_no_ep_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/baseline_model_defaults/decode.log` | 1,274,642 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/baseline_model_defaults/decode.log.gz) |
| `20260903_accuracy_investigation/baseline_model_defaults_with_parsers/decode.log` | 2,076,854 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/baseline_model_defaults_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/fixed_dspark_ep4_forced_temp0_with_parsers/decode.log` | 824,176 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/fixed_dspark_ep4_forced_temp0_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/fixed_dspark_ep4_with_parsers/decode.log` | 745,605 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/fixed_dspark_ep4_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/fixed_dspark_no_ep_with_parsers/decode.log` | 783,356 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/fixed_dspark_no_ep_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/fixed_no_ep_no_dspark_with_parsers/decode.log` | 1,525,111 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/fixed_no_ep_no_dspark_with_parsers/decode.log.gz) |
| `20260903_accuracy_investigation/fixed_original_config_with_parsers/decode.log` | 37,957,624 | [gzip](workspace/remote_document/ser237/20260903_accuracy_investigation/fixed_original_config_with_parsers/decode.log.gz) |
| `20260905_accuracy_investigation/fixed_original_chunk4096_with_parsers/decode.log` | 52,195,396 | [gzip](workspace/remote_document/ser237/20260905_accuracy_investigation/fixed_original_chunk4096_with_parsers/decode.log.gz) |
| `20260905_accuracy_investigation/fixed_original_chunk8192_with_parsers/decode.log` | 50,516,671 | [gzip](workspace/remote_document/ser237/20260905_accuracy_investigation/fixed_original_chunk8192_with_parsers/decode.log.gz) |
| `20260905_accuracy_investigation/repro_6647_exact_defaults/decode.log` | 52,498 | [gzip](workspace/remote_document/ser237/20260905_accuracy_investigation/repro_6647_exact_defaults/decode.log.gz) |

## 已核对的覆盖范围

- 本机 42,522 个映射源路径均存在且大小匹配；6,696 个小文件重新校验 SHA-256，没有差异；95 个大文件本轮未重复哈希。14 个源码工作树版本和状态均与快照一致，测试、文档、脚本未发现新路径。
- 非本任务的 18 个历史会话源和 41 个附件与归档一致，索引和数据库未发现漏掉的历史研究线程。新增会话均是本次归档、追查和子 agent 活动；研究对话快照仍截至北京时间 2026-11-02 17:55:04，不递归追加本次归档操作对话。
- ser237 的 `workspace/document` 共 39 个常规文件已全部逐项比对：7 个内容已有归档，31 份不同日志已补齐，1 个 PID 仅留元数据。没有修改远端文件或测试服务。

## 尚未取回或需要另行准备

- 48 份全量多 Rank Trace、约 127 MiB 的 TP0 全量包，以及旧 ser171 上未独立留存的运行日志，仍需原节点磁盘、容器或当时下载副本。当前没有该原容器的可用访问入口；尚无删除的确证。见[搬运追查](RELOCATION_AUDIT.md)。
- 历史 ShareGPT 数据原件及少量临时输出未找到，详见[原临时文件清单](metadata/artifacts/missing_artifacts.json)。公开数据可重新取得，但不能当作已找回历史同一份输入。ser237 上同名 `decode_health.out` 现为零字节，不能恢复当时健康响应正文。
- 模型权重、凭据、安装包和容器镜像不在本次归档范围。

机器核对记录：[本机](metadata/recovery/remaining_local_sync.json) · [对话与附件](metadata/recovery/remaining_conversation_sync.json) · [远端分类](metadata/recovery/remaining_remote_sync.json) · [远端完整目录清单](metadata/recovery/remote_document_inventory.json)。
