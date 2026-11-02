# 搬运追查与远端日志恢复

2026-11-02 进一步搜索其他 Codex 对话的实际工具调用、执行结果及 Git 历史，并定向检查本机容器层、压缩包和原 Decode 节点。

**首轮三份独立日志已从原 Decode 节点找回；随后完整核对该节点实验目录，又补齐 31 份内容不同的原始日志。** 新增完整日志见[同步状态](SYNC_STATUS.md)。48 份全量 Trace 尚未找回，最后已知环境是旧节点 ser171；不能把当前机器没有文件等同于文件已经被删除。

## 已找回的独立日志

源节点：`ml-pro5000-ser237.qgz02`，SSH 别名 `deepseek-v4-flash-237`，历史及本次核实地址 `10.207.193.166:8015`。原路径均位于 `/sgl-workspace/workspace/document/` 下。

| 原相对路径 | 原件大小 | 已归档文件 |
| --- | ---: | --- |
| `20260829_swa_eviction_e2e/decode_fixed.log` | 1,544,364 字节 | [decode_fixed.log.gz](workspace/document/20260829_swa_eviction_e2e/decode_fixed.log.gz) |
| `20260829_swa_eviction_e2e/decode_first_start_num_token_crash.log` | 155,235 字节 | [启动失败日志](workspace/document/20260829_swa_eviction_e2e/decode_first_start_num_token_crash.log.gz) |
| `20260830_full_kv_abort_only_healthfix_c1024/decode.log` | 35,953,944 字节 | [healthfix Decode 日志](workspace/document/20260830_full_kv_abort_only_healthfix_c1024/decode.log.gz) |

合计原件 **37,653,543 字节**，脱敏检查后以确定性 gzip 保存，共 1,237,534 字节；本次三份日志没有触发凭据脱敏。SCP 下载原件逐一与远端 SHA-256 匹配，文件时间、来源与输出哈希见[恢复清单](metadata/recovery/remote_logs_manifest.json)。原远端文件及服务未改动。

历史调用显示：8/29 在远端把第一次启动的 `decode_fixed.log` 改名为 `decode_first_start_num_token_crash.log`，之后服务重新生成 `decode_fixed.log`；8/30 healthfix 日志也始终由 SSH 远端读取。本次核对前未找到它们被复制回本机的执行记录，所以本机缺少这些文件不代表原件丢失。

healthfix 日志保留的是较早版本的失败诊断，包含 Page 预分配断言，不是后续 page-aligned C1024 成功测试。两者应分别引用。`decode_batch_usage.log` 只出现在文档建议的派生命令中，未发现实际创建证据，本次远端也未找到。

## 48 份全量 Trace 的最后已知位置

当前实际节点为 `ml-pro5000-ser187.qgz02`。旧主会话多次实际工具输出确认当时环境为 `ml-pro5000-ser171.qgz02`、IP `10.207.193.139`；8/19 实跑 Gloo 日志仍引用同一 IP。原 Trace 位于旧实验环境的：

```text
/sgl-workspace/workspace/profile/20260819_dsv4_ep8_dspark_c8/
/sgl-workspace/workspace/profile/20260819_dsv4_split_ab/
/sgl-workspace/workspace/profile/20260819_dsv4_split_e2e_ab/
```

8/19 用户两次明确表示机器要回收、准备归还。实际复制到 `workspace/overall` 并进入 Git 的是 CSV、脚本、日志、轻量 Trace、报告和原件清单。48 份全量多 Rank Trace 没有进入这些已记录的备份操作；8/21 新环境出现的 `deepseek-v4-flash` 是对应的便携 Git 归档。

全量 Trace 中的两份 TP0 原件曾在旧机另行打包，实际 `tar` 成功、成员列表和哈希均在会话中：

```text
/sgl-workspace/workspace/profile/20260819_dsv4_ep8_dspark_c8/delivery/perfetto_tp0_bundle.tar.gz
SHA-256: b239e19c7507c09a4185681ab7d37e999d793958491b304fb33a1c2eafbddcee
```

这个包当时约 127 MiB，包含 C8 Prefill/Decode 的 TP0 全量 Trace。当前本机未找到该包；现存的 `perfetto_lite_bundle.tar.gz` 是另一份精简包，不能替代它。

还有一个客户端线索：8/19 07:23 UTC，紧接原包交付之后，用户在 Mac Chrome 的 Perfetto 中报告了 `NotReadableError`。当时 Mac 的下载目录值得按 `perfetto_tp0_bundle.tar.gz` 或两份 TP0 原件名核对，但这条报错本身不能证明完整下载成功或文件仍在。

搜索保存的其他对话后，没有找到这 48 份全量 Trace 或该包后续被搬走、上传或删除的执行记录。这不能排除在 Codex 之外进行过搬运，也不能确认旧机原件已经不存在。

## 已核对范围与后续入口

- 对话：全部归档原始会话中的路径、`cp/mv/scp/rsync/tar` 等实际工具调用及结果，并交叉检查可读线程。
- 本机存储：原路径、此前 `/data` 文件名搜索；本次另查 1060 个 Docker overlay 层的标准路径、496 个浅层归档根及 27 个候选压缩包，未找到目标全量文件或 TP0 包。
- 旧归档 Git：`deepseek-v4-flash` 全部 465 个本地 Git 对象（含不可达对象）没有目标全量 Trace 或 127 MiB 包；大对象均对应源码 bundle、会话和 CSV 等已知材料。
- 原 Decode 节点 ser237：SSH 逐项检查 48 份 Trace 原路径、`profile` 根目录和 TP0 包，均不存在；只定向检查 50 个明确路径，没有遍历整盘。
- 另有一个 9/5 容器使用宿主目录 `/home/odin/yanghaoxiang_i/yanghaoxiang_sglang_0.5.16_patch1_dsv4_flash/sgl-workspace`，该宿主路径在当前命名空间不可见，其内容未知。
- 旧 ser171 的 `8010` 曾对应另一个 `yanghaoxiang_sglang_0515` 容器；没有证据证明它是生成 Trace 的原容器，历史公钥登录也曾失败。当前未找到原 Trace 容器的可用 SSH 入口。历史上用户曾通过 `op-sven-opsec01.gz01` 上的 `dssh 10.207.193.139` 登录宿主机；若仍保留原磁盘/容器，应从该旧节点核对上述路径或 TP0 包。

逐项证据与边界：[Trace 搬运调用审计](metadata/recovery/trace_moves_audit.json) · [日志远端定位](metadata/recovery/log_moves_audit.json) · [本机容器与包核对](metadata/recovery/local_backup_audit.json) · [ser237 Trace 定向核查](metadata/recovery/ser237_trace_probe.json) · [旧节点连接记录](metadata/recovery/ser171_connection_audit.json) · [48 份原 Trace 清单](metadata/claims/missing_full_traces.json)。

本次只新增取回日志和追查说明，没有重新采集或重造历史 Trace，也没有改写原始对话快照。
