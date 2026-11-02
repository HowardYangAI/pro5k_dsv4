# pro5k_dsv4

本机 Codex 对话、DeepSeek V4 / SGLang 实验、测试、文档和源码状态归档。

对话事件快照截至 **2026-11-02 17:55:04（北京时间，UTC+8）**。后续归档过程不再写回该快照，避免递归收集。源文件保持原样，仓库内文本已检查并脱敏；日志、CSV 和原始事件按需使用 gzip 压缩。

| 内容 | 入口 |
| --- | --- |
| 所有历史对话，按线程阅读 | [对话索引](conversations/README.md) |
| Codex 原始事件和数据库导出 | [原始事件](conversations/raw/) · [数据库投影](conversations/databases/) |
| PD、SWA、CR6647271、accuracy 等测试报告与日志 | [实验文档](workspace/document/) |
| DeepSeek V4 流程图及源文件 | [图文资料](workspace/graph/) |
| 早期压测、Profile、NSYS/Torch 分析和归档会话 | [历史归档](legacy_archive/README.md) |
| 工作树版本、定制代码、补丁与全部测试/文档的离线副本 | [源码状态](source/) |
| `/tmp` 留存的测试结果、补丁和研究资料 | [附加产物](artifacts/tmp/) |
| 覆盖范围、源路径、大小、SHA-256 和已缺失文件 | [元数据](metadata/) |
| 导出和验证脚本 | [工具](tools/) |

## 实验阅读入口

- [PD Decode retraction / rebootstrap 修复报告](workspace/document/20260827_pd_rebootstrap_e2e/PD_DECODE_RETRACTION_FIX_REPORT.md)
- [PD rebootstrap 端到端测试](workspace/document/20260827_pd_rebootstrap_e2e/README.md)
- [SWA eviction 端到端测试](workspace/document/20260829_swa_eviction_e2e/README.md)
- [Full KV retraction 复现](workspace/document/20260830_full_kv_retraction_repro/README.md)
- [Full KV abort-only 测试](workspace/document/20260830_full_kv_abort_only_e2e/README.md)
- [官方 PR #36185 对照](workspace/document/20260830_official_pr36185_e2e/README.md)
- [SWA common force 修复描述](workspace/document/20260830_swa_common_force_e2e/CR_DESCRIPTION.md)
- [早期 Profile 总索引](legacy_archive/artifacts/profile/PROFILE_INDEX.md)

## 保存与恢复

`source/` 保存 Git 版本元数据、定制提交差异、未提交补丁及文件快照。各工作树的测试、文档、benchmark、示例和脚本按内容去重，原始路径与内容对象的对应关系见 [offline_file_index.json.gz](metadata/workspace/offline_file_index.json.gz)。重复内容只存一份，路径映射全部保留。

列出可以恢复的工作树，或将某个工作树的测试、文档与脚本恢复到新目录：

```bash
python3 tools/restore_offline_files.py --list
python3 tools/restore_offline_files.py --repo sglang-dsv4-fixed --dest /tmp/sglang-dsv4-docs-tests
```

大文本文件在原扩展名后增加 `.gz`，可用 `gzip -dc 文件.gz` 阅读。早期 Git bundle 的分片按文件名顺序连接后使用，见 [旧归档恢复说明](legacy_archive/source/RESTORE.md)。源文件与归档文件在压缩、脱敏后可能具有不同的 SHA-256，逐项转换记录保存在各分类 manifest 中。

本次没有归档认证配置、私钥、模型权重、安装包、容器镜像、编译缓存和重复的公共参考仓库。对话中提及但本机已不存在的产物见 [缺失文件清单](metadata/artifacts/missing_artifacts.json)；仅有数据库元数据、原始 rollout 不存在的线程已在对话索引注明。

运行完整校验：

```bash
python3 tools/verify_archive.py
```

校验覆盖 gzip 完整性、JSON/JSONL、文件大小及归档清单哈希。与凭据相关的字段可能已被替换为 `[REDACTED_…]`，使用脚本复现实验时应从环境中重新提供认证。
