# 会话来源与导出方法

当前 Codex JSONL 在 2026-11-02 17:55:04.617618（Asia/Shanghai）建立逐文件字节快照。后续归档工具调用不继续追加到会话归档中。本次同步主线程及当时已存在的子线程均保留。

- `audit.json`：当前本机来源、只读数据库 schema/计数、任务主题、会话引用路径频次和附件清单。
- `byte_snapshot.json`：各源 JSONL 的字节上限及 mtime，供复核活动会话截止点。
- `source_coverage.json`：当前 Codex 与工作区旧会话的覆盖关系、唯一线程、重复副本及目标链接。
- `../../conversations/README.md`：全部历史的统一阅读入口。
- `../../conversations/manifest.json`：原始 JSONL 导出哈希、过滤信息与脱敏统计。
- `../../conversations/databases/manifest.json`：每个数据库只读事务时间、schema、源行数与实际导出行数。

主题主要包括 DeepSeek V4 / DSV4、SGLang 回归测试、split_paged 与 CUDA kernel、CP / MLA / DCP、prefill / decode、SWA eviction / KV cache、MoE / MegaMoE、基准性能及 GLM 权重。

原始 JSONL 逐行解码，应用共享脱敏器，再编码并 gzip。Markdown 仅展示 user / assistant 消息；工具调用、输出、压缩上下文等保留在原始事件中。数据库通过只读事务导出相关表；初始截止时间之后创建的新行不导出，既有行可能反映导出事务时的后续更新。若源只读目录无法打开 SQLite 的 WAL 共享内存，会在私有 /tmp 目录复制稳定 DB / WAL / SHM 后进行只读事务，完成后删除临时副本。

工作区旧归档的五个 rollout 与 primary-session 副本由工作区导出负责保存原始文件，会话导出仅补充 Markdown 与统一索引。primary-session 与其中一个 rollout 是相同字节副本。每个源分段均保留；不能仅按 threads.rollout_path 复制，因为恢复的线程可能有多个历史分段。

不包含认证文件、配置、安装标识、运行日志数据库、队列、shell 快照、插件缓存，也不包含本次任务专用的 `.github-auth` 目录。

复用导出工具（需仍可访问源路径）：

```bash
python tools/export_conversations.py \
  --dest conversations \
  --snapshot metadata/conversations/byte_snapshot.json \
  --audit metadata/conversations/audit.json \
  --common tools/archive_common.py \
  --source-root /root/.codex \
  --legacy-root /sgl-workspace/deepseek-v4-flash/session/all-sessions
```
