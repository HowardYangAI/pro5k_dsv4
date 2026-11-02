# 归档时源码状态

## 主 SGLang worktree

- 路径：`/sgl-workspace/sglang`
- 分支：`upstream_0.5.16-pr-33407`
- HEAD：`3cf4ce7dbe461c9b86447851ca4acef2caab1f37`
- HEAD 内容：回移 PR #32320，SM120 FlashMLA 仅切分实际触达的 SWA pages。
- 未提交修改：`python/sglang/srt/speculative/dspark_components/dspark_draft.py`，补齐 DSpark + EP 所需的 `num_token_non_padded` 元数据。

## three-path worktree

- 分支：`fix/dsv4-pd-retraction-three-path`
- HEAD：`a6b8fad83cc398017de43f9c4e3c51ee80653331`
- 工作区干净，修复已提交并包含在 SGLang bundle 中。

## KV-only worktree

- 分支：`fix/dsv4-pd-retraction-kv-only`
- HEAD：`48886803b94596884d737132a381bd5124ea1cfd`
- 有 24 个 staged 文件，包含 CPU-copy capability、PD rebootstrap/graceful abort、分页容量检查、请求状态清理以及相应测试。
- 完整 staged 差异保存在 `source/kv-only-staged.patch.gz`。

## 在线压测工具

- 仓库：`/sgl-workspace/llm-benchmark-online`
- 分支：`master`
- HEAD：`f45eef0c24f8f4dcb1f75250e04f0b60e9f102cd`
- 工作区干净。

## 其他源码与 CR

- SGLang 存在 `stash@{0}: old experiments before kv-only CR`，已额外导出到 `source/old-experiments-stash.patch.gz`。
- CUPTI kernel monitoring：HEAD `d56ff16`，完整仓库保存在 `source/cupti-kernel-monitoring.bundle`。
- DCL CR：`https://kunpeng.xiaojukeji.com/view/revision/6565881`。
- `/sgl-workspace/DeepEP` 为本机预先存在的第三方依赖/构建目录，不是 Git worktree，且 2026-08-01 后未发现源码修改，因此未复制约 100 MiB 的 build 产物。
