# 源码恢复说明

## 恢复 SGLang 已提交历史

先合并拆分的 bundle：

```bash
cat sglang-all-refs.bundle.part-* > /tmp/sglang-all-refs.bundle
git clone /tmp/sglang-all-refs.bundle sglang-restored
cd sglang-restored
git branch -a
```

该 bundle 包含归档时 SGLang 仓库的全部 refs，包括以下关键分支：

- `upstream_0.5.16`
- `upstream_0.5.16-pr-33407`
- `fix/dsv4-pd-retraction-three-path`
- `fix/dsv4-pd-retraction-kv-only`
- `fix/dsv4-pd-retraction-rebootstrap`

## 恢复未提交修改

patch 分为 staged 和 unstaged 两类。恢复到对应基线/分支后执行：

```bash
gzip -dc /path/to/example.patch.gz | git apply --index
```

关键 patch：

- `main-worktree-unstaged.patch.gz`：主 worktree 中 DSpark + EP 的 `ForwardBatch` 元数据补齐。
- `kv-only-staged.patch.gz`：KV-only worktree 中已暂存的高并发 KV retraction/rebootstrap 修复与测试。
- `three-path-*.patch.gz`：three-path worktree 的未提交状态；归档时实际为空，已提交内容位于 bundle。

如果不希望恢复为 staged 状态，可去掉 `--index`。

## 恢复压测工具

```bash
git clone llm-benchmark-online.bundle llm-benchmark-online
```

## 恢复 CUPTI Profile 工具

```bash
git clone cupti-kernel-monitoring.bundle cupti_kernel_monitoring
```

## 查看旧实验 stash

`old-experiments-stash.patch.gz` 是额外导出的 `stash@{0}`。建议先只查看：

```bash
gzip -dc old-experiments-stash.patch.gz | less
```

确认基线正确后再执行 `git apply`。
