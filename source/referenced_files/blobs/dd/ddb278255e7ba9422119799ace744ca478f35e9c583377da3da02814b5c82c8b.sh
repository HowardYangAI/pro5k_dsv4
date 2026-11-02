#!/usr/bin/env bash
#
# fetch_codegraph_index.sh — 一键从内网网盘下载 CodeGraph 索引 db
#
# 用法:
#   bash tools/codegraph/fetch_codegraph_index.sh          # 下载到默认 .codegraph/codegraph.db
#   bash tools/codegraph/fetch_codegraph_index.sh -f        # 强制覆盖已有 db
#   INDEX_URL=https://... bash tools/codegraph/fetch_codegraph_index.sh
#
# 依赖: curl, sha256sum (校验完整性,可选)
# 网络: 内网 http 直链,无需代理

set -euo pipefail

# ===== 索引 db 下载地址(内网网盘直链) =====
# 如需更新地址,改这里或用环境变量 INDEX_URL 覆盖。
INDEX_URL="${INDEX_URL:-https://s3-nmgpu-inter.didistatic.com/ofs-ifx-biz/codegraph/sglang/0.5.16/codegraph.db?response-content-disposition=&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=[REDACTED_CREDENTIAL]&X-Amz-Date=20260813T091353Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=[REDACTED_CREDENTIAL]"

# ===== 路径 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DB_DIR="$REPO_ROOT/.codegraph"
DB_PATH="$DB_DIR/codegraph.db"

# ===== 选项 =====
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    -f|--force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

echo "CodeGraph 索引 db 下载"
echo "  源:   $INDEX_URL"
echo "  目标: $DB_PATH"
echo ""

# ===== 前置检查 =====
if ! command -v curl >/dev/null 2>&1; then
  echo "✗ 缺少 curl,请先安装" >&2
  exit 1
fi

if [ -f "$DB_PATH" ] && [ "$FORCE" -ne 1 ]; then
  echo "⚠ 已存在 $DB_PATH"
  echo "  覆盖请加 -f,或先删除: rm $DB_PATH"
  echo "  当前 db 大小: $(du -h "$DB_PATH" | awk '{print $1}')"
  exit 0
fi

# ===== 下载 =====
mkdir -p "$DB_DIR"
TMP_PATH="$DB_PATH.download"

echo "下载中(约 420MB,内网应较快)..."
if ! curl -fSL --retry 3 --retry-delay 2 --connect-timeout 15 --max-time 600 \
     -o "$TMP_PATH" "$INDEX_URL"; then
  echo "✗ 下载失败,请检查 INDEX_URL 是否可达:" >&2
  echo "    $INDEX_URL" >&2
  echo "  可手动测试: curl -sI --max-time 10 '$INDEX_URL'" >&2
  rm -f "$TMP_PATH"
  exit 1
fi

# 校验:下载大小应 > 100MB(db 实际约 420MB)
SIZE_BYTES=$(stat -c%s "$TMP_PATH" 2>/dev/null || stat -f%z "$TMP_PATH" 2>/dev/null || echo 0)
SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
if [ "$SIZE_MB" -lt 100 ]; then
  echo "✗ 下载文件仅 ${SIZE_MB}MB,预期约 420MB,可能不完整或地址错误" >&2
  echo "  临时文件保留: $TMP_PATH (可查看内容排查)" >&2
  exit 1
fi

# 原子替换
mv -f "$TMP_PATH" "$DB_PATH"
echo "✅ 下载完成: $DB_PATH (${SIZE_MB}MB)"

# ===== 可选:codegraph 校验索引 =====
if command -v codegraph >/dev/null 2>&1; then
  echo ""
  echo "校验索引..."
  if codegraph status >/dev/null 2>&1; then
    echo "✅ 索引可用:"
    codegraph status | head -8
  else
    echo "⚠ codegraph status 失败,db 可能与当前 codegraph 版本不匹配"
    echo "  可重建: cd $REPO_ROOT && codegraph index"
  fi
else
  echo ""
  echo "ℹ codegraph 未安装,仅下载了 db。安装 codegraph 后即可使用:"
  echo "  见 tools/codegraph/README.md 第二节"
fi
