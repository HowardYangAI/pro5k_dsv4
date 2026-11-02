#!/bin/bash
# ============================================================================
# LLM Benchmark Platform — Environment Setup
# ============================================================================
# Run this script once on a new machine to prepare all dependencies.
# Mounting model directories requires sudo, so run with:
#   sudo bash setup_env.sh
# Or if you only need pip/dataset setup (no mount):
#   bash setup_env.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

echo "========================================="
echo " LLM Benchmark Platform — Setup"
echo "========================================="
echo ""

# ── 1. Python dependencies ─────────────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "       Done."
echo ""

# ── 2. Mount model directories ─────────────────────────────────────────────
MOUNT_SCRIPT="/home/luban/jianshu/ofs/release/current/script/ofs_mount.sh"

if [[ ! -f "$MOUNT_SCRIPT" ]]; then
    echo "[2/4] WARNING: Mount script not found at $MOUNT_SCRIPT"
    echo "       Skipping model directory mounts."
    echo "       If you need model directories, mount them manually or edit config.yaml."
else
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "[2/4] Mounting model directories requires root privileges."
        echo "       Please re-run with: sudo bash $0"
        echo "       Skipping mounts for now."
    else
        echo "[2/4] Mounting model directories..."

        echo "       Mounting ofs-llab-cold -> /nfs/ofs-llab-cold ..."
        bash "$MOUNT_SCRIPT" ofs-llab-cold /nfs/ofs-llab-cold a7ce0339d04d4b008040e3a28e3fc44e nmgllab

        echo "       Mounting luban-data -> /nfs/ofs-luban-data ..."
        bash "$MOUNT_SCRIPT" luban-data /nfs/ofs-luban-data 781e2f270c924746a31aff8a0ed76e97 nmgpu

        echo "       Done."
    fi
fi
echo ""

# ── 3. Download datasets ───────────────────────────────────────────────────
echo "[3/4] Downloading datasets..."

# Read dataset_dir from config.yaml (simple parse — no PyYAML needed)
DATASET_DIR=$(grep '^dataset_dir:' "$CONFIG_FILE" | awk '{print $2}')
if [[ -z "$DATASET_DIR" ]]; then
    DATASET_DIR="/home/luban"
fi
mkdir -p "$DATASET_DIR"

# Download ShareGPT if not present
SHAREGPT_FILE="$DATASET_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ -f "$SHAREGPT_FILE" ]]; then
    echo "       ShareGPT dataset already exists at $SHAREGPT_FILE — skipping."
else
    echo "       Downloading ShareGPT V3 dataset..."
    wget -O "$SHAREGPT_FILE" \
        "https://s3-hnapuhdd-inter.didistatic.com/ifx-dzy-dev/llm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    echo "       Done."
fi
echo ""

# ── 4. Verify ──────────────────────────────────────────────────────────────
echo "[4/4] Verifying setup..."

OK=true
for d in /nfs/ofs-llab-cold /nfs/ofs-luban-data; do
    if [[ -d "$d" ]]; then
        echo "       $d — OK"
    else
        echo "       $d — NOT FOUND (models may not be available)"
        OK=false
    fi
done

if [[ -f "$SHAREGPT_FILE" ]]; then
    echo "       $SHAREGPT_FILE — OK"
else
    echo "       $SHAREGPT_FILE — NOT FOUND"
    OK=false
fi

echo ""
if $OK; then
    echo "========================================="
    echo " Setup complete! Start the server with:"
    echo "   python3 -m uvicorn main:app --host 0.0.0.0 --port 8080"
    echo "========================================="
else
    echo "========================================="
    echo " Setup partially complete — some items missing."
    echo " Check warnings above and edit config.yaml if needed."
    echo "========================================="
fi