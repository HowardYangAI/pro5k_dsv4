#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/sgl-workspace/sglang/python"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NO_PROXY="127.0.0.1,localhost,::1,.xiaojukeji.com,.didi.com,10.0.0.0/8,192.168.0.0/16"

exec python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /data/models/DeepSeek-V4-Flash-0731 \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --ep-size 8 \
  --context-length 131072 \
  --mem-fraction-static 0.6 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 8192 \
  --disable-radix-cache \
  --moe-runner-backend flashinfer_mxfp4 \
  --moe-a2a-backend none \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --max-running-requests 128 \
  --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 6 \
  --speculative-attention-mode decode \
  --host 0.0.0.0 \
  --port 30012
