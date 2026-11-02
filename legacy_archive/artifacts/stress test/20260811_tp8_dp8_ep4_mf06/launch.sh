#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
export SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_RAGGED_VERIFY_MODE=static

exec python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /data/models/DeepSeek-V4-Flash-0731 \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --ep-size 4 \
  --context-length 131072 \
  --mem-fraction-static 0.6 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 8192 \
  --disable-radix-cache \
  --moe-runner-backend flashinfer_mxfp4 \
  --moe-a2a-backend none \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 128 \
  --max-running-requests 512 \
  --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 6 \
  --speculative-attention-mode decode \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --log-requests \
  --log-requests-level 0 \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 30002
