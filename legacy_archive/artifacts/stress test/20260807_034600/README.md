# DeepSeek-V4 Flash DSpark Stress Run

## Run Metadata

- Start time (UTC): 2026-08-07 03:46:00
- Git branch: `upstream_0.5.16-pr-33407`
- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Endpoint: `http://10.207.193.139:30002`
- Topology: 8 GPUs, TP=8, Attention DP=8, FusedMoE EP=8
- MoE runner: `flashinfer_mxfp4`
- MoE A2A: `none` (standard dispatcher)
- Speculative decoding: DSpark, static ragged verify, decode attention mode

## Host Baseline

- Checkpoint filesystem: XFS
- Checkpoint directory size: 166,898,658,872 bytes (155.44 GiB)
- Physical RAM: 1,597,669,093,376 bytes (1.45 TiB)
- Available RAM before startup: 1,565,501,026,304 bytes (1.42 TiB)

## Launch Command

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH:-}"
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
export SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_RAGGED_VERIFY_MODE=static

python3 -m sglang.launch_server \
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
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 16 \
  --max-running-requests 16 \
  --speculative-algorithm DSPARK \
  --speculative-attention-mode decode \
  --log-requests \
  --log-requests-level 0 \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 30002
```

## Runtime Files

- `server.log`: complete server startup and request log
- `gpu_dmon.log`: one-second NVIDIA GPU telemetry
- `vmstat.log`: one-second host CPU, memory, swap, and IO telemetry

## Startup Capacity Results

### Attempt 1 (`server.log`)

- Result: startup failed during the internal warmup request at 03:49:02 UTC.
- Cause: the hand-built DSpark draft batch omitted `num_token_non_padded`, which
  DP-attention CUDA graph replay needs to derive its rank-local token count.
- No CUDA OOM occurred; all GPU memory was released after process cleanup.
- Target weights: 26.87 GB/GPU used, 43.66 GB/GPU available after loading,
  approximately 21-22 seconds.
- DSpark draft weights: 2.05 GB/GPU used, 41.61 GB/GPU available after loading,
  approximately 12-13 seconds.
- DSV4 calculation: `bytes_per_full_token=9259.90`,
  `available_bytes=13.39 GB`, `c128_state_fixed=0.09 GB`,
  `full_token=1542400`.
- DSV4 pools: `full=1542400`, `swa=154112`, `c4=385600`, `c128=12050`,
  `c4_state=19264`, `c128_state=0`.
- Memory pool: 28.26 GB/GPU available after the target pool and 28.00 GB/GPU
  after the draft pool.
- Target CUDA graph: 27.78 GB/GPU before capture; 0.21 GB/GPU used;
  27.58 GB/GPU after capture.
- Draft CUDA graph: 27.58 GB/GPU before capture; 0.08-0.15 GB/GPU used;
  27.43-27.50 GB/GPU after capture.
- Final runtime capacity: `max_total_num_tokens=1542400`,
  `chunked_prefill_size=256`, `max_prefill_tokens=8192`,
  `max_running_requests=2`, `context_len=131072`,
  `available_gpu_mem=27.50 GB`.
- Theoretical concurrency: 47.07x at 32,768 tokens/request; 150.625x at
  10,240 tokens/request. The runtime request cap remains 2 per DP rank.

### Attempt 2 (`server-attempt-2.log`)

- Start time (UTC): 2026-08-07 03:56:48
- Ready time (UTC): 2026-08-07 03:58:09
- Status: ready; `/health` returned HTTP 200 and the built-in eight-request
  DSpark warmup completed successfully.
- Fix applied: hand-built DSpark DP-MoE draft batches now carry their real
  pre-padding token count for DP-attention CUDA graph replay.
- Target weights: 26.87 GB/GPU used, 43.66 GB/GPU available after loading,
  20.28-21.70 seconds.
- DSpark draft weights: 2.05 GB/GPU used, 41.61 GB/GPU available after loading,
  12.37-13.14 seconds.
- DSV4 calculation: `bytes_per_full_token=9259.90`,
  `available_bytes=13.39 GB`, `c128_state_fixed=0.09 GB`,
  `full_token=1542400`.
- DSV4 pools: `full=1542400`, `swa=154112`, `c4=385600`, `c128=12050`,
  `c4_state=19264`, `c128_state=0`.
- Target CUDA graph: 27.81 GB/GPU before capture; 0.20 GB/GPU used;
  27.61 GB/GPU after capture.
- Draft CUDA graph: 27.61 GB/GPU before capture; 0.08 GB/GPU used;
  27.53 GB/GPU after capture.
- Final runtime capacity: `max_total_num_tokens=1542400`,
  `chunked_prefill_size=256`, `max_prefill_tokens=8192`,
  `max_running_requests=2`, `context_len=131072`,
  `available_gpu_mem=27.53 GB`.
- Theoretical concurrency: 47.07x at 32,768 tokens/request; 150.625x at
  10,240 tokens/request. The runtime request cap is 2 per DP rank, 16 global.
- Post-warmup resident GPU memory: approximately 44,646 MiB/GPU, with
  approximately 28,186 MiB/GPU free.
- Telemetry: `gpu-dmon-attempt-2.log` and `vmstat-attempt-2.log` continue to
  sample at one-second intervals while the service is running.
