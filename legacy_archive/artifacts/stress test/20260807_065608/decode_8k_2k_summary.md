# Decode 性能测试

## 测试信息

| 项目 | 内容 | 项目 | 内容 |
|---|---|---|---|
| 机器 | 8 x NVIDIA RTX PRO 5000 72GB Blackwell | 模型 | DeepSeek-V4-Flash-0731 |
| 服务角色 | Decode-only，fake KV transfer | 并行配置 | TP8 / Attention DP8 / EP1 |
| MoE backend | flashinfer_mxfp4 | 推测解码 | DSpark，gamma=5，verify=6 |
| 输入/输出 | 8192 / 2048 tokens | 请求率 | INF |
| KV 传输 | Fake，不含真实 P-D KV 传输开销 | Cache hit rate | 0 |
| Checkpoint | XFS，目录大小 155.44 GiB | 主机内存 | 物理 1.45 TiB，启动前可用 1.42 TiB |

## 服务启动命令

```bash
python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /data/models/DeepSeek-V4-Flash-0731 \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --tp-size 8 --dp-size 8 \
  --enable-dp-attention --enable-dp-lm-head \
  --context-length 131072 --mem-fraction-static 0.6 \
  --chunked-prefill-size 2048 --max-prefill-tokens 8192 \
  --disable-radix-cache \
  --moe-runner-backend flashinfer_mxfp4 --moe-a2a-backend none \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 16 \
  --max-running-requests 16 \
  --speculative-algorithm DSPARK --speculative-attention-mode decode \
  --disaggregation-mode decode --disaggregation-transfer-backend fake \
  --log-requests --log-requests-level 0 \
  --enable-metrics --enable-cache-report \
  --host 0.0.0.0 --port 30002
```

## 显存与容量

```text
[Host] Checkpoint filesystem=XFS, checkpoint directory size=155.44 GiB,
       physical RAM=1.45 TiB, available RAM before startup=1.42 TiB
[DP0 TP0] Load target weight end: mem usage=26.87 GB, available GPU memory=43.66 GB
[DP0 TP0] Load DSpark draft weight end: mem usage=2.05 GB, available GPU memory=41.61 GB
[DP0 TP0] DSV4 memory calculation: bytes_per_full_token=9259.90, available_bytes=13.39 GB,
          c128_state_fixed=0.17 GB, full_token=1,533,440
[DP0 TP0] DSV4 pool sizes: full=1,533,440, swa=153,344, c4=383,360, c128=11,980,
          c4_state=19,168, c128_state=0
[DP0 TP0] Target verify CUDA graph: mem usage=0.22 GB, available GPU memory=27.63 GB
[DP0 TP0] Draft verify CUDA graph: mem usage=0.08 GB, available GPU memory=27.56 GB
[DP0 TP0] Runtime: max_total_num_tokens=1,533,440, max_running_requests=2 per DP rank,
          context_len=131,072, available_gpu_mem=27.56 GB
```

## 延迟

| Max request concurrency | Successful requests | Mean TTFT (ms) | Median TTFT (ms) | P95 TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P95 TPOT (ms) | P99 TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 222.69 | 213.58 | 269.62 | 270.12 | 10.73 | 10.05 | 13.23 | 16.59 |
| 4 | 40 | 301.75 | 289.82 | 346.47 | 349.24 | 12.39 | 10.82 | 22.02 | 24.42 |
| 8 | 80 | 327.23 | 317.93 | 382.17 | 489.11 | 13.92 | 11.87 | 26.54 | 30.22 |
| 16 | 160 | 3,422.55 | 331.60 | 20,408.64 | 26,287.70 | 13.99 | 12.54 | 21.48 | 37.18 |
| 32 | 320 | 28,391.39 | 24,321.82 | 68,212.55 | 76,469.34 | 13.80 | 12.15 | 22.50 | 38.35 |
| 64 | 640 | 77,900.45 | 80,176.96 | 114,488.78 | 126,960.01 | 13.12 | 11.37 | 22.43 | 37.75 |
| 96 | 960 | 122,796.97 | 124,940.69 | 174,875.89 | 218,200.61 | 12.39 | 10.91 | 19.04 | 41.82 |

## 吞吐与 DSpark

| Max request concurrency | Request throughput (req/s) | Input throughput (tok/s) | Output throughput (tok/s) | 实际并发 | DSpark accept rate | Accept length | Total input tokens | Total generated tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 368.92 | 92.23 | 1.00 | 81.56% | 5.72 | 163,840 | 40,960 |
| 4 | 0.15 | 1,235.40 | 308.85 | 3.87 | 71.50% | 5.00 | 327,680 | 81,920 |
| 8 | 0.25 | 2,071.88 | 517.97 | 7.29 | 66.49% | 5.00 | 655,360 | 163,840 |
| 16 | 0.47 | 3,859.50 | 964.88 | 15.10 | 71.59% | 4.51 | 1,310,720 | 327,680 |
| 32 | 0.52 | 4,291.93 | 1,072.98 | 29.68 | 72.27% | 4.52 | 2,621,440 | 655,360 |
| 64 | 0.58 | 4,727.60 | 1,181.90 | 60.46 | 76.06% | 4.72 | 5,242,880 | 1,310,720 |
| 96 | 0.59 | 4,829.82 | 1,207.45 | 87.35 | 80.51% | 4.76 | 7,864,320 | 1,966,080 |

## 口径说明

- DSpark accept rate = `sum(spec_num_correct_drafts) / sum(spec_num_proposed_drafts)`，按每轮正式请求聚合，不含 warmup。
- Accept length 沿用 benchmark 输出；服务日志中的实际 prompt 长度通常为 8196（包含模板/特殊 token），总输入 token 沿用 benchmark 的 8192 口径。
- Fake KV transfer 只用于测试 Decode 调度、kernel 和吞吐，不包含真实 KV 传输，也不用于判断生成正确性。
