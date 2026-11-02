# Benchmark 命令

Split ON/OFF 均运行以下命令三次，仅替换 seed 为 `1787127001`、`1787127002`、`1787127003`：

```bash
python3 bench_serving.py \
  --backend sglang-oai-chat \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --extra-request-body '{"stream_options":{"include_usage":true}}' \
  --seed 1787127001 \
  --host 127.0.0.1 --port 30012 \
  --model /data/models/DeepSeek-V4-Flash-0731 \
  --model-name DeepSeek-V4-Flash-0731 \
  --max-concurrency 8 \
  --num-prompts 40 \
  --request-rate inf \
  --warmup-requests 0
```

两组服务均使用 TP8 / Attention DP8 / EP8、DSpark 6 draft tokens、Decode CUDA Graph。Split OFF 基于 PR #32320 的父提交 `f5868c99b`，并补回与 ON 相同的 DSpark+EP batch 元数据修复。
