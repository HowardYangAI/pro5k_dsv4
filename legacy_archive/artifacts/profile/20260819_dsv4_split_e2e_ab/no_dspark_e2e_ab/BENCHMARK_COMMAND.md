# No-DSpark Split ON/OFF benchmark command

服务端统一使用 TP8 / Attention DP8 / EP8、Decode CUDA Graph，未添加任何 speculative/DSpark 参数。Split ON 为 `3cf4ce7db`，Split OFF 为其父提交 `f5868c99b`。

```bash
python3 bench_serving.py \
  --backend sglang-oai-chat \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --extra-request-body '{"stream_options":{"include_usage":true}}' \
  --seed <1787128001|1787128002|1787128003> \
  --host 127.0.0.1 --port 30013 \
  --model /data/models/DeepSeek-V4-Flash-0731 \
  --model-name DeepSeek-V4-Flash-0731 \
  --max-concurrency 8 \
  --num-prompts 40 \
  --request-rate inf \
  --warmup-requests 0
```

服务器自身已执行启动 warmup 和 CUDA Graph capture；三轮正式 A/B 均在同一服务实例内顺序运行。
