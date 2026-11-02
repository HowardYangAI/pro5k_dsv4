# DeepSeek-V4-Flash EP8 Decode Throughput Run

- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Topology: TP8, Attention DP8, EP8
- Mode: decode-only, fake KV transfer
- MoE runner: `flashinfer_mxfp4`; MoE A2A: `none`
- Speculative decoding: DSpark (`gamma=5`, verify tokens=6)
- Decode CUDA graph maximum batch size: 128
- Global maximum running requests: 512 (64 per DP rank)
- Endpoint: `http://10.207.193.139:30002`

Files:

- `server.log`: service startup and request logs
- `gpu-dmon.log`: per-GPU utilization and memory samples
- `vmstat.log`: host CPU, memory, and I/O samples
