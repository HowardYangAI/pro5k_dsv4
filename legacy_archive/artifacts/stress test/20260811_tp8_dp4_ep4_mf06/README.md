# TP8 / Attention DP4 / EP4 / MF 0.6

- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Endpoint: `http://10.207.193.139:30002`
- Mode: Decode-only with fake KV transfer
- Parallelism: TP8, Attention DP4, EP4 (Attention TP2, MoE TP2)
- DSpark: 6 verify tokens (`gamma=5`)
- Global max running requests: 512 (128 per DP rank)
- CUDA graph max decode batch size: 128
- Purpose: compare the balanced DP4/EP4 configuration and reproduce the KV-full retraction failure under high concurrency.

Files:

- `launch.sh`: exact launch configuration
- `server.log`: server output
- `gpu-dmon.log`: per-GPU monitoring
- `vmstat.log`: host monitoring
