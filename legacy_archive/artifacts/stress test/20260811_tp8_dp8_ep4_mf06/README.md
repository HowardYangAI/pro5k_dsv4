# TP8 / Attention DP8 / EP4 / MF 0.6

- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Endpoint: `http://10.207.193.139:30002`
- Mode: Decode-only with fake KV transfer
- Parallelism: TP8, Attention DP8, EP4 (Attention TP1, MoE TP2)
- DSpark: 6 verify tokens (`gamma=5`)
- Global max running requests: 512 (64 per DP rank)
- CUDA graph max decode batch size: 128
- Purpose: compare EP4 against the existing EP1 and EP8 results while keeping Attention DP8 and DSpark unchanged.

Files:

- `launch.sh`: exact launch configuration
- `server.log`: server output
- `gpu-dmon.log`: per-GPU monitoring
- `vmstat.log`: host monitoring
