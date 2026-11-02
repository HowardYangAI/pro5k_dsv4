# TP8 / Attention DP8 / EP4 / C512 retraction fix verification

- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Endpoint: `http://10.207.193.139:30002`
- Mode: Decode-only with fake KV transfer
- Parallelism: TP8, Attention DP8, EP4
- DSpark: 6 verify tokens (`gamma=5`)
- MF: 0.6
- Global max running requests: 512 (64 per DP rank)
- Purpose: verify that unsupported DeepSeek-V4 CPU KV copy no longer crashes the scheduler when the SWA pool is exhausted.
- Expected overload behavior: only unrecoverable retracted requests are aborted; the service remains healthy.

Files:

- `launch.sh`: exact launch configuration
- `server.log`: first fixed-server verification output
- `server-final.log`: final fixed-server verification output
- `gpu-dmon.log`: per-GPU monitoring
- `vmstat.log`: host monitoring
- `benchmark-c512-final.log`: fixed-token C512 control run (512/512 completed)
- `benchmark-c512-randomids-final.log`: pressure run that exercised retraction

Final pressure result (random IDs, exact 8K input / 2K output, C512):

- 504 requests completed normally; 8 were gracefully aborted with HTTP 500.
- `NotImplementedError`: 0
- `Scheduler hit an exception`: 0
- `state was deleted in TokenizerManager`: 0
- `/v1/models` and a post-pressure 16-token generation both returned HTTP 200.
