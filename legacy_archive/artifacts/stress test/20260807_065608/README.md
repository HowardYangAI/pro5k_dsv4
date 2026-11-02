# DeepSeek-V4-Flash fake-PD decode run (EP1)

- Endpoint: `http://10.207.193.139:30002`
- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Role: decode-only with fake transfer
- Topology: TP8, DP attention 8, EP1 (default; no `--ep-size` argument)
- MoE backend: `flashinfer_mxfp4`; A2A backend: `none`
- Speculative decoding: DSpark
- Client uses `--enable-test-pd-without-transfer`.
- Server log: `server.log`
- GPU monitor: `gpu-dmon.log`
- VM monitor: `vmstat.log`
