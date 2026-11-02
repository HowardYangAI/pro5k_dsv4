# DeepSeek-V4-Flash fake-PD decode run

- Endpoint: `http://10.207.193.139:30002`
- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Role: decode-only (`--disaggregation-mode decode`)
- Topology: TP8, DP attention 8, EP8
- MoE backend: `flashinfer_mxfp4`; A2A backend: `none`
- Speculative decoding: DSpark
- Client must use `--enable-test-pd-without-transfer` to inject the fake bootstrap host.
- Active server log: `server-attempt-2.log`
- `server.log` records the failed first attempt with the default Mooncake backend.
- GPU monitor: `gpu-dmon.log`
- VM monitor: `vmstat.log`
