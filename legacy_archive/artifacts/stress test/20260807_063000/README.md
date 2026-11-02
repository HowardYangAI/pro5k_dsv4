# DeepSeek-V4-Flash service run

- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Topology: TP8, DP attention 8, EP8
- MoE backend: `flashinfer_mxfp4`; A2A: `none`
- Speculative decoding: DSpark
- PD single-service test: `--enable-test-pd-without-transfer` is supplied by the benchmark client, which injects fake bootstrap host `2.2.2.2`.
- Server log: `server.log`
- GPU monitor: `gpu-dmon.log`
- VM monitor: `vmstat.log`
