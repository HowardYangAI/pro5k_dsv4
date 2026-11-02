# DeepSeek-V4-Flash fake-PD decode run (EP8)

- Endpoint: `http://10.207.193.139:30002`
- Model: `/data/models/DeepSeek-V4-Flash-0731`
- Topology: TP8, Attention DP8, EP8
- MoE runner: `flashinfer_mxfp4`; MoE A2A: `none`
- Speculative decoding: DSpark
- Decode-only fake transfer: `--disaggregation-mode decode --disaggregation-transfer-backend fake`
- Decode CUDA graph max batch: 128
- Max running requests: 512
- Benchmark input/output: 8k/2k
