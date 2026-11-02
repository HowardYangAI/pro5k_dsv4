# TP8 + Attention DP8 + EP8 + DSpark Torch Profile

日期：2026-08-19

模型：DeepSeek-V4-Flash-0731，端到端推理，MF=0.6，CUDA Graph decode=full，DSpark draft tokens=6。

## 结论

`bench_serving.py --profile` 会发送一个没有边界的 `/start_profile` 请求。当前 SGLang 默认同时启用 CPU、GPU、Python stack 和 tensor shape，并一直采集到整场 benchmark 结束后再手工 `/stop_profile`。

故障并不是“只要开 Torch Profile 就会触发”：GPU-only、CPU-only，以及设置 `num_steps` 后的 CPU+GPU 联合采集均成功。失败集中在无 `num_steps` 的长时间 CPU+GPU 联合采集及最终手工 stop/flush 阶段。即使关闭 stack 和 shape，该路径仍可触发 GPU Xid 13，因此 stack/shape 不是必要条件，只会进一步增加开销。

截至 2026-08-19，以 `profiler + CUDA Graph + Xid 13/Blackwell`、`Kineto + CUDA Graph`、`EXTRA_INLINE_DATA` 检索 PyTorch、Kineto 和 SGLang 官方 GitHub issue，未找到完全匹配的已知问题。因此当前结论是本地实验确定的触发边界与规避方案，底层具体缺陷仍需向 PyTorch/NVIDIA 上游提供最小复现后确认。

推荐使用有界、自动停止的 Profile：

```bash
curl -X POST http://127.0.0.1:30012/start_profile \
  -H 'Content-Type: application/json' \
  -d '{
    "activities": ["CPU", "GPU"],
    "with_stack": false,
    "record_shapes": false,
    "profile_by_stage": true,
    "num_steps": 5,
    "output_dir": "/sgl-workspace/workspace/profile/20260819_ep8_e2e_torch_profile/traces/cpu_gpu_by_stage_5_8k",
    "profile_id": "cpu_gpu_by_stage_5_8k"
  }'
```

benchmark 命令不要再加 `--profile`，也不需要手工调用 `/stop_profile`。

## 测试矩阵

| 场景 | 结果 | Xid | Trace |
|---|---|---:|---|
| 无 Profile，C4/C8 | 成功 | 无新增 | 无 |
| GPU-only，stack=false，shapes=false | 成功 | 无新增 | 8 rank |
| CPU-only，stack=false，shapes=false | 成功 | 无新增 | 8 rank |
| CPU+GPU，无边界、手工 stop | 请求完成后卡住；stop 时服务失去响应 | 新增 Xid 13 | 未成功落盘 |
| CPU+GPU，`num_steps=1` | 成功 | 无新增 | 8 rank |
| CPU+GPU，`num_steps=10`，8K C4×4 | 成功 | 无新增 | 8 rank |
| CPU+GPU，按阶段各 5 step，8K C4×4 | 成功 | 无新增 | EXTEND/DECODE 分阶段 |
| 关闭 CUDA Graph 对照 | DSpark eager warmup 的 SM120 FP8 GEMM 内部检查失败，未进入 Profile | 无新增 | 对照无效 |

## 文件说明

- `launch.sh`：正常 CUDA Graph 服务启动命令。
- `launch-no-cuda-graph.sh`：关闭 CUDA Graph 的对照启动命令。
- `logs/`：各 benchmark 与服务日志。
- `diagnostics/`：CPU+GPU 无边界采集失败时的 Xid 与进程现场。
- `traces/cpu_gpu_by_stage_5_8k/`：推荐分析的分阶段 trace。
- `PROFILE_SUMMARY.md`：代表 rank 的 GPU kernel 汇总。
