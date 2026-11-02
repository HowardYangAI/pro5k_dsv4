# 分阶段 Profile 摘要

数据源：`cpu_gpu_by_stage_5_8k`，输入 8192、输出 64、C4×4，Prefill/Decode 各采 5 个 scheduler step。

以下为代表 rank TP0/DP0/EP0 的 GPU kernel 累计时间。百分比以该阶段所有 GPU kernel duration 之和为分母，多个 stream 的 duration 可能重叠，因此不等同于端到端墙钟时间。

## EXTEND / Prefill

GPU kernel 累计时间：528.65 ms。

| Kernel 类别 | 时间 | 占比 | 说明 |
|---|---:|---:|---|
| NCCL AllReduce BF16 | 155.38 ms | 29.39% | TP/Attention DP 路径的跨卡结果合并 |
| NCCL Reduce BF16 | 110.97 ms | 20.99% | NCCL reduction 通信 |
| SM120 MXFP4 CUTLASS MoE GEMM | 47.55 ms | 9.00% | MoE Expert 计算 |
| BF16 GEMM | 26.17 ms | 4.95% | Dense/投影矩阵计算 |
| 另一组 SM120 MoE GEMM | 26.03 ms | 4.92% | MoE Expert 计算/输出处理 |
| FlashMLA sparse prefill kernels | 30.61 ms | 5.79% | 两类 sparse MLA prefill kernel 合计 |

该阶段最明显的瓶颈是通信：仅 AllReduce + Reduce 已占 GPU kernel duration 的约 50.38%。

## DECODE

GPU kernel 累计时间：236.04 ms。

| Kernel 类别 | 时间 | 占比 | 说明 |
|---|---:|---:|---|
| cuBLAS small-N GEMM | 64.10 ms | 27.16% | Decode 小 batch 矩阵计算 |
| SM120 MXFP4 CUTLASS MoE GEMM | 30.88 ms | 13.08% | MoE Expert 计算 |
| NCCL Reduce BF16 | 25.11 ms | 10.64% | 跨卡 reduction |
| NCCL AllGather | 22.81 ms | 9.66% | 收集各 rank 的切分结果 |
| SM120 blockwise GEMM | 22.51 ms | 9.54% | 量化矩阵计算 |
| BF16 GEMM | 16.70 ms | 7.07% | 投影/输出矩阵计算 |
| FlashInfer sparse MLA decode | 5.00 ms | 2.12% | Decode attention |
| FlashMLA `_page_split_kernel` | 3.45 ms | 1.46% | PR #32320 涉及的 SWA page split 路径 |

Decode 中 NCCL Reduce + AllGather 合计约 20.30%；小 batch GEMM 与 MoE GEMM 是主要计算项。`_page_split_kernel` 在当前采样中仅占 1.46%，不是整体性能主项，也不支持“PR #32320 导致 Profile Xid”的假设。

## 当前根因判断

证据指向 PyTorch/Kineto 在 SM120 上进行长时间 CPU+CUDA 联合采集，并在无边界手工 stop/flush 时与 CUDA runtime/driver 状态交互异常。表现为 `Xid 13: EXTRA_INLINE_DATA`，Python 随后可能只在下一次 CUDA API 调用处看到异步错误。

目前已排除：普通推理负载本身、单独 CPU activity、单独 GPU activity、stack/shape 是唯一触发条件、PR #32320 是直接根因。

尚未精确定位到 PyTorch、CUPTI 或 NVIDIA driver 中的具体缺陷；但有界 `num_steps` + 自动停止已形成稳定规避方案，并能产出可分析的 CPU+GPU trace。

