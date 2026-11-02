# DeepSeek-V4-Flash TP8/DP8/EP8 + DSpark Torch Profile

## 采集口径

- 代表 Rank：TP0 / DP0 / EP0
- Prefill：8192 -> 1，warmup 10，active profile 5 scheduler steps
- Decode：1 -> 128，warmup 10，active profile 5 scheduler steps
- Profile：CPU + GPU，自动停止；服务开启 DSpark（draft tokens=6）
- 百分比口径：所选 Rank 内 GPU kernel duration 求和；多 stream 可重叠，不等同于墙钟时间

## 三张交付表

1. `01_prefill_timeline_tp0.csv` 与 `01_decode_timeline_tp0.csv`：按 GPU 时间戳排序的完整执行序列。
2. `02_prefill_kernel_aggregate_tp0.csv`：Prefill kernel 聚合排名。
3. `02_decode_kernel_aggregate_tp0.csv`：Decode kernel 聚合排名。

时间顺序 CSV 预留了 FLOP、带宽和 AI 字段。Torch Trace 能可靠给出 duration、调用关系、部分 shape 与源码位置；对复杂 MoE/Attention kernel，FLOP 和访存量需要结合模型语义人工校核，因此本报告不填入未经验证的估算值。

## Prefill 聚合 Top 30

| 算子简称 | 调用次数 | 总时间(ms) | 平均时间(ms) | 时间占比(%) | 累计占比(%) | Python源码 |
|---|---|---|---|---|---|---|
| NCCL AllReduce | 258 | 900.705456 | 3.491106 | 88.383695 | 88.383695 | python/sglang/srt/distributed/parallel_state.py:872 _all_reduce_in_place |
| NCCL Reduce | 258 | 70.03351 | 0.271448 | 6.872191 | 95.255886 | python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py:555 ncclGroupEnd |
| SM120 CUTLASS GEMM | 258 | 15.302183 | 0.059311 | 1.50156 | 96.757446 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| SM120 CUTLASS GEMM | 258 | 9.249353 | 0.03585 | 0.907613 | 97.665059 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| NCCL AllReduce | 6 | 9.129288 | 1.521548 | 0.895831 | 98.560891 | python/sglang/srt/distributed/parallel_state.py:872 _all_reduce_in_place |
| SM120 CUTLASS GEMM | 258 | 5.081231 | 0.019695 | 0.498607 | 99.059498 | python/sglang/jit_kernel/fp8_blockwise_gemm.py:73 fp8_blockwise_scaled_mm |
| SM120 CUTLASS GEMM | 258 | 1.358114 | 0.005264 | 0.133268 | 99.192765 | python/sglang/jit_kernel/fp8_blockwise_gemm.py:73 fp8_blockwise_scaled_mm |
| void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_32x6_tn_align8> | 258 | 1.325999 | 0.00514 | 0.130117 | 99.322882 | python/sglang/jit_kernel/dsv4/gemm.py:16 linear_bf16_fp32 |
| void tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16, tensorrt_ll | 258 | 0.854651 | 0.003313 | 0.083865 | 99.406746 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_fp8_e4m3, __nv_fp8_e4m3, (tensorrt_llm::kernels: | 258 | 0.801378 | 0.003106 | 0.078637 | 99.485383 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| void tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_b | 258 | 0.775503 | 0.003006 | 0.076098 | 99.561481 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_e4m3fn, true, false, false, fl | 516 | 0.672299 | 0.001303 | 0.065971 | 99.627452 | python/sglang/kernels/ops/quantization/fp8_kernel.py:669 sglang_per_token_group_quant_fp8_row_padded |
| _router_triton_kernel | 240 | 0.576256 | 0.002401 | 0.056546 | 99.683999 | python/sglang/jit_kernel/moe_fused_gate.py:254 moe_fused_gate |
| void tensorrt_llm::kernels::cutlass_kernels::fusedBuildExpertMapsSortFirstTokenKernel<256, 6, 6> | 258 | 0.562703 | 0.002181 | 0.055216 | 99.739215 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> > | 258 | 0.548035 | 0.002124 | 0.053777 | 99.792992 | python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py:365 maybe_fuse_routed_scale_and_shared_add |
| void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::BlockScaleQuantizationType)2, __nv_bfloat16, 32, tru | 258 | 0.539096 | 0.00209 | 0.0529 | 99.845892 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| Memset (Device) | 516 | 0.516252 | 0.001 | 0.050658 | 99.89655 | unresolved |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal | 258 | 0.449219 | 0.001741 | 0.044081 | 99.940631 | python/sglang/jit_kernel/dsv4/gemm.py:16 linear_bf16_fp32 |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> > | 258 | 0.309692 | 0.0012 | 0.030389 | 99.97102 | python/sglang/srt/layers/dp_attention.py:422 _dp_gather_via_all_reduce |
| void (anonymous namespace)::silu_mul_clamp_kernel<__nv_bfloat16, true> | 258 | 0.224255 | 0.000869 | 0.022006 | 99.993026 | python/sglang/jit_kernel/dsv4/moe.py:176 silu_and_mul_clamp |
| Top-k / routing | 18 | 0.02701 | 0.001501 | 0.00265 | 99.995676 | python/sglang/jit_kernel/dsv4/moe.py:112 hash_topk |
| void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::array<char*, 1ul> > | 12 | 0.009856 | 0.000821 | 0.000967 | 99.996643 | python/sglang/srt/layers/dp_attention.py:384 get_dp_local_info |
| void at_cuda_detail::cub::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long, long, long, unsigned int, | 6 | 0.007839 | 0.001307 | 0.000769 | 99.997413 | python/sglang/srt/layers/dp_attention.py:384 get_dp_local_info |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorItera | 6 | 0.00768 | 0.00128 | 0.000754 | 99.998166 | python/sglang/srt/model_executor/forward_batch_info.py:221 compute_local_num_token_non_padded |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> > | 6 | 0.00768 | 0.00128 | 0.000754 | 99.99892 | python/sglang/srt/model_executor/forward_batch_info.py:221 compute_local_num_token_non_padded |
| Memcpy HtoD (Pageable -> Device) | 18 | 0.004767 | 0.000265 | 0.000468 | 99.999388 | python/sglang/srt/model_executor/forward_batch_info.py:642 init_new |
| void at_cuda_detail::cub::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true> > | 6 | 0.004705 | 0.000784 | 0.000462 | 99.999849 | python/sglang/srt/layers/dp_attention.py:384 get_dp_local_info |
| Memcpy HtoD (Pinned -> Device) | 6 | 0.001536 | 0.000256 | 0.000151 | 100.0 | python/sglang/srt/model_executor/forward_batch_info.py:1160 prepare_mlp_sync_batch |

## Decode 聚合 Top 30

| 算子简称 | 调用次数 | 总时间(ms) | 平均时间(ms) | 时间占比(%) | 累计占比(%) | Python源码 |
|---|---|---|---|---|---|---|
| NCCL AllReduce | 43 | 166.539448 | 3.87301 | 43.684843 | 43.684843 | python/sglang/srt/distributed/parallel_state.py:872 _all_reduce_in_place |
| cuBLAS small-N GEMM | 465 | 64.474596 | 0.138655 | 16.912285 | 60.597128 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| NCCL Reduce | 273 | 24.849337 | 0.091023 | 6.518212 | 67.11534 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| SM120 CUTLASS GEMM | 1356 | 22.956981 | 0.01693 | 6.021829 | 73.137169 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8> | 230 | 16.758506 | 0.072863 | 4.395912 | 77.533081 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| SM120 CUTLASS GEMM | 503 | 16.602979 | 0.033008 | 4.355116 | 81.888197 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| NCCL AllGather | 235 | 16.36792 | 0.069651 | 4.293457 | 86.181654 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8> | 10 | 8.695415 | 0.869542 | 2.280888 | 88.462542 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8> | 310 | 4.648819 | 0.014996 | 1.219428 | 89.68197 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| _page_split_kernel | 230 | 3.476352 | 0.015115 | 0.911879 | 90.59385 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| FlashMLA sparse attention | 215 | 3.230235 | 0.015024 | 0.847321 | 91.44117 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8> | 105 | 3.220093 | 0.030668 | 0.84466 | 92.285831 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, floa | 25 | 2.055495 | 0.08222 | 0.539175 | 92.825006 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_e4m3fn, true, false, false, fl | 1356 | 1.714052 | 0.001264 | 0.449612 | 93.274618 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| hc_split_sinkhorn_kernel__kernel | 460 | 1.695665 | 0.003686 | 0.444789 | 93.719407 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| mhc_post_tilelang_kernel | 460 | 1.457022 | 0.003167 | 0.38219 | 94.101597 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_fun | 465 | 1.330462 | 0.002861 | 0.348993 | 94.45059 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8> | 230 | 1.327883 | 0.005773 | 0.348316 | 94.798906 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<float> >(at::Tens | 896 | 1.25394 | 0.001399 | 0.32892 | 95.127826 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal | 435 | 1.020916 | 0.002347 | 0.267796 | 95.395621 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| RMSNorm | 470 | 1.013607 | 0.002157 | 0.265879 | 95.6615 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| triton_red_fused__to_copy_mean_pow_view_0 | 460 | 0.7951 | 0.001728 | 0.208562 | 95.870062 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::Float8_e4m3fn>, std::array<char*, 1ul> > | 896 | 0.786493 | 0.000878 | 0.206304 | 96.076366 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| SM120 CUTLASS GEMM | 43 | 0.786146 | 0.018282 | 0.206213 | 96.28258 | python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py:295 fused_experts_none_to_flashinfer_mxfp4 |
| void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_fp8_e4m3, __nv_fp8_e4m3, (tensorrt_llm::kernels: | 273 | 0.75113 | 0.002751 | 0.197028 | 96.479608 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16, tensorrt_ll | 273 | 0.745309 | 0.00273 | 0.195501 | 96.67511 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| MoE kernel | 230 | 0.741433 | 0.003224 | 0.194485 | 96.869594 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_b | 273 | 0.65453 | 0.002398 | 0.171689 | 97.041284 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, f | 475 | 0.629526 | 0.001325 | 0.165131 | 97.206414 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |
| _router_triton_kernel | 255 | 0.61449 | 0.00241 | 0.161186 | 97.367601 | python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123 replay |

## Perfetto

Perfetto 可直接打开 `.trace.json.gz`。代表 Trace：

- Prefill TP0：`prefill/1787121565.2203753/dsv4-ep8-dspark-prefill-1787121596.3962104-TP-0-DP-0-EP-0.trace.json.gz`
- Decode TP0：`decode/1787121646.3158555/dsv4-ep8-dspark-decode-1787121646.6495135-TP-0-DP-0-EP-0.trace.json.gz`

完整压缩包包含 Prefill/Decode 的 8-rank 原始 Trace、三张表及本说明。
