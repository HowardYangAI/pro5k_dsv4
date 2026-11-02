# DeepSeek-V4-Flash TP8/DP8/EP8 + DSpark Torch Profile

## 采集口径

- 代表 Rank：TP0 / DP0 / EP0
- Prefill：C8 / 8 requests / 8192 -> 1，active profile 6 scheduler steps
- Decode：C8 / 8 requests / 1 -> 128，active profile 8 scheduler steps
- Profile：CPU + GPU，自动停止；服务开启 DSpark（draft tokens=6）
- 百分比口径：所选 Rank 内 GPU kernel duration 求和；多 stream 可重叠，不等同于墙钟时间

## 三张交付表

1. `01_operator_timeline_tp0.csv`：Prefill 与 Decode 分阶段、按 GPU 时间戳排序的完整执行序列；另保留两份分阶段 timeline CSV。
2. `02_prefill_kernel_aggregate_tp0.csv`：Prefill kernel 聚合排名。
3. `03_decode_kernel_aggregate_tp0.csv`：Decode kernel 聚合排名。

时间顺序 CSV 预留了 FLOP、带宽和 AI 字段。Torch Trace 能可靠给出 duration、调用关系、部分 shape 与源码位置；对复杂 MoE/Attention kernel，FLOP 和访存量需要结合模型语义人工校核，因此本报告不填入未经验证的估算值。

## Prefill 聚合 Top 30

| 算子简称 | 调用次数 | 总时间(ms) | 平均时间(ms) | 时间占比(%) | 累计占比(%) | Python源码 |
|---|---|---|---|---|---|---|
| NCCL AllReduce | 258 | 336.897017 | 1.305802 | 38.842162 | 38.842162 | unresolved |
| NCCL Reduce | 258 | 203.437449 | 0.788517 | 23.455092 | 62.297254 | unresolved |
| SM120 CUTLASS GEMM | 516 | 132.012989 | 0.255839 | 15.220289 | 77.517543 | unresolved |
| SM120 CUTLASS GEMM | 865 | 33.283936 | 0.038479 | 3.837434 | 81.354977 | unresolved |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8> | 215 | 26.131647 | 0.121543 | 3.012819 | 84.367796 | unresolved |
| FlashMLA sparse attention | 105 | 17.88537 | 0.170337 | 2.062074 | 86.42987 | unresolved |
| FlashMLA sparse attention | 100 | 12.502939 | 0.125029 | 1.441512 | 87.871382 | unresolved |
| SM120 CUTLASS GEMM | 411 | 8.459932 | 0.020584 | 0.975378 | 88.84676 | unresolved |
| void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_tn_align1> | 430 | 7.864898 | 0.01829 | 0.906775 | 89.753535 | unresolved |
| void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x128_32x4_tn_align8> | 320 | 7.141608 | 0.022318 | 0.823384 | 90.576918 | unresolved |
| _page_mark_kernel | 215 | 5.343073 | 0.024852 | 0.616024 | 91.192942 | unresolved |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, f | 860 | 5.115831 | 0.005949 | 0.589824 | 91.782766 | unresolved |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}: | 435 | 4.921984 | 0.011315 | 0.567475 | 92.350241 | unresolved |
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, floa | 5 | 4.42691 | 0.885382 | 0.510396 | 92.860636 | unresolved |
| mhc_post_tilelang_kernel | 430 | 4.247842 | 0.009879 | 0.48975 | 93.350386 | unresolved |
| MoE kernel | 258 | 3.69494 | 0.014321 | 0.426004 | 93.77639 | unresolved |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> > | 258 | 3.427585 | 0.013285 | 0.39518 | 94.17157 | unresolved |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_fun | 430 | 3.358181 | 0.00781 | 0.387178 | 94.558748 | unresolved |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal | 998 | 3.255516 | 0.003262 | 0.375341 | 94.934089 | unresolved |
| NCCL AllReduce | 6 | 3.15328 | 0.525547 | 0.363554 | 95.297642 | unresolved |
| void tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16, tensorrt_ll | 258 | 3.086173 | 0.011962 | 0.355817 | 95.653459 | unresolved |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float | 430 | 2.938467 | 0.006834 | 0.338787 | 95.992247 | unresolved |
| _page_split_kernel | 215 | 2.840195 | 0.01321 | 0.327457 | 96.319704 | unresolved |
| void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_e4m3fn, true, false, false, fl | 1276 | 2.713201 | 0.002126 | 0.312815 | 96.632519 | unresolved |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsi | 430 | 2.275163 | 0.005291 | 0.262312 | 96.894831 | unresolved |
| void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_fp8_e4m3, __nv_fp8_e4m3, (tensorrt_llm::kernels: | 258 | 1.820954 | 0.007058 | 0.209945 | 97.104776 | unresolved |
| void (anonymous namespace)::fused_q_norm_rope<__nv_bfloat16, 512l, 64l, long, true> | 215 | 1.790538 | 0.008328 | 0.206438 | 97.311214 | unresolved |
| hc_split_sinkhorn_kernel__kernel | 430 | 1.587669 | 0.003692 | 0.183049 | 97.494263 | unresolved |
| void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_256x64_32x4_tn_align8> | 100 | 1.525888 | 0.015259 | 0.175926 | 97.670188 | unresolved |
| void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::BlockScaleQuantizationType)2, __nv_bfloat16, 32, tru | 258 | 1.390814 | 0.005391 | 0.160352 | 97.830541 | unresolved |

## Decode 聚合 Top 30

| 算子简称 | 调用次数 | 总时间(ms) | 平均时间(ms) | 时间占比(%) | 累计占比(%) | Python源码 |
|---|---|---|---|---|---|---|
| NCCL AllReduce | 86 | 139.702169 | 1.624444 | 28.983503 | 28.983503 | unresolved |
| NCCL Reduce | 362 | 85.73938 | 0.236849 | 17.788038 | 46.771541 | unresolved |
| cuBLAS small-N GEMM | 558 | 77.494623 | 0.138879 | 16.077529 | 62.84907 | unresolved |
| SM120 CUTLASS GEMM | 638 | 32.89349 | 0.051557 | 6.824293 | 69.673363 | unresolved |
| SM120 CUTLASS GEMM | 1860 | 31.733501 | 0.017061 | 6.583634 | 76.256997 | unresolved |
| NCCL AllGather | 282 | 20.781949 | 0.073695 | 4.311556 | 80.568553 | unresolved |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8> | 276 | 20.076099 | 0.072739 | 4.165116 | 84.733668 | unresolved |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8> | 12 | 10.434368 | 0.869531 | 2.164781 | 86.898449 | unresolved |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8> | 372 | 5.579741 | 0.014999 | 1.157609 | 88.056058 | unresolved |
| _page_split_kernel | 319 | 4.670872 | 0.014642 | 0.969049 | 89.025107 | unresolved |
| FlashMLA sparse attention | 301 | 4.367557 | 0.01451 | 0.906121 | 89.931228 | unresolved |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8> | 126 | 3.83405 | 0.030429 | 0.795436 | 90.726664 | unresolved |
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, floa | 64 | 2.565954 | 0.040093 | 0.532349 | 91.259013 | unresolved |
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, floa | 30 | 2.467169 | 0.082239 | 0.511855 | 91.770868 | unresolved |
| hc_split_sinkhorn_kernel__kernel | 638 | 2.361225 | 0.003701 | 0.489875 | 92.260743 | unresolved |
| void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_e4m3fn, true, false, false, fl | 1860 | 2.308349 | 0.001241 | 0.478905 | 92.739648 | unresolved |
| mhc_post_tilelang_kernel | 638 | 2.071717 | 0.003247 | 0.429812 | 93.169459 | unresolved |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_fun | 644 | 1.974212 | 0.003066 | 0.409583 | 93.579042 | unresolved |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8> | 276 | 1.595075 | 0.005779 | 0.330924 | 93.909966 | unresolved |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<float> >(at::Tens | 1136 | 1.536655 | 0.001353 | 0.318804 | 94.22877 | unresolved |
| RMSNorm | 658 | 1.369928 | 0.002082 | 0.284214 | 94.512984 | unresolved |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal | 522 | 1.221756 | 0.002341 | 0.253473 | 94.766458 | unresolved |
| MoE kernel | 276 | 1.102721 | 0.003995 | 0.228778 | 94.995235 | unresolved |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::Float8_e4m3fn>, std::array<char*, 1ul> > | 1136 | 1.001045 | 0.000881 | 0.207683 | 95.202918 | unresolved |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, f | 742 | 0.971022 | 0.001309 | 0.201454 | 95.404373 | unresolved |
| triton_red_fused__to_copy_mean_pow_view_0 | 552 | 0.964737 | 0.001748 | 0.20015 | 95.604523 | unresolved |
| void tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16, tensorrt_ll | 362 | 0.955238 | 0.002639 | 0.19818 | 95.802703 | unresolved |
| void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_fp8_e4m3, __nv_fp8_e4m3, (tensorrt_llm::kernels: | 362 | 0.942654 | 0.002604 | 0.195569 | 95.998272 | unresolved |
| std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, floa | 1 | 0.878944 | 0.878944 | 0.182351 | 96.180623 | unresolved |
| void tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_b | 362 | 0.846912 | 0.00234 | 0.175706 | 96.356329 | unresolved |

## Perfetto

Perfetto 可直接打开 `.trace.json.gz`。代表 Trace：

- Prefill TP0：`prefill/dsv4-ep8-dspark-c8-prefill-dsv4-ep8-dspark-c8-prefill-TP-0-DP-0-EP-0.trace.json.gz`
- Decode TP0：`decode/dsv4-ep8-dspark-c8-decode-1787122755.7801342-TP-0-DP-0-EP-0.trace.json.gz`

Perfetto 交付包只包含代表 Rank 的两份 Trace、三张表及本说明；完整 8-rank Trace 保留在上级 `prefill/` 与 `decode/` 目录。
