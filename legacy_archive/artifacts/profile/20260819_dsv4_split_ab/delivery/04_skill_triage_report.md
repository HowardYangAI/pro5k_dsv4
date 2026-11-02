# Split ON

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /sgl-workspace/workspace/profile/20260819_dsv4_split_ab/split_on/decode/dsv4-split-on-c8-w40-n80-decode-1787124772.5090497-TP-0-DP-0-EP-0.trace.json.gz

Kernel Table
##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 152.09 ms | 22.6% | 86 | unresolved | cuLaunchKernelEx |
| void gemmSN_TN_kernel<float, 128, 16, 2, 4, 6, 7, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> > | gemm | 128.93 ms | 19.1% | 930 | unresolved | cudaGraphLaunch |
| ncclDevKernel_Reduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 101.69 ms | 15.1% | 546 | unresolved | cudaGraphLaunch |
| _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17GroupProblemShapeIN4cute5tupleIJlllEEEEENS1_10collective13CollectiveMmaINS1_47MainloopSm120ArrayTmaWarpSpecializedBlockScaledILi4ELi3ENS6_IJNS5_1CILi1EEESD_SD_EEENS1_59KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120ILi3EEEEENS6_IJNSC_ILi128EEENSC_ILi32EEESI_EEENS6_IJNS_12float_e2m1_tENS_13float_ue8m0_tEEEENS6_IJPNS6_IJlSD_NSC_ILi0EEEEEEPNS5_6LayoutINS6_IJNS6_IJNS6_IJSJ_NSC_ILi4EEEEEEiEEESU_NS6_IJSD_iEEEEEENS6_IJNS6_IJNS6_IJNSC_ILi16EEESS_EEEiEEENS6_IJNS6_IJSO_SD_EEENSC_ILi512EEEEEENS6_IJSO_iEEEEEEEEEEENS6_IJNS_12float_e4m3_tESM_EEES17_NS5_8TiledMMAINS5_8MMA_AtomIJNS5_5SM12011BLOCKSCALED19SM120_16x8x32_TN_VSISL_S18_fSM_Li32EEEEEENSR_INS6_IJSS_NSC_ILi2EEESD_EEENS6_IJSD_SS_SO_EEEEENS6_IJSI_NSR_INS6_IJNSC_ILi8EEES1H_S1H_EEENS6_IJSD_SX_S1L_EEEEESJ_EEEEENS6_IJNS5_13SM90_TMA_LOADES1R_EEENS6_IJNS5_14ComposedLayoutINS5_7SwizzleILi3ELi4ELi3EEENS5_18smem_ptr_flag_bitsILi8EEENSR_INS6_IJS1L_SI_EEENS6_IJSI_SD_EEEEEEENSR_INS6_IJNS6_IJST_SD_EEENS6_IJNS6_IJSJ_SD_EEESS_SD_EEEEEENS6_IJNS6_IJSY_S11_EEENS6_IJS10_SD_S11_EEEEEEEEEEENS6_IJNS5_9Copy_AtomIJNS5_26SM100_SU4_DU8x16_x4_LDSM_NEhEEENS2B_IJNS5_13UniversalCopyISM_SM_EESM_EEEEEENS5_8identityES1S_S2A_NS6_IJNS2B_IJNS5_17SM75_U32x4_LDSM_NEhEEES2G_EEES2I_EENS_8epilogue10collective18CollectiveEpilogueINS2N_30Sm90PtrArrayTmaWarpSpecializedILi3ELi2ELi4ELb1ELb0ELi2EEEJSK_NS6_IJNSC_ILi64EEESJ_EEENS_10bfloat16_tEPNS6_IJSD_lSO_EEES2U_S2W_NS2N_6fusion15FusionCallbacksINS2N_31Sm120PtrArrayTmaWarpSpecializedILi3ELi2ELi4ELb1ELb0ELi2EEENS2X_17LinearCombinationIS2U_fS2U_fLNS_15FloatRoundStyleE2EEESK_S2T_JEEES1R_NS1T_IS1V_NS1W_ILi16EEENSR_INS6_IJS2S_S1L_EEENS6_IJSD_S2S_EEEEEEENS5_17SM75_U16x4_LDSM_TENS5_14SM90_TMA_STOREES39_NS5_17SM90_U16x4_STSM_TENS2B_IJNS5_17SM90_U32x2_STSM_NENS_6half_tEEEEvEEEvvEEEEvNT_6ParamsE | moe | 59.41 ms | 8.8% | 1092 | unresolved | cudaGraphLaunch |
| _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective13CollectiveMmaINS1_47MainloopSm120TmaWarpSpecializedBlockwiseScalingILi4ELi2ENS5_IJNS4_1CILi1EEESB_SB_EEENS1_56KernelTmaWarpSpecializedCooperativeBlockwiseScalingSm120ILi2EEEEENS5_IJNSA_ILi128EEENSA_ILi32EEESG_EEENS_12float_e4m3_tENS5_IJNS5_IJlSB_lEEENS4_6LayoutINS5_IJNS5_IJSG_iEEESM_iEEENS5_IJNS5_IJNSA_ILi0EEEiEEENS5_IJSO_SB_EEEiEEEEEEEESJ_NS5_IJSK_NSL_INS5_IJNS5_IJSB_iEEESM_iEEENS5_IJSQ_SP_iEEEEEEEENS4_8TiledMMAINS4_8MMA_AtomIJNS4_16SM120_16x8x32_TNISJ_SJ_fEEEEENSL_INS5_IJNSA_ILi4EEENSA_ILi2EEESB_EEENS5_IJSB_S14_SO_EEEEENS5_IJSG_SH_SH_EEEEENS4_13SM90_TMA_LOADENS4_14ComposedLayoutINS4_7SwizzleILi3ELi4ELi3EEENS4_18smem_ptr_flag_bitsILi8EEENSL_INS5_IJNSA_ILi8EEESG_EEENS5_IJSG_SB_EEEEEEENS4_9Copy_AtomIJNS4_17SM75_U32x4_LDSM_NEhEEENS4_8identityES1B_S1L_S1O_S1P_EENS_8epilogue10collective18CollectiveEpilogueINS1R_22Sm90TmaWarpSpecializedILi2ELi2ELi4ELb0ELb1EEEJSI_NS5_IJNSA_ILi64EEESH_EEEvNS5_IJSB_llEEENS_10bfloat16_tES1Y_NS1R_6fusion15FusionCallbacksINS1R_23Sm120TmaWarpSpecializedILi2ELi2ELi4ELb0ELb1EEENS20_17LinearCombinationIS1Z_fvfLNS_15FloatRoundStyleE2EEESI_S1X_JEEES1B_NS1C_IS1E_NS1F_ILi16EEENSL_INS5_IJS1W_S1H_EEENS5_IJSB_S1W_EEEEEEENS4_17SM75_U16x4_LDSM_TENS4_14SM90_TMA_STOREES2C_NS4_17SM90_U16x4_STSM_TENS1M_IJNS4_17SM90_U32x2_STSM_NENS_6half_tEEEEvEEEvvEEEEvNT_6ParamsE | gemm | 49.79 ms | 7.4% | 2884 | unresolved | cudaGraphLaunch |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 34.14 ms | 5.1% | 470 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8> | gemm | 33.47 ms | 5.0% | 460 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8> | gemm | 17.39 ms | 2.6% | 20 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8> | gemm | 9.29 ms | 1.4% | 620 | unresolved | cudaGraphLaunch |
| _page_split_kernel | other | 7.58 ms | 1.1% | 503 | unresolved | cudaGraphLaunch |
| void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 64, 128, 64> | attention | 6.95 ms | 1.0% | 473 | unresolved | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |


# Split OFF

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /sgl-workspace/workspace/profile/20260819_dsv4_split_ab/split_off/decode/dsv4-split-off-c8-w40-n80-decode-1787125268.685384-TP-0-DP-0-EP-0.trace.json.gz

Kernel Table
##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 141.54 ms | 19.0% | 86 | unresolved | cuLaunchKernelEx |
| void gemmSN_TN_kernel<float, 128, 16, 2, 4, 6, 7, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> > | gemm | 130.04 ms | 17.4% | 930 | unresolved | cudaGraphLaunch |
| ncclDevKernel_Reduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 114.52 ms | 15.3% | 546 | unresolved | cudaGraphLaunch |
| _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17GroupProblemShapeIN4cute5tupleIJlllEEEEENS1_10collective13CollectiveMmaINS1_47MainloopSm120ArrayTmaWarpSpecializedBlockScaledILi4ELi3ENS6_IJNS5_1CILi1EEESD_SD_EEENS1_59KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120ILi3EEEEENS6_IJNSC_ILi128EEENSC_ILi32EEESI_EEENS6_IJNS_12float_e2m1_tENS_13float_ue8m0_tEEEENS6_IJPNS6_IJlSD_NSC_ILi0EEEEEEPNS5_6LayoutINS6_IJNS6_IJNS6_IJSJ_NSC_ILi4EEEEEEiEEESU_NS6_IJSD_iEEEEEENS6_IJNS6_IJNS6_IJNSC_ILi16EEESS_EEEiEEENS6_IJNS6_IJSO_SD_EEENSC_ILi512EEEEEENS6_IJSO_iEEEEEEEEEEENS6_IJNS_12float_e4m3_tESM_EEES17_NS5_8TiledMMAINS5_8MMA_AtomIJNS5_5SM12011BLOCKSCALED19SM120_16x8x32_TN_VSISL_S18_fSM_Li32EEEEEENSR_INS6_IJSS_NSC_ILi2EEESD_EEENS6_IJSD_SS_SO_EEEEENS6_IJSI_NSR_INS6_IJNSC_ILi8EEES1H_S1H_EEENS6_IJSD_SX_S1L_EEEEESJ_EEEEENS6_IJNS5_13SM90_TMA_LOADES1R_EEENS6_IJNS5_14ComposedLayoutINS5_7SwizzleILi3ELi4ELi3EEENS5_18smem_ptr_flag_bitsILi8EEENSR_INS6_IJS1L_SI_EEENS6_IJSI_SD_EEEEEEENSR_INS6_IJNS6_IJST_SD_EEENS6_IJNS6_IJSJ_SD_EEESS_SD_EEEEEENS6_IJNS6_IJSY_S11_EEENS6_IJS10_SD_S11_EEEEEEEEEEENS6_IJNS5_9Copy_AtomIJNS5_26SM100_SU4_DU8x16_x4_LDSM_NEhEEENS2B_IJNS5_13UniversalCopyISM_SM_EESM_EEEEEENS5_8identityES1S_S2A_NS6_IJNS2B_IJNS5_17SM75_U32x4_LDSM_NEhEEES2G_EEES2I_EENS_8epilogue10collective18CollectiveEpilogueINS2N_30Sm90PtrArrayTmaWarpSpecializedILi3ELi2ELi4ELb1ELb0ELi2EEEJSK_NS6_IJNSC_ILi64EEESJ_EEENS_10bfloat16_tEPNS6_IJSD_lSO_EEES2U_S2W_NS2N_6fusion15FusionCallbacksINS2N_31Sm120PtrArrayTmaWarpSpecializedILi3ELi2ELi4ELb1ELb0ELi2EEENS2X_17LinearCombinationIS2U_fS2U_fLNS_15FloatRoundStyleE2EEESK_S2T_JEEES1R_NS1T_IS1V_NS1W_ILi16EEENSR_INS6_IJS2S_S1L_EEENS6_IJSD_S2S_EEEEEEENS5_17SM75_U16x4_LDSM_TENS5_14SM90_TMA_STOREES39_NS5_17SM90_U16x4_STSM_TENS2B_IJNS5_17SM90_U32x2_STSM_NENS_6half_tEEEEvEEEvvEEEEvNT_6ParamsE | moe | 67.39 ms | 9.0% | 1049 | unresolved | cudaGraphLaunch |
| _page_split_kernel | other | 57.22 ms | 7.7% | 503 | unresolved | cudaGraphLaunch |
| _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective13CollectiveMmaINS1_47MainloopSm120TmaWarpSpecializedBlockwiseScalingILi4ELi2ENS5_IJNS4_1CILi1EEESB_SB_EEENS1_56KernelTmaWarpSpecializedCooperativeBlockwiseScalingSm120ILi2EEEEENS5_IJNSA_ILi128EEENSA_ILi32EEESG_EEENS_12float_e4m3_tENS5_IJNS5_IJlSB_lEEENS4_6LayoutINS5_IJNS5_IJSG_iEEESM_iEEENS5_IJNS5_IJNSA_ILi0EEEiEEENS5_IJSO_SB_EEEiEEEEEEEESJ_NS5_IJSK_NSL_INS5_IJNS5_IJSB_iEEESM_iEEENS5_IJSQ_SP_iEEEEEEEENS4_8TiledMMAINS4_8MMA_AtomIJNS4_16SM120_16x8x32_TNISJ_SJ_fEEEEENSL_INS5_IJNSA_ILi4EEENSA_ILi2EEESB_EEENS5_IJSB_S14_SO_EEEEENS5_IJSG_SH_SH_EEEEENS4_13SM90_TMA_LOADENS4_14ComposedLayoutINS4_7SwizzleILi3ELi4ELi3EEENS4_18smem_ptr_flag_bitsILi8EEENSL_INS5_IJNSA_ILi8EEESG_EEENS5_IJSG_SB_EEEEEEENS4_9Copy_AtomIJNS4_17SM75_U32x4_LDSM_NEhEEENS4_8identityES1B_S1L_S1O_S1P_EENS_8epilogue10collective18CollectiveEpilogueINS1R_22Sm90TmaWarpSpecializedILi2ELi2ELi4ELb0ELb1EEEJSI_NS5_IJNSA_ILi64EEESH_EEEvNS5_IJSB_llEEENS_10bfloat16_tES1Y_NS1R_6fusion15FusionCallbacksINS1R_23Sm120TmaWarpSpecializedILi2ELi2ELi4ELb0ELb1EEENS20_17LinearCombinationIS1Z_fvfLNS_15FloatRoundStyleE2EEESI_S1X_JEEES1B_NS1C_IS1E_NS1F_ILi16EEENSL_INS5_IJS1W_S1H_EEENS5_IJSB_S1W_EEEEEEENS4_17SM75_U16x4_LDSM_TENS4_14SM90_TMA_STOREES2C_NS4_17SM90_U16x4_STSM_TENS1M_IJNS4_17SM90_U32x2_STSM_NENS_6half_tEEEEvEEEvvEEEEvNT_6ParamsE | gemm | 51.50 ms | 6.9% | 2884 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8> | gemm | 36.78 ms | 4.9% | 460 | unresolved | cudaGraphLaunch |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 35.12 ms | 4.7% | 470 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8> | gemm | 17.51 ms | 2.3% | 20 | unresolved | cudaGraphLaunch |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8> | gemm | 10.17 ms | 1.4% | 620 | unresolved | cudaGraphLaunch |
| void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<(ModelType)1, 64, 128, 64> | attention | 8.11 ms | 1.1% | 473 | unresolved | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |
