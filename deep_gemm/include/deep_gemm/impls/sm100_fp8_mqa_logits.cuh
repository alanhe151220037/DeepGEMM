#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cuda_bf16.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits, bool kEnableFilterV2,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k, const uint32_t stride_logits,
                          uint32_t* cu_seq_len_k_start,
                          uint32_t* cu_seq_len_k_end,
                          logits_dtype_t* logits,
                          const __nv_bfloat16* filter_thresholds,
                          __nv_bfloat16* candidate_scores,
                          int32_t* candidate_indices,
                          int32_t* candidate_counts,
                          int32_t* candidate_overflow,
                          const uint32_t candidate_capacity,
                          const int32_t* q_block_starts,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // Normally, `h (kNumHeads) == 32` and `d (kHeadDim) == 64`
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto num_q_blocks =
        q_block_starts == nullptr ? math::ceil_div(seq_len, BLOCK_Q) : seq_len;
    constexpr uint32_t kNumProducerWarps = kNumMathThreads / 32;

    // Types
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    // Align to 512 bytes for swizzle-64B
    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    // TMA configs
    constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });

    // Tensor memory allocation
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2);

    // Initialize barriers
    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Block scheduler
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_block_q_start = [&](const uint32_t idx) -> uint32_t {
        return q_block_starts == nullptr
                   ? idx * BLOCK_Q
                   : static_cast<uint32_t>(q_block_starts[idx]);
    };
    const auto has_block = [&](const uint32_t idx) -> bool {
        return idx < num_q_blocks and get_block_q_start(idx) < seq_len;
    };
    const auto get_block_q_end = [&](const uint32_t idx) -> uint32_t {
        const auto q_start = get_block_q_start(idx);
        if (q_block_starts == nullptr)
            return min(q_start + BLOCK_Q, seq_len);
        const auto next_start =
            idx + 1 < num_q_blocks
                ? static_cast<uint32_t>(q_block_starts[idx + 1])
                : seq_len;
        return min(q_start + BLOCK_Q, min(next_start, seq_len));
    };
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        const auto block_q_start = get_block_q_start(block_q_idx);
        const auto block_q_end = get_block_q_end(block_q_idx);

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_start + i, block_q_end - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, math::ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    // UMMA settings
    // Construct instruction with layout D
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Prefetch
        const auto issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            const auto q_start = get_block_q_start(block_idx);
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_start * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_start);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and has_block(block_q_idx))
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (has_block(block_q_idx)) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx());
                    has_block(next_block_q_idx))
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                            smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                              smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        while (has_block(block_q_idx)) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Issue UMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i]->wait(((num_total_kv_blocks + kv_block_idx) & 1) ^ 1);
                    ptx::tcgen05_after_thread_sync();
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[q_stage_idx], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // UMMA warp must also arrive on empty_q to prevent running ahead
            // of math warps in the Q pipeline
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Offsets
        const auto tmem_start = warpgroup_idx * UMMA_N;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Local register buffers
        float weights[BLOCK_Q][kNumHeads];
        float filter_threshold_values[BLOCK_Q] = {};
        uint32_t filter_lane_counts[BLOCK_Q] = {};
        uint32_t filter_lane_bases[BLOCK_Q] = {};
        bool filter_q_valid[BLOCK_Q] = {};
        const uint32_t filter_lane_capacity = candidate_capacity / 32;

        while (has_block(block_q_idx)) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            const uint32_t block_q_start = get_block_q_start(block_q_idx);
            const uint32_t block_q_end = get_block_q_end(block_q_idx);
            uint32_t block_window_start = seq_k_start[0];
            uint32_t block_window_end = seq_k_end[0];
            #pragma unroll
            for (uint32_t row = 1; row < BLOCK_Q; ++row) {
                block_window_start = min(block_window_start, seq_k_start[row]);
                block_window_end = max(block_window_end, seq_k_end[row]);
            }

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
            }

            if constexpr (kEnableFilterV2) {
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                    filter_lane_counts[i] = 0;
                    const uint32_t raw_q_idx = block_q_start + i;
                    const uint32_t q_idx = min(raw_q_idx, seq_len - 1);
                    filter_q_valid[i] = raw_q_idx < block_q_end;
                    filter_threshold_values[i] =
                        reinterpret_cast<const float*>(
                            filter_thresholds)[q_idx];
                    const uint32_t filter_output_base =
                        (q_idx * kNumProducerWarps + warp_idx) *
                        candidate_capacity;
                    filter_lane_bases[i] =
                        filter_output_base +
                        lane_idx * filter_lane_capacity;
                }
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

                // Wait UMMA arrival
                full_umma_barriers[warpgroup_idx]->wait((num_total_kv_blocks + kv_block_idx) & 1);
                ptx::tcgen05_after_thread_sync();

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto kv_block_start =
                    kv_start + kv_block_idx * BLOCK_KV;
                const auto kv_offset =
                    kv_block_start + math_thread_idx;
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Load accumulator from TMEM
                    float accum[kNumHeads];
                    tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum);

                    // Release TMEM empty
                    if (i == BLOCK_Q - 1) {
                        ptx::tcgen05_before_thread_sync();
                        empty_umma_barriers[warpgroup_idx]->arrive();
                    }

                    // Accumulate weighted ReLU in parallel
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                        auto b = make_float2(weights[i][j], weights[i][j + 1]);
                        return __ffma2_rn(a, b, sum);
                    };

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);
                    }

                    auto sum = __fadd2_rn(sum_0, sum_1);
                    const float result_fp32 = scale_kv * (sum.x + sum.y);

                    if constexpr (kEnableFilterV2) {
                        const bool valid =
                            filter_q_valid[i] and
                            seq_k_start[i] <= kv_offset and
                            kv_offset < seq_k_end[i];
                        const bool accepted =
                            valid and isfinite(result_fp32) and
                            result_fp32 >= filter_threshold_values[i];
                        const uint32_t lane_count = filter_lane_counts[i];
                        const bool has_capacity =
                            lane_count < filter_lane_capacity;
                        const uint32_t should_store =
                            accepted and has_capacity;
                        const uint32_t store_slot =
                            has_capacity ? lane_count : 0;
                        const uint32_t dst =
                            filter_lane_bases[i] + store_slot;
                        const uint32_t filter_candidate_index =
                            kv_offset - seq_k_start[i];
                        ptx::st_no_allocate_if(
                            reinterpret_cast<float*>(candidate_scores) + dst,
                            result_fp32,
                            should_store);
                        ptx::st_no_allocate_if(
                            reinterpret_cast<uint32_t*>(candidate_indices) + dst,
                            filter_candidate_index,
                            should_store);
                        filter_lane_counts[i] =
                            lane_count + static_cast<uint32_t>(accepted);
                    } else {
                        auto result = static_cast<logits_dtype_t>(result_fp32);
                        const uint32_t q_idx = block_q_start + i;
                        const auto q_offset = q_idx * static_cast<uint64_t>(stride_logits);
                        if constexpr (kIsCompressedLogits) {
                            if (q_idx < block_q_end and
                                block_window_start <= kv_offset and
                                kv_offset < block_window_end)
                                logits[
                                    q_offset + kv_offset -
                                    block_window_start] = result;
                        } else {
                            logits[q_offset + kv_offset] = result;
                        }
                    }
                    __syncwarp();
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            if constexpr (kEnableFilterV2) {
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                    const uint32_t q_idx = block_q_start + i;
                    if (q_idx < block_q_end) {
                        const uint32_t lane_count = filter_lane_counts[i];
                        const uint64_t output_offset =
                            (static_cast<uint64_t>(q_idx) *
                                 kNumProducerWarps +
                             warp_idx) *
                                32 +
                            lane_idx;
                        candidate_counts[output_offset] =
                            static_cast<int32_t>(
                                min(lane_count, filter_lane_capacity));
                        candidate_overflow[output_offset] =
                            lane_count > filter_lane_capacity ? 1 : 0;
                    }
                }
            }
            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
