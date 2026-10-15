#pragma once

#include <deep_gemm/impls/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_long_context_mqa_logits.cuh>

namespace deep_gemm {

// Only scheduling and output differ from v0.1.5. TMA, UMMA, ring pipelines,
// TMEM lifetime and FP32 head reduction are shared with the stock core.
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits, bool kEnableFilter,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_long_context_mqa_logits(
        const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
        const uint32_t logits_stride,
        const uint32_t* starts, const uint32_t* ends, logits_dtype_t* logits,
        const float* thresholds, float* candidate_scores,
        int32_t* candidate_indices, int32_t* candidate_counts,
        int32_t* candidate_overflow, const uint32_t candidate_capacity,
        const int32_t* q_block_starts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_q,
        const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
        const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
        const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* row_starts, uint32_t* row_ends) {
        return sched::SM100LongContextMQALogitsScheduler<BLOCK_Q, BLOCK_KV, kNumSMs>(
            sm_idx, num_q_tokens, num_kv_tokens, starts, ends, q_block_starts, row_starts, row_ends);
    };
    const MQACandidateOutput output = {
        thresholds, candidate_scores, candidate_indices, candidate_counts,
        candidate_overflow, candidate_capacity
    };
    // FP8 does not consume sf_q; use the KV-scale descriptor just as v0.1.5 does.
    sm100_mqa_logits_core_impl<false, kNumHeads, kHeadDim, kIsCompressedLogits,
        BLOCK_Q, BLOCK_KV, kNumQStages, kNumKVStages, kNumSMs,
        kNumSpecializedThreads, kNumMathThreads, logits_dtype_t, float,
        decltype(make_scheduler), kNumMathThreads / 128, kEnableFilter>(
            logits_stride, logits, tensor_map_q, tensor_map_kv_scales,
            tensor_map_kv, tensor_map_kv_scales, tensor_map_weights, make_scheduler, output);
}

} // namespace deep_gemm
