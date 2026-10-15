#pragma once

#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

struct MQACandidateOutput {
    const float* thresholds = nullptr;
    float* scores = nullptr;
    int32_t* indices = nullptr;
    int32_t* counts = nullptr;
    int32_t* overflow = nullptr;
    uint32_t capacity_per_warp = 0;
};

// No candidate state in the standard/backward logits instantiations.
template <uint32_t BLOCK_Q, bool kEnabled>
struct MQACandidateWriter {
    CUTLASS_DEVICE MQACandidateWriter(const MQACandidateOutput&, uint32_t, uint32_t) {}
};

template <uint32_t BLOCK_Q>
struct MQACandidateWriter<BLOCK_Q, true> {
    MQACandidateOutput output;
    uint32_t warp, lane, lane_capacity;
    float thresholds[BLOCK_Q];
    uint32_t counts[BLOCK_Q];
    // The host bounds the total candidate allocation to UINT32_MAX elements.
    uint32_t bases[BLOCK_Q];

    CUTLASS_DEVICE MQACandidateWriter(const MQACandidateOutput& output, uint32_t warp, uint32_t lane):
        output(output), warp(warp), lane(lane), lane_capacity(output.capacity_per_warp / 32) {}

    template <uint32_t kNumTokens, typename Scheduler>
    CUTLASS_DEVICE void begin(const Scheduler& scheduler, uint32_t block) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumTokens; ++i) {
            const auto row = scheduler.get_logits_row(block, i);
            thresholds[i] = output.thresholds[row];
            counts[i] = 0;
            bases[i] = (row * 8 + warp) * output.capacity_per_warp
                     + lane * lane_capacity;
        }
    }

    CUTLASS_DEVICE void store(uint32_t i, uint32_t kv, uint32_t start, uint32_t end, float score) {
        // Row bounds must be checked before compaction. A block-wide interval
        // would admit another row's backward-window values or another request.
        const bool accepted = kv - start < end - start and isfinite(score) and score >= thresholds[i];
        const auto count = counts[i];
        const bool has_capacity = count < lane_capacity;
        const uint32_t should_store = accepted and has_capacity;
        const auto dst = bases[i] + (has_capacity ? count : 0);
        ptx::st_no_allocate_if(output.scores + dst, score, should_store);
        ptx::st_no_allocate_if(reinterpret_cast<uint32_t*>(output.indices) + dst, kv - start, should_store);
        counts[i] = count + static_cast<uint32_t>(accepted);
    }

    template <uint32_t kNumTokens, typename Scheduler>
    CUTLASS_DEVICE void finish(const Scheduler& scheduler, uint32_t block) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumTokens; ++i) {
            const auto row = scheduler.get_logits_row(block, i);
            const auto offset = (static_cast<uint64_t>(row) * 8 + warp) * 32 + lane;
            output.counts[offset] = cute::min(counts[i], lane_capacity);
            output.overflow[offset] = counts[i] > lane_capacity;
        }
    }
};

} // namespace deep_gemm
