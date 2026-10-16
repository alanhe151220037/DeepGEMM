#pragma once

#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

namespace deep_gemm::sched {

// Persistent Q-only scheduling, without dev's cross-CTA KV splitting.
// Optional starts are request-aware block starts, padded to num_q_tokens with -1.
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs, bool kHasQMask = false>
struct SM100LongContextMQALogitsScheduler {
    static constexpr bool kIsPaged = false;
    static constexpr bool kHasPartialBlock = true;
    static constexpr uint32_t kPageKV = 0;

    uint32_t current_q_block_idx, num_q_blocks, num_q_tokens, num_kv_tokens;
    uint32_t block_window_start, block_window_end;
    const uint32_t* starts;
    const uint32_t* ends;
    const int32_t* q_block_starts;
    uint32_t* seq_k_start;
    uint32_t* seq_k_end;
    const bool* q_mask;
    uint32_t active_rows;

    CUTLASS_DEVICE SM100LongContextMQALogitsScheduler(
            const uint32_t sm_idx, const uint32_t num_q_tokens,
            const uint32_t num_kv_tokens, const uint32_t* starts,
            const uint32_t* ends, const int32_t* q_block_starts,
            uint32_t* seq_k_start, uint32_t* seq_k_end,
            const bool* q_mask = nullptr):
        current_q_block_idx(sm_idx),
        num_q_blocks(q_block_starts == nullptr ? math::ceil_div(num_q_tokens, BLOCK_Q) : num_q_tokens),
        num_q_tokens(num_q_tokens), num_kv_tokens(num_kv_tokens),
        starts(starts), ends(ends), q_block_starts(q_block_starts),
        seq_k_start(seq_k_start), seq_k_end(seq_k_end), q_mask(q_mask) {}

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t q_block_idx) const {
        return q_block_starts == nullptr ? q_block_idx * BLOCK_Q
                                         : static_cast<uint32_t>(q_block_starts[q_block_idx]);
    }

    CUTLASS_DEVICE uint32_t get_num_block_tokens(const uint32_t q_block_idx) const {
        const auto start = get_q_tma_token_base(q_block_idx);
        auto end = cute::min(start + BLOCK_Q, num_q_tokens);
        if (q_block_starts != nullptr and q_block_idx + 1 < num_q_blocks) {
            const auto next = q_block_starts[q_block_idx + 1];
            if (next >= 0)
                end = cute::min(end, static_cast<uint32_t>(next));
        }
        return end - start;
    }

    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_base, uint32_t& num_kv_splits) {
        while (true) {
            if (current_q_block_idx >= num_q_blocks)
                return false;
            q_block_idx = current_q_block_idx;
            const auto q_start = get_q_tma_token_base(q_block_idx);
            if (q_start >= num_q_tokens)
                return false;
            current_q_block_idx += kNumSMs;

            const auto q_end = q_start + get_num_block_tokens(q_block_idx);
            uint32_t start = num_kv_tokens, end = 0;
            uint32_t output_start = num_kv_tokens, output_end = 0;
            bool any_active = false;
            active_rows = 0;
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                // Duplicate the last real row only for scheduling; the math epilogue
                // dispatches the actual token count and never stores padded rows.
                const auto row = cute::min(q_start + i, q_end - 1);
                seq_k_start[i] = cute::min(starts[row], num_kv_tokens);
                seq_k_end[i] = cute::min(ends[row], num_kv_tokens);
                output_start = cute::min(output_start, seq_k_start[i]);
                output_end = cute::max(output_end, seq_k_end[i]);
                if constexpr (kHasQMask) {
                    if (q_start + i >= q_end or not q_mask[row])
                        continue;
                    active_rows |= 1u << i;
                }
                any_active = true;
                start = cute::min(start, seq_k_start[i]);
                end = cute::max(end, seq_k_end[i]);
            }
            if constexpr (kHasQMask) {
                if (not any_active)
                    continue;
            }
            // Preserve the metadata's shared column origin even for partial masks.
            block_window_start = output_start;
            block_window_end = output_end;
            kv_base = start / 4 * 4;
            num_kv_splits = end > start ? math::ceil_div(end - kv_base, SPLIT_KV) : 0;
            return true;
        }
    }

    CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t q_block_idx, const uint32_t i) const {
        return get_q_tma_token_base(q_block_idx) + i;
    }

    CUTLASS_DEVICE bool is_row_active(uint32_t /*block*/, uint32_t token) const {
        return (active_rows >> token) & 1u;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t base, const uint32_t split) {
        return base + split * SPLIT_KV;
    }

    CUTLASS_DEVICE static uint32_t get_logits_col(
            const uint32_t base, const uint32_t split, const uint32_t thread) {
        return get_kv_tma_offset(base, split) + thread;
    }
};

} // namespace deep_gemm::sched
