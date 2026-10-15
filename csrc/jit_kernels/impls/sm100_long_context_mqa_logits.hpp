#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"
#include <deep_gemm/layout/mqa_logits.cuh>

namespace deep_gemm {

class SM100LongContextMQALogitsRuntime final: public LaunchRuntime<SM100LongContextMQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;
        bool enable_filter_v2;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;
        void* filter_thresholds;
        void* candidate_scores;
        int* candidate_indices;
        int* candidate_counts;
        int* candidate_overflow;
        int candidate_capacity;
        int* q_block_starts;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_long_context_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_long_context_mqa_logits<
        {}, {}, {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
    args.num_heads, args.head_dim,
    args.is_compressed_logits, args.enable_filter_v2,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.filter_thresholds, args.candidate_scores,
            args.candidate_indices, args.candidate_counts,
            args.candidate_overflow, args.candidate_capacity,
            args.q_block_starts,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void sm100_long_context_mqa_logits(const torch::Tensor& q,
                                const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                const torch::Tensor& weights,
                                const torch::Tensor& cu_seq_len_k_start,
                                const torch::Tensor& cu_seq_len_k_end,
                                const torch::Tensor& logits,
                                const at::ScalarType& logits_dtype,
                                const int& seq_len, const int& seq_len_kv,
                                const int& max_seqlen_k, const int& stride_logits,
                                const int& num_heads, const int& head_dim,
                                const int& block_q, const int& block_kv,
                                const torch::Tensor* q_block_starts = nullptr) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 3, num_kv_stages = 5;
    constexpr int num_math_threads = 256;
    DG_HOST_ASSERT(num_heads == 32 and head_dim == 128 and block_q == 4 and block_kv == 256);
    if (seq_len == 0 or seq_len_kv == 0)
        return;

    // Use compressed logits format when max_seqlen_k is specified
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv, head_dim, head_dim);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                       get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                       1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q, static_cast<int>(weights.stride(0)), 0);

    // Calculate shared memory size
    const int smem_size = sizeof(layout::MQALogitsSharedStorage<false, 32, 128, 4, 256, 3, 5, 3>);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SM100LongContextMQALogitsRuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .enable_filter_v2 = false,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .filter_thresholds = nullptr,
        .candidate_scores = nullptr,
        .candidate_indices = nullptr,
        .candidate_counts = nullptr,
        .candidate_overflow = nullptr,
        .candidate_capacity = 0,
        .q_block_starts =
            q_block_starts == nullptr ? nullptr : q_block_starts->data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100LongContextMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_long_context_mqa_logits", code);
    SM100LongContextMQALogitsRuntime::launch(runtime, args);
}

static void sm100_fp8_mqa_filter_candidates(
        const torch::Tensor& q,
        const torch::Tensor& kv,
        const torch::Tensor& kv_scales,
        const torch::Tensor& weights,
        const torch::Tensor& cu_seq_len_k_start,
        const torch::Tensor& cu_seq_len_k_end,
        const torch::Tensor& thresholds,
        torch::Tensor& candidate_scores,
        torch::Tensor& candidate_indices,
        torch::Tensor& candidate_counts,
        torch::Tensor& candidate_overflow,
        const int& candidate_capacity) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_math_threads = 256;
    constexpr int num_q_stages = 3;
    constexpr int num_kv_stages = 5;
    constexpr int block_q = 4;
    constexpr int block_kv = 256;
    constexpr int num_producer_warps = num_math_threads / 32;

    const auto [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto [seq_len_kv, kv_head_dim] = get_shape<2>(kv);
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(num_heads == 32 and head_dim == 128);
    DG_HOST_ASSERT(kv_head_dim == head_dim);
    if (seq_len == 0)
        return;
    DG_HOST_ASSERT(candidate_capacity > 0);
    DG_HOST_ASSERT(candidate_capacity % 32 == 0);
    if (seq_len_kv == 0) {
        candidate_counts.zero_();
        candidate_overflow.zero_();
        return;
    }
    DG_HOST_ASSERT(
        static_cast<uint64_t>(seq_len) * num_producer_warps *
            candidate_capacity <=
        0xffffffffull);

    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, seq_len * num_heads,
        head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, head_dim, seq_len_kv,
        head_dim, block_kv, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales,
        get_tma_aligned_size(
            seq_len_kv, static_cast<int>(kv_scales.element_size())),
        1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, seq_len,
        num_heads, block_q, static_cast<int>(weights.stride(0)), 0);

    const int smem_size = sizeof(layout::MQALogitsSharedStorage<false, 32, 128, 4, 256, 3, 5, 3>);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    const SM100LongContextMQALogitsRuntime::Args args = {
        .seq_len = static_cast<int>(seq_len),
        .seq_len_kv = static_cast<int>(seq_len_kv),
        .stride_logits = 0,
        .num_heads = static_cast<int>(num_heads),
        .head_dim = static_cast<int>(head_dim),
        .is_compressed_logits = false,
        .enable_filter_v2 = true,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = nullptr,
        .filter_thresholds = thresholds.data_ptr(),
        .candidate_scores = candidate_scores.data_ptr(),
        .candidate_indices = candidate_indices.data_ptr<int>(),
        .candidate_counts = candidate_counts.data_ptr<int>(),
        .candidate_overflow = candidate_overflow.data_ptr<int>(),
        .candidate_capacity = candidate_capacity,
        .q_block_starts = nullptr,
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = torch::kFloat,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(
            device_runtime->get_num_sms(),
            num_specialized_threads + num_math_threads,
            smem_size)
    };
    const auto code = SM100LongContextMQALogitsRuntime::generate(args);
    const auto runtime =
        compiler->build("sm100_fp8_mqa_filter_candidates", code);
    SM100LongContextMQALogitsRuntime::launch(runtime, args);
}


} // namespace deep_gemm
