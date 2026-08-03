#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace deep_gemm {

class SMXXMQAFinalTopKRuntime final : public LaunchRuntime<SMXXMQAFinalTopKRuntime> {
public:
    struct Args {
        const float* prefix_scores;
        const int32_t* prefix_indices;
        const float* candidate_scores;
        const int32_t* candidate_indices;
        const int32_t* candidate_counts;
        const int32_t* page_table;
        const int32_t* cu_seqlens_q;
        const int32_t* transform_offsets;
        float* output_scores;
        int32_t* output_indices;
        int32_t candidate_capacity;
        int64_t page_table_stride;
        int32_t prefill_bs;
        bool transform_indices;
        bool use_transform_offsets;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace deep_gemm {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr int kNumWarps = 8;
constexpr int kWarpSize = 32;
constexpr int kRadix = 256;
constexpr int kSmemInputSize = 4096;
constexpr float kNegInf = -3.4028234663852886e38f;

__device__ __forceinline__ uint32_t score_to_key(float x) {
    const uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint8_t score_to_coarse_bin(float x) {
    const __half h = __float2half_rn(x);
    const uint16_t bits = __half_as_ushort(h);
    const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                                          : static_cast<uint16_t>(bits | 0x8000u);
    return static_cast<uint8_t>(key >> 8);
}

__global__ __launch_bounds__(kThreadsPerBlock)
void smxx_mqa_final_topk(
    const float* __restrict__ prefix_scores,
    const int32_t* __restrict__ prefix_indices,
    const float* __restrict__ candidate_scores,
    const int32_t* __restrict__ candidate_indices,
    const int32_t* __restrict__ candidate_counts,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ transform_offsets,
    float* __restrict__ output_scores,
    int32_t* __restrict__ output_indices,
    int candidate_capacity,
    int64_t page_table_stride,
    int prefill_bs,
    bool transform_indices,
    bool use_transform_offsets) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_capacity = candidate_capacity / kWarpSize;

    const float* prefix_score_row = prefix_scores + static_cast<uint64_t>(row) * TopK;
    const int32_t* prefix_index_row = prefix_indices + static_cast<uint64_t>(row) * TopK;
    const float* candidate_score_row =
        candidate_scores + static_cast<uint64_t>(row) * kNumWarps * candidate_capacity;
    const int32_t* candidate_index_row =
        candidate_indices + static_cast<uint64_t>(row) * kNumWarps * candidate_capacity;
    const int32_t* count_row =
        candidate_counts + static_cast<uint64_t>(row) * kNumWarps * kWarpSize;
    float* output_score_row = output_scores + static_cast<uint64_t>(row) * TopK;
    int32_t* output_index_row = output_indices + static_cast<uint64_t>(row) * TopK;

    alignas(128) __shared__ int s_histogram_buf[2][kRadix + 128];
    alignas(128) __shared__ int s_indices[TopK];
    alignas(128) __shared__ int s_input_idx[2][kSmemInputSize];
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_threshold_bin_id;
    alignas(128) __shared__ int s_num_input[2];
    alignas(128) __shared__ int s_last_remain;
    __shared__ const int32_t* s_page_entry;
    __shared__ int32_t s_transform_offset;

    auto& hist = s_histogram_buf[0];

    if (tid == 0) {
        s_page_entry = page_table;
        s_transform_offset = 0;
    }
    __syncthreads();
    if (prefill_bs <= kThreadsPerBlock) {
        if (tid < prefill_bs && row >= cu_seqlens_q[tid] && row < cu_seqlens_q[tid + 1]) {
            if (use_transform_offsets)
                s_transform_offset = transform_offsets[tid];
            else
                s_page_entry = page_table + static_cast<int64_t>(tid) * page_table_stride;
        }
    } else {
        for (int i = tid; i < prefill_bs; i += kThreadsPerBlock) {
            if (row >= cu_seqlens_q[i] && row < cu_seqlens_q[i + 1]) {
                if (use_transform_offsets)
                    s_transform_offset = transform_offsets[i];
                else
                    s_page_entry = page_table + static_cast<int64_t>(i) * page_table_stride;
            }
        }
    }
    __syncthreads();
    const int32_t* page_entry = s_page_entry;

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            if (tid < kRadix) {
                const int delta = 1 << i;
                const int src_idx = i & 1;
                int value = s_histogram_buf[src_idx][tid];
                if (tid + delta < kRadix)
                    value += s_histogram_buf[src_idx][tid + delta];
                s_histogram_buf[src_idx ^ 1][tid] = value;
            }
            __syncthreads();
        }
    };

    const auto load_score_by_id = [&] (int id) {
        if (id < TopK)
            return prefix_score_row[id];
        return candidate_score_row[id - TopK];
    };

    const auto load_index_by_id = [&] (int id) {
        if (id < TopK)
            return prefix_index_row[id];
        return candidate_index_row[id - TopK];
    };

    const int consumer_warp = tid / kWarpSize;
    const int consumer_lane = tid - consumer_warp * kWarpSize;

    for (int i = tid; i < TopK; i += kThreadsPerBlock)
        s_indices[i] = -1;
    if (tid < kRadix + 1)
        hist[tid] = 0;
    __syncthreads();

    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
        if (prefix_index_row[i] >= 0)
            atomicAdd(&hist[score_to_coarse_bin(prefix_score_row[i])], 1);
    }
    for (int producer_group = 0; producer_group < 8; ++producer_group) {
        const int producer_linear = producer_group * 32 + consumer_warp;
        const int producer_warp = producer_linear / kWarpSize;
        const int producer_lane = producer_linear - producer_warp * kWarpSize;
        const int count = count_row[producer_warp * kWarpSize + producer_lane];
        const int candidate_base =
            producer_warp * candidate_capacity + producer_lane * lane_capacity;
        for (int lane_slot = consumer_lane; lane_slot < count; lane_slot += kWarpSize) {
            const int offset = candidate_base + lane_slot;
            if (candidate_index_row[offset] >= 0)
                atomicAdd(&hist[score_to_coarse_bin(candidate_score_row[offset])], 1);
        }
    }
    __syncthreads();

    run_cumsum();
    if (tid < kRadix && hist[tid] > TopK && hist[tid + 1] <= TopK) {
        s_threshold_bin_id = tid;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    int topk = TopK;
    int threshold_bin = s_threshold_bin_id;
    topk -= hist[threshold_bin + 1];
    if (topk == 0) {
        for (int i = tid; i < TopK; i += kThreadsPerBlock) {
            if (prefix_index_row[i] >= 0 &&
                static_cast<int>(score_to_coarse_bin(prefix_score_row[i])) > threshold_bin) {
                const int pos = atomicAdd(&s_counter, 1);
                if (pos < TopK)
                    s_indices[pos] = i;
            }
        }
        for (int producer_group = 0; producer_group < 8; ++producer_group) {
            const int producer_linear = producer_group * 32 + consumer_warp;
            const int producer_warp = producer_linear / kWarpSize;
            const int producer_lane = producer_linear - producer_warp * kWarpSize;
            const int count = count_row[producer_warp * kWarpSize + producer_lane];
            const int candidate_base =
                producer_warp * candidate_capacity + producer_lane * lane_capacity;
            for (int lane_slot = consumer_lane; lane_slot < count; lane_slot += kWarpSize) {
                const int offset = candidate_base + lane_slot;
                const int id = TopK + offset;
                if (candidate_index_row[offset] >= 0 &&
                    static_cast<int>(score_to_coarse_bin(candidate_score_row[offset])) > threshold_bin) {
                    const int pos = atomicAdd(&s_counter, 1);
                    if (pos < TopK)
                        s_indices[pos] = id;
                }
            }
        }
        __syncthreads();
    } else {
        __syncthreads();
        if (tid < kRadix + 1)
            hist[tid] = 0;
        __syncthreads();

        for (int i = tid; i < TopK; i += kThreadsPerBlock) {
            if (prefix_index_row[i] < 0)
                continue;
            const int bin = static_cast<int>(score_to_coarse_bin(prefix_score_row[i]));
            if (bin > threshold_bin) {
                const int pos = atomicAdd(&s_counter, 1);
                if (pos < TopK)
                    s_indices[pos] = i;
            } else if (bin == threshold_bin) {
                const int pos = atomicAdd(&s_num_input[0], 1);
                if (pos < kSmemInputSize) {
                    s_input_idx[0][pos] = i;
                    atomicAdd(&hist[(score_to_key(prefix_score_row[i]) >> 24) & 0xff], 1);
                }
            }
        }
        for (int producer_group = 0; producer_group < 8; ++producer_group) {
            const int producer_linear = producer_group * 32 + consumer_warp;
            const int producer_warp = producer_linear / kWarpSize;
            const int producer_lane = producer_linear - producer_warp * kWarpSize;
            const int count = count_row[producer_warp * kWarpSize + producer_lane];
            const int candidate_base =
                producer_warp * candidate_capacity + producer_lane * lane_capacity;
            for (int lane_slot = consumer_lane; lane_slot < count; lane_slot += kWarpSize) {
                const int offset = candidate_base + lane_slot;
                const int id = TopK + offset;
                if (candidate_index_row[offset] < 0)
                    continue;
                const float score = candidate_score_row[offset];
                const int bin = static_cast<int>(score_to_coarse_bin(score));
                if (bin > threshold_bin) {
                    const int pos = atomicAdd(&s_counter, 1);
                    if (pos < TopK)
                        s_indices[pos] = id;
                } else if (bin == threshold_bin) {
                    const int pos = atomicAdd(&s_num_input[0], 1);
                    if (pos < kSmemInputSize) {
                        s_input_idx[0][pos] = id;
                        atomicAdd(&hist[(score_to_key(score) >> 24) & 0xff], 1);
                    }
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int round = 0; round < 4; ++round) {
            const int r_idx = round & 1;
            const int raw_num_input = s_num_input[r_idx];
            const int num_input = raw_num_input < kSmemInputSize ? raw_num_input : kSmemInputSize;

            run_cumsum();
            if (tid < kRadix && hist[tid] > topk && hist[tid + 1] <= topk) {
                s_threshold_bin_id = tid;
                s_num_input[r_idx ^ 1] = 0;
                s_last_remain = topk - hist[tid + 1];
            }
            __syncthreads();

            threshold_bin = s_threshold_bin_id;
            topk -= hist[threshold_bin + 1];
            const int shift = 24 - round * 8;

            if (topk == 0) {
                for (int i = tid; i < num_input; i += kThreadsPerBlock) {
                    const int id = s_input_idx[r_idx][i];
                    const int bin = (score_to_key(load_score_by_id(id)) >> shift) & 0xff;
                    if (bin > threshold_bin) {
                        const int pos = atomicAdd(&s_counter, 1);
                        if (pos < TopK)
                            s_indices[pos] = id;
                    }
                }
                __syncthreads();
                break;
            }

            __syncthreads();
            if (tid < kRadix + 1)
                hist[tid] = 0;
            __syncthreads();
            for (int i = tid; i < num_input; i += kThreadsPerBlock) {
                const int id = s_input_idx[r_idx][i];
                const uint32_t key = score_to_key(load_score_by_id(id));
                const int bin = (key >> shift) & 0xff;
                if (bin > threshold_bin) {
                    const int pos = atomicAdd(&s_counter, 1);
                    if (pos < TopK)
                        s_indices[pos] = id;
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const int pos = atomicAdd(&s_last_remain, -1);
                        if (pos > 0 && pos <= TopK)
                            s_indices[TopK - pos] = id;
                    } else {
                        const int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                        if (pos < kSmemInputSize) {
                            s_input_idx[r_idx ^ 1][pos] = id;
                            atomicAdd(&hist[(key >> (shift - 8)) & 0xff], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
        const int id = s_indices[i];
        if (id >= 0) {
            const float score = load_score_by_id(id);
            const int32_t index = load_index_by_id(id);
            output_score_row[i] = score;
            output_index_row[i] =
                index < 0 ? -1 :
                (!transform_indices ? index :
                 (use_transform_offsets ? index + s_transform_offset
                                        : page_entry[index]));
        } else {
            output_score_row[i] = kNegInf;
            output_index_row[i] = -1;
        }
    }
}

} // namespace deep_gemm
)";
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        auto prefix_scores = const_cast<float*>(args.prefix_scores);
        auto prefix_indices = const_cast<int32_t*>(args.prefix_indices);
        auto candidate_scores = const_cast<float*>(args.candidate_scores);
        auto candidate_indices = const_cast<int32_t*>(args.candidate_indices);
        auto candidate_counts = const_cast<int32_t*>(args.candidate_counts);
        auto page_table = const_cast<int32_t*>(args.page_table);
        auto cu_seqlens_q = const_cast<int32_t*>(args.cu_seqlens_q);
        auto transform_offsets =
            const_cast<int32_t*>(args.transform_offsets);
        auto output_scores = args.output_scores;
        auto output_indices = args.output_indices;
        auto candidate_capacity = args.candidate_capacity;
        auto page_table_stride = args.page_table_stride;
        auto prefill_bs = args.prefill_bs;
        auto transform_indices = args.transform_indices;
        auto use_transform_offsets = args.use_transform_offsets;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            prefix_scores,
            prefix_indices,
            candidate_scores,
            candidate_indices,
            candidate_counts,
            page_table,
            cu_seqlens_q,
            transform_offsets,
            output_scores,
            output_indices,
            candidate_capacity,
            page_table_stride,
            prefill_bs,
            transform_indices,
            use_transform_offsets));
    }
};

static void smxx_mqa_final_topk(
        const torch::Tensor& prefix_scores,
        const torch::Tensor& prefix_indices,
        const torch::Tensor& candidate_scores,
        const torch::Tensor& candidate_indices,
        const torch::Tensor& candidate_counts,
        const torch::Tensor& page_table,
        const torch::Tensor& cu_seqlens_q,
        const torch::Tensor& transform_offsets,
        const bool& transform_indices,
        torch::Tensor& output_scores,
        torch::Tensor& output_indices) {
    DG_HOST_ASSERT(prefix_scores.is_cuda() and prefix_indices.is_cuda());
    DG_HOST_ASSERT(candidate_scores.is_cuda() and candidate_indices.is_cuda());
    DG_HOST_ASSERT(candidate_counts.is_cuda() and output_scores.is_cuda());
    DG_HOST_ASSERT(page_table.is_cuda() and cu_seqlens_q.is_cuda());
    DG_HOST_ASSERT(transform_offsets.is_cuda());
    DG_HOST_ASSERT(output_indices.is_cuda());
    DG_HOST_ASSERT(prefix_scores.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(candidate_scores.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(prefix_indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(candidate_indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(candidate_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(page_table.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seqlens_q.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(transform_offsets.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(output_scores.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(output_indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(prefix_scores.dim() == 2 and prefix_indices.dim() == 2);
    DG_HOST_ASSERT(candidate_scores.dim() == 3 and candidate_indices.dim() == 3);
    DG_HOST_ASSERT(candidate_counts.dim() == 3);
    DG_HOST_ASSERT(page_table.dim() == 2);
    DG_HOST_ASSERT(cu_seqlens_q.dim() == 1);
    DG_HOST_ASSERT(transform_offsets.dim() == 1);
    DG_HOST_ASSERT(output_scores.dim() == 2 and output_indices.dim() == 2);
    DG_HOST_ASSERT(prefix_scores.size(1) == 2048);
    DG_HOST_ASSERT(prefix_indices.sizes() == prefix_scores.sizes());
    DG_HOST_ASSERT(output_scores.sizes() == prefix_scores.sizes());
    DG_HOST_ASSERT(output_indices.sizes() == prefix_scores.sizes());
    DG_HOST_ASSERT(candidate_scores.size(0) == prefix_scores.size(0));
    DG_HOST_ASSERT(candidate_scores.size(1) == 8);
    DG_HOST_ASSERT(candidate_indices.sizes() == candidate_scores.sizes());
    DG_HOST_ASSERT(candidate_counts.size(0) == prefix_scores.size(0));
    DG_HOST_ASSERT(candidate_counts.size(1) == 8 and candidate_counts.size(2) == 32);
    DG_HOST_ASSERT(candidate_scores.size(2) % 32 == 0);
    DG_HOST_ASSERT(cu_seqlens_q.size(0) >= 2);
    DG_HOST_ASSERT(cu_seqlens_q.size(0) == page_table.size(0) + 1);
    DG_HOST_ASSERT(
        transform_offsets.numel() == 0 or
        transform_offsets.numel() == page_table.size(0));
    DG_HOST_ASSERT(prefix_scores.is_contiguous() and prefix_indices.is_contiguous());
    DG_HOST_ASSERT(candidate_scores.is_contiguous() and candidate_indices.is_contiguous());
    DG_HOST_ASSERT(candidate_counts.is_contiguous());
    DG_HOST_ASSERT(page_table.is_contiguous() and cu_seqlens_q.is_contiguous());
    DG_HOST_ASSERT(transform_offsets.is_contiguous());
    DG_HOST_ASSERT(output_scores.is_contiguous() and output_indices.is_contiguous());

    constexpr int threads = 1024;
    const SMXXMQAFinalTopKRuntime::Args args = {
        .prefix_scores = prefix_scores.data_ptr<float>(),
        .prefix_indices = prefix_indices.data_ptr<int32_t>(),
        .candidate_scores = candidate_scores.data_ptr<float>(),
        .candidate_indices = candidate_indices.data_ptr<int32_t>(),
        .candidate_counts = candidate_counts.data_ptr<int32_t>(),
        .page_table = page_table.data_ptr<int32_t>(),
        .cu_seqlens_q = cu_seqlens_q.data_ptr<int32_t>(),
        .transform_offsets = transform_offsets.data_ptr<int32_t>(),
        .output_scores = output_scores.data_ptr<float>(),
        .output_indices = output_indices.data_ptr<int32_t>(),
        .candidate_capacity = static_cast<int32_t>(candidate_scores.size(2)),
        .page_table_stride = page_table.stride(0),
        .prefill_bs = static_cast<int32_t>(cu_seqlens_q.size(0) - 1),
        .transform_indices = transform_indices,
        .use_transform_offsets = transform_offsets.numel() != 0,
        .launch_args = LaunchArgs(static_cast<int>(prefix_scores.size(0)), threads, 0)
    };
    const auto code = SMXXMQAFinalTopKRuntime::generate(args);
    const auto runtime = compiler->build("smxx_mqa_final_topk", code);
    SMXXMQAFinalTopKRuntime::launch(runtime, args);
}

} // namespace deep_gemm
