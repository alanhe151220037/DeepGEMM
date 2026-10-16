#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace deep_gemm {

class SMXXMQABackwardTopKRuntime final : public LaunchRuntime<SMXXMQABackwardTopKRuntime> {
public:
    struct Args {
        const void* logits;
        const int32_t* lengths;
        const int32_t* row_starts;
        const int32_t* window_starts;
        void* scores;
        int32_t* indices;
        uint32_t stride_logits;
        const bool* q_mask;
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
constexpr int kRadix = 256;
constexpr int kSmem = 8 * 1024 * sizeof(uint32_t);

__device__ __forceinline__ uint8_t load_coarse_bin(const float* input, int idx) {
    const __half h = __float2half_rn(input[idx]);
    const uint16_t bits = __half_as_ushort(h);
    const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                                          : static_cast<uint16_t>(bits | 0x8000u);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t load_order_key(const float* input, int idx) {
    const float x = input[idx];
    const unsigned int bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void naive_topk_gather(
    const float* __restrict__ score,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_indices,
    int row_start,
    int window_start,
    int length) {
    const int tx = threadIdx.x;
    for (int i = tx; i < TopK; i += kThreadsPerBlock) {
        if (i < length) {
            const int local_idx = i;
            out_scores[i] = score[row_start + local_idx];
            out_indices[i] = window_start + local_idx;
        } else {
            out_scores[i] = -3.4028234663852886e38f;
            out_indices[i] = -1;
        }
    }
}

__device__ void fast_topk_select_float(
    const float* __restrict__ input,
    int* __restrict__ index,
    int* __restrict__ radix_storage,
    int row_start,
    int length) {
    int topk = TopK;
    constexpr int SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

    alignas(128) __shared__ int s_histogram_buf[2][kRadix + 128];
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_threshold_bin_id;
    alignas(128) __shared__ int s_num_input[2];

    auto& s_histogram = s_histogram_buf[0];
    auto s_input_idx = reinterpret_cast<int (*)[SMEM_INPUT_SIZE]>(radix_storage);
    const int tx = threadIdx.x;

    for (int i = tx; i < TopK; i += kThreadsPerBlock) index[i] = -1;
    if (tx < kRadix + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += kThreadsPerBlock) {
        const auto bin = load_coarse_bin(input, idx + row_start);
        atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            if (tx < kRadix) {
                const int j = 1 << i;
                const int k = i & 1;
                int value = s_histogram_buf[k][tx];
                if (tx < kRadix - j) value += s_histogram_buf[k][tx + j];
                s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum();
    if (tx < kRadix && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    int threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];
    if (topk == 0) {
        for (int idx = tx; idx < length; idx += kThreadsPerBlock) {
            const auto bin = static_cast<int>(load_coarse_bin(input, idx + row_start));
            if (bin > threshold_bin) {
                const int pos = atomicAdd(&s_counter, 1);
                if (pos < TopK) index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    }

    __syncthreads();
    if (tx < kRadix + 1) s_histogram[tx] = 0;
    __syncthreads();
    for (int idx = tx; idx < length; idx += kThreadsPerBlock) {
        const auto bin = static_cast<int>(load_coarse_bin(input, idx + row_start));
        if (bin > threshold_bin) {
            const int pos = atomicAdd(&s_counter, 1);
            if (pos < TopK) index[pos] = idx;
        } else if (bin == threshold_bin) {
            const int pos = atomicAdd(&s_num_input[0], 1);
            if (pos < SMEM_INPUT_SIZE) {
                s_input_idx[0][pos] = idx;
                const auto key = load_order_key(input, idx + row_start);
                const auto sub_bin = (key >> 24) & 0xff;
                atomicAdd(&s_histogram[sub_bin], 1);
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int round = 0; round < 4; ++round) {
        __shared__ int s_last_remain;
        const int r_idx = round & 1;
        const int raw_num_input = s_num_input[r_idx];
        const int num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

        run_cumsum();
        if (tx < kRadix && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
            s_threshold_bin_id = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = topk - s_histogram[tx + 1];
        }
        __syncthreads();

        threshold_bin = s_threshold_bin_id;
        topk -= s_histogram[threshold_bin + 1];
        const int offset = 24 - round * 8;

        if (topk == 0) {
            for (int i = tx; i < num_input; i += kThreadsPerBlock) {
                const int idx = s_input_idx[r_idx][i];
                const auto bin = (load_order_key(input, idx + row_start) >> offset) & 0xff;
                if (bin > threshold_bin) {
                    const int pos = atomicAdd(&s_counter, 1);
                    if (pos < TopK) index[pos] = idx;
                }
            }
            __syncthreads();
            break;
        }

        __syncthreads();
        if (tx < kRadix + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += kThreadsPerBlock) {
            const int idx = s_input_idx[r_idx][i];
            const auto bin = (load_order_key(input, idx + row_start) >> offset) & 0xff;
            if (bin > threshold_bin) {
                const int pos = atomicAdd(&s_counter, 1);
                if (pos < TopK) index[pos] = idx;
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    const int pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0 && pos <= TopK) index[TopK - pos] = idx;
                } else {
                    const int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    if (pos < SMEM_INPUT_SIZE) {
                        s_input_idx[r_idx ^ 1][pos] = idx;
                        const auto key = load_order_key(input, idx + row_start);
                        const auto sub_bin = (key >> (offset - 8)) & 0xff;
                        atomicAdd(&s_histogram[sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void smxx_mqa_backward_topk(
    const float* __restrict__ logits,
    const int32_t* __restrict__ lengths,
    const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ window_starts,
    float* __restrict__ scores,
    int32_t* __restrict__ indices,
    uint32_t stride_logits, const bool* __restrict__ q_mask) {
    const uint32_t row = blockIdx.x;
    if (q_mask != nullptr and not q_mask[row]) {
        for (int i = threadIdx.x; i < TopK; i += kThreadsPerBlock) {
            scores[row * TopK + i] = -3.4028234663852886e38f;
            indices[row * TopK + i] = -1;
        }
        return;
    }
    const int length = lengths[row];
    const int row_start = row_starts[row];
    const int window_start = window_starts[row];
    const float* row_logits = logits + row * stride_logits;
    float* row_scores = scores + row * TopK;
    int32_t* row_indices = indices + row * TopK;

    extern __shared__ int s_storage[];
    int* s_indices = s_storage;
    int* s_radix_storage = s_storage + TopK;

    if (length <= TopK) {
        naive_topk_gather(row_logits, row_scores, row_indices, row_start, window_start, length);
        return;
    }

    fast_topk_select_float(row_logits, s_indices, s_radix_storage, row_start, length);
    __syncthreads();
    const int tx = threadIdx.x;
    for (int i = tx; i < TopK; i += kThreadsPerBlock) {
        const int local_idx = s_indices[i];
        if (local_idx >= 0) {
            row_scores[i] = row_logits[row_start + local_idx];
            row_indices[i] = window_start + local_idx;
        } else {
            row_scores[i] = -3.4028234663852886e38f;
            row_indices[i] = -1;
        }
    }
}

} // namespace deep_gemm
)";
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        auto logits = const_cast<void*>(args.logits);
        auto lengths = const_cast<int32_t*>(args.lengths);
        auto row_starts = const_cast<int32_t*>(args.row_starts);
        auto window_starts = const_cast<int32_t*>(args.window_starts);
        auto scores = args.scores;
        auto indices = args.indices;
        auto stride_logits = args.stride_logits;
        auto q_mask = args.q_mask;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            logits,
            lengths,
            row_starts,
            window_starts,
            scores,
            indices,
            stride_logits, q_mask));
    }
};

static void smxx_mqa_backward_topk(
        const torch::Tensor& logits,
        const torch::Tensor& lengths,
        const torch::Tensor& row_starts,
        const torch::Tensor& window_starts,
        torch::Tensor& scores,
        torch::Tensor& indices,
        const std::optional<torch::Tensor>& q_mask = std::nullopt) {
    DG_HOST_ASSERT(logits.is_cuda() and lengths.is_cuda() and row_starts.is_cuda());
    DG_HOST_ASSERT(window_starts.is_cuda() and scores.is_cuda() and indices.is_cuda());
    DG_HOST_ASSERT(logits.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(lengths.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(row_starts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(window_starts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(scores.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(logits.dim() == 2 and scores.dim() == 2 and indices.dim() == 2);
    DG_HOST_ASSERT(lengths.numel() == logits.size(0));
    DG_HOST_ASSERT(row_starts.numel() == logits.size(0));
    DG_HOST_ASSERT(window_starts.numel() == logits.size(0));
    DG_HOST_ASSERT(scores.size(0) == logits.size(0) and scores.size(1) == 2048);
    DG_HOST_ASSERT(indices.size(0) == logits.size(0) and indices.size(1) == 2048);
    DG_HOST_ASSERT(logits.stride(1) == 1 and logits.stride(0) >= logits.size(1));
    DG_HOST_ASSERT(lengths.is_contiguous());
    DG_HOST_ASSERT(row_starts.is_contiguous() and window_starts.is_contiguous());
    DG_HOST_ASSERT(scores.is_contiguous() and indices.is_contiguous());

    if (logits.size(0) == 0)
        return;

    constexpr int threads = 1024;
    constexpr int dynamic_smem = 2048 * sizeof(int32_t) + 8 * 1024 * sizeof(uint32_t);
    const SMXXMQABackwardTopKRuntime::Args args = {
        .logits = logits.data_ptr(),
        .lengths = lengths.data_ptr<int32_t>(),
        .row_starts = row_starts.data_ptr<int32_t>(),
        .window_starts = window_starts.data_ptr<int32_t>(),
        .scores = scores.data_ptr(),
        .indices = indices.data_ptr<int32_t>(),
        .stride_logits = static_cast<uint32_t>(logits.stride(0)),
        .q_mask = q_mask.has_value() ? q_mask->data_ptr<bool>() : nullptr,
        .launch_args = LaunchArgs(static_cast<int>(logits.size(0)), threads, dynamic_smem)
    };
    const auto code = SMXXMQABackwardTopKRuntime::generate(args);
    const auto runtime = compiler->build("smxx_mqa_backward_topk", code);
    SMXXMQABackwardTopKRuntime::launch(runtime, args);
}

} // namespace deep_gemm
