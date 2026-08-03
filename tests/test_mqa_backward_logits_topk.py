import argparse
import math

import torch

import deep_gemm


LOGITS_BYTES = 4


def ceil_div(a, b):
    return (a + b - 1) // b


def compute_full_logits_rows(num_q, num_k, free_mem_bytes):
    bytes_per_row = num_k * LOGITS_BYTES
    return min(
        num_q,
        max(1, int((free_mem_bytes * 0.5) // max(bytes_per_row, 1))),
    )


def compare_topk_sets(lhs, rhs):
    lhs_sorted = torch.sort(lhs.to(torch.int32), dim=1).values
    rhs_sorted = torch.sort(rhs.to(torch.int32), dim=1).values
    pos = torch.searchsorted(rhs_sorted, lhs_sorted)
    pos = pos.clamp(max=rhs_sorted.shape[1] - 1)
    matched = rhs_sorted.gather(1, pos) == lhs_sorted
    matched &= lhs_sorted >= 0
    lhs_valid = lhs_sorted >= 0
    per_row = lhs_valid.sum(dim=1) - matched.sum(dim=1)
    mismatched = per_row > 0
    values = per_row[mismatched]
    p50, p99 = _quantiles(values, qs=(0.5, 0.99))
    return {
        "rows": int(mismatched.sum().item()),
        "total_rows": lhs.shape[0],
        "replaced": int(values.sum().item()),
        "p50": p50,
        "p99": p99,
    }


def selected_ranks(logits, selected_indices, top_scores, rank_depth):
    valid = selected_indices >= 0
    safe_indices = selected_indices.clamp(min=0).to(torch.int64)
    selected_scores = torch.gather(logits, 1, safe_indices)
    ranks = torch.searchsorted(
        (-top_scores).contiguous(),
        (-selected_scores).contiguous(),
        right=False,
    ).to(torch.int32)
    ranks.add_(1)
    ranks.masked_fill_(~valid, 0)
    ranks.masked_fill_(ranks > rank_depth, rank_depth + 1)
    return ranks


def rank_percentiles(ranks, rank_depth, topk):
    ranks = torch.sort(ranks[ranks > 0]).values
    if ranks.numel() == 0:
        return {
            "p50": "n/a",
            "p95": "n/a",
            "p99": "n/a",
            "outside_topk": 0,
            "censored": 0,
            "count": 0,
        }

    def nearest_rank(percent):
        index = ceil_div(percent * ranks.numel(), 100) - 1
        value = int(ranks[index].item())
        return f">{rank_depth}" if value > rank_depth else str(value)

    return {
        "p50": nearest_rank(50),
        "p95": nearest_rank(95),
        "p99": nearest_rank(99),
        "outside_topk": int((ranks > topk).sum().item()),
        "censored": int((ranks > rank_depth).sum().item()),
        "count": ranks.numel(),
    }


def print_rank_stats(label, stats, rank_depth, topk):
    print(
        f"{label}: P50={stats['p50']}, P95={stats['p95']}, "
        f"P99={stats['p99']}, >{topk}="
        f"{stats['outside_topk']:,}/{stats['count']:,}, "
        f">{rank_depth}="
        f"{stats['censored']:,}/{stats['count']:,}"
    )


def build_ranges(requests, device):
    ks_parts = []
    ke_parts = []
    k_base = 0
    for seq_len, extend_len in requests:
        ks_parts.append(
            torch.full((extend_len,), k_base, dtype=torch.int32, device=device)
        )
        lengths = torch.arange(
            seq_len - extend_len + 1,
            seq_len + 1,
            dtype=torch.int32,
            device=device,
        )
        ke_parts.append(lengths + k_base)
        k_base += seq_len
    return torch.cat(ks_parts), torch.cat(ke_parts), k_base


def _quantiles(values, qs=(0.5, 0.95, 0.99)):
    if values.numel() == 0:
        return [float("nan")] * len(qs)
    return [float(torch.quantile(values.float(), q).item()) for q in qs]


def compute_torch_threshold_stats(
    logits, custom_thresholds, row_starts, lengths, topk, rows_per_chunk
):
    q_count, width = logits.shape
    torch_thresholds = torch.empty(
        q_count, device=logits.device, dtype=torch.float32
    )
    custom_ranks = torch.empty(q_count, device=logits.device, dtype=torch.int32)
    torch_ranks = torch.empty(q_count, device=logits.device, dtype=torch.int32)
    columns = torch.arange(width, device=logits.device)
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for row_start in range(0, q_count, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, q_count)
        rows = slice(row_start, row_end)
        valid = (
            (columns.unsqueeze(0) >= row_starts[rows].to(torch.int64).unsqueeze(1))
            & (
                columns.unsqueeze(0)
                < (
                    row_starts[rows].to(torch.int64)
                    + lengths[rows].to(torch.int64)
                ).unsqueeze(1)
            )
        )
        clean_logits = logits[rows].masked_fill(~valid, neg_inf)
        topk_values = torch.topk(clean_logits, topk, dim=1).values
        chunk_torch_thresholds = topk_values[:, -1].to(torch.float32)
        torch_thresholds[rows] = chunk_torch_thresholds
        custom_ranks[rows] = (
            ((clean_logits > custom_thresholds[rows].unsqueeze(1)).sum(dim=1) + 1)
            .to(torch.int32)
        )
        torch_ranks[rows] = (
            ((clean_logits > chunk_torch_thresholds.unsqueeze(1)).sum(dim=1) + 1)
            .to(torch.int32)
        )
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    threshold_delta = custom_thresholds - torch_thresholds
    delta_p50, delta_p95, delta_p99 = _quantiles(threshold_delta)
    rank_p50, rank_p95, rank_p99 = _quantiles(custom_ranks)
    torch_rank_p50, torch_rank_p95, torch_rank_p99 = _quantiles(torch_ranks)
    lower_rows = int((custom_thresholds < torch_thresholds).sum().item())
    higher_rows = int((custom_thresholds > torch_thresholds).sum().item())
    equal_rows = q_count - lower_rows - higher_rows
    return {
        "elapsed_ms": elapsed_ms,
        "delta_p50": delta_p50,
        "delta_p95": delta_p95,
        "delta_p99": delta_p99,
        "rank_p50": rank_p50,
        "rank_p95": rank_p95,
        "rank_p99": rank_p99,
        "torch_rank_p50": torch_rank_p50,
        "torch_rank_p95": torch_rank_p95,
        "torch_rank_p99": torch_rank_p99,
        "lower_rows": lower_rows,
        "equal_rows": equal_rows,
        "higher_rows": higher_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward-len", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--total-seq-len", type=int, default=1024 * 1024)
    parser.add_argument("--chunk-len", type=int, default=64 * 1024)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--data-mode",
        choices=("random", "old"),
        default="random",
        help=(
            "Synthetic input distribution. random keeps this script's current "
            "randn q/k, random k_scale, random weights behavior. old matches "
            "test_nsa_fp8_mqa_topk_chunk.py: clipped bf16 q/k, k_scale=1, "
            "uniform head weights."
        ),
    )
    parser.add_argument("--v2-capacity-mult", type=float, default=3.0)
    parser.add_argument("--v2-capacity-per-warp", type=int, default=0)
    parser.add_argument(
        "--free-mem-gb",
        type=float,
        default=None,
        help=(
            "Free memory budget used to choose original full-logits Q chunks. "
            "Matches test_nsa_fp8_mqa_topk_chunk.py: use 50%% of this budget "
            "for logits workspace."
        ),
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument(
        "--profile-logits-checksum",
        action="store_true",
        help=(
            "Profile the checksum-only FP8 MQA logits kernel. This path does "
            "not produce valid logits and is not connected to topk."
        ),
    )
    parser.add_argument(
        "--profile-logits-checksum-early-release",
        action="store_true",
        help=(
            "Profile checksum-only logits with empty_umma released before "
            "math warps finish reading TMEM. Output is invalid."
        ),
    )
    parser.add_argument(
        "--profile-logits-no-epilogue",
        action="store_true",
        help=(
            "Profile the FP8 MQA logits kernel with the math epilogue removed. "
            "This path only benchmarks the TMA/UMMA/barrier pipeline."
        ),
    )
    parser.add_argument(
        "--profile-logits-early-release",
        action="store_true",
        help=(
            "Profile the original FP8 MQA logits kernel with empty_umma "
            "released before math warps finish reading TMEM. Output is invalid."
        ),
    )
    parser.add_argument(
        "--profile-logits-unsafe-decoupled",
        action="store_true",
        help=(
            "Profile an unsafe FP8 MQA logits kernel that skips both "
            "full_umma and empty_umma producer/consumer waits. Output is invalid."
        ),
    )
    parser.add_argument(
        "--original-rows-per-chunk",
        type=int,
        default=0,
        help="Manual override for original full-logits Q chunk rows; 0 means derive from --free-mem-gb.",
    )
    parser.add_argument("--skip-threshold-check", action="store_true")
    parser.add_argument("--threshold-check-rows-per-chunk", type=int, default=512)
    parser.add_argument(
        "--rank-depth",
        type=int,
        default=16384,
        help="Exact full-logits rank depth used by the torch.topk Golden path.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device, e.g. cuda or cuda:1")
    torch.cuda.set_device(device)
    if args.total_seq_len < args.chunk_len:
        raise ValueError("total_seq_len must be >= chunk_len")
    if args.chunk_len % args.cp_size != 0:
        raise ValueError("chunk_len must be divisible by cp_size")
    if not 0 <= args.cp_rank < args.cp_size:
        raise ValueError("cp_rank must be in [0, cp_size)")
    if args.rank_depth < args.topk:
        raise ValueError("rank_depth must be >= topk")
    if args.free_mem_gb is not None:
        free_mem_bytes = int(args.free_mem_gb * 1024**3)
    else:
        free_mem_bytes, _ = torch.cuda.mem_get_info(device)

    total_k = args.total_seq_len
    q_count = args.chunk_len // args.cp_size
    prefix_len = total_k - args.chunk_len
    ks = torch.zeros(q_count, dtype=torch.int32, device=device)
    # Model one CP rank as strided positions in the final extend chunk.
    ke = torch.arange(
        prefix_len + args.cp_rank + 1,
        prefix_len + args.cp_rank + 1 + q_count * args.cp_size,
        args.cp_size,
        dtype=torch.int32,
        device=device,
    )
    window_starts, q_block_starts, topk_row_starts, logits_width = (
        deep_gemm.build_mqa_backward_metadata(
            ks, ke, [q_count], backward_len=args.backward_len
        )
    )
    filter_ke = window_starts.contiguous()
    filter_kv_len = int(filter_ke.max().item()) if filter_ke.numel() != 0 else 0
    if args.v2_capacity_per_warp > 0:
        v2_capacity_per_warp = args.v2_capacity_per_warp
    else:
        expected_per_lane = (
            filter_kv_len
            * args.topk
            / max(args.backward_len, 1)
            / 8
            / 32
        )
        lane_capacity = int(math.ceil(expected_per_lane * args.v2_capacity_mult))
        lane_capacity = max(32, ((lane_capacity + 31) // 32) * 32)
        v2_capacity_per_warp = lane_capacity * 32
    v2_capacity_per_warp = max(32, ((v2_capacity_per_warp + 31) // 32) * 32)

    if args.data_mode == "old":
        q = (
            torch.randn(
                q_count, 32, 128, dtype=torch.bfloat16, device=device
            )
            .clamp(-2, 2)
            .to(torch.float8_e4m3fn)
        )
        k = (
            torch.randn(total_k, 128, dtype=torch.bfloat16, device=device)
            .clamp(-2, 2)
            .to(torch.float8_e4m3fn)
        )
        k_scale = torch.ones(total_k, device=device, dtype=torch.float32)
        weights = torch.ones(q_count, 32, device=device, dtype=torch.float32)
        weights /= math.sqrt(32 * 128)
    else:
        q = torch.randn(q_count, 32, 128, device=device).to(
            torch.float8_e4m3fn
        )
        k = torch.randn(total_k, 128, device=device).to(torch.float8_e4m3fn)
        k_scale = torch.rand(total_k, device=device, dtype=torch.float32) + 0.5
        weights = torch.randn(q_count, 32, device=device, dtype=torch.float32)
    page_table_size_1 = torch.arange(
        total_k, dtype=torch.int32, device=device
    ).unsqueeze(0).contiguous()
    cu_seqlens_q_topk = torch.tensor(
        [0, q_count], dtype=torch.int32, device=device
    )

    for _ in range(args.warmup):
        backward_logits = deep_gemm.fp8_mqa_backward_logits(
            q,
            (k, k_scale),
            weights,
            window_starts,
            ke,
            q_block_starts,
            args.backward_len,
            logits_width,
        )
        scores, global_indices = deep_gemm.fp8_mqa_backward_topk(
            backward_logits,
            ks,
            ke,
            topk_row_starts,
            backward_len=args.backward_len,
            topk=args.topk,
        )
        if filter_kv_len != 0:
            thresholds = scores.min(dim=1).values.to(torch.float32).contiguous()
            candidate_scores, candidate_indices, candidate_counts, candidate_overflow = (
                deep_gemm.fp8_mqa_filter_candidates(
                    q,
                    (k[:filter_kv_len], k_scale[:filter_kv_len]),
                    weights,
                    ks,
                    filter_ke,
                    thresholds,
                    candidate_capacity_per_warp=v2_capacity_per_warp,
                    index_base=0,
                )
            )
            final_scores, final_indices = deep_gemm.mqa_final_topk(
                scores,
                global_indices,
                candidate_scores,
                candidate_indices,
                candidate_counts,
                page_table_size_1,
                cu_seqlens_q_topk,
            )
    torch.cuda.synchronize()

    logits_start = torch.cuda.Event(enable_timing=True)
    logits_end = torch.cuda.Event(enable_timing=True)
    topk_start = torch.cuda.Event(enable_timing=True)
    topk_end = torch.cuda.Event(enable_timing=True)
    v2_start = torch.cuda.Event(enable_timing=True)
    v2_end = torch.cuda.Event(enable_timing=True)
    final_start = torch.cuda.Event(enable_timing=True)
    final_end = torch.cuda.Event(enable_timing=True)

    logits_start.record()
    for _ in range(args.iters):
        backward_logits = deep_gemm.fp8_mqa_backward_logits(
            q,
            (k, k_scale),
            weights,
            window_starts,
            ke,
            q_block_starts,
            args.backward_len,
            logits_width,
        )
    logits_end.record()
    logits_end.synchronize()
    logits_ms = logits_start.elapsed_time(logits_end) / args.iters

    topk_start.record()
    for _ in range(args.iters):
        scores, global_indices = deep_gemm.fp8_mqa_backward_topk(
            backward_logits,
            ks,
            ke,
            topk_row_starts,
            backward_len=args.backward_len,
            topk=args.topk,
        )
    topk_end.record()
    topk_end.synchronize()
    topk_ms = topk_start.elapsed_time(topk_end) / args.iters
    thresholds = scores.min(dim=1).values.to(torch.float32).contiguous()
    lengths = (ke - ks).clamp(min=0, max=args.backward_len).to(torch.int32)

    if filter_kv_len != 0:
        v2_start.record()
        for _ in range(args.iters):
            candidate_scores, candidate_indices, candidate_counts, candidate_overflow = (
                deep_gemm.fp8_mqa_filter_candidates(
                    q,
                    (k[:filter_kv_len], k_scale[:filter_kv_len]),
                    weights,
                    ks,
                    filter_ke,
                    thresholds,
                    candidate_capacity_per_warp=v2_capacity_per_warp,
                    index_base=0,
                )
            )
        v2_end.record()
        v2_end.synchronize()
        v2_ms = v2_start.elapsed_time(v2_end) / args.iters

        final_start.record()
        for _ in range(args.iters):
            final_scores, final_indices = deep_gemm.mqa_final_topk(
                scores,
                global_indices,
                candidate_scores,
                candidate_indices,
                candidate_counts,
                page_table_size_1,
                cu_seqlens_q_topk,
            )
        final_end.record()
        final_end.synchronize()
        final_ms = final_start.elapsed_time(final_end) / args.iters
    else:
        candidate_scores = candidate_indices = candidate_counts = candidate_overflow = None
        v2_ms = 0.0
        final_scores = scores
        final_indices = global_indices
        final_ms = 0.0

    original_logits_ms = 0.0
    original_topk_ms = 0.0
    original_indices = None
    original_compare = None
    golden_indices = None
    golden_compare = None
    original_golden_compare = None
    ours_ranks = None
    original_ranks = None
    golden_ranks = None
    golden_diagnostic_ms = 0.0
    original_checksum_ms = None
    original_checksum_early_release_ms = None
    original_no_epilogue_ms = None
    original_early_release_ms = None
    original_unsafe_decoupled_ms = None
    checksum_rows_per_chunk = None
    checksum_early_release_rows_per_chunk = None
    no_epilogue_rows_per_chunk = None
    early_release_rows_per_chunk = None
    unsafe_decoupled_rows_per_chunk = None

    if args.profile_logits_checksum:
        original_checksum_ms = 0.0
        checksum_rows_per_chunk = (
            compute_full_logits_rows(q_count, total_k, free_mem_bytes)
            if args.original_rows_per_chunk <= 0
            else min(args.original_rows_per_chunk, q_count)
        )
        checksum_start = torch.cuda.Event(enable_timing=True)
        checksum_end = torch.cuda.Event(enable_timing=True)

        for _ in range(args.warmup):
            for row_start in range(0, q_count, checksum_rows_per_chunk):
                row_end = min(row_start + checksum_rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                checksum_logits = deep_gemm.fp8_mqa_logits_checksum(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                )
                del checksum_logits
        torch.cuda.synchronize()

        for _ in range(args.iters):
            for row_start in range(0, q_count, checksum_rows_per_chunk):
                row_end = min(row_start + checksum_rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                checksum_start.record()
                checksum_logits = deep_gemm.fp8_mqa_logits_checksum(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                )
                checksum_end.record()
                checksum_end.synchronize()
                original_checksum_ms += checksum_start.elapsed_time(checksum_end)
                del checksum_logits
        original_checksum_ms /= args.iters

    if args.profile_logits_checksum_early_release:
        original_checksum_early_release_ms = 0.0
        checksum_early_release_rows_per_chunk = (
            compute_full_logits_rows(q_count, total_k, free_mem_bytes)
            if args.original_rows_per_chunk <= 0
            else min(args.original_rows_per_chunk, q_count)
        )
        checksum_early_start = torch.cuda.Event(enable_timing=True)
        checksum_early_end = torch.cuda.Event(enable_timing=True)

        for _ in range(args.warmup):
            for row_start in range(
                0, q_count, checksum_early_release_rows_per_chunk
            ):
                row_end = min(
                    row_start + checksum_early_release_rows_per_chunk,
                    q_count,
                )
                rows = slice(row_start, row_end)
                checksum_logits = (
                    deep_gemm.fp8_mqa_logits_checksum_early_release(
                        q[rows],
                        (k, k_scale),
                        weights[rows],
                        ks[rows],
                        ke[rows],
                    )
                )
                del checksum_logits
        torch.cuda.synchronize()

        for _ in range(args.iters):
            for row_start in range(
                0, q_count, checksum_early_release_rows_per_chunk
            ):
                row_end = min(
                    row_start + checksum_early_release_rows_per_chunk,
                    q_count,
                )
                rows = slice(row_start, row_end)
                checksum_early_start.record()
                checksum_logits = (
                    deep_gemm.fp8_mqa_logits_checksum_early_release(
                        q[rows],
                        (k, k_scale),
                        weights[rows],
                        ks[rows],
                        ke[rows],
                    )
                )
                checksum_early_end.record()
                checksum_early_end.synchronize()
                original_checksum_early_release_ms += (
                    checksum_early_start.elapsed_time(checksum_early_end)
                )
                del checksum_logits
        original_checksum_early_release_ms /= args.iters

    if args.profile_logits_no_epilogue:
        original_no_epilogue_ms = 0.0
        no_epilogue_rows_per_chunk = (
            compute_full_logits_rows(q_count, total_k, free_mem_bytes)
            if args.original_rows_per_chunk <= 0
            else min(args.original_rows_per_chunk, q_count)
        )
        no_epilogue_start = torch.cuda.Event(enable_timing=True)
        no_epilogue_end = torch.cuda.Event(enable_timing=True)

        for _ in range(args.warmup):
            for row_start in range(0, q_count, no_epilogue_rows_per_chunk):
                row_end = min(row_start + no_epilogue_rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                marker = deep_gemm.fp8_mqa_logits_no_epilogue(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                )
                del marker
        torch.cuda.synchronize()

        for _ in range(args.iters):
            for row_start in range(0, q_count, no_epilogue_rows_per_chunk):
                row_end = min(row_start + no_epilogue_rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                no_epilogue_start.record()
                marker = deep_gemm.fp8_mqa_logits_no_epilogue(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                )
                no_epilogue_end.record()
                no_epilogue_end.synchronize()
                original_no_epilogue_ms += no_epilogue_start.elapsed_time(
                    no_epilogue_end
                )
                del marker
        original_no_epilogue_ms /= args.iters

    if args.profile_logits_early_release:
        original_early_release_ms = 0.0
        early_release_rows_per_chunk = (
            compute_full_logits_rows(q_count, total_k, free_mem_bytes)
            if args.original_rows_per_chunk <= 0
            else min(args.original_rows_per_chunk, q_count)
        )
        early_release_start = torch.cuda.Event(enable_timing=True)
        early_release_end = torch.cuda.Event(enable_timing=True)

        for _ in range(args.warmup):
            for row_start in range(0, q_count, early_release_rows_per_chunk):
                row_end = min(row_start + early_release_rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                early_logits = deep_gemm.fp8_mqa_logits_early_release(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                    False,
                    0,
                )
                del early_logits
        torch.cuda.synchronize()

        for _ in range(args.iters):
            for row_start in range(0, q_count, early_release_rows_per_chunk):
                row_end = min(row_start + early_release_rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                early_release_start.record()
                early_logits = deep_gemm.fp8_mqa_logits_early_release(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                    False,
                    0,
                )
                early_release_end.record()
                early_release_end.synchronize()
                original_early_release_ms += early_release_start.elapsed_time(
                    early_release_end
                )
                del early_logits
        original_early_release_ms /= args.iters

    if args.profile_logits_unsafe_decoupled:
        original_unsafe_decoupled_ms = 0.0
        unsafe_decoupled_rows_per_chunk = (
            compute_full_logits_rows(q_count, total_k, free_mem_bytes)
            if args.original_rows_per_chunk <= 0
            else min(args.original_rows_per_chunk, q_count)
        )
        unsafe_start = torch.cuda.Event(enable_timing=True)
        unsafe_end = torch.cuda.Event(enable_timing=True)

        for _ in range(args.warmup):
            for row_start in range(0, q_count, unsafe_decoupled_rows_per_chunk):
                row_end = min(
                    row_start + unsafe_decoupled_rows_per_chunk, q_count
                )
                rows = slice(row_start, row_end)
                unsafe_logits = deep_gemm.fp8_mqa_logits_unsafe_decoupled(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                    False,
                    0,
                )
                del unsafe_logits
        torch.cuda.synchronize()

        for _ in range(args.iters):
            for row_start in range(0, q_count, unsafe_decoupled_rows_per_chunk):
                row_end = min(
                    row_start + unsafe_decoupled_rows_per_chunk, q_count
                )
                rows = slice(row_start, row_end)
                unsafe_start.record()
                unsafe_logits = deep_gemm.fp8_mqa_logits_unsafe_decoupled(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                    False,
                    0,
                )
                unsafe_end.record()
                unsafe_end.synchronize()
                original_unsafe_decoupled_ms += unsafe_start.elapsed_time(
                    unsafe_end
                )
                del unsafe_logits
        original_unsafe_decoupled_ms /= args.iters

    if not args.skip_original:
        try:
            from sgl_kernel import fast_topk_transform_fused
        except ImportError as exc:
            raise RuntimeError(
                "Original path requires sgl_kernel.fast_topk_transform_fused. "
                "Use --skip-original to disable it."
            ) from exc

        original_indices = torch.empty(
            (q_count, args.topk), dtype=torch.int32, device=device
        )
        diagnostic_original_indices = torch.empty_like(original_indices)
        golden_indices = torch.empty_like(original_indices)
        ours_ranks = torch.empty_like(original_indices)
        original_ranks = torch.empty_like(original_indices)
        golden_ranks = torch.empty_like(original_indices)
        rank_depth = min(args.rank_depth, total_k)
        full_columns = torch.arange(total_k, dtype=torch.int64, device=device)
        rows_per_chunk = (
            compute_full_logits_rows(q_count, total_k, free_mem_bytes)
            if args.original_rows_per_chunk <= 0
            else min(args.original_rows_per_chunk, q_count)
        )
        original_lengths = (ke - ks).to(torch.int32).contiguous()
        original_row_starts = ks.to(torch.int32).contiguous()
        original_cu_seqlens_q = torch.empty(2, dtype=torch.int32, device=device)
        original_cu_seqlens_q[0] = 0

        original_logits_start = torch.cuda.Event(enable_timing=True)
        original_logits_end = torch.cuda.Event(enable_timing=True)
        original_topk_start = torch.cuda.Event(enable_timing=True)
        original_topk_end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.warmup):
            for row_start in range(0, q_count, rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                original_cu_seqlens_q[1] = row_end - row_start
                full_logits = deep_gemm.fp8_mqa_logits(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                    clean_logits=False,
                    max_seqlen_k=total_k,
                )
                chunk_indices = fast_topk_transform_fused(
                    score=full_logits,
                    lengths=original_lengths[rows],
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=original_cu_seqlens_q,
                    topk=args.topk,
                    row_starts=original_row_starts[rows],
                )
                original_indices[rows] = chunk_indices
                del full_logits, chunk_indices
        torch.cuda.synchronize()

        for iteration in range(args.iters):
            for row_start in range(0, q_count, rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, q_count)
                rows = slice(row_start, row_end)
                original_cu_seqlens_q[1] = row_end - row_start

                original_logits_start.record()
                full_logits = deep_gemm.fp8_mqa_logits(
                    q[rows],
                    (k, k_scale),
                    weights[rows],
                    ks[rows],
                    ke[rows],
                    clean_logits=False,
                    max_seqlen_k=total_k,
                )
                original_logits_end.record()
                original_logits_end.synchronize()
                original_logits_ms += original_logits_start.elapsed_time(
                    original_logits_end
                )

                original_topk_start.record()
                chunk_indices = fast_topk_transform_fused(
                    score=full_logits,
                    lengths=original_lengths[rows],
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=original_cu_seqlens_q,
                    topk=args.topk,
                    row_starts=original_row_starts[rows],
                )
                original_topk_end.record()
                original_topk_end.synchronize()
                original_topk_ms += original_topk_start.elapsed_time(
                    original_topk_end
                )
                original_indices[rows] = chunk_indices
                del full_logits, chunk_indices
        original_logits_ms /= args.iters
        original_topk_ms /= args.iters

        # Correctness diagnostics are intentionally separate from the timed
        # original path. Each path gets its own fresh logits buffer because
        # fast_topk_transform_fused may use its score input as workspace.
        for row_start in range(0, q_count, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, q_count)
            rows = slice(row_start, row_end)
            original_cu_seqlens_q[1] = row_end - row_start

            sglang_logits = deep_gemm.fp8_mqa_logits(
                q[rows],
                (k, k_scale),
                weights[rows],
                ks[rows],
                ke[rows],
                clean_logits=False,
                max_seqlen_k=total_k,
            )
            diagnostic_original_indices[rows] = fast_topk_transform_fused(
                score=sglang_logits,
                lengths=original_lengths[rows],
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=original_cu_seqlens_q,
                topk=args.topk,
                row_starts=original_row_starts[rows],
            )
            del sglang_logits

            diagnostic_start = torch.cuda.Event(enable_timing=True)
            diagnostic_end = torch.cuda.Event(enable_timing=True)
            diagnostic_start.record()
            golden_logits = deep_gemm.fp8_mqa_logits(
                q[rows],
                (k, k_scale),
                weights[rows],
                ks[rows],
                ke[rows],
                clean_logits=False,
                max_seqlen_k=total_k,
            )
            valid = (
                (
                    full_columns.unsqueeze(0)
                    >= ks[rows].to(torch.int64).unsqueeze(1)
                )
                & (
                    full_columns.unsqueeze(0)
                    < ke[rows].to(torch.int64).unsqueeze(1)
                )
            )
            golden_logits.masked_fill_(~valid, float("-inf"))
            rank_result = torch.topk(golden_logits, rank_depth, dim=1)
            golden_indices[rows] = rank_result.indices[:, : args.topk].to(
                torch.int32
            )
            ours_ranks[rows] = selected_ranks(
                golden_logits,
                final_indices[rows],
                rank_result.values,
                rank_depth,
            )
            original_ranks[rows] = selected_ranks(
                golden_logits,
                diagnostic_original_indices[rows],
                rank_result.values,
                rank_depth,
            )
            golden_ranks[rows] = selected_ranks(
                golden_logits,
                golden_indices[rows],
                rank_result.values,
                rank_depth,
            )
            diagnostic_end.record()
            diagnostic_end.synchronize()
            golden_diagnostic_ms += diagnostic_start.elapsed_time(
                diagnostic_end
            )
            del golden_logits, rank_result

        original_compare = compare_topk_sets(
            final_indices, diagnostic_original_indices
        )
        golden_compare = compare_topk_sets(final_indices, golden_indices)
        original_golden_compare = compare_topk_sets(
            diagnostic_original_indices, golden_indices
        )

    assert backward_logits.shape == (q_count, logits_width)
    assert scores.shape == (q_count, args.topk)
    assert global_indices.shape == (q_count, args.topk)

    local_indices = global_indices - window_starts.unsqueeze(1).to(torch.int64)
    local_indices.masked_fill_(global_indices < 0, -1)
    safe_indices = (
        local_indices.clamp_min(0) + topk_row_starts.unsqueeze(1)
    ).to(torch.int64)
    gathered = torch.gather(backward_logits, 1, safe_indices)
    gathered.masked_fill_(global_indices < 0, float("-inf"))
    torch.testing.assert_close(scores, gathered, rtol=0, atol=0)
    threshold_stats = None
    if not args.skip_threshold_check:
        threshold_stats = compute_torch_threshold_stats(
            backward_logits,
            thresholds,
            topk_row_starts,
            lengths,
            args.topk,
            args.threshold_check_rows_per_chunk,
        )
    if filter_kv_len != 0:
        assert candidate_scores.shape == (
            q_count,
            8,
            v2_capacity_per_warp,
        )
        assert candidate_indices.shape == candidate_scores.shape
        assert candidate_counts.shape == (q_count, 8, 32)
        assert candidate_overflow.shape == candidate_counts.shape
    assert final_scores.shape == (q_count, args.topk)
    assert final_indices.shape == (q_count, args.topk)
    if original_indices is not None:
        assert original_indices.shape == (q_count, args.topk)
        assert golden_indices.shape == (q_count, args.topk)

    print("=== Shape Plan ===")
    print(
        f"Q={q_count:,}, K={total_k:,}, chunk={args.chunk_len:,}, "
        f"cp_size={args.cp_size}, cp_rank={args.cp_rank}, "
        f"backward_len={args.backward_len:,}, "
        f"topk={args.topk}, data_mode={args.data_mode}"
    )
    print(f"backward logits: [{q_count:,}, {logits_width:,}]")
    print(
        f"full logits: [{q_count:,}, {total_k:,}] = "
        f"{q_count * total_k * LOGITS_BYTES / 1024**3:.2f} GiB"
    )
    planned_original_rows = (
        compute_full_logits_rows(q_count, total_k, free_mem_bytes)
        if args.original_rows_per_chunk <= 0
        else min(args.original_rows_per_chunk, q_count)
    )
    print(
        "full-logits Q chunks="
        f"{ceil_div(q_count, planned_original_rows)} "
        f"(rows/chunk={planned_original_rows:,}, "
        f"free_mem_for_heuristic={free_mem_bytes / 1024**3:.2f} GiB)"
    )
    print(
        "backward row offset: "
        f"max={int(topk_row_starts.max().item()) if topk_row_starts.numel() else 0}, "
        f"extra={logits_width - args.backward_len}"
    )
    print(f"V2 filter KV: [0, {filter_kv_len:,})")
    print(
        "V2 capacity: "
        f"per_warp={v2_capacity_per_warp:,}, "
        f"per_lane={v2_capacity_per_warp // 32:,}"
    )
    print(
        "V2 candidates: "
        f"scores={tuple(candidate_scores.shape) if candidate_scores is not None else None}, "
        f"indices={tuple(candidate_indices.shape) if candidate_indices is not None else None}, "
        f"counts={tuple(candidate_counts.shape) if candidate_counts is not None else None}"
    )
    print("\n=== Timing ===")
    print(f"fp8_mqa_backward_logits: {logits_ms:.3f} ms")
    print(f"fp8_mqa_backward_topk:   {topk_ms:.3f} ms")
    print(f"fp8_mqa_filter_v2:       {v2_ms:.3f} ms")
    print(f"mqa_final_topk:          {final_ms:.3f} ms")
    print(f"combined:                {logits_ms + topk_ms + v2_ms + final_ms:.3f} ms")
    if original_indices is not None:
        print("SGLang original")
        print(f"  fp8_mqa_logits:        {original_logits_ms:.3f} ms")
        if original_checksum_ms is not None:
            print(
                f"  fp8_mqa_logits_checksum:"
                f" {original_checksum_ms:.3f} ms"
            )
        if original_checksum_early_release_ms is not None:
            print(
                f"  fp8_mqa_logits_checksum_early_release:"
                f" {original_checksum_early_release_ms:.3f} ms "
                "(invalid output)"
            )
        if original_no_epilogue_ms is not None:
            print(
                f"  fp8_mqa_logits_no_epilogue:"
                f" {original_no_epilogue_ms:.3f} ms"
            )
        if original_early_release_ms is not None:
            print(
                f"  fp8_mqa_logits_early_release:"
                f" {original_early_release_ms:.3f} ms (invalid output)"
            )
        if original_unsafe_decoupled_ms is not None:
            print(
                f"  fp8_mqa_logits_unsafe_decoupled:"
                f" {original_unsafe_decoupled_ms:.3f} ms (invalid output)"
            )
        print(f"  fast_topk_transform:   {original_topk_ms:.3f} ms")
        print(
            f"  total operators:       "
            f"{original_logits_ms + original_topk_ms:.3f} ms"
        )
        print(
            "  chunks:                "
            f"rows_per_chunk={rows_per_chunk:,}, "
            f"num_chunks={ceil_div(q_count, rows_per_chunk)}"
        )
    elif original_checksum_ms is not None:
        print("SGLang original checksum-only")
        print(f"  fp8_mqa_logits_checksum: {original_checksum_ms:.3f} ms")
        if original_checksum_early_release_ms is not None:
            print(
                f"  fp8_mqa_logits_checksum_early_release: "
                f"{original_checksum_early_release_ms:.3f} ms "
                "(invalid output)"
            )
        if original_no_epilogue_ms is not None:
            print(
                f"  fp8_mqa_logits_no_epilogue: "
                f"{original_no_epilogue_ms:.3f} ms"
            )
        if original_early_release_ms is not None:
            print(
                f"  fp8_mqa_logits_early_release: "
                f"{original_early_release_ms:.3f} ms (invalid output)"
            )
        if original_unsafe_decoupled_ms is not None:
            print(
                f"  fp8_mqa_logits_unsafe_decoupled: "
                f"{original_unsafe_decoupled_ms:.3f} ms (invalid output)"
            )
        print(
            "  chunks:                  "
            f"rows_per_chunk={checksum_rows_per_chunk:,}, "
            f"num_chunks={ceil_div(q_count, checksum_rows_per_chunk)}"
        )
    elif original_checksum_early_release_ms is not None:
        print("SGLang original checksum early-release benchmark")
        print(
            f"  fp8_mqa_logits_checksum_early_release: "
            f"{original_checksum_early_release_ms:.3f} ms (invalid output)"
        )
        print(
            "  chunks:                                "
            f"rows_per_chunk={checksum_early_release_rows_per_chunk:,}, "
            f"num_chunks={ceil_div(q_count, checksum_early_release_rows_per_chunk)}"
        )
    elif original_no_epilogue_ms is not None:
        print("SGLang original no-epilogue")
        print(
            f"  fp8_mqa_logits_no_epilogue: "
            f"{original_no_epilogue_ms:.3f} ms"
        )
        if original_early_release_ms is not None:
            print(
                f"  fp8_mqa_logits_early_release: "
                f"{original_early_release_ms:.3f} ms (invalid output)"
            )
        if original_unsafe_decoupled_ms is not None:
            print(
                f"  fp8_mqa_logits_unsafe_decoupled: "
                f"{original_unsafe_decoupled_ms:.3f} ms (invalid output)"
            )
        print(
            "  chunks:                     "
            f"rows_per_chunk={no_epilogue_rows_per_chunk:,}, "
            f"num_chunks={ceil_div(q_count, no_epilogue_rows_per_chunk)}"
        )
    elif original_early_release_ms is not None:
        print("SGLang original early-release benchmark")
        print(
            f"  fp8_mqa_logits_early_release: "
            f"{original_early_release_ms:.3f} ms (invalid output)"
        )
        print(
            "  chunks:                         "
            f"rows_per_chunk={early_release_rows_per_chunk:,}, "
            f"num_chunks={ceil_div(q_count, early_release_rows_per_chunk)}"
        )
    elif original_unsafe_decoupled_ms is not None:
        print("SGLang original unsafe-decoupled benchmark")
        print(
            f"  fp8_mqa_logits_unsafe_decoupled: "
            f"{original_unsafe_decoupled_ms:.3f} ms (invalid output)"
        )
        print(
            "  chunks:                          "
            f"rows_per_chunk={unsafe_decoupled_rows_per_chunk:,}, "
            f"num_chunks={ceil_div(q_count, unsafe_decoupled_rows_per_chunk)}"
        )
    if candidate_counts is not None:
        print(
            "V2 candidate stats: "
            f"accepted={int(candidate_counts.sum().item()):,}, "
            f"overflow_lanes={int(candidate_overflow.sum().item()):,}"
        )
    if threshold_stats is not None:
        print("\n=== Threshold Check Against torch.topk ===")
        print(
            "custom_threshold - torch_threshold: "
            f"P50={threshold_stats['delta_p50']:.6g}, "
            f"P95={threshold_stats['delta_p95']:.6g}, "
            f"P99={threshold_stats['delta_p99']:.6g}"
        )
        print(
            "custom threshold rank in valid backward logits: "
            f"P50={threshold_stats['rank_p50']:.0f}, "
            f"P95={threshold_stats['rank_p95']:.0f}, "
            f"P99={threshold_stats['rank_p99']:.0f}"
        )
        print(
            "torch threshold rank in valid backward logits:  "
            f"P50={threshold_stats['torch_rank_p50']:.0f}, "
            f"P95={threshold_stats['torch_rank_p95']:.0f}, "
            f"P99={threshold_stats['torch_rank_p99']:.0f}"
        )
        print(
            "threshold rows: "
            f"custom<torch={threshold_stats['lower_rows']:,}, "
            f"custom=torch={threshold_stats['equal_rows']:,}, "
            f"custom>torch={threshold_stats['higher_rows']:,}"
        )
        print(
            "threshold check time: "
            f"{threshold_stats['elapsed_ms']:.3f} ms "
            "(diagnostic, not included above)"
        )
    if original_compare is not None:
        print("\n=== Correctness Against torch.topk Golden ===")
        print(
            "ours vs torch.topk: "
            f"rows={golden_compare['rows']:,}/{golden_compare['total_rows']:,}, "
            f"replaced-indices={golden_compare['replaced']:,}, "
            "replacements-per-mismatched-row "
            f"P50={golden_compare['p50']:.0f}, "
            f"P99={golden_compare['p99']:.0f}"
        )
        print(
            "SGLang vs torch.topk: "
            f"rows={original_golden_compare['rows']:,}/"
            f"{original_golden_compare['total_rows']:,}, "
            f"replaced-indices={original_golden_compare['replaced']:,}, "
            "replacements-per-mismatched-row "
            f"P50={original_golden_compare['p50']:.0f}, "
            f"P99={original_golden_compare['p99']:.0f}"
        )
        print(
            "ours vs SGLang: "
            f"rows={original_compare['rows']:,}/{original_compare['total_rows']:,}, "
            f"replaced-indices={original_compare['replaced']:,}, "
            "replacements-per-mismatched-row "
            f"P50={original_compare['p50']:.0f}, "
            f"P99={original_compare['p99']:.0f}"
        )
        effective_rank_depth = min(args.rank_depth, total_k)
        print("\n=== Selected Index Rank In Full FP32 Logits ===")
        print_rank_stats(
            "ours",
            rank_percentiles(
                ours_ranks, effective_rank_depth, args.topk
            ),
            effective_rank_depth,
            args.topk,
        )
        print_rank_stats(
            "SGLang",
            rank_percentiles(
                original_ranks, effective_rank_depth, args.topk
            ),
            effective_rank_depth,
            args.topk,
        )
        print_rank_stats(
            "torch.topk",
            rank_percentiles(
                golden_ranks, effective_rank_depth, args.topk
            ),
            effective_rank_depth,
            args.topk,
        )
        print(
            "torch Golden + rank analysis: "
            f"{golden_diagnostic_ms:.3f} ms "
            "(diagnostic, not included above)"
        )
    print("\nPASS")


if __name__ == "__main__":
    main()