"""Regression cases for the four long-context ops on the v0.1.5 core.

Run with an SM100 CUDA build:
    python -m unittest discover -s tests -p test_long_context_mqa_migration.py -v

Metadata cases only need PyTorch CPU. CUDA cases compare against the stock
v0.1.5 FP8 logits, so both sides use the same head-reduction definition.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_metadata_helper(package, function_name="build_mqa_backward_metadata", **extra):
    # Exercise both shipped wrappers without importing the CUDA extension.
    tree = ast.parse((ROOT / package / "__init__.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = {"torch": torch, **extra}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(ROOT / package), "exec"
        ),
        namespace,
    )
    return namespace[function.name]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MetadataTests(unittest.TestCase):
    def test_final_wrapper_preserves_supplied_output_and_mask(self):
        for package in ("deep_gemm", "sgl_deep_gemm"):
            seen = []

            def final(*args):
                seen.append(args)
                output, mask = args[-2:]
                output[mask] = 17

            wrapper = load_metadata_helper(
                package, "mqa_final_topk", _C=SimpleNamespace(mqa_final_topk=final)
            )
            scores = torch.zeros(3, 2048)
            indices = torch.zeros(3, 2048, dtype=torch.int32)
            output = torch.full_like(indices, -123)
            mask = torch.tensor([True, False, True])
            _, actual = wrapper(
                scores,
                indices,
                None,
                None,
                None,
                None,
                None,
                q_mask=mask,
                out_indices=output,
            )
            self.assertIs(actual, output)
            self.assertIs(seen[0][-1], mask)
            self.assertTrue(bool((output[mask] == 17).all()))
            self.assertTrue(bool((output[~mask] == -123).all()))

    def test_cp_stride_and_request_tails(self):
        for package in ("deep_gemm", "sgl_deep_gemm"):
            with self.subTest(package=package):
                helper = load_metadata_helper(package)
                ks = torch.tensor([0] * 5 + [100000] * 3, dtype=torch.int32)
                ke = torch.tensor(
                    [70000, 70008, 70016, 70024, 70032, 170000, 170008, 170016],
                    dtype=torch.int32,
                )
                windows, blocks, rows, width = helper(ks, ke, [0, 5, 0, 3])
                self.assertEqual(windows.tolist(), (ke - 65536).tolist())
                self.assertEqual(blocks.tolist(), [0, 4, 5, -1, -1, -1, -1, -1])
                self.assertEqual(rows.tolist(), [0, 8, 16, 24, 0, 0, 8, 16])
                self.assertEqual(width, 65792)

    def test_empty_short_and_nonmonotonic_windows(self):
        for package in ("deep_gemm", "sgl_deep_gemm"):
            helper = load_metadata_helper(package)
            empty = torch.empty(0, dtype=torch.int32)
            windows, blocks, rows, width = helper(empty, empty, [0], backward_len=4097)
            self.assertEqual(
                (windows.numel(), blocks.numel(), rows.numel(), width), (0, 0, 0, 4352)
            )
            ks = torch.tensor([7, 7, 7], dtype=torch.int32)
            ke = torch.tensor([9000, 20, 8000], dtype=torch.int32)
            windows, blocks, rows, width = helper(ks, ke, [3], backward_len=4096)
            self.assertEqual(windows.tolist(), [4904, 7, 3904])
            self.assertEqual(blocks.tolist(), [0, -1, -1])
            self.assertEqual(rows.tolist(), [4897, 0, 3897])
            self.assertEqual(width, 9216)

    def test_default_output_width_includes_block_union(self):
        # Validate the optional wrapper fallback without a CUDA extension.
        for package in ("deep_gemm", "sgl_deep_gemm"):
            wrapper = load_metadata_helper(
                package,
                "fp8_mqa_backward_logits",
                _parse_tensor_or_tuple=lambda value: value,
                _C=SimpleNamespace(fp8_mqa_backward_logits=lambda *args: args[-2]),
            )
            for starts, ends, blocks, window, expected in (
                (
                    [100, 108, 116, 124, 90000],
                    [4196, 4204, 4212, 4220, 94096],
                    [0, 4, -1, -1, -1],
                    4096,
                    4352,
                ),
                ([4904, 7, 3904], [9000, 20, 8000], [0, -1, -1], 4096, 9216),
                ([], [], [], 4097, 4352),
            ):
                starts, ends, blocks = [
                    torch.tensor(x, dtype=torch.int32) for x in (starts, ends, blocks)
                ]
                self.assertEqual(
                    wrapper(
                        None,
                        (None, None),
                        None,
                        starts,
                        ends,
                        blocks,
                        backward_len=window,
                    ),
                    expected,
                )

    def test_invalid_metadata(self):
        helper = load_metadata_helper("deep_gemm")
        values = torch.zeros(3, dtype=torch.int32)
        for args, kwargs in (
            ((values, values, [2]), {}),
            ((values, values, [-1, 4]), {}),
            ((values, values[:2], [3]), {}),
            ((values, values, [3]), {"backward_len": 0}),
            ((values, values, [3]), {"block_q": 2}),
        ):
            with self.assertRaises(ValueError):
                helper(*args, **kwargs)

    def test_topk_wrapper_uses_cached_lengths_without_host_scalar_reads(self):
        for package in ("deep_gemm", "sgl_deep_gemm"):
            calls = []
            wrapper = load_metadata_helper(
                package,
                "fp8_mqa_backward_topk",
                _C=SimpleNamespace(
                    fp8_mqa_backward_topk=lambda *args: calls.append(args)
                ),
            )
            ks = torch.tensor([0, 0], dtype=torch.int32)
            ke = torch.tensor([70000, 70008], dtype=torch.int32)
            rows = torch.tensor([0, 8], dtype=torch.int32)
            lengths = torch.full((2,), 65536, dtype=torch.int32)
            logits = torch.empty(2, 65792)
            with patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("width validation synchronizes"),
            ), patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("D2H read")
            ):
                wrapper(logits, ks, ke, rows, lengths=lengths)
                wrapper(logits, ks, ke, rows)
            self.assertIs(calls[0][1], lengths)
            torch.testing.assert_close(calls[1][1], lengths)
            with self.assertRaisesRegex(ValueError, "lengths"):
                wrapper(logits, ks, ke, rows, lengths=lengths[:1])
            with self.assertRaisesRegex(ValueError, "row_starts"):
                wrapper(logits, ks, ke, rows[:1])


HAS_SM100 = (
    torch is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)


@unittest.skipUnless(HAS_SM100, "SM100 CUDA GPU is required")
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import deep_gemm

        cls.dg = deep_gemm

    def make_inputs(self, q_count, k_count):
        torch.manual_seed(42)
        q = (torch.randn(q_count, 32, 128, device="cuda") * 0.25).to(
            torch.float8_e4m3fn
        )
        kv = (torch.randn(k_count, 128, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.75, 1.25, k_count, device="cuda")
        # A strided FP32 weights view also checks the TMA row stride.
        weights = torch.rand(q_count, 64, device="cuda")[:, :32]
        return q, (kv, scales), weights

    def test_masked_logits_and_custom_pipeline_preserve_shared_rows(self):
        dg = self.dg
        request_q = [5, 3, 1, 2, 3]
        q, kv, weights = self.make_inputs(sum(request_q), 40960)
        starts, ends, flags = [], [], []
        for request, n in enumerate(request_q):
            base = 3 + request * 8192
            starts.extend([base] * n)
            length = 6000 if request in (1, 3, 4) else 3000
            ends.extend(base + length + i * 8 for i in range(n))
            flags.extend([request in (1, 3, 4)] * n)
        ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
        ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
        mask = torch.tensor(flags, dtype=torch.bool, device="cuda")
        full = dg.fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False)
        for active, stride in (
            (mask, 6144),
            (~mask, 3072),
            (torch.zeros_like(mask), 1),
        ):
            compact = dg.fp8_mqa_logits(
                q,
                kv,
                weights,
                ks,
                ke,
                clean_logits=False,
                max_seqlen_k=stride,
                q_mask=active,
            )
            for row in range(q.shape[0]):
                if bool(active[row]):
                    start, end = starts[row], ends[row]
                    torch.testing.assert_close(
                        compact[row, : end - start],
                        full[row, start:end],
                        rtol=0,
                        atol=0,
                    )

        windows, blocks, offsets, width = dg.build_mqa_backward_metadata(
            ks, ke, request_q, backward_len=4096
        )
        page_table = torch.arange(8192, dtype=torch.int32, device="cuda").repeat(5, 1)
        cu_q = torch.tensor([0, 5, 8, 9, 11, 14], dtype=torch.int32, device="cuda")

        def run(active, out=None):
            logits = dg.fp8_mqa_backward_logits(
                q,
                kv,
                weights,
                windows,
                ke,
                blocks,
                backward_len=4096,
                output_len=width,
                q_mask=active,
            )
            for row in range(q.shape[0]):
                if active is None or bool(active[row]):
                    lo, hi, col = int(windows[row]), ends[row], int(offsets[row])
                    torch.testing.assert_close(
                        logits[row, col : col + hi - lo],
                        full[row, lo:hi],
                        rtol=0,
                        atol=0,
                    )
            ps, pi = dg.fp8_mqa_backward_topk(
                logits,
                ks,
                ke,
                offsets,
                backward_len=4096,
                index_starts=windows - ks,
                q_mask=active,
            )
            if active is not None:
                self.assertTrue(bool((pi[~active] == -1).all()))
                self.assertTrue(bool(torch.isfinite(ps).all()))
            cs, ci, cc, overflow = dg.fp8_mqa_filter_candidates(
                q,
                kv,
                weights,
                ks,
                windows,
                ps.amin(1),
                candidate_capacity_per_warp=1024,
                q_mask=active,
            )
            self.assertFalse(bool(overflow.any()))
            if active is not None:
                self.assertFalse(bool(cc[~active].any()))
            return dg.mqa_final_topk(
                ps,
                pi,
                cs,
                ci,
                cc,
                page_table,
                cu_q,
                transform_indices=False,
                q_mask=active,
                out_indices=out,
            )[1]

        expected = run(None)
        partial = mask.clone()
        partial[5] = False  # Keep the original shared origin for rows 6 and 7.
        partial_out = torch.full_like(expected, -123)
        run(partial, partial_out)
        torch.testing.assert_close(
            partial_out[partial].sort(1).values, expected[partial].sort(1).values
        )
        self.assertTrue(bool((partial_out[~partial] == -123).all()))
        shared = torch.full_like(expected, -123)
        self.assertEqual(run(mask, shared).data_ptr(), shared.data_ptr())
        torch.testing.assert_close(
            shared[mask].sort(1).values, expected[mask].sort(1).values
        )
        self.assertTrue(bool((shared[~mask] == -123).all()))
        run(~mask, shared)
        torch.testing.assert_close(shared.sort(1).values, expected.sort(1).values)
        snapshot = shared.clone()
        run(torch.zeros_like(mask), shared)
        torch.testing.assert_close(shared, snapshot)

    def test_four_ops_cp_ragged_and_index_transforms(self):
        dg = self.dg
        q, kv, weights = self.make_inputs(8, 16384)
        ks = torch.tensor([3] * 5 + [8195] * 3, dtype=torch.int32, device="cuda")
        ke = torch.tensor(
            [7200, 7208, 7216, 7224, 7232, 15392, 15400, 15408],
            dtype=torch.int32,
            device="cuda",
        )
        windows, blocks, rows, width = dg.build_mqa_backward_metadata(
            ks, ke, [5, 3], backward_len=4096
        )
        self.assertEqual(rows.tolist(), [0, 8, 16, 24, 0, 0, 8, 16])
        self.assertEqual(width, 4352)
        full = dg.fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False)
        backward = dg.fp8_mqa_backward_logits(
            q, kv, weights, windows, ke, blocks, backward_len=4096, output_len=width
        )
        for row in range(8):
            start, end = int(windows[row]), int(ke[row])
            offset = int(rows[row])
            torch.testing.assert_close(
                backward[row, offset : offset + end - start],
                full[row, start:end],
                rtol=0,
                atol=0,
            )

        # All rows must write the same union, including columns outside their
        # own window. Request tails use their own origin, even within a chunk.
        union_starts = windows - rows
        union_ends = torch.tensor(
            [7224] * 4 + [7232] + [15408] * 3, dtype=torch.int32, device="cuda"
        )
        union_ref = dg.fp8_mqa_logits(
            q,
            kv,
            weights,
            union_starts,
            union_ends,
            clean_logits=False,
            max_seqlen_k=width,
        )
        for row in range(8):
            length = int(union_ends[row] - union_starts[row])
            torch.testing.assert_close(
                backward[row, :length], union_ref[row, :length], rtol=0, atol=0
            )
        auto_width = dg.fp8_mqa_backward_logits(
            q, kv, weights, windows, ke, blocks, backward_len=4096
        )
        self.assertEqual(auto_width.shape, backward.shape)
        for row in range(8):
            length = int(union_ends[row] - union_starts[row])
            torch.testing.assert_close(
                auto_width[row, :length], backward[row, :length], rtol=0, atol=0
            )

        prefix_scores, prefix_ids = dg.fp8_mqa_backward_topk(
            backward,
            ks,
            ke,
            rows,
            backward_len=4096,
            index_starts=windows - ks,
            lengths=(ke - ks).clamp(0, 4096).to(torch.int32),
        )
        # The selection algorithm is unchanged; score/index association must be exact.
        absolute = prefix_ids.to(torch.int64) + ks[:, None]
        torch.testing.assert_close(
            prefix_scores, full.gather(1, absolute), rtol=0, atol=0
        )
        torch.testing.assert_close(
            prefix_scores.min(1).values,
            torch.stack(
                [
                    backward[r, int(rows[r]) : int(rows[r]) + 4096]
                    .topk(2048)
                    .values[-1]
                    for r in range(8)
                ]
            ),
            rtol=0,
            atol=0,
        )

        thresholds = prefix_scores.min(1).values.contiguous()
        capacity = 512
        scores, ids, counts, overflow = dg.fp8_mqa_filter_candidates(
            q, kv, weights, ks, windows, thresholds, capacity
        )
        self.assertFalse(bool(overflow.any()))
        lanes = scores.view(8, 8, 32, capacity // 32)
        lane_ids = ids.view_as(lanes)
        # Compare complete candidate sets and exact scores, not aggregate counts alone.
        slots = torch.arange(capacity // 32, device="cuda")
        for row in range(8):
            valid = slots < counts[row, :, :, None]
            candidate_ids = lane_ids[row][valid].long()
            start, end = int(ks[row]), int(windows[row])
            ref_ids = torch.nonzero(full[row, start:end] >= thresholds[row]).flatten()
            torch.testing.assert_close(
                candidate_ids.sort().values, ref_ids, rtol=0, atol=0
            )
            torch.testing.assert_close(
                lanes[row][valid], full[row, start + candidate_ids], rtol=0, atol=0
            )
            # Poison unused slots: final TopK must honor each lane's count.
            lanes[row].masked_fill_(~valid, float("inf"))
            lane_ids[row].masked_fill_(~valid, -2147483648)

        lengths = (ke - ks).long()
        cu_q = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
        page_table = torch.stack(
            (
                torch.arange(8192, device="cuda"),
                torch.arange(8192, device="cuda") + 20000,
            )
        ).to(torch.int32)
        final_scores, final_ids = dg.mqa_final_topk(
            prefix_scores,
            prefix_ids,
            scores,
            ids,
            counts,
            page_table,
            cu_q,
            transform_indices=False,
        )
        expected_scores = torch.stack(
            [full[r, int(ks[r]) : int(ke[r])].topk(2048).values for r in range(8)]
        )
        torch.testing.assert_close(
            final_scores.sort(descending=True).values, expected_scores, rtol=0, atol=0
        )
        self.assertTrue(bool(((final_ids >= 0) & (final_ids < lengths[:, None])).all()))
        torch.testing.assert_close(
            final_scores, full.gather(1, final_ids.long() + ks[:, None]), rtol=0, atol=0
        )
        transformed_scores, transformed_ids = dg.mqa_final_topk(
            prefix_scores, prefix_ids, scores, ids, counts, page_table, cu_q
        )
        table_rows = torch.tensor([0] * 5 + [1] * 3, device="cuda")
        # Atomic selection does not promise output order or a stable tie choice.
        table_bases = torch.tensor([0, 20000], device="cuda")
        local_ids = transformed_ids.long() - table_bases[table_rows, None]
        torch.testing.assert_close(
            transformed_scores.sort(descending=True).values,
            expected_scores,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            transformed_scores, full.gather(1, local_ids + ks[:, None]), rtol=0, atol=0
        )
        offsets = torch.tensor([100, 10000], dtype=torch.int32, device="cuda")
        offset_scores, offset_ids = dg.mqa_final_topk(
            prefix_scores,
            prefix_ids,
            scores,
            ids,
            counts,
            page_table,
            cu_q,
            transform_offsets=offsets,
        )
        local_ids = offset_ids.long() - offsets[table_rows, None]
        torch.testing.assert_close(
            offset_scores.sort(descending=True).values, expected_scores, rtol=0, atol=0
        )
        torch.testing.assert_close(
            offset_scores, full.gather(1, local_ids + ks[:, None]), rtol=0, atol=0
        )

    def test_filter_overflow_and_empty_intervals(self):
        dg = self.dg
        q, kv, weights = self.make_inputs(5, 769)
        ks = torch.tensor([3, 7, 13, 769, 31], dtype=torch.int32, device="cuda")
        ke = torch.tensor([769, 7, 769, 769, 31], dtype=torch.int32, device="cuda")
        thresholds = torch.full((5,), -1.0, device="cuda")
        scores, ids, counts, overflow = dg.fp8_mqa_filter_candidates(
            q, kv, weights, ks, ke, thresholds, 32
        )
        self.assertTrue(bool(overflow[[0, 2]].any()))
        self.assertTrue(bool((counts <= 1).all()))
        self.assertEqual(int(counts[[1, 3, 4]].sum()), 0)
        self.assertEqual(int(overflow[[1, 3, 4]].sum()), 0)
        valid = counts.bool()
        selected = ids.view(5, 8, 32)[valid]
        lengths = (ke - ks)[:, None, None].expand_as(counts)[valid]
        self.assertTrue(bool(((selected >= 0) & (selected < lengths)).all()))
        self.assertTrue(bool(torch.isfinite(scores.view(5, 8, 32)[valid]).all()))

    def test_filter_nonfinite_scores_and_thresholds(self):
        dg = self.dg
        q, kv, weights = self.make_inputs(3, 260)
        kv[1][5] = float("nan")
        kv[1][6] = float("inf")
        ks = torch.full((3,), 3, dtype=torch.int32, device="cuda")
        ke = torch.full((3,), 260, dtype=torch.int32, device="cuda")
        thresholds = torch.tensor(
            [float("-inf"), float("inf"), float("nan")], device="cuda"
        )
        scores, ids, counts, overflow = dg.fp8_mqa_filter_candidates(
            q, kv, weights, ks, ke, thresholds, 64
        )
        self.assertFalse(bool(overflow.any()))
        self.assertEqual(int(counts[1:].sum()), 0)
        full = dg.fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False)
        lanes = scores.view(3, 8, 32, 2)
        lane_ids = ids.view_as(lanes)
        valid = torch.arange(2, device="cuda") < counts[0, :, :, None]
        selected = lane_ids[0][valid].long()
        expected = torch.nonzero(torch.isfinite(full[0, 3:260])).flatten()
        torch.testing.assert_close(selected.sort().values, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            lanes[0][valid], full[0, selected + 3], rtol=0, atol=0
        )

    def test_final_topk_at_or_below_k(self):
        prefix_lengths = [0, 17, 2048, 2040]
        candidate_lengths = [0, 5, 0, 8]
        prefix_scores = torch.full(
            (4, 2048), -torch.finfo(torch.float32).max, device="cuda"
        )
        prefix_ids = torch.full((4, 2048), -1, dtype=torch.int32, device="cuda")
        scores = torch.full((4, 8, 32), float("inf"), device="cuda")
        ids = torch.full((4, 8, 32), 1000000, dtype=torch.int32, device="cuda")
        counts = torch.zeros((4, 8, 32), dtype=torch.int32, device="cuda")
        for row, (prefix_len, candidate_len) in enumerate(
            zip(prefix_lengths, candidate_lengths)
        ):
            prefix_ids[row, :prefix_len] = torch.arange(prefix_len, device="cuda")
            prefix_scores[row, :prefix_len] = prefix_ids[row, :prefix_len].float()
            ids[row, 0, :candidate_len] = torch.arange(
                prefix_len, prefix_len + candidate_len, device="cuda"
            )
            scores[row, 0, :candidate_len] = ids[row, 0, :candidate_len].float()
            counts[row, 0, :candidate_len] = 1
        table = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(4, 1)
        cu_q = torch.arange(5, device="cuda", dtype=torch.int32)
        out_scores, out_ids = self.dg.mqa_final_topk(
            prefix_scores,
            prefix_ids,
            scores,
            ids,
            counts,
            table,
            cu_q,
            transform_indices=False,
        )
        for row, (prefix_len, candidate_len) in enumerate(
            zip(prefix_lengths, candidate_lengths)
        ):
            valid = out_ids[row] >= 0
            expected = torch.arange(
                prefix_len + candidate_len, device="cuda", dtype=torch.int32
            )
            torch.testing.assert_close(
                out_ids[row][valid].sort().values, expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                out_scores[row][valid], out_ids[row][valid].float(), rtol=0, atol=0
            )
            self.assertTrue(
                bool((out_scores[row][~valid] == -torch.finfo(torch.float32).max).all())
            )

    def test_short_windows_and_empty_queries(self):
        dg = self.dg
        for q_count in (0, 3):
            q, kv, weights = self.make_inputs(q_count, 257)
            ks = torch.full((q_count,), 3, dtype=torch.int32, device="cuda")
            ke = torch.tensor(
                [] if q_count == 0 else [3, 13, 257], dtype=torch.int32, device="cuda"
            )
            windows, blocks, rows, width = dg.build_mqa_backward_metadata(
                ks, ke, [q_count], backward_len=257
            )
            logits = dg.fp8_mqa_backward_logits(
                q, kv, weights, windows, ke, blocks, backward_len=257, output_len=257
            )
            self.assertEqual(logits.stride(0), 512)
            scores, ids = dg.fp8_mqa_backward_topk(
                logits, ks, ke, rows, backward_len=257, index_starts=windows - ks
            )
            self.assertEqual(tuple(scores.shape), (q_count, 2048))
            for row in range(q_count):
                length = int(ke[row] - ks[row])
                self.assertEqual(ids[row, :length].tolist(), list(range(length)))
                self.assertTrue(bool((ids[row, length:] == -1).all()))
                torch.testing.assert_close(
                    scores[row, :length], logits[row, :length], rtol=0, atol=0
                )
            candidates = dg.fp8_mqa_filter_candidates(
                q, kv, weights, ks, windows, torch.zeros(q_count, device="cuda"), 32
            )
            cu_q = torch.tensor([0, q_count], dtype=torch.int32, device="cuda")
            table = torch.arange(257, device="cuda", dtype=torch.int32)[None, :]
            out_scores, out_ids = dg.mqa_final_topk(
                scores, ids, *candidates[:3], table, cu_q, transform_indices=False
            )
            self.assertEqual(tuple(out_scores.shape), (q_count, 2048))
            self.assertEqual(tuple(out_ids.shape), (q_count, 2048))


if __name__ == "__main__":
    unittest.main()
