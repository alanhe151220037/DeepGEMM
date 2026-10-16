"""CPU checks for benchmark planning and actual SGLang backend dispatch."""

import ast
import contextlib
from enum import Enum, IntEnum, auto
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

PATH = Path(__file__).with_name("bench_sglang_long_context_topk.py")
SPEC = importlib.util.spec_from_file_location("long_context_bench", PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
SGLANG_ROOT = bench.parse_args([]).sglang_root


class PlanningTests(unittest.TestCase):
    def test_mixed_preset_exercises_both_routes_at_boundary(self):
        args = bench.parse_args(["--preset", "mixed", "--modes", "all"])
        self.assertEqual(args.min_kv_length, 262144)
        self.assertEqual(
            [kv > args.min_kv_length for kv, _ in args.requests],
            [False, True, False, True],
        )
        self.assertEqual(
            bench.local_q_lengths(args.requests, args.cp_size, args.cp_rank),
            [5, 7, 3, 9],
        )
        self.assertEqual(bench.parse_args([]).min_kv_length, 0)

    def test_requests_and_cp_tails(self):
        requests = bench.parse_requests("32768:41,24576:25")
        self.assertEqual(bench.local_q_lengths(requests, 8, 0), [6, 4])
        self.assertEqual(bench.local_q_lengths(requests, 8, 7), [5, 3])
        with self.assertRaises(ValueError):
            bench.local_q_lengths([(8, 1)], 8, 7)

    def test_presets_and_arguments(self):
        args = bench.parse_args(["--preset", "long"])
        self.assertEqual(
            bench.local_q_lengths(args.requests, args.cp_size, args.cp_rank), [8192]
        )
        self.assertEqual(args.backward_len, 65536)
        self.assertEqual(args.max_rows, 0)
        args = bench.parse_args(["--preset", "chunked", "--free-mem-gb", "0.125"])
        self.assertEqual(args.max_rows, 128)
        self.assertEqual(
            bench.parse_args(["--preset", "chunked", "--max-rows", "0"]).max_rows, 0
        )
        self.assertGreater(
            sum(bench.local_q_lengths(args.requests, args.cp_size, 0))
            * sum(k for k, _ in args.requests),
            8_000_000,
        )
        for argv in (
            ["--warmup", "0"],
            ["--rank-depth", "1"],
            ["--atol", "nan"],
            ["--memory-policy", "live", "--free-mem-gb", "1"],
            ["--max-rows", "-1"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                bench.parse_args(argv)

    def test_rank_quantiles_and_empty_ranks(self):
        self.assertEqual(
            bench.rank_quantiles([0, 2, 1, 1]), {"p50": 1, "p95": 3, "p99": 3}
        )
        self.assertEqual(
            bench.rank_quantiles([0] * 3), {"p50": None, "p95": None, "p99": None}
        )
        self.assertIsNone(bench.clean_json({"value": float("nan")})["value"])

    def test_fixed_cuda_view_forwards_other_methods(self):
        fake = SimpleNamespace(
            cuda=SimpleNamespace(get_device_properties=lambda index: ("device", index))
        )
        with patch.object(bench, "torch", fake):
            view = bench.TorchView(123, 456)
            self.assertEqual(view.cuda.mem_get_info(0), (123, 456))
            self.assertEqual(view.cuda.get_device_properties(0), ("device", 0))


class KernelProfileTests(unittest.TestCase):
    def test_only_kernel_activity_durations_are_averaged(self):
        events = [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void test_kernel<int>()",
                "dur": duration,
                "args": {"device": 0, "grid": [4, 1, 1], "block": [128, 1, 1]},
            }
            for duration in (2.0, 8.0)
        ]
        events += [
            {"ph": "X", "cat": category, "name": "excluded", "dur": 100000}
            for category in (
                "cpu_op",
                "cuda_runtime",
                "cuda_driver",
                "gpu_memcpy",
                "gpu_memset",
            )
        ]
        events.append({"ph": "s", "cat": "kernel", "name": "flow", "dur": 100000})
        result = bench.summarize_kernel_trace({"traceEvents": events})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["gpu_us_per_launch"], 5.0)
        self.assertEqual(result[0]["launches"], 2)
        self.assertEqual(result[0]["name"], "void test_kernel<int>()")

    def test_variants_preserve_symbol_and_launch_geometry(self):
        events = [
            {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "dur": 3,
                "args": {"device": 0, "grid": grid, "block": [128, 1, 1]},
            }
            for name, grid in (
                ("void test_kernel<int>()", [1, 1, 1]),
                ("void test_kernel<float>()", [1, 1, 1]),
                ("void test_kernel<int>()", [2, 1, 1]),
            )
        ]
        result = bench.summarize_kernel_trace({"traceEvents": events})
        self.assertEqual(
            [item["label"] for item in result],
            [f"test_kernel [v{i}]" for i in (1, 2, 3)],
        )
        self.assertEqual(len(result), 3)
        self.assertIn(
            "backward_logits",
            bench.kernel_label(
                "void deep_gemm::sm100_long_context_mqa_logits<32, 128, true, false, 4>()"
            ),
        )
        self.assertIn(
            "filter_candidates",
            bench.kernel_label(
                "void deep_gemm::sm100_long_context_mqa_logits<32, 128, false, true, 4>()"
            ),
        )

    def test_missing_or_invalid_cuda_activities_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "CUPTI"):
            bench.summarize_kernel_trace(
                {
                    "traceEvents": [
                        {"ph": "X", "cat": "cpu_op", "name": "aten::add", "dur": 50}
                    ]
                }
            )
        for duration in (-1, float("nan"), None):
            with self.assertRaisesRegex(RuntimeError, "Invalid CUDA duration"):
                bench.summarize_kernel_trace(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "test",
                                "dur": duration,
                            }
                        ]
                    }
                )

    def test_console_has_one_timing_metric_and_separate_accuracy_lines(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bench.print_kernel_profile(
                "custom", [{"label": "kernel [v1]", "gpu_us_per_launch": 12.5}]
            )
            bench.print_fields(
                {
                    "paths": {"custom": {"bad_rows": 0, "invalid_ids": 0}},
                    "overflow_lanes": 0,
                }
            )
        lines = output.getvalue().splitlines()
        self.assertIn("    kernel [v1]: 12.500", lines)
        self.assertIn("        bad_rows: 0", lines)
        self.assertIn("        invalid_ids: 0", lines)
        self.assertIn("    overflow_lanes: 0", lines)
        self.assertNotIn("stream_ms", output.getvalue())


@unittest.skipUnless(SGLANG_ROOT.is_dir(), "sibling SGLang source checkout is required")
class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = SimpleNamespace(
            SGLANG_DSA_FUSE_TOPK=bench.Flag(True),
            SGLANG_OPT_USE_TOPK_V2=bench.Flag(True),
            SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=bench.Flag(0.45),
        )
        cls.namespace = {
            "torch": None,
            "contextlib": contextlib,
            "Enum": Enum,
            "IntEnum": IntEnum,
            "auto": auto,
            "envs": cls.envs,
            "get_is_capture_mode": lambda: False,
        }
        cls.config, cls.indexer, cls.metadata, cls.hashes = bench.source_contract(
            SGLANG_ROOT, cls.namespace
        )

    def test_source_methods_and_request_chunks(self):
        self.assertTrue(bench.INDEXER_METHODS <= set(vars(self.indexer)))
        self.assertEqual(len(self.hashes), 5)
        chunks = list(self.config.iter_request_q_chunks([5, 3], 4))
        self.assertEqual(chunks, [(0, 4, 0, 1, [4]), (4, 8, 0, 2, [1, 3])])

    def test_shared_output_and_mask_reach_each_topk_backend_mode(self):
        calls = []
        fake = ModuleType("sgl_kernel")

        def capture(*args, **kwargs):
            calls.append(kwargs)
            return kwargs["out"]

        fake.fast_topk_transform_fused = capture
        fake.fast_topk_transform_ragged_fused = capture
        fake.fast_topk_v2 = capture
        metadata = SimpleNamespace(page_table_1="original_page_table")
        backend = self.namespace["DSATopKBackend"].SGL_KERNEL
        methods = self.namespace["TopkTransformMethod"]
        mask, output = object(), object()
        with patch.dict(sys.modules, {"sgl_kernel": fake}):
            for method, raw in (
                (methods.PAGED, False),
                (methods.RAGGED, False),
                (methods.PAGED, True),
            ):
                result = backend.topk_transform(
                    "compressed_logits",
                    "lengths",
                    2048,
                    method,
                    metadata,
                    cu_seqlens_q_topk="original_cu_q",
                    topk_indices_offset="original_offsets",
                    row_starts="zeros",
                    force_unfused_topk=raw,
                    q_mask=mask,
                    out=output,
                )
                self.assertIs(result, output)
                self.assertIs(calls[-1]["q_mask"], mask)
                self.assertEqual(calls[-1]["row_starts"], "zeros")
            self.assertEqual(calls[0]["page_table_size_1"], "original_page_table")
            self.assertEqual(calls[1]["topk_indices_offset"], "original_offsets")

    def test_real_source_memory_policy(self):
        gpu = SimpleNamespace(
            mem_get_info=lambda _: (10 * 2**30, 80 * 2**30),
            get_device_properties=lambda _: SimpleNamespace(total_memory=80 * 2**30),
        )
        self.namespace["torch"] = SimpleNamespace(cuda=gpu)
        instance = self.indexer()
        self.assertEqual(instance._should_chunk_mqa_logits(8, 128, 0), (False, 0))
        with patch.dict("os.environ", {}, clear=True):
            need_chunk, budget = instance._should_chunk_mqa_logits(8192, 1048576, 0)
        self.assertTrue(need_chunk)
        self.assertEqual(budget, int(10 * 2**30 * 0.45))

    def test_paged_ragged_and_raw_use_expected_sgl_kernels(self):
        calls = []
        fake = ModuleType("sgl_kernel")
        fake.fast_topk_transform_fused = lambda **kw: calls.append(("paged", kw))
        fake.fast_topk_transform_ragged_fused = lambda **kw: calls.append(
            ("ragged", kw)
        )
        fake.fast_topk_v2 = lambda *a, **kw: calls.append(("raw", kw))

        class Table:
            def __getitem__(self, key):
                self.key = key
                return "expanded_page_table"

        table = Table()
        metadata = SimpleNamespace(page_table_1=table)
        backend = self.namespace["DSATopKBackend"].SGL_KERNEL
        methods = self.namespace["TopkTransformMethod"]
        with patch.dict(sys.modules, {"sgl_kernel": fake}):
            backend.topk_transform(
                "logits",
                "lengths",
                2048,
                methods.PAGED,
                metadata,
                cu_seqlens_q_topk="cu_q",
                row_starts="ks",
                batch_idx_list=[0, 0, 1],
            )
            self.assertEqual(table.key, [0, 0, 1])
            self.assertEqual(calls[-1][1]["page_table_size_1"], "expanded_page_table")
            backend.topk_transform(
                "logits",
                "lengths",
                2048,
                methods.RAGGED,
                metadata,
                topk_indices_offset="offsets",
                row_starts="ks",
            )
            self.assertEqual(calls[-1][1]["topk_indices_offset"], "offsets")
            backend.topk_transform(
                "logits",
                "lengths",
                2048,
                methods.PAGED,
                metadata,
                row_starts="ks",
                force_unfused_topk=True,
            )
        self.assertEqual([name for name, _ in calls], ["paged", "ragged", "raw"])


@unittest.skipIf(torch is None, "PyTorch CPU is required")
class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.torch_patch = patch.object(bench, "torch", torch)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    @unittest.skipUnless(SGLANG_ROOT.is_dir(), "SGLang source checkout is required")
    def test_sglang_plan_matches_real_deepgemm_metadata_for_every_chunk(self):
        config_module = bench.load_module(
            SGLANG_ROOT / "python/sglang/srt/layers/attention/dsa/long_context_topk.py",
            "_test_real_sglang_metadata_plan",
        )
        for package in ("deep_gemm", "sgl_deep_gemm"):
            path = PATH.parents[1] / package / "__init__.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "build_mqa_backward_metadata"
            )
            namespace = {"torch": torch}
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
            helper = namespace[node.name]
            for cap, lengths in ((8, [5, 7]), (3, [5, 7]), (0, [33, 17])):
                with self.subTest(package=package, cap=cap):
                    ks = torch.tensor(
                        [0] * lengths[0] + [10000] * lengths[1], dtype=torch.int32
                    )
                    ke = ks + torch.tensor(
                        [7000 + 8 * row for length in lengths for row in range(length)],
                        dtype=torch.int32,
                    )
                    calls = []

                    def build(*args, **kwargs):
                        calls.append(1)
                        return helper(*args, **kwargs)

                    config = config_module.NSALongContextTopKConfig(
                        backward_length=4096, max_rows_per_chunk=cap
                    )
                    plan = config_module.build_mqa_backward_metadata(
                        ks,
                        ke,
                        lengths,
                        config,
                        2048,
                        SimpleNamespace(build_mqa_backward_metadata=build),
                    )
                    self.assertEqual(len(calls), 1)
                    ranges = list(
                        config_module.iter_request_q_chunks(lengths, plan.max_rows)
                    )
                    for chunk, (start, end, _, _, fragment_lengths) in zip(
                        plan.chunks, ranges
                    ):
                        windows, blocks, rows, width = helper(
                            ks[start:end],
                            ke[start:end],
                            fragment_lengths,
                            backward_len=4096,
                        )
                        torch.testing.assert_close(chunk.window_starts, windows)
                        torch.testing.assert_close(chunk.q_block_starts, blocks)
                        torch.testing.assert_close(chunk.topk_row_starts, rows)
                        self.assertGreaterEqual(chunk.logits_width, width)
                        self.assertTrue(
                            bool(
                                (
                                    chunk.topk_row_starts + chunk.topk_lengths
                                    <= chunk.logits_width
                                ).all()
                            )
                        )

    def test_window_audit_uses_each_rows_offset_in_shared_output(self):
        rows = torch.tensor([0, 2, 3], dtype=torch.int32)
        starts = torch.tensor([10, 12, 13], dtype=torch.int32)
        ends = starts + 4
        q = torch.arange(3)[:, None, None]
        weights = torch.ones(3, 1)
        shared = torch.arange(10, 18)[None, :] + torch.arange(3)[:, None] * 100
        args = SimpleNamespace(check_max_rows=0, check_rows=2, atol=0, rtol=0)
        dg = SimpleNamespace(
            fp8_mqa_logits=lambda q, kv, weights, ks, ke, **kw: (
                torch.arange(kw["max_seqlen_k"])[None, :]
                + ks[:, None]
                + q[:, 0, 0, None] * 100
            ).float()
        )
        for output, correct in ((shared, True), (shared + rows[:, None], False)):
            audit = bench.Audit(dg, args, 3)
            audit.check_window_logits(
                (q, None, weights, starts, ends), output.float(), rows
            )
            self.assertEqual(audit.stats["window_logit_bad_values"] == 0, correct)

    def compare(self, full, outputs, mode="raw", data=None):
        if data is None:
            rows, width = full.shape
            data = {
                "q_fp8": torch.arange(rows)[:, None, None],
                "k_fp8": torch.empty(width, 1),
                "k_scale": torch.ones(width),
                "weights": torch.ones(rows, 1),
                "ks": torch.zeros(rows, dtype=torch.int32),
                "ke": torch.full((rows,), width, dtype=torch.int32),
                "request_q_lengths": [rows],
                "request_k_lengths": [width],
            }
        dg = SimpleNamespace(
            fp8_mqa_logits=lambda q, *a, **kw: full[q[:, 0, 0].long()].clone()
        )
        return bench.compare_outputs(dg, bench.parse_args([]), data, outputs, mode)

    def test_tied_scores_allow_different_unique_indices(self):
        full = torch.zeros(1, 4096)
        outputs = {
            "original": torch.arange(2048)[None, :],
            "custom": torch.arange(2048, 4096)[None, :],
        }
        result = self.compare(full, outputs)
        self.assertEqual(result["custom_original_index_overlap"], 0)
        self.assertTrue(result["full_coverage"])
        self.assertTrue(all(p["bad_rows"] == 0 for p in result["paths"].values()))

    def test_rejects_duplicate_invalid_missing_and_lower_score_indices(self):
        full = torch.arange(4096, dtype=torch.float32)[None, :]
        original = torch.arange(2048, 4096)[None, :]
        for value, metric in (
            (2049, "duplicate_ids"),
            (-2, "invalid_ids"),
            (-1, "wrong_valid_count_rows"),
            (0, "max_selected_score_error"),
        ):
            with self.subTest(value=value):
                custom = original.clone()
                custom[0, 0] = value
                paths = self.compare(full, {"original": original, "custom": custom})[
                    "paths"
                ]
                self.assertEqual(paths["original"]["bad_rows"], 0)
                self.assertEqual(paths["custom"]["bad_rows"], 1)
                self.assertGreater(paths["custom"][metric], 0)

    def test_multi_request_index_transforms_and_short_rows(self):
        ks = torch.tensor([0, 0, 4096], dtype=torch.int32)
        ke = torch.tensor([4096, 17, 10240], dtype=torch.int32)
        data = {
            "q_fp8": torch.arange(3)[:, None, None],
            "k_fp8": torch.empty(10240, 1),
            "k_scale": torch.ones(10240),
            "weights": torch.ones(3, 1),
            "ks": ks,
            "ke": ke,
            "request_q_lengths": [2, 1],
            "request_k_lengths": [4096, 6144],
            "page_table_1": torch.stack(
                (torch.arange(6144).flip(0) * 2, torch.arange(6144) * 3 + 30000)
            ).int(),
        }
        full = torch.arange(10240, dtype=torch.float32).repeat(3, 1)
        local = torch.full((3, 2048), -1, dtype=torch.int64)
        local[0] = torch.arange(2048, 4096)
        local[1, :17] = torch.arange(17)
        local[2] = torch.arange(4096, 6144)
        for mode in ("raw", "ragged", "paged"):
            with self.subTest(mode=mode):
                if mode == "raw":
                    output = local
                elif mode == "ragged":
                    output = torch.where(local >= 0, local + ks[:, None], local)
                else:
                    table = data["page_table_1"][[0, 0, 1]].long()
                    output = torch.where(
                        local >= 0, table.gather(1, local.clamp(min=0)), local
                    )
                result = self.compare(
                    full, {"original": output, "custom": output.clone()}, mode, data
                )
                self.assertTrue(
                    all(p["bad_rows"] == 0 for p in result["paths"].values())
                )
        # A physical ID from another request must not decode as valid padding.
        invalid = torch.full((3, 1), -1, dtype=torch.int32)
        invalid[2, 0] = 4095
        self.assertEqual(int(bench.decode_indices(invalid, data, "ragged")[2, 0]), -2)
        invalid[2, 0] = 30001  # Not a slot in request 1's physical page table.
        self.assertEqual(int(bench.decode_indices(invalid, data, "paged")[2, 0]), -2)


if __name__ == "__main__":
    unittest.main()
