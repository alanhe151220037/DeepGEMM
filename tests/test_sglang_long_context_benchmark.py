"""CPU checks for benchmark planning and actual SGLang backend dispatch."""

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
        args = bench.parse_args(["--preset", "chunked", "--free-mem-gb", "0.125"])
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
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                bench.parse_args(argv)

    def test_statistics_and_empty_ranks(self):
        self.assertEqual(bench.summary([4, 1, 3, 2])["p50"], 2)
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
