"""Compare the actual SGLang ragged TopK call paths using prepared indexer inputs.

Examples (run after building/installing this DeepGEMM and sgl_kernel):
  python tests/bench_sglang_long_context_topk.py --sglang-root ../sglang-taas-approaching --preset smoke --modes all
  python tests/bench_sglang_long_context_topk.py --sglang-root ../sglang-taas-approaching --preset long --json long.json
  python tests/bench_sglang_long_context_topk.py --requests 32768:41,24576:25 --cp-size 8 --backward-len 4096 --modes all
  python tests/bench_sglang_long_context_topk.py --preset chunked --max-rows 128 --modes all
  python tests/bench_sglang_long_context_topk.py --preset mixed --modes all

The exercised path starts with Q/KV/weights and request metadata already
prepared. It includes the original indexer methods' allocations, metadata,
chunking, .item() synchronizations, threshold reduction and index transforms.
KV-cache gathering, projections/quantization, distributed CP communication,
and downstream sparse attention are outside this operator-chain benchmark.
CP here means strided Q positions, not the separate SGLang CP dispatch path.

Selected methods are compiled directly from --sglang-root, without importing
the serving engine. Their bodies are not rewritten. The pool adapter returns
the prepared KV, and PP receive is disabled. The real sgl_kernel backend is
used. Source hashes are recorded to identify the code actually exercised.

The original path retains SGLang's memory planning; --memory-policy and
--free-mem-gb control only that path. The custom path uses only the optional
--max-rows cap (128 for the chunked preset, otherwise 0 for all Q rows).
The first warmup builds its metadata once; measured runs reuse it, matching
subsequent layers of the same forward.
Profiled runs disable overflow logging. Checks run separately and always
inspect overflow, irrespective of that log switch.

--input accepts a torch.save dictionary (weights_only=True): q_fp8, k_fp8,
k_scale, weights, ks, ke, request_q_lengths, request_k_lengths, and optionally
page_table_1. Q/K use E4M3FN, weights/scales FP32, bounds/page table int32.
Bounds address concatenated request KV; ke is exclusive. Q lengths are local
rows already selected for the CP rank. A nonidentity page table is generated
when absent. Synthetic inputs test kernel behavior, not model-level quality.

Accuracy uses fresh stock FP8 DeepGEMM logits + torch.topk, not unquantized
model scores. Tied index choices are accepted when selected score multisets
match. Duplicates, invalid IDs, missing entries, score mismatches or candidate
overflow fail the run. --check-max-rows limits score checks to a prefix and is
reported as partial coverage; overflow is still checked for every row.
Performance output contains only actual CUDA kernel execution durations from
torch.profiler/CUPTI, averaged per launch for each kernel and launch geometry.
CPU work, launch gaps, memcpy and memset events are excluded. Warmup and
accuracy checks run outside profiling. These numbers do not measure end-to-end
latency. Full kernel symbols and launch counts are available with --json;
numbered variants in the console distinguish different symbols/geometries.
CUDA profiling support is required; there is no CUDA-event timing fallback.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
from dataclasses import replace
from enum import Enum, IntEnum, auto
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

torch = None  # --help and --dry-run also work without PyTorch/CUDA.
TOPK = 2048
PRESETS = {
    "smoke": ([(16384, 33), (8192, 17)], 1, 4096),
    "ragged": ([(32768, 41), (24576, 25)], 8, 4096),
    "chunked": ([(65536, 129), (49152, 97)], 1, 4096),
    "long": ([(1048576, 65536)], 8, 65536),
    "mixed": ([(262144, 5), (262145, 7), (8192, 3), (1048576, 9)], 1, 65536),
}
INDEXER_METHODS = {
    "_get_topk_ragged",
    "_get_topk_ragged_long_context",
    "_get_topk_ragged_short_context",
    "_should_chunk_mqa_logits",
    "_get_mqa_logits_budget_bytes",
    "_mqa_logits_free_mem_fraction",
    "_with_real_sm_count",
    "_pad_heads_for_deep_gemm",
    "_mask_init_and_local_tokens",
}
METADATA_METHODS = {
    "get_page_table_64",
    "get_page_table_1",
    "get_seqlens_expanded",
    "get_indexer_kvcache_range",
    "get_mqa_backward_metadata",
    "get_mqa_routing_metadata",
    "get_indexer_seq_len",
    "get_indexer_seq_len_cpu",
    "get_dsa_extend_len_cpu",
    "get_token_to_batch_idx",
    "get_topk_transform_offsets",
    "topk_transform",
}


def parse_requests(value):
    try:
        result = [tuple(map(int, item.split(":"))) for item in value.split(",")]
        if not result or any(len(x) != 2 or not 0 < x[1] <= x[0] for x in result):
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected KV_LENGTH:EXTEND_LENGTH[, ...], with 0 < extend <= KV"
        ) from exc


def local_q_lengths(requests, cp_size, cp_rank):
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("CP rank must be in [0, cp_size)")
    result = [len(range(cp_rank, extend, cp_size)) for _, extend in requests]
    if any(n == 0 for n in result):
        raise ValueError(
            "every request must have at least one Q row on the selected CP rank"
        )
    return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_nodes(nodes, namespace, filename):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)


def extract_class(path, source_name, target_name, methods, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    source = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == source_name
    )
    body = [
        n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    found = {n.name for n in body}
    if found != methods:
        raise RuntimeError(
            f"SGLang source contract changed: {path}: missing {methods-found}"
        )
    # Keep the source memory-planning constants as well as method bodies.
    body += [
        n
        for n in source.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id.startswith("_MQA_LOGITS_")
            for t in n.targets
        )
    ]
    node = ast.ClassDef(
        name=target_name, bases=[], keywords=[], body=body, decorator_list=[]
    )
    compile_nodes([node], namespace, path)
    return namespace[target_name]


def source_contract(root, namespace):
    attention = root / "python/sglang/srt/layers/attention"
    paths = {
        "indexer": attention / "dsa/dsa_indexer.py",
        "metadata": attention / "dsa_backend.py",
        "topk_backend": attention / "dsa/dsa_topk_backend.py",
        "config": attention / "dsa/long_context_topk.py",
        "cu_seqlens": attention / "dsa/dsa_backend_mtp_precompute.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"SGLang source file missing: {path}")
    config = load_module(paths["config"], "_bench_sglang_long_context_config")
    namespace.update(build_mqa_backward_metadata=config.build_mqa_backward_metadata)
    namespace.update(build_mqa_routing_metadata=config.build_mqa_routing_metadata)
    backend_tree = ast.parse(paths["topk_backend"].read_text(encoding="utf-8"))
    classes = [
        n
        for n in backend_tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in {"TopkTransformMethod", "DSATopKBackend"}
    ]
    if len(classes) != 2:
        raise RuntimeError("SGLang TopK backend API changed")
    compile_nodes(classes, namespace, paths["topk_backend"])
    cu_tree = ast.parse(paths["cu_seqlens"].read_text(encoding="utf-8"))
    compile_nodes(
        [
            next(
                n
                for n in cu_tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "compute_cu_seqlens"
            )
        ],
        namespace,
        paths["cu_seqlens"],
    )
    indexer = extract_class(
        paths["indexer"], "Indexer", "SourceIndexer", INDEXER_METHODS, namespace
    )
    metadata = extract_class(
        paths["metadata"],
        "DSAIndexerMetadata",
        "SourceMetadata",
        METADATA_METHODS,
        namespace,
    )
    hashes = {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in paths.items()
    }
    return config, indexer, metadata, hashes


class Flag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class CudaView:
    def __init__(self, free_bytes, total_bytes):
        self.free_bytes, self.total_bytes = free_bytes, total_bytes

    def mem_get_info(self, *args, **kwargs):
        return self.free_bytes, self.total_bytes

    def __getattr__(self, name):
        return getattr(torch.cuda, name)


class TorchView:
    """Override only source-method memory queries; all tensor ops are real torch."""

    def __init__(self, free_bytes=None, total_bytes=None):
        self.cuda = (
            CudaView(free_bytes, total_bytes) if free_bytes is not None else torch.cuda
        )

    def __getattr__(self, name):
        return getattr(torch, name)


class Recorder:
    def __init__(self):
        self.audit = None

    def call(self, name, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if self.audit is not None:
            self.audit.observe(name, args, kwargs, result)
        return result


class DeepGemmView:
    def __init__(self, module, recorder):
        self.module, self.recorder = module, recorder

    def __getattr__(self, name):
        fn = getattr(self.module, name)
        if self.recorder.audit is None:
            return fn
        return lambda *a, **k: self.recorder.call(name, fn, *a, **k)


def make_page_table(k_lengths, device):
    # Nonidentity page mapping detects missing/wrong index transforms.
    pages = [math.ceil(n / 64) for n in k_lengths]
    permutation = torch.randperm(sum(pages), device=device, dtype=torch.int64)
    table = torch.full(
        (len(pages), max(k_lengths)), -1, device=device, dtype=torch.int32
    )
    offset = 0
    for r, (n, count) in enumerate(zip(k_lengths, pages)):
        logical = torch.arange(n, device=device)
        table[r, :n] = (
            permutation[offset : offset + count][logical // 64] * 64 + logical % 64
        ).int()
        offset += count
    return table.contiguous()


def prepare_inputs(args, device):
    if args.input:
        data = torch.load(args.input, map_location=device, weights_only=True)
        required = {
            "q_fp8",
            "k_fp8",
            "k_scale",
            "weights",
            "ks",
            "ke",
            "request_q_lengths",
            "request_k_lengths",
        }
        if not required <= data.keys():
            raise ValueError(f"input capture missing {required-data.keys()}")
        data = dict(data)
    else:
        q_lengths = local_q_lengths(args.requests, args.cp_size, args.cp_rank)
        k_lengths = [k for k, _ in args.requests]
        num_q, num_k = sum(q_lengths), sum(k_lengths)
        data = {
            "q_fp8": torch.randn(num_q, 32, 128, device=device).to(torch.float8_e4m3fn),
            "k_fp8": torch.randn(num_k, 128, device=device).to(torch.float8_e4m3fn),
            "k_scale": torch.rand(num_k, device=device) + 0.5,
            "weights": torch.randn(num_q, 32, device=device),
            "request_q_lengths": q_lengths,
            "request_k_lengths": k_lengths,
        }
        if args.distribution == "uniform":
            data["weights"].fill_(1 / math.sqrt(32 * 128))
            data["k_scale"].fill_(1)
        elif args.distribution == "ties":
            data["weights"].zero_()
        starts, ends, base = [], [], 0
        for (k_len, extend), q_len in zip(args.requests, q_lengths):
            starts.append(torch.full((q_len,), base, dtype=torch.int32, device=device))
            ends.append(
                torch.arange(
                    k_len - extend + args.cp_rank + 1,
                    k_len + 1,
                    args.cp_size,
                    dtype=torch.int32,
                    device=device,
                )
                + base
            )
            base += k_len
        data["ks"], data["ke"] = torch.cat(starts), torch.cat(ends)
    q_lengths = list(map(int, data["request_q_lengths"]))
    k_lengths = list(map(int, data["request_k_lengths"]))
    if (
        len(q_lengths) != len(k_lengths)
        or not q_lengths
        or min(q_lengths + k_lengths) <= 0
    ):
        raise ValueError("request Q/K lengths must be positive matching lists")
    data["request_q_lengths"], data["request_k_lengths"] = q_lengths, k_lengths
    data["weights"] = (
        data["weights"].squeeze(-1) if data["weights"].ndim == 3 else data["weights"]
    )
    expected = {
        "q_fp8": ((sum(q_lengths), 32, 128), torch.float8_e4m3fn),
        "k_fp8": ((sum(k_lengths), 128), torch.float8_e4m3fn),
        "k_scale": ((sum(k_lengths),), torch.float32),
        "weights": ((sum(q_lengths), 32), torch.float32),
        "ks": ((sum(q_lengths),), torch.int32),
        "ke": ((sum(q_lengths),), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        value = data[name]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"{name}: expected {shape}, {dtype}; got {value.shape}, {value.dtype}"
            )
        data[name] = value.contiguous()
    q_base = k_base = 0
    for nq, nk in zip(q_lengths, k_lengths):
        ks, ke = data["ks"][q_base : q_base + nq], data["ke"][q_base : q_base + nq]
        if not bool(((ks == k_base) & (ke >= ks) & (ke <= k_base + nk)).all()):
            raise ValueError(
                "ks/ke do not describe concatenated request-local KV ranges"
            )
        q_base += nq
        k_base += nk
    if "page_table_1" not in data:
        data["page_table_1"] = make_page_table(k_lengths, device)
    table = data["page_table_1"]
    if (
        table.dtype != torch.int32
        or table.ndim != 2
        or table.shape[0] != len(k_lengths)
        or table.shape[1] < max(k_lengths)
    ):
        raise ValueError("page_table_1 must be int32 [requests, >= max(K lengths)]")
    data["page_table_1"] = table.contiguous()
    return data


def make_metadata(cls, namespace, data, mode):
    device = data["q_fp8"].device
    q_lengths, k_lengths = data["request_q_lengths"], data["request_k_lengths"]
    q_tensor = torch.tensor(q_lengths, dtype=torch.int32, device=device)
    k_tensor = torch.tensor(k_lengths, dtype=torch.int32, device=device)
    cu_q = torch.cat((q_tensor.new_zeros(1), q_tensor.cumsum(0, dtype=torch.int32)))
    cu_k = torch.cat((k_tensor.new_zeros(1), k_tensor.cumsum(0, dtype=torch.int32)))
    batch_ids = [r for r, n in enumerate(q_lengths) for _ in range(n)]
    metadata = cls()
    metadata.force_unfused_topk = mode == "raw"
    methods = namespace["TopkTransformMethod"]
    metadata.topk_transform_method = (
        methods.RAGGED if mode == "ragged" else methods.PAGED
    )
    metadata.topk_backend = namespace["DSATopKBackend"].SGL_KERNEL
    metadata.attn_metadata = SimpleNamespace(
        page_table_1=data["page_table_1"],
        real_page_table=(data["page_table_1"][:, ::64] // 64).contiguous(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        topk_indices_offset=data["ks"] if mode == "ragged" else None,
        dsa_seqlens_expanded=(data["ke"] - data["ks"]).contiguous(),
        indexer_k_start_end=(data["ks"], data["ke"]),
        mqa_backward_metadata_cache={},
        mqa_routing_metadata_cache={},
        indexer_seq_lens=k_tensor,
        indexer_seq_lens_cpu=k_tensor.cpu(),
        dsa_extend_seq_lens_list=q_lengths,
        token_to_batch_idx=batch_ids,
    )
    return metadata


def build_runner(cls, config, namespace, data, metadata, custom):
    indexer = cls()
    indexer.n_heads, indexer.head_dim, indexer.index_topk = 32, 128, TOPK
    indexer.num_init_tokens = indexer.num_local_tokens = 0
    indexer.layer_id, indexer.logits_with_pp_recv = 0, False
    indexer.long_context_topk_config = replace(config, enabled=custom)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        seq_lens_cpu=metadata.get_indexer_seq_len_cpu(),
        extend_seq_lens_cpu=data["request_q_lengths"],
    )
    weights = data["weights"].unsqueeze(-1)
    return lambda: indexer._get_topk_ragged(
        False, forward_batch, 0, data["q_fp8"], weights, metadata
    )


def warmup_runners(runners, warmup):
    for _ in range(warmup):
        for run in runners.values():
            result = run()
            del result
    torch.cuda.synchronize()


def kernel_label(name):
    # Keep the actual function name, omitting the often very long C++ signature.
    symbol = name.split("<", 1)[0].split("(", 1)[0].split()[-1]
    if "sm100_long_context_mqa_logits<" in name:
        flags = name.split("sm100_long_context_mqa_logits<", 1)[1].split(",")[:4]
        if len(flags) == 4:
            if flags[3].strip() == "true":
                symbol += " (filter_candidates)"
            elif flags[2].strip() == "true":
                symbol += " (backward_logits)"
    return symbol


def summarize_kernel_trace(trace):
    """Aggregate GPU kernel activities, never CPU/runtime API event durations."""
    groups = {}
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or "kernel" not in {
            part.strip().lower() for part in event.get("cat", "").split(",")
        }:
            continue
        name, duration = event.get("name"), event.get("dur")
        if not isinstance(name, str) or not name:
            raise RuntimeError("CUDA kernel activity has no name")
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise RuntimeError(f"Invalid CUDA duration for {name}: {duration!r}")
        args = event.get("args", {})
        geometry = {key: args.get(key) for key in ("device", "grid", "block")}
        key = (name, json.dumps(geometry, sort_keys=True))
        entry = groups.setdefault(
            key, {"name": name, **geometry, "launches": 0, "total_us": 0.0}
        )
        entry["launches"] += 1
        entry["total_us"] += duration
    if not groups:
        raise RuntimeError(
            "No CUDA kernel activities in the profiler trace. Check this PyTorch "
            "build's CUPTI support and profiler warnings; CPU/API timings cannot "
            "substitute for GPU kernel durations."
        )
    result, variants = [], {}
    for entry in groups.values():
        label = kernel_label(entry["name"])
        variants[label] = variants.get(label, 0) + 1
        result.append(
            {
                **{k: v for k, v in entry.items() if k != "total_us"},
                "label": f"{label} [v{variants[label]}]",
                "gpu_us_per_launch": entry["total_us"] / entry["launches"],
            }
        )
    return result


def profile_kernels(run, iterations):
    if (
        torch.profiler.ProfilerActivity.CUDA
        not in torch.profiler.supported_activities()
    ):
        raise RuntimeError("torch.profiler CUDA/CUPTI support is required")
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(iterations):
            output = run()
            del output
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix="mqa-kernel-trace-") as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text(encoding="utf-8"))
    return summarize_kernel_trace(trace)


def print_fields(values, indent=4):
    """One accuracy/config field per line, including nested path diagnostics."""
    for name, value in values.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{name}:")
            print_fields(value, indent + 2)
        else:
            print(f"{' ' * indent}{name}: {value}")


def print_kernel_profile(name, kernels):
    print(f"  {name} kernels (mean GPU us/launch):")
    for kernel in kernels:
        print(f"    {kernel['label']}: {kernel['gpu_us_per_launch']:.3f}")


class Audit:
    def __init__(self, dg, args, num_q):
        self.dg, self.args = dg, args
        self.limit = min(num_q, args.check_max_rows or num_q)
        self.cursor = self.chunk_start = 0
        self.stats = {
            "window_logit_max_abs_error": 0.0,
            "window_logit_bad_values": 0,
            "prefix_threshold_max_abs_error": 0.0,
            "prefix_threshold_bad_rows": 0,
            "prefix_score_index_bad_values": 0,
            "overflow_lanes": 0,
            "stored_candidates": 0,
            "invalid_candidates": 0,
            "bad_candidate_counts": 0,
            "max_lane_utilization": 0.0,
            "capacity_per_warp_per_chunk": [],
        }

    def check_window_logits(self, a, out, row_starts, q_mask=None):
        n = max(0, min(a[0].shape[0], self.limit - self.chunk_start))
        for start in range(0, n, self.args.check_rows):
            end = min(n, start + self.args.check_rows)
            ref = self.dg.fp8_mqa_logits(
                a[0][start:end],
                a[1],
                a[2][start:end],
                a[3][start:end],
                a[4][start:end],
                clean_logits=False,
                max_seqlen_k=out.shape[1],
            )
            valid = (
                torch.arange(out.shape[1], device=out.device)[None, :]
                < (a[4][start:end] - a[3][start:end])[:, None]
            )
            if q_mask is not None:
                valid &= q_mask[start:end, None]
            columns = torch.arange(out.shape[1], device=out.device)[None, :]
            offsets = columns + row_starts[start:end, None]
            actual = out[start:end].gather(
                1, offsets.long().clamp(max=out.shape[1] - 1)
            )
            delta = (ref - actual).abs()[valid]
            self.stats["window_logit_max_abs_error"] = max(
                self.stats["window_logit_max_abs_error"],
                float(delta.max()) if delta.numel() else 0,
            )
            self.stats["window_logit_bad_values"] += int(
                (
                    ~torch.isclose(
                        ref,
                        actual,
                        rtol=self.args.rtol,
                        atol=self.args.atol,
                    )
                    & valid
                ).sum()
            )

    def observe(self, name, a, kw, out):
        if name == "fp8_mqa_backward_logits":
            self.chunk_start, self.cursor = self.cursor, self.cursor + a[0].shape[0]
            self.pending_window_logits = (a, out)
        elif name == "fp8_mqa_backward_topk":
            logits, ks, ke, row_starts = a[:4]
            self.check_window_logits(
                *self.pending_window_logits, row_starts, kw.get("q_mask")
            )
            self.pending_window_logits = None
            scores, ids = out
            n = max(0, min(logits.shape[0], self.limit - self.chunk_start))
            lengths = (ke - ks).clamp(0, self.args.backward_len)
            if kw.get("q_mask") is not None:
                lengths = lengths.masked_fill(~kw["q_mask"], 0)
            for start in range(0, n, self.args.check_rows):
                end = min(n, start + self.args.check_rows)
                length = lengths[start:end]
                columns = torch.arange(logits.shape[1], device=logits.device)[None, :]
                valid = (columns >= row_starts[start:end, None]) & (
                    columns < (row_starts[start:end] + length)[:, None]
                )
                ref = logits[start:end].masked_fill(~valid, float("-inf"))
                thresholds = ref.topk(min(TOPK, ref.shape[1]), dim=1).values[:, -1]
                thresholds = torch.where(
                    length >= TOPK,
                    thresholds,
                    torch.full_like(thresholds, -torch.finfo(torch.float32).max),
                )
                actual = scores[start:end].amin(1)
                self.stats["prefix_threshold_max_abs_error"] = max(
                    self.stats["prefix_threshold_max_abs_error"],
                    float((actual - thresholds).abs().max()),
                )
                self.stats["prefix_threshold_bad_rows"] += int(
                    (
                        ~torch.isclose(
                            actual, thresholds, rtol=self.args.rtol, atol=self.args.atol
                        )
                    ).sum()
                )
                local = (
                    ids[start:end].long()
                    - kw["index_starts"][start:end, None]
                    + row_starts[start:end, None]
                )
                gathered = logits[start:end].gather(
                    1, local.clamp(0, logits.shape[1] - 1)
                )
                self.stats["prefix_score_index_bad_values"] += int(
                    (
                        (ids[start:end] >= 0)
                        & ~torch.isclose(
                            scores[start:end],
                            gathered,
                            rtol=self.args.rtol,
                            atol=self.args.atol,
                        )
                    ).sum()
                )
        elif name == "fp8_mqa_filter_candidates":
            scores, ids, counts, overflow = out
            capacity = scores.shape[2]
            lane_capacity = capacity // 32
            self.stats["capacity_per_warp_per_chunk"].append(capacity)
            self.stats["overflow_lanes"] += int(overflow.sum())
            self.stats["stored_candidates"] += int(counts.sum())
            self.stats["bad_candidate_counts"] += int(
                ((counts < 0) | (counts > lane_capacity)).sum()
            )
            self.stats["max_lane_utilization"] = max(
                self.stats["max_lane_utilization"], float(counts.max()) / lane_capacity
            )
            n = max(0, min(scores.shape[0], self.limit - self.chunk_start))
            slots = torch.arange(lane_capacity, device=scores.device)
            for start in range(0, n, self.args.check_rows):
                end = min(n, start + self.args.check_rows)
                values = scores[start:end].view(-1, 8, 32, lane_capacity)
                indices = ids[start:end].view_as(values)
                valid = slots < counts[start:end, :, :, None]
                bad = (
                    ~torch.isfinite(values)
                    | (values < a[5][start:end, None, None, None])
                    | (indices < 0)
                    | (
                        indices
                        >= (a[4][start:end] - a[3][start:end])[:, None, None, None]
                    )
                )
                self.stats["invalid_candidates"] += int((valid & bad).sum())


def decode_indices(indices, data, mode):
    result = indices.long().clone()
    if mode == "raw":
        return result
    if mode == "ragged":
        local = result - data["ks"][:, None]
        # A positive transformed ID below this request's base is invalid,
        # including base-1; it must not turn into a legitimate -1 padding slot.
        local = torch.where(local >= 0, local, torch.full_like(local, -2))
        return torch.where(result >= 0, local, result)
    start = 0
    for request, (nq, nk) in enumerate(
        zip(data["request_q_lengths"], data["request_k_lengths"])
    ):
        table = data["page_table_1"][request, :nk].long()
        ordered, logical = table.sort()
        if bool((ordered < 0).any()) or bool((ordered[1:] == ordered[:-1]).any()):
            raise ValueError(
                "each request page table must contain unique nonnegative physical slots"
            )
        values = result[start : start + nq]
        pos = torch.searchsorted(ordered, values.contiguous()).clamp(max=nk - 1)
        decoded = logical[pos]
        decoded = torch.where(
            ordered[pos] == values, decoded, torch.full_like(decoded, -2)
        )
        result[start : start + nq] = torch.where(values == -1, values, decoded)
        start += nq
    return result


def rank_quantiles(histogram):
    total = sum(histogram)
    if not total:
        return {"p50": None, "p95": None, "p99": None}
    result = {}
    for p in (0.5, 0.95, 0.99):
        target, cumulative = math.ceil(p * total), 0
        for rank, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                result[f"p{int(p*100)}"] = rank
                break
    return result


def compare_outputs(dg, args, data, outputs, mode):
    decoded = {name: decode_indices(out, data, mode) for name, out in outputs.items()}
    num_q, num_k = data["q_fp8"].shape[0], data["k_fp8"].shape[0]
    limit = min(num_q, args.check_max_rows or num_q)
    depth = min(args.rank_depth, num_k)
    metrics = {
        name: {
            "bad_rows": 0,
            "invalid_ids": 0,
            "duplicate_ids": 0,
            "wrong_valid_count_rows": 0,
            "max_selected_score_error": 0.0,
            "rank_above_topk": 0,
            "rank_censored": 0,
            "rank_histogram": [0] * (depth + 2),
        }
        for name in outputs
    }
    overlap_total = overlap_denominator = 0
    for start in range(0, limit, args.check_rows):
        end = min(limit, start + args.check_rows)
        ks, ke = data["ks"][start:end], data["ke"][start:end]
        full = dg.fp8_mqa_logits(
            data["q_fp8"][start:end],
            (data["k_fp8"], data["k_scale"]),
            data["weights"][start:end],
            ks,
            ke,
            clean_logits=False,
        )
        columns = torch.arange(num_k, device=full.device)[None, :]
        full.masked_fill_(
            (columns < ks[:, None]) | (columns >= ke[:, None]), float("-inf")
        )
        ranked = full.topk(depth, dim=1).values
        golden = ranked[:, :TOPK]
        if golden.shape[1] < TOPK:
            golden = torch.nn.functional.pad(
                golden, (0, TOPK - golden.shape[1]), value=float("-inf")
            )
        lengths = (ke - ks).long()
        expected_count = lengths.clamp(max=TOPK)
        for name, local_all in decoded.items():
            local = local_all[start:end]
            entry = metrics[name]
            valid = (local >= 0) & (local < lengths[:, None])
            invalid = (local != -1) & ~valid
            selected = full.gather(1, (local + ks[:, None]).clamp(0, num_k - 1))
            selected = selected.masked_fill(~valid, float("-inf"))
            sorted_scores = selected.sort(dim=1, descending=True).values
            close = torch.isclose(sorted_scores, golden, rtol=args.rtol, atol=args.atol)
            finite = torch.isfinite(sorted_scores) & torch.isfinite(golden)
            errors = (sorted_scores - golden).abs()[finite]
            entry["max_selected_score_error"] = max(
                entry["max_selected_score_error"],
                float(errors.max()) if errors.numel() else 0,
            )
            sorted_ids = local.masked_fill(~valid, -1).sort(dim=1).values
            duplicates = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (
                sorted_ids[:, 1:] >= 0
            )
            wrong_count = valid.sum(1) != expected_count
            bad_rows = ~close.all(1) | invalid.any(1) | duplicates.any(1) | wrong_count
            entry["bad_rows"] += int(bad_rows.sum())
            entry["invalid_ids"] += int(invalid.sum())
            entry["duplicate_ids"] += int(duplicates.sum())
            entry["wrong_valid_count_rows"] += int(wrong_count.sum())
            ranks = (
                torch.searchsorted(
                    (-ranked).contiguous(), (-selected).contiguous(), right=False
                )
                + 1
            )
            valid_ranks = ranks[valid]
            entry["rank_above_topk"] += int((valid_ranks > TOPK).sum())
            entry["rank_censored"] += int((valid_ranks > depth).sum())
            hist = torch.bincount(valid_ranks, minlength=depth + 2).cpu().tolist()
            entry["rank_histogram"] = [
                a + b for a, b in zip(entry["rank_histogram"], hist)
            ]
        ours, original = decoded["custom"][start:end], decoded["original"][start:end]
        sorted_original = original.sort(dim=1).values
        pos = torch.searchsorted(sorted_original.contiguous(), ours.contiguous()).clamp(
            max=TOPK - 1
        )
        overlap_total += int(
            ((ours >= 0) & (sorted_original.gather(1, pos) == ours)).sum()
        )
        overlap_denominator += int(expected_count.sum())
    for entry in metrics.values():
        entry["rank_quantiles"] = rank_quantiles(entry.pop("rank_histogram"))
    return {
        "checked_rows": limit,
        "total_rows": num_q,
        "full_coverage": limit == num_q,
        "rank_depth": depth,
        "censored_rank_value": depth + 1,
        "custom_original_index_overlap": overlap_total / max(overlap_denominator, 1),
        "paths": metrics,
    }


def git_revision(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sglang-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "sglang-taas-approaching",
    )
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--requests", type=parse_requests)
    parser.add_argument("--cp-size", type=int)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument("--backward-len", type=int)
    parser.add_argument(
        "--min-kv-length",
        type=int,
        help="request routing threshold; default 256K for mixed, otherwise 0 to exercise custom kernels",
    )
    parser.add_argument("--candidate-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="custom row cap; 0 uses all Q rows (default: 128 for chunked, otherwise 0)",
    )
    parser.add_argument(
        "--distribution", choices=("random", "uniform", "ties"), default="random"
    )
    parser.add_argument(
        "--modes", choices=("paged", "ragged", "raw", "all"), default="paged"
    )
    parser.add_argument("--memory-policy", choices=("fixed", "live"), default="fixed")
    parser.add_argument(
        "--free-mem-gb",
        type=float,
        help="original-path planning budget in GiB; defaults to live free memory after input setup",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--check-rows",
        type=int,
        default=16,
        help="rows of full logits per accuracy chunk",
    )
    parser.add_argument(
        "--check-max-rows",
        type=int,
        default=0,
        help="0 checks all rows; otherwise check this prefix only",
    )
    parser.add_argument("--rank-depth", type=int, default=16384)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check source contracts and shape plan without importing torch",
    )
    args = parser.parse_args(argv)
    requests, cp, backward = PRESETS[args.preset]
    args.requests = requests if args.requests is None else args.requests
    args.cp_size = cp if args.cp_size is None else args.cp_size
    args.backward_len = backward if args.backward_len is None else args.backward_len
    args.min_kv_length = (
        (256 * 1024 if args.preset == "mixed" else 0)
        if args.min_kv_length is None
        else args.min_kv_length
    )
    args.max_rows = (
        (128 if args.preset == "chunked" else 0)
        if args.max_rows is None
        else args.max_rows
    )
    if (
        min(args.warmup, args.iters, args.check_rows, args.backward_len) < 1
        or args.check_max_rows < 0
        or args.max_rows < 0
        or args.min_kv_length < 0
    ):
        parser.error(
            "warmup/iters/check-rows/backward-len must be positive; check-max-rows/max-rows must be nonnegative"
        )
    if args.rank_depth < TOPK or any(
        not math.isfinite(x) or x < 0 for x in (args.atol, args.rtol)
    ):
        parser.error("rank-depth must be >= 2048 and tolerances must be nonnegative")
    if not math.isfinite(args.candidate_multiplier) or args.candidate_multiplier <= 0:
        parser.error("candidate-multiplier must be finite and positive")
    if args.free_mem_gb is not None and (
        args.memory_policy != "fixed"
        or not math.isfinite(args.free_mem_gb)
        or args.free_mem_gb <= 0
    ):
        parser.error(
            "free-mem-gb requires a positive finite value and fixed memory policy"
        )
    try:
        local_q_lengths(args.requests, args.cp_size, args.cp_rank)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    global torch
    args = parse_args(argv)
    envs = SimpleNamespace(
        SGLANG_DSA_FUSE_TOPK=Flag(True),
        SGLANG_DSA_LONG_CONTEXT_TOPK_WARN_OVERFLOW=Flag(False),
        SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=Flag(
            float(os.environ.get("SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION", ".45"))
        ),
        SGLANG_OPT_USE_TOPK_V2=Flag(True),
    )
    namespace = {
        "torch": None,
        "contextlib": contextlib,
        "Enum": Enum,
        "IntEnum": IntEnum,
        "auto": auto,
        "envs": envs,
        "TYPE_CHECKING": False,
        "_is_cuda": True,
        "_is_hip": False,
        "_is_fp8_fnuz": False,
        "get_is_capture_mode": lambda: False,
        "get_server_args": lambda: SimpleNamespace(
            dsa_topk_backend="sgl-kernel", mem_fraction_static=None
        ),
        "ceil_align": lambda x, n: (x + n - 1) // n * n,
        "logger": logging.getLogger(__name__),
    }
    config_module, indexer_cls, metadata_cls, hashes = source_contract(
        args.sglang_root.resolve(), namespace
    )
    if args.dry_run:
        q_lengths = local_q_lengths(args.requests, args.cp_size, args.cp_rank)
        print(
            json.dumps(
                {
                    "source_contract": "OK",
                    "source_files": hashes,
                    "requests": args.requests,
                    "local_q_lengths": q_lengths,
                    "routing_threshold": args.min_kv_length,
                    "long_request_mask": [
                        k > args.min_kv_length for k, _ in args.requests
                    ],
                    "full_logits_gib": sum(q_lengths)
                    * sum(k for k, _ in args.requests)
                    * 4
                    / 2**30,
                    "capture_not_loaded": bool(args.input),
                },
                indent=2,
            )
        )
        return 0
    import torch as torch_module
    import deep_gemm

    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --dry-run for CPU source checks")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("The custom logits kernels require SM100")
    torch.manual_seed(args.seed)
    data = prepare_inputs(args, device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    planning_budget = (
        int(args.free_mem_gb * 2**30) if args.free_mem_gb is not None else free_bytes
    )
    if planning_budget > free_bytes:
        raise ValueError("free-mem-gb exceeds actual free GPU memory")
    namespace["torch"] = (
        TorchView(planning_budget, total_bytes)
        if args.memory_policy == "fixed"
        else torch
    )
    recorder = Recorder()
    namespace["deep_gemm"] = DeepGemmView(deep_gemm, recorder)
    namespace["get_token_to_kv_pool"] = lambda: SimpleNamespace(
        page_size=64,
        get_index_k_scale_buffer=lambda *a: (data["k_fp8"], data["k_scale"][:, None]),
    )
    config = config_module.NSALongContextTopKConfig.from_value(
        {
            "min_kv_length": args.min_kv_length,
            "backward_length": args.backward_len,
            "candidate_buffer_multiplier": args.candidate_multiplier,
            "max_rows_per_chunk": args.max_rows,
        }
    )
    report = {
        "settings": vars(args),
        "source_files": hashes,
        "sglang_commit": git_revision(args.sglang_root),
        "deepgemm_commit": git_revision(Path(__file__).resolve().parents[1]),
        "deepgemm_module": deep_gemm.__file__,
        "deepgemm_version": getattr(deep_gemm, "__version__", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "planning_budget_gib": planning_budget / 2**30,
        "request_q_lengths": data["request_q_lengths"],
        "request_k_lengths": data["request_k_lengths"],
        "execution_boundary": "prepared Q/KV/weights and request metadata -> transformed TopK indices; eager execution",
        "performance_metric": "mean CUDA kernel execution time per launch in microseconds (CUPTI)",
        "overflow_logging_during_profiling": False,
        "results": {},
    }
    modes = ("paged", "ragged", "raw") if args.modes == "all" else (args.modes,)
    failed = False
    for mode in modes:
        print(
            f"Running {mode}: Q={data['q_fp8'].shape[0]}, K={data['k_fp8'].shape[0]} ...",
            flush=True,
        )
        metadata = make_metadata(metadata_cls, namespace, data, mode)
        runners = {
            name: build_runner(indexer_cls, config, namespace, data, metadata, custom)
            for name, custom in (("original", False), ("custom", True))
        }
        warmup_runners(runners, args.warmup)
        profiles = {
            name: profile_kernels(run, args.iters) for name, run in runners.items()
        }
        outputs = {"original": runners["original"]()}
        audit = Audit(deep_gemm, args, data["q_fp8"].shape[0])
        recorder.audit = audit
        try:
            outputs["custom"] = runners["custom"]()
        finally:
            recorder.audit = None
        accuracy = compare_outputs(deep_gemm, args, data, outputs, mode)
        mode_failed = any(p["bad_rows"] for p in accuracy["paths"].values()) or any(
            audit.stats[k]
            for k in (
                "window_logit_bad_values",
                "prefix_threshold_bad_rows",
                "prefix_score_index_bad_values",
                "overflow_lanes",
                "invalid_candidates",
                "bad_candidate_counts",
            )
        )
        failed |= mode_failed
        report["results"][mode] = {
            "passed": not mode_failed,
            "kernels": profiles,
            "audit": audit.stats,
            "accuracy": accuracy,
        }
        print(
            f"\n{mode}: {'FAIL' if mode_failed else 'PASS'}",
            flush=True,
        )
        for name, profile_result in profiles.items():
            print_kernel_profile(name, profile_result)
        print("  audit:")
        print_fields(audit.stats)
        print("  accuracy:")
        print_fields(accuracy)
        print(flush=True)
        del outputs, metadata, runners, audit
        gc.collect()
        torch.cuda.empty_cache()
    report["passed"] = not failed
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"Saved {args.json.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
