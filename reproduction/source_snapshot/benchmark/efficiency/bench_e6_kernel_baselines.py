"""E6 kernel-only baselines for dense, sparse, and GAS GQA decode.

The benchmark intentionally keeps support construction and library planning out of
the timed region.  It records unavailable external backends instead of replacing
them with synthetic timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import (
    affine_tokens,
    balanced_counts,
    cuda_event_samples,
    error_metrics,
    fp32_reference,
    load_profile_spec,
    padded_tensor,
    quantile,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "benchmark" / "results_efficiency" / "e6_kernel_baselines"
CUDA_SOURCE = ROOT / "csrc" / "src" / "gqa_group_decode.cu"

METHODS = (
    "fa2",
    "flashinfer",
    "quest",
    "quest_twi",
    "gas_per_head",
    "gas",
)
MAIN_METHODS = ("fa2", "flashinfer", "quest", "quest_twi", "gas")
ALL_METHODS = METHODS


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def get_gas_module():
    return load(
        name="twilight_e6_gqa_group_decode",
        sources=[str(CUDA_SOURCE)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


def dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference dense GQA decode, computed in FP32 outside the timed path."""
    group_size = q.shape[1] // k.shape[1]
    scale = 1.0 / math.sqrt(q.shape[-1])
    output = torch.empty_like(q, dtype=torch.float32)
    for batch_idx in range(q.shape[0]):
        for kv_head in range(k.shape[1]):
            key = k[batch_idx, kv_head].float()
            value = v[batch_idx, kv_head].float()
            query = q[batch_idx, kv_head * group_size : (kv_head + 1) * group_size].float()
            probs = torch.softmax(torch.matmul(query, key.transpose(0, 1)) * scale, dim=-1)
            output[batch_idx, kv_head * group_size : (kv_head + 1) * group_size] = torch.matmul(probs, value)
    return output


def support_lists(tensor: torch.Tensor) -> List[List[int]]:
    values = tensor.detach().cpu().tolist()
    return [[int(token) for token in row if int(token) >= 0] for batch in values for row in batch]


def build_union_from_heads(
    head_tensor: torch.Tensor, batch_size: int, num_heads: int, num_kv_heads: int
) -> tuple[torch.Tensor, List[int], dict]:
    group_size = num_heads // num_kv_heads
    head_lists = support_lists(head_tensor)
    union_lists: List[List[int]] = []
    for batch_idx in range(batch_size):
        for kv_head in range(num_kv_heads):
            merged: List[int] = []
            group_start = batch_idx * num_heads + kv_head * group_size
            seen = set(merged)
            for local_head in range(group_size):
                head = head_lists[group_start + local_head]
                for token in head:
                    if token not in seen:
                        seen.add(token)
                        merged.append(token)
            union_lists.append(merged)
    width = max(len(values) for values in union_lists)
    result = torch.full(
        (batch_size, num_kv_heads, width),
        -1,
        dtype=torch.int32,
        device=head_tensor.device,
    )
    for linear, values in enumerate(union_lists):
        batch_idx = linear // num_kv_heads
        kv_head = linear % num_kv_heads
        result[batch_idx, kv_head, : len(values)] = torch.tensor(
            values, dtype=torch.int32, device=result.device
        )
    stats = {
        "union_min": min(map(len, union_lists)),
        "union_mean": statistics.fmean(map(len, union_lists)),
        "union_max": max(map(len, union_lists)),
        "union_width": width,
    }
    return result.contiguous(), [len(values) for values in union_lists], stats


def build_consistent_indices(
    spec: dict,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    exception_budget: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Build synthetic masks that preserve the profile's set relationships.

    The old E4 generator independently sampled each head.  Here the original
    head supports are generated first, then group shared tokens and exceptions
    are selected as subsets of those supports.
    """
    length = spec["length"]
    group_size = num_heads // num_kv_heads
    head_counts = balanced_counts(spec["headwise_support_mean"], batch_size * num_heads)
    pre_union_target = round(spec["profile_pre_loaded_kv_mean"] / num_kv_heads)
    pre_union_counts = balanced_counts(pre_union_target, batch_size * num_kv_heads)
    shared_counts = balanced_counts(spec["shared_support_mean"], batch_size * num_kv_heads)
    exception_mean = spec["complement_support_mean"]
    exception_counts = balanced_counts(exception_mean, batch_size * num_heads)

    head_lists: List[List[int]] = []
    shared_lists: List[List[int]] = []
    exception_lists: List[List[int]] = []
    gas_lists: List[List[int]] = []
    original_union_counts: List[int] = []
    final_union_counts: List[int] = []
    distinct_exception_counts: List[int] = []

    for batch_idx in range(batch_size):
        for kv_head in range(num_kv_heads):
            group_linear = batch_idx * num_kv_heads + kv_head
            union_tokens = affine_tokens(
                length,
                pre_union_counts[group_linear],
                seed + 100003 + group_linear * 29,
            )
            per_head: List[List[int]] = [[] for _ in range(group_size)]
            # Cover every token in the target union once, then add overlap
            # memberships until each head reaches its profile count.
            for token_idx, token in enumerate(union_tokens):
                per_head[token_idx % group_size].append(token)
            for local_head in range(group_size):
                q_head = kv_head * group_size + local_head
                head_linear = batch_idx * num_heads + q_head
                target = head_counts[head_linear]
                if target < len(per_head[local_head]):
                    raise ValueError("Profile head support is too small to cover the target union")
                order = affine_tokens(
                    len(union_tokens),
                    len(union_tokens),
                    seed + 200003 + head_linear * 43,
                )
                for position in order:
                    token = union_tokens[position]
                    if token not in per_head[local_head]:
                        per_head[local_head].append(token)
                        if len(per_head[local_head]) == target:
                            break
                if len(per_head[local_head]) != target:
                    raise ValueError("Could not construct the requested profile head support")
                head_lists.append(per_head[local_head])

            shared_target = shared_counts[group_linear]
            shared = union_tokens[:shared_target]
            shared_lists.append(shared)
            shared_set = set(shared)
            local_exceptions: List[List[int]] = []
            for local_head in range(group_size):
                q_head = kv_head * group_size + local_head
                head_linear = batch_idx * num_heads + q_head
                candidates = [token for token in per_head[local_head] if token not in shared_set]
                exception_target = exception_counts[head_linear]
                if exception_target > exception_budget or exception_target > len(candidates):
                    raise ValueError("Profile exception target exceeds available head support")
                exception = candidates[:exception_target]
                local_exceptions.append(exception)
                exception_lists.append(exception)
                gas_lists.append(shared + exception)
            original_union_counts.append(len(set().union(*map(set, per_head))))
            final_union_counts.append(len(set().union(*map(set, [shared] + local_exceptions))))
            distinct_exception_counts.append(len({tuple(values) for values in local_exceptions}))

    if any(len(values) != len(set(values)) for values in head_lists + shared_lists + exception_lists):
        raise RuntimeError("Consistent synthetic generator produced duplicate support tokens")
    for group_linear, shared in enumerate(shared_lists):
        for local_head in range(group_size):
            exception = exception_lists[group_linear * group_size + local_head]
            if set(shared).intersection(exception):
                raise RuntimeError("Shared and exception support overlap")

    stats = {
        "path_a_valid_support_min": min(map(len, head_lists)),
        "path_a_valid_support_mean": statistics.fmean(map(len, head_lists)),
        "path_a_valid_support_max": max(map(len, head_lists)),
        "gas_valid_support_min": min(map(len, gas_lists)),
        "gas_valid_support_mean": statistics.fmean(map(len, gas_lists)),
        "gas_valid_support_max": max(map(len, gas_lists)),
        "shared_valid_support_min": min(map(len, shared_lists)),
        "shared_valid_support_mean": statistics.fmean(map(len, shared_lists)),
        "shared_valid_support_max": max(map(len, shared_lists)),
        "complement_valid_support_min": min(map(len, exception_lists)),
        "complement_valid_support_mean": statistics.fmean(map(len, exception_lists)),
        "complement_valid_support_max": max(map(len, exception_lists)),
        "distinct_complement_sets_per_group_min": min(distinct_exception_counts),
        "distinct_complement_sets_per_group_mean": statistics.fmean(distinct_exception_counts),
        "distinct_complement_sets_per_group_max": max(distinct_exception_counts),
        "actual_group_union_min": min(original_union_counts),
        "actual_group_union_mean": statistics.fmean(original_union_counts),
        "actual_group_union_max": max(original_union_counts),
        "actual_final_group_union_min": min(final_union_counts),
        "actual_final_group_union_mean": statistics.fmean(final_union_counts),
        "actual_final_group_union_max": max(final_union_counts),
        "padding_value": -1,
        "internal_duplicate_count": 0,
        "group_exception_overlap_count": 0,
        "profile_headwise_support_mean": spec["headwise_support_mean"],
        "profile_pre_group_union_mean": pre_union_target,
        "profile_shared_support_mean": spec["shared_support_mean"],
        "profile_complement_support_mean": spec["complement_support_mean"],
        "profile_post_group_union_mean": spec["profile_post_group_union_mean"],
    }
    return {
        "a": padded_tensor(head_lists, (batch_size, num_heads), device),
        "b": padded_tensor(gas_lists, (batch_size, num_heads), device),
        "group": padded_tensor(shared_lists, (batch_size, num_kv_heads), device),
        "exception": padded_tensor(exception_lists, (batch_size, num_heads), device, width=exception_budget),
        "a_counts": [len(values) for values in head_lists],
        "b_counts": [len(values) for values in gas_lists],
        "stats": stats,
    }


def group_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    group_size = q.shape[1] // k.shape[1]
    output = torch.empty_like(q, dtype=torch.float32)
    for batch_idx in range(q.shape[0]):
        for kv_head in range(k.shape[1]):
            tokens = [int(x) for x in indices[batch_idx, kv_head].detach().cpu().tolist() if int(x) >= 0]
            selected = torch.tensor(tokens, dtype=torch.long, device=q.device)
            key = k[batch_idx, kv_head].float().index_select(0, selected)
            value = v[batch_idx, kv_head].float().index_select(0, selected)
            query = q[batch_idx, kv_head * group_size : (kv_head + 1) * group_size].float()
            probs = torch.softmax(torch.matmul(query, key.transpose(0, 1)) / math.sqrt(q.shape[-1]), dim=-1)
            output[batch_idx, kv_head * group_size : (kv_head + 1) * group_size] = torch.matmul(probs, value)
    return output


def backend_fa2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Callable[[], torch.Tensor]:
    from flash_attn import flash_attn_with_kvcache

    q_cache = q.unsqueeze(1).contiguous()
    k_cache = k.permute(0, 2, 1, 3).contiguous()
    v_cache = v.permute(0, 2, 1, 3).contiguous()
    return lambda: flash_attn_with_kvcache(
        q_cache,
        k_cache,
        v_cache,
        causal=False,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
    ).squeeze(1)


def backend_flashinfer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Callable[[], torch.Tensor]:
    import flashinfer

    wrapper_class = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if wrapper_class is None:
        raise RuntimeError("flashinfer.BatchDecodeWithPagedKVCacheWrapper is unavailable")
    page_size = 256
    if k.shape[2] % page_size:
        raise RuntimeError("FlashInfer E6 adapter requires lengths divisible by page_size=256")
    pages_per_request = k.shape[2] // page_size
    # Prepare paged HND KV and plan outside the timed closure.  The values are
    # identical to the dense cache; only the storage layout changes.
    k_pages = k.view(q.shape[0], k.shape[1], pages_per_request, page_size, k.shape[3])
    k_pages = k_pages.permute(0, 2, 1, 3, 4).reshape(-1, k.shape[1], page_size, k.shape[3]).contiguous()
    v_pages = v.view(q.shape[0], v.shape[1], pages_per_request, page_size, v.shape[3])
    v_pages = v_pages.permute(0, 2, 1, 3, 4).reshape(-1, v.shape[1], page_size, v.shape[3]).contiguous()
    indptr = torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device) * pages_per_request
    page_indices = torch.arange(q.shape[0] * pages_per_request, dtype=torch.int32, device=q.device)
    last_page_len = torch.full((q.shape[0],), page_size, dtype=torch.int32, device=q.device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = wrapper_class(workspace, kv_layout="HND")
    wrapper.plan(
        indptr,
        page_indices,
        last_page_len,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
    )
    paged_cache = (k_pages, v_pages)
    return lambda: wrapper.run(q, paged_cache)


def backend_gas(
    module,
    method: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: dict,
    union: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    if method == "quest":
        return lambda: module.batched_per_head_sparse_decode(q, k, v, indices["a"])
    if method == "quest_twi":
        return lambda: module.batched_gqa_group_shared_decode(q, k, v, union)
    if method == "gas_per_head":
        return lambda: module.batched_per_head_sparse_decode(q, k, v, indices["b"])
    if method == "gas":
        return lambda: module.batched_gqa_group_exception_decode(
            q, k, v, indices["group"], indices["exception"]
        )
    raise ValueError(f"unknown GAS method: {method}")


def method_metadata(method: str) -> dict:
    return {
        "fa2": {"label": "FA2", "support": "dense", "backend": "flash_attn_with_kvcache"},
        "flashinfer": {
            "label": "FlashInfer",
            "support": "dense",
            "backend": "BatchDecodeWithPagedKVCacheWrapper",
            "settings": {"kv_layout": "HND", "page_size": 256, "backend": "auto", "split_kv": "default"},
        },
        "quest": {"label": "Quest (E4 profile adapter)", "support": "per-head I0_h", "backend": "twilight_e6_gqa_group_decode.batched_per_head_sparse_decode"},
        "quest_twi": {"label": "Quest+Twilight (group-varlen)", "support": "group union U_g", "backend": "twilight_e6_gqa_group_decode.batched_gqa_group_shared_decode"},
        "gas_per_head": {"label": "GAS (per-head)", "support": "per-head F_h", "backend": "twilight_e6_gqa_group_decode.batched_per_head_sparse_decode"},
        "gas": {"label": "GAS (shared+exception fused)", "support": "G_g + E_h", "backend": "twilight_e6_gqa_group_decode.batched_gqa_group_exception_decode"},
    }[method]


def make_protocol(args, specs: Sequence[dict]) -> dict:
    return {
        "protocol": "E6 kernel baselines",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "model": "Meta-Llama-3.1-8B-Instruct profile-sized synthetic decode supports",
        "shape": {
            "q_len": 1,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "dtype": "float16",
            "lengths": args.lengths,
            "batch_sizes": args.batch_sizes,
        },
        "timing": {"warmup": args.warmups, "repeats": args.repeats, "rounds": args.rounds, "timer": "CUDA events"},
        "correctness": {"atol": args.atol, "rtol": args.rtol, "reference": "FP32 gather/QK/softmax/PV"},
        "methods": {method: method_metadata(method) for method in METHODS},
        "profiles": specs,
        "hardware": {"torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform()},
        "limitations": [
            "Inputs and support masks are profile-sized synthetic replays.",
            "Quest label denotes the E4 profile-sized per-head adapter unless a real Quest backend is added.",
            "External library methods are omitted from timing when unavailable and retain an availability record.",
        ],
    }


def shape_paths(output: Path, length: int, batch: int) -> dict[str, Path]:
    stem = f"L{length}_B{batch}"
    return {
        "correctness": output / "shards" / f"{stem}_correctness.csv",
        "timings": output / "shards" / f"{stem}_timings_raw.csv",
        "stats": output / "shards" / f"{stem}_support.json",
    }


def shape_complete(output: Path, length: int, batch: int, args) -> bool:
    paths = shape_paths(output, length, batch)
    if not paths["correctness"].exists() or not paths["stats"].exists():
        return False
    correctness = read_csv(paths["correctness"])
    if len(correctness) != len(METHODS) or any(row.get("status") == "failed" for row in correctness):
        return False
    timing_rows = read_csv(paths["timings"]) if paths["timings"].exists() else []
    available = sum(row.get("status") == "available" for row in correctness)
    return len(timing_rows) == available * args.rounds * args.repeats


def summarize_rows(output: Path, args) -> None:
    timing_rows = read_csv(output / "E6_timings_raw.csv") if (output / "E6_timings_raw.csv").exists() else []
    correctness_rows = read_csv(output / "E6_correctness.csv") if (output / "E6_correctness.csv").exists() else []
    summary: List[dict] = []
    for method in METHODS:
        for length in args.lengths:
            for batch in args.batch_sizes:
                samples = [float(row["latency_us"]) for row in timing_rows if row["method"] == method and int(row["length"]) == length and int(row["batch_size"]) == batch]
                checks = [row for row in correctness_rows if row["method"] == method and int(row["length"]) == length and int(row["batch_size"]) == batch]
                status = checks[0]["status"] if checks else "missing"
                values = sorted(samples)
                summary.append({
                    "method": method,
                    "label": method_metadata(method)["label"],
                    "length": length,
                    "batch_size": batch,
                    "status": status,
                    "support": method_metadata(method)["support"],
                    "n": len(values),
                    "pooled_median_us": statistics.median(values) if values else "",
                    "mean_us": statistics.fmean(values) if values else "",
                    "p10_us": quantile(values, 0.10) if values else "",
                    "p90_us": quantile(values, 0.90) if values else "",
                    "round_median_us": "",
                    "speedup_vs_gas": "",
                    "speedup_vs_quest_twi": "",
                })
    by_key = {(row["method"], int(row["length"]), int(row["batch_size"])): row for row in summary}
    for row in summary:
        if not row["pooled_median_us"]:
            continue
        key = (int(row["length"]), int(row["batch_size"]))
        gas = by_key.get(("gas", *key), {})
        twi = by_key.get(("quest_twi", *key), {})
        if gas.get("pooled_median_us"):
            row["speedup_vs_gas"] = float(gas["pooled_median_us"]) / float(row["pooled_median_us"])
        if twi.get("pooled_median_us"):
            row["speedup_vs_quest_twi"] = float(twi["pooled_median_us"]) / float(row["pooled_median_us"])
    fields = list(summary[0]) if summary else []
    write_csv(output / "E6_summary.csv", summary, fields)

    lines = [
        "# E6 Kernel Baselines Summary",
        "",
        "Kernel-only FP16 decode benchmark on profile-sized synthetic supports.",
        "The primary comparison uses pooled median CUDA-event latency over completed repeats.",
        "",
        "## Timing results",
        "",
        "| method | length | batch | status | median us | mean us | p10 us | p90 us | speedup vs GAS | speedup vs Quest+Twilight |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(value):
            return "" if value == "" else f"{float(value):.3f}"
        lines.append("| {method} | {length} | {batch_size} | {status} | {median} | {mean} | {p10} | {p90} | {gas} | {twi} |".format(
            method=row["label"], length=row["length"], batch_size=row["batch_size"], status=row["status"],
            median=fmt(row["pooled_median_us"]), mean=fmt(row["mean_us"]), p10=fmt(row["p10_us"]), p90=fmt(row["p90_us"]),
            gas=fmt(row["speedup_vs_gas"]), twi=fmt(row["speedup_vs_quest_twi"]),
        ))
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- `gas_per_head` and `gas` use exactly the same final per-head GAS support; their ratio isolates execution organization.",
        "- `quest_twi` uses the group union support and the group-varlen shared kernel, matching the GQA union execution semantics.",
        "- Quest is labeled as an E4 profile adapter unless a real Quest backend is available and recorded.",
        "- Dense and sparse support sizes differ; their latency comparison combines work reduction and kernel implementation.",
        "- These measurements exclude support construction, metadata preparation, planning, copies, output projection, and end-to-end serving overhead.",
        "",
        "Unavailable methods and failures are retained in `E6_correctness.csv` and `E6_availability.csv`.",
    ]
    atomic_write(output / "E6_Result_Summary.md", "\n".join(lines) + "\n")


def run_shape(args, module, specs_by_length: dict[int, dict], length: int, batch: int, output: Path) -> None:
    paths = shape_paths(output, length, batch)
    if not args.force and shape_complete(output, length, batch, args):
        print(f"skip completed length={length} batch={batch}", flush=True)
        return
    seed = args.seed + length * 1009 + batch * 9176
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(batch, args.num_heads, args.head_dim, dtype=torch.float16, device=device, generator=generator)
    k = torch.randn(batch, args.num_kv_heads, length, args.head_dim, dtype=torch.float16, device=device, generator=generator)
    v = torch.randn(batch, args.num_kv_heads, length, args.head_dim, dtype=torch.float16, device=device, generator=generator)
    indices = build_consistent_indices(
        specs_by_length[length],
        batch,
        args.num_heads,
        args.num_kv_heads,
        args.exception_budget,
        seed,
        device,
    )
    quest_union, quest_union_counts, quest_union_stats = build_union_from_heads(
        indices["a"], batch, args.num_heads, args.num_kv_heads
    )
    gas_union, gas_union_counts, gas_union_stats = build_union_from_heads(
        indices["b"], batch, args.num_heads, args.num_kv_heads
    )
    refs = {
        "fa2": dense_reference(q, k, v),
        "flashinfer": None,
        "quest": fp32_reference(q, k, v, indices["a"], indices["a_counts"], args.num_heads // args.num_kv_heads),
        "quest_twi": group_reference(q, k, v, quest_union),
        "gas_per_head": fp32_reference(q, k, v, indices["b"], indices["b_counts"], args.num_heads // args.num_kv_heads),
        "gas": fp32_reference(q, k, v, indices["b"], indices["b_counts"], args.num_heads // args.num_kv_heads),
    }
    backends: dict[str, Callable[[], torch.Tensor]] = {}
    unavailable: dict[str, str] = {}
    for method in ("fa2", "flashinfer"):
        try:
            backends[method] = backend_fa2(q, k, v) if method == "fa2" else backend_flashinfer(q, k, v)
        except Exception as exc:
            unavailable[method] = f"{type(exc).__name__}: {exc}"
    for method in ("quest", "quest_twi", "gas_per_head", "gas"):
        backends[method] = backend_gas(module, method, q, k, v, indices, quest_union)

    correctness: List[dict] = []
    availability: List[dict] = []
    for method in METHODS:
        if method in unavailable:
            correctness.append({"method": method, "length": length, "batch_size": batch, "status": "unavailable", "reason": unavailable[method], "pass": ""})
            availability.append({"method": method, "length": length, "batch_size": batch, "status": "unavailable", "reason": unavailable[method]})
            continue
        try:
            result = backends[method]()
            torch.cuda.synchronize(device)
            if method in ("fa2", "flashinfer"):
                reference = refs["fa2"]
            else:
                reference = refs[method]
            metrics = error_metrics(result, reference, args.atol, args.rtol)
            correctness.append({"method": method, "length": length, "batch_size": batch, "status": "available", "reason": "", **metrics})
            availability.append({"method": method, "length": length, "batch_size": batch, "status": "available", "reason": ""})
        except Exception as exc:
            correctness.append({"method": method, "length": length, "batch_size": batch, "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "pass": False})
            availability.append({"method": method, "length": length, "batch_size": batch, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    write_csv(paths["correctness"], correctness, list(correctness[0]))
    paths["stats"].parent.mkdir(parents=True, exist_ok=True)
    paths["stats"].write_text(
        json.dumps(
            {
                "length": length,
                "batch_size": batch,
                "seed": seed,
                "indices": indices["stats"],
                "quest_twi_union": quest_union_stats,
                "quest_twi_union_counts": quest_union_counts,
                "gas_final_union": gas_union_stats,
                "gas_final_union_counts": gas_union_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    timing_rows: List[dict] = []
    usable = {
        row["method"]
        for row in correctness
        if row.get("status") == "available" and str(row.get("pass", "false")).lower() == "true"
    }
    for round_idx in range(args.rounds):
        order = list(METHODS)
        random.Random(seed + 500009 + round_idx).shuffle(order)
        for method in order:
            if method not in usable:
                continue
            samples = cuda_event_samples(backends[method], device, args.warmups, args.repeats)
            timing_rows.extend({"method": method, "length": length, "batch_size": batch, "round": round_idx + 1, "repeat": repeat_idx + 1, "latency_us": value, "order": ">".join(order)} for repeat_idx, value in enumerate(samples))
            print(f"finished method={method} length={length} batch={batch} round={round_idx + 1}/{args.rounds}", flush=True)
    if timing_rows:
        write_csv(paths["timings"], timing_rows, list(timing_rows[0]))
    print(f"finished length={length} batch={batch}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--exception-budget", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.lengths != [16384, 32768]:
        raise ValueError("E6 profile sources support lengths [16384, 32768]")
    if not args.batch_sizes or any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("E6 requires CUDA")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = [load_profile_spec(length) for length in args.lengths]
    protocol = make_protocol(args, specs)
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_hash = hashlib.sha256(protocol_text.encode()).hexdigest()
    protocol["protocol_hash"] = protocol_hash
    atomic_write(output / "config.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    module = get_gas_module()
    for length in args.lengths:
        for batch in args.batch_sizes:
            run_shape(args, module, {spec["length"]: spec for spec in specs}, length, batch, output)
    correctness = []
    timings = []
    availability = []
    for path in sorted((output / "shards").glob("*_correctness.csv")):
        correctness.extend(read_csv(path))
    for path in sorted((output / "shards").glob("*_timings_raw.csv")):
        timings.extend(read_csv(path))
    for path in sorted((output / "shards").glob("*_correctness.csv")):
        availability.extend({k: row.get(k, "") for k in ("method", "length", "batch_size", "status", "reason")} for row in read_csv(path))
    if correctness:
        write_csv(output / "E6_correctness.csv", correctness, list(correctness[0]))
    if timings:
        write_csv(output / "E6_timings_raw.csv", timings, list(timings[0]))
    if availability:
        write_csv(output / "E6_availability.csv", availability, list(availability[0]))
    summarize_rows(output, args)
    print(f"E6 complete: {output}", flush=True)


if __name__ == "__main__":
    main()
