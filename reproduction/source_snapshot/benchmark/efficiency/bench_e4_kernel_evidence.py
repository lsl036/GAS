import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "csrc" / "src" / "gqa_group_decode.cu"
DEFAULT_OUTPUT = ROOT / "benchmark" / "results_efficiency" / "e4_kernel_evidence"
MODEL_CONFIG = ROOT / "benchmark" / "configs" / "config_quest_twi_gqa_group_max_p080_exception16.json"
PROFILE_SOURCES = {
    16384: ROOT
    / "benchmark"
    / "results_passkey_ours"
    / "phase3_group_max_p080_exception16_llama31_16k"
    / "profile_summary.json",
    32768: ROOT
    / "benchmark"
    / "results_passkey_ours"
    / "phase_long_group_max_p080_exception16_llama31_32k"
    / "profile_summary.json",
}
PATHS = ("A", "B", "C")
PATH_LABELS = {
    "A": "per_head_headwise_support",
    "B": "per_head_gas_support",
    "C": "shared_complement_fused",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_text(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def get_module():
    return load(
        name="twilight_e4_gqa_decode",
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


def metric(summary: dict, name: str) -> float:
    try:
        return float(summary["gqa"][name]["mean_us"])
    except KeyError as exc:
        raise ValueError(f"Missing gqa.{name}.mean_us in profile summary") from exc


def load_profile_spec(length: int) -> dict:
    try:
        source = PROFILE_SOURCES[length]
    except KeyError as exc:
        raise ValueError(f"No frozen profile source for length={length}") from exc
    summary = json.loads(source.read_text(encoding="utf-8"))
    raw_profile = source.with_name("profile.jsonl")
    if not raw_profile.exists():
        raise ValueError(f"Missing raw profile paired with {source}")
    with raw_profile.open("r", encoding="utf-8") as f:
        first_record = json.loads(next(f))
    recorded_config = Path(first_record["config_name"]).resolve()
    if recorded_config != MODEL_CONFIG.resolve():
        raise ValueError(
            f"Profile {raw_profile} records config {recorded_config}, expected {MODEL_CONFIG}"
        )
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    expected = {
        "selector": (config["selector"]["type"], config["selector"]["token_budget"], config["selector"]["chunk_size"]),
        "weight_pruner": (config["weight_pruner"]["type"], config["weight_pruner"]["threshold"]),
        "gqa": (
            config["gqa_refinement"]["mode"],
            config["gqa_refinement"]["threshold"],
            config["gqa_refinement"]["exception_budget"],
        ),
    }
    if expected != {
        "selector": ("quest", 8192, 16),
        "weight_pruner": ("top_p", 0.85),
        "gqa": ("group_max", 0.8, 16),
    }:
        raise ValueError(f"Frozen E4 config no longer matches the required protocol: {expected}")

    pre_head = metric(summary, "pre_per_head_selected_mean")
    shared = metric(summary, "post_group_shared_count_mean")
    post_head = metric(summary, "post_per_head_selected_mean")
    complement = post_head - shared
    if not 0.0 <= complement <= 16.0 + 1e-6:
        raise ValueError(
            f"Invalid profile-derived complement mean for length={length}: {complement}"
        )
    return {
        "length": length,
        "profile_summary": str(source.relative_to(ROOT)),
        "profile_summary_sha256": sha256(source),
        "raw_profile": str(raw_profile.relative_to(ROOT)),
        "raw_profile_sha256": sha256(raw_profile),
        "model_config": str(MODEL_CONFIG.relative_to(ROOT)),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "profile_seq_len_mean": float(summary["seq_len"]["mean_us"]),
        "headwise_support_mean": pre_head,
        "shared_support_mean": shared,
        "complement_support_mean": complement,
        "gas_per_head_support_mean": post_head,
        "profile_post_group_union_mean": metric(summary, "post_group_union_count_mean"),
        "profile_pre_loaded_kv_mean": metric(summary, "pre_loaded_kv_tokens_mean"),
        "profile_post_loaded_kv_mean": metric(summary, "post_loaded_kv_tokens_mean"),
        "index_source": "profile-sized synthetic",
    }


def balanced_counts(target_mean: float, count: int) -> List[int]:
    low = math.floor(target_mean)
    high_count = round((target_mean - low) * count)
    values = [low + 1] * high_count + [low] * (count - high_count)
    return values


def complement_counts(
    target_mean: float, batch_size: int, num_heads: int, group_size: int, budget: int
) -> List[int]:
    values: List[int] = []
    groups = batch_size * (num_heads // group_size)
    total = round(target_mean * groups * group_size)
    group_low = total // groups
    high_groups = total - group_low * groups
    if group_low + (1 if high_groups else 0) > budget * group_size:
        raise ValueError("Profile-derived complement mean exceeds exception budget")
    for group_idx in range(groups):
        group_total = group_low + (group_idx < high_groups)
        per_head_low = group_total // group_size
        high_heads = group_total - per_head_low * group_size
        local = [per_head_low] * group_size
        for offset in range(high_heads):
            local[(offset + group_idx) % group_size] += 1
        if max(local) > budget:
            raise ValueError("Profile-derived complement length exceeds exception budget")
        values.extend(local)
    return values


def coprime_step(length: int, value: int) -> int:
    step = value % length
    if step == 0:
        step = 1
    while math.gcd(step, length) != 1:
        step += 1
    return step


def affine_tokens(length: int, count: int, seed: int, excluded: Iterable[int] = ()) -> List[int]:
    excluded_set = set(excluded)
    if count > length - len(excluded_set):
        raise ValueError("Requested support exceeds available unique token positions")
    start = (seed * 2654435761) % length
    step = coprime_step(length, 2 * seed + 1)
    result: List[int] = []
    cursor = 0
    while len(result) < count:
        token = (start + cursor * step) % length
        cursor += 1
        if token not in excluded_set:
            result.append(token)
    return result


def padded_tensor(
    lists: Sequence[Sequence[int]],
    outer_shape: Tuple[int, int],
    device,
    width: int = 0,
) -> torch.Tensor:
    minimum_width = max(len(values) for values in lists)
    width = width or minimum_width
    if width < minimum_width:
        raise ValueError(f"Padding width {width} is smaller than valid support {minimum_width}")
    tensor = torch.full((*outer_shape, width), -1, dtype=torch.int32)
    for linear, values in enumerate(lists):
        row = linear // outer_shape[1]
        col = linear % outer_shape[1]
        if values:
            tensor[row, col, : len(values)] = torch.tensor(values, dtype=torch.int32)
    return tensor.to(device=device, non_blocking=False).contiguous()


def describe(values: Sequence[int], prefix: str) -> dict:
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_max": max(values),
    }


def build_indices(
    spec: dict,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    exception_budget: int,
    seed: int,
    device,
) -> dict:
    length = spec["length"]
    group_size = num_heads // num_kv_heads
    a_counts = balanced_counts(spec["headwise_support_mean"], batch_size * num_heads)
    shared_counts = balanced_counts(spec["shared_support_mean"], batch_size * num_kv_heads)
    e_counts = complement_counts(
        spec["complement_support_mean"],
        batch_size,
        num_heads,
        group_size,
        exception_budget,
    )

    a_lists: List[List[int]] = []
    group_lists: List[List[int]] = []
    exception_lists: List[List[int]] = []
    full_lists: List[List[int]] = []
    union_counts: List[int] = []
    distinct_complement_counts: List[int] = []

    for linear, count in enumerate(a_counts):
        a_lists.append(affine_tokens(length, count, seed + 101 + linear * 17))

    for batch_idx in range(batch_size):
        for kv_head in range(num_kv_heads):
            group_linear = batch_idx * num_kv_heads + kv_head
            shared = affine_tokens(
                length,
                shared_counts[group_linear],
                seed + 100003 + group_linear * 29,
            )
            group_lists.append(shared)
            local_exceptions: List[List[int]] = []
            for local_head in range(group_size):
                q_head = kv_head * group_size + local_head
                head_linear = batch_idx * num_heads + q_head
                complement = affine_tokens(
                    length,
                    e_counts[head_linear],
                    seed + 200003 + head_linear * 43,
                    excluded=shared,
                )
                local_exceptions.append(complement)
                exception_lists.append(complement)
                full_lists.append(shared + complement)
            if sum(bool(values) for values in local_exceptions) > 1:
                nonempty = [tuple(values) for values in local_exceptions if values]
                if len(set(nonempty)) == 1:
                    raise RuntimeError("Synthetic generator produced identical per-head complements")
            distinct_complement_counts.append(len(set(tuple(values) for values in local_exceptions)))
            union = set(shared)
            for complement in local_exceptions:
                union.update(complement)
            union_counts.append(len(union))

    if any(len(values) != len(set(values)) for values in a_lists + group_lists + exception_lists):
        raise RuntimeError("Index generator produced a duplicate within a support list")
    for head_linear, complement in enumerate(exception_lists):
        batch_idx = head_linear // num_heads
        q_head = head_linear % num_heads
        kv_head = q_head // group_size
        shared = group_lists[batch_idx * num_kv_heads + kv_head]
        if set(shared).intersection(complement):
            raise RuntimeError("Group and exception supports overlap")

    stats = {
        **describe(a_counts, "path_a_valid_support"),
        **describe([len(values) for values in full_lists], "gas_valid_support"),
        **describe(shared_counts, "shared_valid_support"),
        **describe(e_counts, "complement_valid_support"),
        **describe(distinct_complement_counts, "distinct_complement_sets_per_group"),
        **describe(union_counts, "actual_group_union"),
        "padding_value": -1,
        "internal_duplicate_count": 0,
        "group_exception_overlap_count": 0,
        "profile_headwise_support_mean": spec["headwise_support_mean"],
        "profile_shared_support_mean": spec["shared_support_mean"],
        "profile_complement_support_mean": spec["complement_support_mean"],
        "profile_gas_per_head_support_mean": spec["gas_per_head_support_mean"],
        "profile_post_group_union_mean": spec["profile_post_group_union_mean"],
    }
    group_width = max(shared_counts)
    return {
        "a": padded_tensor(a_lists, (batch_size, num_heads), device),
        "b": padded_tensor(
            full_lists,
            (batch_size, num_heads),
            device,
            width=group_width + exception_budget,
        ),
        "group": padded_tensor(
            group_lists, (batch_size, num_kv_heads), device, width=group_width
        ),
        "exception": padded_tensor(
            exception_lists,
            (batch_size, num_heads),
            device,
            width=exception_budget,
        ),
        "a_counts": a_counts,
        "b_counts": [len(values) for values in full_lists],
        "stats": stats,
    }


@torch.no_grad()
def fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    counts: Sequence[int],
    group_size: int,
) -> torch.Tensor:
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(q.shape[-1])
    num_heads = q.shape[1]
    for batch_idx in range(q.shape[0]):
        for head_idx in range(num_heads):
            count = counts[batch_idx * num_heads + head_idx]
            selected = indices[batch_idx, head_idx, :count].to(torch.long)
            kv_head = head_idx // group_size
            key = k[batch_idx, kv_head].index_select(0, selected).float()
            value = v[batch_idx, kv_head].index_select(0, selected).float()
            logits = torch.mv(key, q[batch_idx, head_idx].float()) * scale
            output[batch_idx, head_idx] = torch.mv(value.transpose(0, 1), torch.softmax(logits, dim=0))
    return output


def error_metrics(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    actual_fp32 = actual.float()
    difference = actual_fp32 - expected
    reference_norm = torch.linalg.vector_norm(expected)
    relative_l2 = torch.linalg.vector_norm(difference) / torch.clamp(reference_norm, min=1e-12)
    return {
        "max_abs_error": float(difference.abs().max().item()),
        "relative_l2_error": float(relative_l2.item()),
        "finite": bool(torch.isfinite(actual_fp32).all().item()),
        "pass": bool(torch.allclose(actual_fp32, expected, atol=atol, rtol=rtol)),
    }


def correctness_rows(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: dict,
    outputs: Dict[str, torch.Tensor],
    length: int,
    batch_size: int,
    seed: int,
    protocol_hash: str,
    atol: float,
    rtol: float,
) -> List[dict]:
    group_size = q.shape[1] // k.shape[1]
    module.batched_per_head_sparse_decode_out(q, k, v, indices["a"], outputs["A"])
    module.batched_per_head_sparse_decode_out(q, k, v, indices["b"], outputs["B"])
    module.batched_gqa_group_exception_decode_out(
        q, k, v, indices["group"], indices["exception"], outputs["C"]
    )
    torch.cuda.synchronize(q.device)
    reference_a = fp32_reference(q, k, v, indices["a"], indices["a_counts"], group_size)
    reference_bc = fp32_reference(q, k, v, indices["b"], indices["b_counts"], group_size)
    torch.cuda.synchronize(q.device)

    bc_error = float((outputs["B"].float() - outputs["C"].float()).abs().max().item())
    bc_pass = bool(
        torch.allclose(outputs["B"].float(), outputs["C"].float(), atol=atol, rtol=rtol)
    )
    rows: List[dict] = []
    for path in PATHS:
        reference = reference_a if path == "A" else reference_bc
        row = {
            "protocol_hash": protocol_hash,
            "length": length,
            "batch_size": batch_size,
            "path": path,
            "path_name": PATH_LABELS[path],
            "seed": seed,
            "atol": atol,
            "rtol": rtol,
            "same_support_as_c": path in ("B", "C"),
            "bc_max_abs_error": bc_error if path in ("B", "C") else "",
            "bc_pass": bc_pass if path in ("B", "C") else "",
            **error_metrics(outputs[path], reference, atol, rtol),
            **indices["stats"],
        }
        rows.append(row)
    del reference_a, reference_bc
    return rows


def cuda_event_samples(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmups: int,
    repeats: int,
) -> List[float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for repeat_idx in range(repeats):
        starts[repeat_idx].record(stream)
        function()
        ends[repeat_idx].record(stream)
    stream.synchronize()
    return [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(repeats)]


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def shard_paths(output_dir: Path, length: int, batch_size: int) -> Tuple[Path, Path, Path]:
    stem = f"L{length}_B{batch_size}"
    shard_dir = output_dir / "shards"
    return (
        shard_dir / f"{stem}_correctness.csv",
        shard_dir / f"{stem}_timings_raw.csv",
        shard_dir / f"{stem}_index_stats.json",
    )


def shard_complete(output_dir: Path, length: int, batch_size: int, protocol_hash: str, repeats: int, rounds: int) -> bool:
    correctness_path, timing_path, stats_path = shard_paths(output_dir, length, batch_size)
    if not correctness_path.exists() or not timing_path.exists() or not stats_path.exists():
        return False
    correctness = read_csv(correctness_path)
    timings = read_csv(timing_path)
    expected_timing_rows = len(PATHS) * rounds * repeats
    return (
        len(correctness) == len(PATHS)
        and len(timings) == expected_timing_rows
        and all(row["protocol_hash"] == protocol_hash for row in correctness + timings)
        and all(row["pass"].lower() == "true" for row in correctness)
    )


def run_shape(
    module,
    spec: dict,
    batch_size: int,
    args,
    device: torch.device,
    protocol_hash: str,
) -> None:
    length = spec["length"]
    correctness_path, timing_path, stats_path = shard_paths(args.output_dir, length, batch_size)
    if not args.force and shard_complete(
        args.output_dir, length, batch_size, protocol_hash, args.repeats, args.rounds
    ):
        print(f"skip completed length={length} batch={batch_size}", flush=True)
        return

    seed = args.seed + length * 1009 + batch_size * 9176
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        batch_size,
        args.num_heads,
        args.head_dim,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        batch_size,
        args.num_kv_heads,
        length,
        args.head_dim,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        batch_size,
        args.num_kv_heads,
        length,
        args.head_dim,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    indices = build_indices(
        spec,
        batch_size,
        args.num_heads,
        args.num_kv_heads,
        args.exception_budget,
        seed,
        device,
    )
    outputs = {path: torch.empty_like(q) for path in PATHS}
    functions = {
        "A": lambda: module.batched_per_head_sparse_decode_out(
            q, k, v, indices["a"], outputs["A"]
        ),
        "B": lambda: module.batched_per_head_sparse_decode_out(
            q, k, v, indices["b"], outputs["B"]
        ),
        "C": lambda: module.batched_gqa_group_exception_decode_out(
            q, k, v, indices["group"], indices["exception"], outputs["C"]
        ),
    }

    correctness = correctness_rows(
        module,
        q,
        k,
        v,
        indices,
        outputs,
        length,
        batch_size,
        seed,
        protocol_hash,
        args.atol,
        args.rtol,
    )
    write_csv(correctness_path, correctness, list(correctness[0]))
    atomic_write_text(
        stats_path,
        json.dumps(
            {
                "protocol_hash": protocol_hash,
                "length": length,
                "batch_size": batch_size,
                "seed": seed,
                "source": spec,
                "actual": indices["stats"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    if not all(row["pass"] for row in correctness):
        raise RuntimeError(
            f"Correctness failed for length={length}, batch={batch_size}; "
            f"failure rows retained in {correctness_path}"
        )

    timing_rows: List[dict] = []
    for round_idx in range(args.rounds):
        order = list(PATHS)
        order_seed = seed + 500009 + round_idx
        random.Random(order_seed).shuffle(order)
        order_text = ">".join(order)
        for order_position, path in enumerate(order):
            samples = cuda_event_samples(functions[path], device, args.warmups, args.repeats)
            for repeat_idx, latency_us in enumerate(samples):
                timing_rows.append(
                    {
                        "protocol_hash": protocol_hash,
                        "length": length,
                        "batch_size": batch_size,
                        "path": path,
                        "path_name": PATH_LABELS[path],
                        "round": round_idx,
                        "repeat": repeat_idx,
                        "latency_us": latency_us,
                        "execution_order": order_text,
                        "order_position": order_position,
                        "input_seed": seed,
                        "order_seed": order_seed,
                    }
                )
        print(
            f"finished length={length} batch={batch_size} round={round_idx + 1}/{args.rounds}",
            flush=True,
        )
    write_csv(timing_path, timing_rows, list(timing_rows[0]))
    del q, k, v, indices, outputs
    torch.cuda.empty_cache()


def summarize(output_dir: Path, lengths: Sequence[int], batch_sizes: Sequence[int], protocol_hash: str, repeats: int, rounds: int) -> List[dict]:
    correctness_rows_all: List[dict] = []
    timing_rows_all: List[dict] = []
    for length in lengths:
        for batch_size in batch_sizes:
            if not shard_complete(output_dir, length, batch_size, protocol_hash, repeats, rounds):
                raise RuntimeError(f"Incomplete E4 shard: length={length}, batch={batch_size}")
            correctness_path, timing_path, _ = shard_paths(output_dir, length, batch_size)
            correctness_rows_all.extend(read_csv(correctness_path))
            timing_rows_all.extend(read_csv(timing_path))

    write_csv(output_dir / "correctness.csv", correctness_rows_all, list(correctness_rows_all[0]))
    write_csv(output_dir / "timings_raw.csv", timing_rows_all, list(timing_rows_all[0]))

    summary_rows: List[dict] = []
    for length in lengths:
        for batch_size in batch_sizes:
            shape_rows = [
                row
                for row in timing_rows_all
                if int(row["length"]) == length and int(row["batch_size"]) == batch_size
            ]
            stats: dict = {
                "protocol_hash": protocol_hash,
                "length": length,
                "batch_size": batch_size,
                "input_kind": "profile-sized synthetic",
            }
            round_medians: Dict[str, List[float]] = {}
            for path in PATHS:
                values = [float(row["latency_us"]) for row in shape_rows if row["path"] == path]
                path_rounds = []
                for round_idx in range(rounds):
                    round_values = [
                        float(row["latency_us"])
                        for row in shape_rows
                        if row["path"] == path and int(row["round"]) == round_idx
                    ]
                    path_rounds.append(statistics.median(round_values))
                round_medians[path] = path_rounds
                prefix = path.lower()
                stats.update(
                    {
                        f"{prefix}_median_us": statistics.median(values),
                        f"{prefix}_mean_us": statistics.fmean(values),
                        f"{prefix}_p10_us": quantile(values, 0.10),
                        f"{prefix}_p90_us": quantile(values, 0.90),
                        f"{prefix}_round_medians_us": json.dumps(path_rounds),
                    }
                )
            a_speedups = [
                round_medians["A"][idx] / round_medians["C"][idx] for idx in range(rounds)
            ]
            b_speedups = [
                round_medians["B"][idx] / round_medians["C"][idx] for idx in range(rounds)
            ]
            stats.update(
                {
                    "speedup_a_over_c_median": statistics.median(a_speedups),
                    "speedup_a_over_c_min": min(a_speedups),
                    "speedup_a_over_c_max": max(a_speedups),
                    "speedup_a_over_c_rounds": json.dumps(a_speedups),
                    "speedup_b_over_c_median": statistics.median(b_speedups),
                    "speedup_b_over_c_min": min(b_speedups),
                    "speedup_b_over_c_max": max(b_speedups),
                    "speedup_b_over_c_rounds": json.dumps(b_speedups),
                }
            )
            c_correctness = next(
                row
                for row in correctness_rows_all
                if int(row["length"]) == length
                and int(row["batch_size"]) == batch_size
                and row["path"] == "C"
            )
            stats.update(
                {
                    "c_max_abs_error": c_correctness["max_abs_error"],
                    "c_relative_l2_error": c_correctness["relative_l2_error"],
                    "c_correctness_pass": c_correctness["pass"],
                    "path_a_valid_support_mean": c_correctness["path_a_valid_support_mean"],
                    "gas_valid_support_mean": c_correctness["gas_valid_support_mean"],
                    "shared_valid_support_mean": c_correctness["shared_valid_support_mean"],
                    "complement_valid_support_mean": c_correctness["complement_valid_support_mean"],
                    "actual_group_union_mean": c_correctness["actual_group_union_mean"],
                }
            )
            summary_rows.append(stats)
    write_csv(output_dir / "summary.csv", summary_rows, list(summary_rows[0]))
    return summary_rows


def render_report(rows: Sequence[dict], config: dict, output_dir: Path) -> str:
    lines = [
        "# E4 Kernel Evidence Summary",
        "",
        "## Scope",
        "",
        "Kernel-only FP16 decode microbenchmark on profile-sized synthetic supports. "
        "Q/K/V, indices, outputs, and CUDA events are allocated outside timed regions. "
        "Path B and C use exactly the same per-head support; Path A uses the profile-sized "
        "head-wise support and therefore measures both support-size and execution-organization changes.",
        "",
        f"- GPU: `{config['hardware']['torch_device_name']}`",
        f"- Shape: Hq={config['shape']['num_heads']}, Hkv={config['shape']['num_kv_heads']}, "
        f"d={config['shape']['head_dim']}, q_len=1, dtype=FP16",
        f"- Timing: warmup {config['timing']['warmups']}, repeats {config['timing']['repeats']}, "
        f"rounds {config['timing']['rounds']}, CUDA events",
        f"- Correctness: FP32 gather/QK/joint-softmax/PV, atol={config['correctness']['atol']}, "
        f"rtol={config['correctness']['rtol']}",
        "",
        "## Main results",
        "",
        "| length | batch | A median us | B median us | C median us | A/C | B/C | C max abs | C rel L2 | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {length} | {batch_size} | {a_median_us:.3f} | {b_median_us:.3f} | "
            "{c_median_us:.3f} | {speedup_a_over_c_median:.3f}x | "
            "{speedup_b_over_c_median:.3f}x | {c_max_abs_error} | "
            "{c_relative_l2_error} | {c_correctness_pass} |".format(**row)
        )
    c_pass_count = sum(str(row["c_correctness_pass"]).lower() == "true" for row in rows)
    b_wins = sum(float(row["speedup_b_over_c_median"]) > 1.0 for row in rows)
    a_wins = sum(float(row["speedup_a_over_c_median"]) > 1.0 for row in rows)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Shared + complement correctness passed on {c_pass_count}/{len(rows)} frozen shapes.",
            f"- C is faster than the same-support per-head path B on {b_wins}/{len(rows)} shapes; this isolates shared reuse/fused execution.",
            f"- C is faster than the head-wise path A on {a_wins}/{len(rows)} shapes; this comparison also includes the changed support size.",
            f"- Speedups are medians of {config['timing']['rounds']} same-round ratios. A value above 1 means C is faster.",
            "- These are kernel-only results. They exclude index construction, compaction, copies, output projection, prefill, and serving latency.",
            "- The inputs replay profile-derived mean sizes, not real token IDs or a real-mask trace.",
            "",
            "## Artifacts",
            "",
            "- `correctness.csv`: FP32-reference errors and support/union diagnostics.",
            "- `timings_raw.csv`: every CUDA-event sample with round and execution order.",
            "- `summary.csv`: six shape rows and paired-round speedups.",
            "- `shards/`: resumable per-shape records.",
            "- `E2_PrePost_Export.csv`: E2 A1-A8 same-trajectory pre/post loaded-KV analysis.",
            "- `config.json`: source hashes, environment, hardware, protocol, and kernel symbols.",
            "",
        ]
    )
    return "\n".join(lines)


def hardware_metadata(device: torch.device) -> dict:
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_device_index": device.index if device.index is not None else torch.cuda.current_device(),
        "torch_device_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version,pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,power.limit,persistence_mode",
                "--format=csv,noheader",
            ]
        ),
    }


def build_config(args, specs: Sequence[dict], hardware: dict, protocol_hash: str) -> dict:
    return {
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "protocol_hash": protocol_hash,
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_text(["git", "status", "--short"]),
        "source": {
            "benchmark": str(Path(__file__).resolve().relative_to(ROOT)),
            "benchmark_sha256": sha256(Path(__file__).resolve()),
            "cuda": str(CUDA_SOURCE.relative_to(ROOT)),
            "cuda_sha256": sha256(CUDA_SOURCE),
            "run_script": str((args.output_dir / "run_e4.sh")),
            "run_script_sha256": (
                sha256(args.output_dir / "run_e4.sh")
                if (args.output_dir / "run_e4.sh").exists()
                else "unavailable"
            ),
            "profile_specs": list(specs),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nvcc": run_text(["nvcc", "--version"]),
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", "unset"),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "unset"),
            "cuda_compile_flags": [
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
        "hardware": hardware,
        "shape": {
            "lengths": args.lengths,
            "batch_sizes": args.batch_sizes,
            "q_len": 1,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "group_size": args.num_heads // args.num_kv_heads,
            "head_dim": args.head_dim,
            "dtype": "float16",
            "layout": "contiguous [B,H,L,D] KV and [B,H,D] Q",
        },
        "method": {
            "selector": "quest",
            "token_budget": 8192,
            "chunk_size": 16,
            "p_t": 0.85,
            "p_g": 0.80,
            "exception_budget": args.exception_budget,
            "index_source": "profile-sized synthetic",
            "padding": -1,
        },
        "paths": {
            "A": {
                "description": "fused per-head sparse decode on head-wise support S_h",
                "kernel_symbol": "BatchedPerHeadSparseDecodeKernel",
            },
            "B": {
                "description": "fused per-head sparse decode on exact GAS support F_h",
                "kernel_symbol": "BatchedPerHeadSparseDecodeKernel",
            },
            "C": {
                "description": "fused shared + per-head complement decode on G_g and E_h",
                "kernel_symbol": "BatchedGQAGroupExceptionDecodeKernel",
            },
        },
        "timing": {
            "tool": "torch.cuda.Event",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "rounds": args.rounds,
            "order": "deterministically shuffled A/B/C each round",
            "excluded": [
                "compile",
                "allocation",
                "reference",
                "random generation",
                "index construction",
                "duplicate removal",
                "compaction",
                "CPU-GPU copies",
            ],
        },
        "correctness": {
            "reference": "FP32 gather/QK/joint-softmax/PV on each exact per-head support",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "completed_shapes": [],
    }


def protocol_hash(args, specs: Sequence[dict], hardware: dict) -> str:
    payload = {
        "lengths": args.lengths,
        "batch_sizes": args.batch_sizes,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "exception_budget": args.exception_budget,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rounds": args.rounds,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "profile_hashes": [spec["profile_summary_sha256"] for spec in specs],
        "raw_profile_hashes": [spec["raw_profile_sha256"] for spec in specs],
        "model_config_hashes": [spec["model_config_sha256"] for spec in specs],
        "cuda_sha256": sha256(CUDA_SOURCE),
        "benchmark_sha256": sha256(Path(__file__).resolve()),
        "gpu_uuid": hardware["uuid"],
        "gpu_name": hardware["torch_device_name"],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E4 kernel-only A/B/C benchmark for GQA shared + complement decode."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--exception-budget", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.lengths != [16384, 32768] or args.batch_sizes != [1, 32, 64]:
        print("warning: running a non-paper shape set", file=sys.stderr)
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")
    if args.exception_budget != 16:
        raise ValueError("E4 P0 freezes exception budget at 16")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E4 requires an available CUDA device")
    torch.cuda.set_device(device)
    specs = [load_profile_spec(length) for length in args.lengths]
    hardware = hardware_metadata(device)
    current_protocol_hash = protocol_hash(args, specs, hardware)
    config = build_config(args, specs, hardware, current_protocol_hash)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output_dir / "config.json", json.dumps(config, indent=2) + "\n")

    module = get_module()
    for spec in specs:
        for batch_size in args.batch_sizes:
            run_shape(module, spec, batch_size, args, device, current_protocol_hash)
            config["completed_shapes"].append(
                {"length": spec["length"], "batch_size": batch_size}
            )
            atomic_write_text(
                args.output_dir / "config.json", json.dumps(config, indent=2) + "\n"
            )

    summary_rows = summarize(
        args.output_dir,
        args.lengths,
        args.batch_sizes,
        current_protocol_hash,
        args.repeats,
        args.rounds,
    )
    config["status"] = "complete"
    config["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_write_text(args.output_dir / "config.json", json.dumps(config, indent=2) + "\n")
    atomic_write_text(
        args.output_dir / "E4_Result_Summary.md",
        render_report(summary_rows, config, args.output_dir),
    )
    print(f"E4 complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
