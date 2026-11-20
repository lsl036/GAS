"""Component-based Estimated TPOT benchmark for Method B.

This is a profile-calibrated synthetic projection.  It measures attention and
selection components on the current CUDA implementations, measures common
Llama-shaped matrix work separately, and sums the components without dividing
by batch size.  It is deliberately labelled as Estimated TPOT rather than
end-to-end serving latency.
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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import torch

# Keep imports valid both when this file is run as a script and when the CPU
# smoke test imports it as ``benchmark.efficiency.bench_e7_estimated_tpot``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_e6_kernel_baselines import (
    backend_fa2,
    backend_flashinfer,
    build_consistent_indices,
    build_union_from_heads,
    dense_reference,
    error_metrics,
    fp32_reference,
    group_reference,
    load_profile_spec,
    support_lists,
)
from bench_e4_kernel_evidence import affine_tokens, cuda_event_samples, padded_tensor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "benchmark" / "results_efficiency" / "e7_estimated_tpot"
MODEL_CONFIG_PATH = Path("/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct/config.json")
PROFILE_SOURCES = {
    16384: ROOT / "benchmark/results_passkey_ours/phase3_group_max_p080_exception16_llama31_16k/profile_summary.json",
    32768: ROOT / "benchmark/results_passkey_ours/phase_long_group_max_p080_exception16_llama31_32k/profile_summary.json",
}
METHODS = ("fa2", "flashinfer", "quest", "quest_twi", "gas")
SPARSE_METHODS = ("quest", "quest_twi", "gas")


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def profile_specs() -> Dict[int, dict]:
    specs = {}
    for length, path in PROFILE_SOURCES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        gqa = data["gqa"]
        spec = load_profile_spec(length)
        spec.update({
            "length": length,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": sha256(path),
            "headwise_support_mean": float(gqa["pre_per_head_selected_mean"]["mean_us"]),
            "pre_union_mean": float(gqa["pre_group_union_count_mean"]["mean_us"]),
            "shared_mean": float(gqa["post_group_shared_count_mean"]["mean_us"]),
            "complement_mean": float(gqa["post_per_head_selected_mean"]["mean_us"] - gqa["post_group_shared_count_mean"]["mean_us"]),
            "gas_union_mean": float(gqa["post_group_union_count_mean"]["mean_us"]),
            "gas_support_mean": float(gqa["post_per_head_selected_mean"]["mean_us"]),
            "loaded_kv_pre_mean": float(gqa["pre_loaded_kv_tokens_mean"]["mean_us"]),
            "loaded_kv_post_mean": float(gqa["post_loaded_kv_tokens_mean"]["mean_us"]),
            "profile_pre_loaded_kv_mean": float(gqa["pre_loaded_kv_tokens_mean"]["mean_us"]),
            "profile_post_group_union_mean": float(gqa["post_group_union_count_mean"]["mean_us"]),
            "shared_support_mean": float(gqa["post_group_shared_count_mean"]["mean_us"]),
            "complement_support_mean": float(gqa["post_per_head_selected_mean"]["mean_us"] - gqa["post_group_shared_count_mean"]["mean_us"]),
        })
        specs[length] = spec
    return specs


def model_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    }
    actual = {
        "hidden_size": int(data["hidden_size"]),
        "intermediate_size": int(data["intermediate_size"]),
        "num_hidden_layers": int(data["num_hidden_layers"]),
        "num_attention_heads": int(data["num_attention_heads"]),
        "num_key_value_heads": int(data["num_key_value_heads"]),
        "head_dim": int(data["hidden_size"] // data["num_attention_heads"]),
        "vocab_size": int(data["vocab_size"]),
    }
    if actual != expected:
        raise ValueError(f"Unexpected model shape: {actual}; expected {expected}")
    return {"path": str(path), "sha256": sha256(path), "actual": actual}


def make_candidate_indices(batch: int, heads: int, length: int, budget: int, device: torch.device) -> torch.Tensor:
    count = min(length, budget)
    lists = [affine_tokens(length, count, 9001 + linear * 17) for linear in range(batch * heads)]
    return padded_tensor(lists, (batch, heads), device, width=count)


def selection_callable(
    method: str,
    scores: torch.Tensor,
    group_size: int,
    candidate_budget: int,
) -> Callable[[], torch.Tensor]:
    batch, heads, _ = scores.shape
    top_k = min(scores.shape[-1], candidate_budget)

    def quest() -> torch.Tensor:
        _, indices = torch.topk(scores, top_k, dim=-1, sorted=False)
        return indices

    def quest_twi() -> torch.Tensor:
        values, indices = torch.topk(scores, top_k, dim=-1, sorted=False)
        values = torch.softmax(values.float(), dim=-1).to(scores.dtype)
        group_values = values.view(batch, heads // group_size, group_size, top_k).amax(dim=2)
        return indices, group_values.cumsum(dim=-1)

    def gas() -> torch.Tensor:
        values, indices = torch.topk(scores, top_k, dim=-1, sorted=False)
        values = torch.softmax(values.float(), dim=-1).to(scores.dtype)
        group_values = values.view(batch, heads // group_size, group_size, top_k).amax(dim=2)
        group_mass = group_values.sum(dim=-1, keepdim=True)
        residual = values - values.view(batch, heads // group_size, group_size, top_k).amax(dim=2).repeat_interleave(group_size, dim=1)
        _, exception = torch.topk(residual, min(16, top_k), dim=-1, sorted=False)
        return indices, group_mass, exception

    return {"quest": quest, "quest_twi": quest_twi, "gas": gas}[method]


def measure_component(fn: Callable[[], object], device: torch.device, warmups: int, repeats: int) -> List[float]:
    return cuda_event_samples(fn, device, warmups, repeats)


def common_model_callable(batch: int, cfg: dict, device: torch.device) -> tuple[Callable[[], object], Callable[[], object]]:
    hidden = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    x = torch.randn(batch, hidden, device=device, dtype=torch.float16)
    w_qkv = torch.randn(hidden, hidden + 2 * (hidden // 4), device=device, dtype=torch.float16)
    w_o = torch.randn(hidden, hidden, device=device, dtype=torch.float16)
    w_gate = torch.randn(hidden, intermediate, device=device, dtype=torch.float16)
    w_up = torch.randn(hidden, intermediate, device=device, dtype=torch.float16)
    w_down = torch.randn(intermediate, hidden, device=device, dtype=torch.float16)
    w_lm = torch.randn(hidden, vocab, device=device, dtype=torch.float16)

    def layer() -> object:
        qkv = torch.matmul(x, w_qkv)
        projected = torch.matmul(x, w_o)
        gate = torch.matmul(x, w_gate)
        up = torch.matmul(x, w_up)
        mlp = torch.matmul(torch.nn.functional.silu(gate) * up, w_down)
        norm = x * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + 1e-5).to(x.dtype)
        return qkv, projected, mlp, norm

    def output() -> object:
        return torch.matmul(x, w_lm)

    return layer, output


def ref_for_method(method: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: dict, quest: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    group = q.shape[1] // k.shape[1]
    if method == "fa2" or method == "flashinfer":
        return dense_reference(q, k, v)
    if method == "quest":
        return fp32_reference(q, k, v, quest, [quest.shape[-1]] * (q.shape[0] * q.shape[1]), group)
    if method == "quest_twi":
        return group_reference(q, k, v, union)
    return fp32_reference(q, k, v, indices["b"], indices["b_counts"], group)


def attention_callable(method: str, module, q, k, v, indices, quest, union):
    if method == "fa2":
        return backend_fa2(q, k, v)
    if method == "flashinfer":
        return backend_flashinfer(q, k, v)
    if method == "quest":
        return lambda: module.batched_per_head_sparse_decode(q, k, v, quest)
    if method == "quest_twi":
        return lambda: module.batched_gqa_group_shared_decode(q, k, v, union)
    return lambda: module.batched_gqa_group_exception_decode(q, k, v, indices["group"], indices["exception"])


def shape_paths(output: Path, length: int, batch: int) -> dict[str, Path]:
    stem = f"L{length}_B{batch}"
    return {
        "timings": output / "shards" / f"{stem}_timings.csv",
        "components": output / "shards" / f"{stem}_components.csv",
        "estimates": output / "shards" / f"{stem}_estimates.csv",
        "correctness": output / "shards" / f"{stem}_correctness.csv",
        "support": output / "shards" / f"{stem}_support.csv",
    }


def shape_complete(output: Path, length: int, batch: int, args) -> bool:
    paths = shape_paths(output, length, batch)
    if not all(path.exists() for path in paths.values()):
        return False
    timing_rows = read_csv(paths["timings"])
    expected = args.rounds * args.repeats * (len(METHODS) + len(SPARSE_METHODS) + 2)
    return len(timing_rows) >= expected


def run_shape(args, output: Path, module, cfg: dict, spec: dict, length: int, batch: int, device: torch.device) -> None:
    if not args.force and shape_complete(output, length, batch, args):
        print(f"skip completed length={length} batch={batch}", flush=True)
        return
    paths = shape_paths(output, length, batch)
    seed = args.seed + length * 1009 + batch * 9176
    torch.manual_seed(seed)
    group_size = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
    q = torch.randn(batch, cfg["num_attention_heads"], cfg["head_dim"], device=device, dtype=torch.float16)
    k = torch.randn(batch, cfg["num_key_value_heads"], length, cfg["head_dim"], device=device, dtype=torch.float16)
    v = torch.randn(batch, cfg["num_key_value_heads"], length, cfg["head_dim"], device=device, dtype=torch.float16)
    indices = build_consistent_indices(spec, batch, cfg["num_attention_heads"], cfg["num_key_value_heads"], 16, seed, device)
    union, _, union_stats = build_union_from_heads(indices["a"], batch, cfg["num_attention_heads"], cfg["num_key_value_heads"])
    quest = make_candidate_indices(batch, cfg["num_attention_heads"], length, 8192, device)
    scores = torch.randn(batch, cfg["num_attention_heads"], length, device=device, dtype=torch.float16)
    funcs = {
        method: selection_callable(method, scores, group_size, 8192)
        for method in SPARSE_METHODS
    }
    funcs["fa2"] = None
    funcs["flashinfer"] = None
    attention: dict[str, Callable[[], object]] = {}
    unavailable: dict[str, str] = {}
    for method in METHODS:
        try:
            attention[method] = attention_callable(method, module, q, k, v, indices, quest, union)
        except Exception as exc:
            unavailable[method] = f"{type(exc).__name__}: {exc}"

    correctness_rows: List[dict] = []
    check_batch = min(batch, args.correctness_batch)
    q_check, k_check, v_check = q[:check_batch], k[:check_batch], v[:check_batch]
    check_indices = {key: value[:check_batch] if torch.is_tensor(value) and value.shape[0] == batch else value for key, value in indices.items()}
    check_quest = quest[:check_batch]
    check_union = union[:check_batch]
    for method in METHODS:
        if method in unavailable:
            correctness_rows.append({"method": method, "length": length, "batch_size": batch, "check_batch_size": check_batch, "status": "unavailable", "reason": unavailable[method], "pass": ""})
            continue
        try:
            result = attention[method]() if method in attention else None
            torch.cuda.synchronize(device)
            reference = ref_for_method(method, q_check, k_check, v_check, check_indices, check_quest, check_union)
            metrics = error_metrics(result[:check_batch], reference, 2e-2, 2e-2)
            correctness_rows.append({"method": method, "length": length, "batch_size": batch, "check_batch_size": check_batch, "status": "available", "reason": "", **metrics})
        except Exception as exc:
            correctness_rows.append({"method": method, "length": length, "batch_size": batch, "check_batch_size": check_batch, "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "pass": False})
    write_csv(paths["correctness"], correctness_rows, list(correctness_rows[0]))
    # Correctness wrappers may retain backend workspaces.  Timing creates one
    # backend at a time so high-batch FlashInfer/FA2 workspaces do not coexist.
    del attention
    torch.cuda.empty_cache()

    support_rows = []
    for method in METHODS:
        if method == "fa2" or method == "flashinfer":
            candidate = length
            final = length
        elif method == "quest":
            candidate = quest.shape[-1]
            final = quest.shape[-1]
        elif method == "quest_twi":
            candidate = int(round(spec["headwise_support_mean"]))
            final = union_stats["union_mean"]
        else:
            candidate = int(round(spec["headwise_support_mean"]))
            final = spec["gas_support_mean"]
        support_rows.append({
            "length": length,
            "batch_size": batch,
            "method": method,
            "candidate_tokens": candidate,
            "head_selected_mean": spec["headwise_support_mean"] if method not in ("fa2", "flashinfer") else length,
            "union_tokens_mean": union_stats["union_mean"] if method == "quest_twi" else (spec["gas_union_mean"] if method == "gas" else ""),
            "shared_tokens_mean": spec["shared_mean"] if method == "gas" else "",
            "complement_tokens_mean": spec["complement_mean"] if method == "gas" else "",
            "final_tokens_mean": final,
            "input_type": "profile_calibrated_synthetic",
        })
    write_csv(paths["support"], support_rows, list(support_rows[0]))

    timing_rows: List[dict] = []
    component_rows: List[dict] = []
    layer_fn, output_fn = common_model_callable(batch, cfg, device)
    common_layer_samples: List[float] = []
    output_samples_all: List[float] = []
    selection_samples_by_method: Dict[str, List[float]] = {method: [] for method in SPARSE_METHODS}
    attention_samples_by_method: Dict[str, List[float]] = {method: [] for method in METHODS}
    # Measure common model work once per round and release its large matrix
    # closures before allocating/using attention backend workspaces.
    for round_idx in range(args.rounds):
        layer_samples = measure_component(layer_fn, device, args.warmups, args.repeats)
        output_samples = measure_component(output_fn, device, args.warmups, args.repeats)
        common_layer_samples.extend(layer_samples)
        output_samples_all.extend(output_samples)
        layer_order = ["common_layer", "output"]
        for component in layer_order:
            samples = layer_samples if component == "common_layer" else output_samples
            for repeat_idx, value in enumerate(samples):
                timing_rows.append({"run_id": f"L{length}_B{batch}", "method": "common", "batch_size": batch, "length": length, "trace_id": "synthetic_trace_0", "layer_id": "all_layers", "component": component, "round": round_idx + 1, "repeat": repeat_idx + 1, "latency_us": value, "timing_scope": "component", "order": ">".join(layer_order)})
    del layer_fn, output_fn
    torch.cuda.empty_cache()

    # Run one backend at a time.  The method order is still randomized per
    # round in the record, but backend workspaces are released after every
    # method to keep B128 within the available device memory.
    for round_idx in range(args.rounds):
        order = list(METHODS)
        random.Random(seed + 500009 + round_idx).shuffle(order)
        for method in order:
            if method in unavailable:
                continue
            try:
                method_attention = attention_callable(method, module, q, k, v, indices, quest, union)
                if method in SPARSE_METHODS:
                    samples = measure_component(funcs[method], device, args.warmups, args.repeats)
                    selection_samples_by_method[method].extend(samples)
                    for repeat_idx, value in enumerate(samples):
                        timing_rows.append({"run_id": f"L{length}_B{batch}", "method": method, "batch_size": batch, "length": length, "trace_id": "synthetic_trace_0", "layer_id": "all_layers", "component": "selection_and_metadata", "round": round_idx + 1, "repeat": repeat_idx + 1, "latency_us": value, "timing_scope": "component", "order": ">".join(order)})
                samples = cuda_event_samples(method_attention, device, args.warmups, args.repeats)
                attention_samples_by_method[method].extend(samples)
                for repeat_idx, value in enumerate(samples):
                    timing_rows.append({"run_id": f"L{length}_B{batch}", "method": method, "batch_size": batch, "length": length, "trace_id": "synthetic_trace_0", "layer_id": "all_layers", "component": "attention", "round": round_idx + 1, "repeat": repeat_idx + 1, "latency_us": value, "timing_scope": "component", "order": ">".join(order)})
                print(f"finished method={method} length={length} batch={batch} round={round_idx + 1}/{args.rounds}", flush=True)
            except Exception as exc:
                unavailable[method] = f"{type(exc).__name__}: {exc}"
                print(f"unavailable method={method} length={length} batch={batch}: {unavailable[method]}", flush=True)
            finally:
                if "method_attention" in locals():
                    del method_attention
                torch.cuda.empty_cache()
    common_layer_us = statistics.fmean(common_layer_samples)
    output_us = statistics.fmean(output_samples_all)
    for method in METHODS:
        if method in unavailable:
            component_rows.append({"method": method, "length": length, "batch_size": batch, "common_ms": "", "selection_and_metadata_ms": "", "attention_ms": "", "output_ms": "", "path_overhead_ms": "", "estimated_tpot_ms": "", "common_layer_us": "", "attention_one_layer_us": "", "selection_one_layer_us": "", "sparse_layers": "", "dense_attention_layers": "", "timing_scope": "component_sum", "input_type": "profile_calibrated_synthetic", "status": "unavailable", "reason": unavailable[method]})
            continue
        attention_us = statistics.fmean(attention_samples_by_method[method])
        selection_us = statistics.fmean(selection_samples_by_method[method]) if method in SPARSE_METHODS else 0.0
        dense_layers = 32 if method in ("fa2", "flashinfer") else 2
        sparse_layers = 0 if method in ("fa2", "flashinfer") else 30
        dense_attention_us = attention_us * dense_layers
        sparse_attention_us = attention_us * sparse_layers
        selection_total_us = selection_us * sparse_layers
        common_total_us = common_layer_us * 32
        total_us = common_total_us + output_us + dense_attention_us + sparse_attention_us + selection_total_us
        component_rows.append({"method": method, "length": length, "batch_size": batch, "common_ms": common_total_us / 1000.0, "selection_and_metadata_ms": selection_total_us / 1000.0, "attention_ms": (dense_attention_us + sparse_attention_us) / 1000.0, "output_ms": output_us / 1000.0, "path_overhead_ms": 0.0, "estimated_tpot_ms": total_us / 1000.0, "common_layer_us": common_layer_us, "attention_one_layer_us": attention_us, "selection_one_layer_us": selection_us, "sparse_layers": sparse_layers, "dense_attention_layers": dense_layers, "timing_scope": "component_sum", "input_type": "profile_calibrated_synthetic", "status": "available"})
    write_csv(paths["timings"], timing_rows, list(timing_rows[0]))
    # One component row per method/shape is sufficient for the step aggregation.
    unique_components = {(row["method"], row["length"], row["batch_size"]): row for row in component_rows}
    write_csv(paths["components"], list(unique_components.values()), list(next(iter(unique_components.values()))))
    estimate_rows = []
    for row in unique_components.values():
        for trace_id in range(args.num_traces):
            estimate_rows.append({"method": row["method"], "length": length, "batch_size": batch, "trace_id": f"synthetic_trace_{trace_id}", "round": 0, "common_ms": row["common_ms"], "selection_and_metadata_ms": row["selection_and_metadata_ms"], "attention_ms": row["attention_ms"], "dense_layer_ms": 0.0, "output_ms": row["output_ms"], "path_overhead_ms": row["path_overhead_ms"], "estimated_tpot_ms": row["estimated_tpot_ms"], "representative_layer_weight": 1.0, "formula": "32*common_layer + attention_layers + sparse_layers*selection + output", "input_type": row["input_type"]})
    write_csv(paths["estimates"], estimate_rows, list(estimate_rows[0]))
    del q, k, v, scores, quest, union
    torch.cuda.empty_cache()


def aggregate(output: Path, args, cfg: dict, specs: Dict[int, dict]) -> None:
    timing_rows: List[dict] = []
    component_rows: List[dict] = []
    estimate_rows: List[dict] = []
    correctness_rows: List[dict] = []
    support_rows: List[dict] = []
    for path in sorted((output / "shards").glob("*_timings.csv")):
        timing_rows.extend(read_csv(path))
    for path in sorted((output / "shards").glob("*_components.csv")):
        component_rows.extend(read_csv(path))
    for path in sorted((output / "shards").glob("*_estimates.csv")):
        estimate_rows.extend(read_csv(path))
    for path in sorted((output / "shards").glob("*_correctness.csv")):
        correctness_rows.extend(read_csv(path))
    for path in sorted((output / "shards").glob("*_support.csv")):
        support_rows.extend(read_csv(path))
    if timing_rows:
        write_csv(output / "E7_timings_raw.csv", timing_rows, list(timing_rows[0]))
    if component_rows:
        write_csv(output / "E7_components.csv", component_rows, list(component_rows[0]))
    if estimate_rows:
        write_csv(output / "E7_step_estimates.csv", estimate_rows, list(estimate_rows[0]))
    if correctness_rows:
        write_csv(output / "E7_correctness.csv", correctness_rows, list(correctness_rows[0]))
    if support_rows:
        write_csv(output / "E7_support_stats.csv", support_rows, list(support_rows[0]))

    method_values = {
        (row["method"], int(row["length"]), int(row["batch_size"])): float(row["estimated_tpot_ms"])
        for row in component_rows
        if row.get("status") == "available" and row.get("estimated_tpot_ms") not in (None, "")
    }
    rows = []
    for method, length, batch in ((m, l, b) for l in args.lengths for b in args.batch_sizes for m in METHODS):
        values = [row for row in component_rows if row["method"] == method and int(row["length"]) == length and int(row["batch_size"]) == batch]
        if not values or values[0].get("status") != "available":
            status = values[0].get("status", "missing") if values else "missing"
            rows.append({"method": method, "batch_size": batch, "length": length, "status": status})
            continue
        row = values[0]
        rows.append({**row, "speedup_vs_fa2": method_values.get(("fa2", length, batch), 0.0) / float(row["estimated_tpot_ms"]) if ("fa2", length, batch) in method_values else "", "speedup_vs_flashinfer": method_values.get(("flashinfer", length, batch), 0.0) / float(row["estimated_tpot_ms"]) if ("flashinfer", length, batch) in method_values else "", "speedup_vs_quest": method_values.get(("quest", length, batch), 0.0) / float(row["estimated_tpot_ms"]) if ("quest", length, batch) in method_values else "", "speedup_vs_twilight": method_values.get(("quest_twi", length, batch), 0.0) / float(row["estimated_tpot_ms"]) if ("quest_twi", length, batch) in method_values else "", "estimated_tokens_per_second": batch * 1000.0 / float(row["estimated_tpot_ms"])})
    fields = ["method", "batch_size", "length", "estimated_tpot_ms", "common_ms", "selection_and_metadata_ms", "attention_ms", "output_ms", "path_overhead_ms", "speedup_vs_fa2", "speedup_vs_flashinfer", "speedup_vs_quest", "speedup_vs_twilight", "estimated_tokens_per_second", "timing_scope", "input_type", "status"]
    write_csv(output / "E7_summary.csv", rows, fields)
    write_csv(output / "E7_optimization.csv", [{"phase": "baseline", "optimization": "none", "status": "frozen", "scope": "profile_calibrated_synthetic"}], ["phase", "optimization", "status", "scope"])
    render_report(output, rows, correctness_rows, timing_rows)


def render_report(output: Path, rows: Sequence[dict], correctness: Sequence[dict], timings: Sequence[dict]) -> None:
    lines = [
        "# E7 Estimated TPOT Result Summary",
        "",
        "> AUDIT 2026-10-23: aggregate TPOT and speedups are not publication-ready.",
        "> Selection is disconnected from attention and dense-layer costs are miscounted.",
        "> See VALIDITY_NOTICE.md in the archived E7 directory and thoughts/Paper/Quest_TPOT_Audit_20261023.md.",
        "",
        "This is a component-based Estimated TPOT projection, not measured end-to-end serving latency.",
        "Inputs are profile-calibrated synthetic replays using LLaMA-3.1-8B dimensions and current CUDA attention backends.",
        "Estimated TPOT is a complete single-step estimate and is not divided by batch size.",
        "",
        "## Main results",
        "",
        "| method | length | batch | estimated TPOT ms | common ms | selection/metadata ms | attention ms | output ms | vs FA2 | vs FlashInfer | vs Quest | vs Twilight | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(key):
            value = row.get(key, "")
            return "" if value in (None, "") else f"{float(value):.4f}"
        lines.append("| {method} | {length} | {batch_size} | {tpot} | {common} | {selection} | {attention} | {output} | {fa2} | {fi} | {quest} | {twi} | {status} |".format(method=row["method"], length=row["length"], batch_size=row["batch_size"], tpot=fmt("estimated_tpot_ms"), common=fmt("common_ms"), selection=fmt("selection_and_metadata_ms"), attention=fmt("attention_ms"), output=fmt("output_ms"), fa2=fmt("speedup_vs_fa2"), fi=fmt("speedup_vs_flashinfer"), quest=fmt("speedup_vs_quest"), twi=fmt("speedup_vs_twilight"), status=row.get("status", "")))
    passed = sum(row.get("pass", "").lower() == "true" for row in correctness)
    lines += [
        "",
        "## Scope and limits",
        "",
        f"- Correctness checks: {passed}/{len(correctness)} rows passed on the representative correctness batch.",
        f"- Raw component timing rows: {len(timings)}.",
        "- The common model cost uses random FP16 matrices with the exact LLaMA-3.1-8B dimensions; it is a shape-faithful component replay, not a checkpoint-weight replay.",
        "- The sparse support counts come from the completed 16K/32K GAS profile summaries and are labelled profile-calibrated synthetic.",
        "- The estimate sums 32 common transformer layers, two dense attention layers for sparse methods, 30 sparse attention layers, method-specific selection cost, and one output projection.",
        "- JIT, first allocation, CPU loading, tokenizer, prefill, network, queueing, and inter-request overlap are excluded.",
        "- Full KV cache residency at these batch/context combinations is not claimed on a single 40GB GPU.",
        "",
        "The five method labels are FA2, FlashInfer, Quest with the full 8192-token candidate set, Quest+Twilight with group-union execution, and GAS with group-max p_g=0.80 and exception budget 16.",
    ]
    (output / "E7_Result_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--correctness-batch", type=int, default=4)
    parser.add_argument("--num-traces", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help="Allow a small smoke matrix instead of the full E7 matrix")
    args = parser.parse_args()
    if not args.allow_subset and (args.lengths != [16384, 32768] or args.batch_sizes != [32, 64, 96, 128]):
        raise ValueError("E7 requires lengths [16384, 32768] and batches [32, 64, 96, 128]")
    if not torch.cuda.is_available():
        raise RuntimeError("E7 requires CUDA")
    cfg = model_config(args.model_config)
    specs = profile_specs()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods = {
        "fa2": {"label": "FA2", "support": "dense", "backend": "flash_attn_with_kvcache"},
        "flashinfer": {"label": "FlashInfer", "support": "dense", "backend": "BatchDecodeWithPagedKVCacheWrapper"},
        "quest": {"label": "Quest (candidate I0_h)", "support": "I0_h, budget 8192", "backend": "batched_per_head_sparse_decode"},
        "quest_twi": {"label": "Quest+Twilight", "support": "U_g=union(S_h)", "backend": "batched_gqa_group_shared_decode"},
        "gas": {"label": "GAS", "support": "F_h=G_g union E_h", "backend": "batched_gqa_group_exception_decode"},
    }
    atomic_json(output / "methods.json", methods)
    atomic_json(output / "config.json", {"protocol": "E7 Estimated TPOT", "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(), "model": cfg, "shape": {"lengths": args.lengths, "batch_sizes": args.batch_sizes, "q_len": 1, "dtype": "float16", **cfg["actual"]}, "sparsity": {"selector": "Quest token_budget=8192 chunk_size=16", "twilight_p_t": 0.85, "gas_p_g": 0.80, "gas_exception_budget": 16, "sparse_layers": 30, "dense_layers": 2}, "timing": {"warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds, "estimated_tpot_no_batch_division": True}, "input_type": "profile_calibrated_synthetic", "assumptions": ["Common model work uses shape-faithful random FP16 matrices.", "Support counts are calibrated from completed LLaMA profile summaries.", "Results are component sums and are not measured serving TPOT.", "All methods are run sequentially to reduce GPU residency pressure."]})
    atomic_json(output / "E7_trace_manifest.json", {"input_type": "profile_calibrated_synthetic", "real_trace_available": False, "lengths": args.lengths, "batch_sizes": args.batch_sizes, "num_traces": args.num_traces, "seed": args.seed, "profile_sources": {str(length): spec for length, spec in specs.items()}, "representative_layer": "all 32 layers by weighted component replication", "dense_layers": [0, 1], "sparse_layers": list(range(2, 32))})
    device = torch.device("cuda", args.device)
    from bench_e6_kernel_baselines import get_gas_module
    module = get_gas_module()
    for length in args.lengths:
        for batch in args.batch_sizes:
            run_shape(args, output, module, cfg["actual"], specs[length], length, batch, device)
    aggregate(output, args, cfg["actual"], specs)
    print(f"E7 complete: {output}", flush=True)


if __name__ == "__main__":
    main()
