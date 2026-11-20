#!/usr/bin/env python3
"""E6-style kernel replay at matched per-KV-group union cardinality."""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import (affine_tokens, cuda_event_samples, error_metrics,
                                      fp32_reference, load_profile_spec, quantile)
from bench_e6_kernel_baselines import (build_consistent_indices, build_union_from_heads,
                                      get_gas_module, group_reference, write_csv)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmark/results_efficiency/lessismore_e6_matched"
K_BY_LENGTH = {16384: 1101, 32768: 1059}
METHODS = ("twilight_per_head", "twilight_union_proxy", "less_global_shared", "less_global_per_head", "gas_fused")


def less_indices(length, batch, kv_heads, query_heads, budget, recent_ratio, seed, device):
    recent = budget - int(budget * (1 - recent_ratio))
    middle = budget - recent
    selections = []
    for index in range(batch):
        history = affine_tokens(length - recent, middle, seed + 990001 + index * 43)
        selections.append(history + list(range(length - recent, length)))
    global_ids = torch.tensor(selections, dtype=torch.int32, device=device)
    return (global_ids[:, None, :].expand(batch, kv_heads, budget).contiguous(),
            global_ids[:, None, :].expand(batch, query_heads, budget).contiguous())


def paths(output, length, batch):
    root = output / "shards"
    stem = f"L{length}_B{batch}"
    return root / f"{stem}_correctness.csv", root / f"{stem}_timings.csv", root / f"{stem}_support.json"


def run_shape(args, module, spec, length, batch):
    k_budget = K_BY_LENGTH[length]
    seed = args.seed + length * 1009 + batch * 9176
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, 32, 128, dtype=torch.float16, device=device, generator=generator)
    k = torch.randn(batch, 8, length, 128, dtype=torch.float16, device=device, generator=generator)
    v = torch.randn(batch, 8, length, 128, dtype=torch.float16, device=device, generator=generator)
    indices = build_consistent_indices(spec, batch, 32, 8, 16, seed, device)
    union, _, stats = build_union_from_heads(indices["a"], batch, 32, 8)
    _, _, gas_stats = build_union_from_heads(indices["b"], batch, 32, 8)
    if stats["union_min"] != k_budget or stats["union_max"] != k_budget:
        raise ValueError(f"E6 union no longer matches frozen K={k_budget}: {stats}")
    group, head = less_indices(length, batch, 8, 32, k_budget, 0.25, seed, device)
    funcs = {
        "twilight_per_head": lambda: module.batched_per_head_sparse_decode(q, k, v, indices["a"]),
        "twilight_union_proxy": lambda: module.batched_gqa_group_shared_decode(q, k, v, union),
        "less_global_shared": lambda: module.batched_gqa_group_shared_decode(q, k, v, group),
        "less_global_per_head": lambda: module.batched_per_head_sparse_decode(q, k, v, head),
        "gas_fused": lambda: module.batched_gqa_group_exception_decode(
            q, k, v, indices["group"], indices["exception"]),
    }
    check_batch = min(batch, args.correctness_batch)
    qc, kc, vc = q[:check_batch], k[:check_batch], v[:check_batch]
    refs = {
        "twilight_per_head": fp32_reference(qc, kc, vc, indices["a"][:check_batch],
                                              indices["a_counts"][:check_batch * 32], 4),
        "twilight_union_proxy": group_reference(qc, kc, vc, union[:check_batch]),
        "less_global_shared": group_reference(qc, kc, vc, group[:check_batch]),
        "gas_fused": fp32_reference(qc, kc, vc, indices["b"][:check_batch],
                                      indices["b_counts"][:check_batch * 32], 4),
    }
    refs["less_global_per_head"] = refs["less_global_shared"]
    correctness = []
    outputs = {}
    for method in METHODS:
        out = funcs[method]()
        torch.cuda.synchronize(device)
        metrics = error_metrics(out[:check_batch], refs[method], 2e-2, 2e-2)
        metrics["finite_all_batch"] = bool(torch.isfinite(out).all().item())
        correctness.append({"method": method, "length": length, "batch_size": batch,
                            "checked_batch": check_batch, **metrics})
        if not metrics["pass"] or not metrics["finite_all_batch"]:
            raise RuntimeError(f"Correctness failed for {method}: {metrics}")
        if method.startswith("less_"):
            outputs[method] = out[:check_batch].clone()
    equivalence = error_metrics(outputs["less_global_shared"], outputs["less_global_per_head"].float(), 2e-2, 2e-2)
    if not equivalence["pass"]:
        raise RuntimeError(f"Same-support LessIsMore kernels disagree: {equivalence}")
    correct_path, timing_path, stats_path = paths(args.output, length, batch)
    write_csv(correct_path, correctness, list(correctness[0]))
    stats_path.write_text(json.dumps({"length": length, "batch": batch, "seed": seed,
        "less_K": k_budget, "less_recent_ratio": 0.25, "less_global_shared_across_all_query_heads": True,
        "twilight_group_union": stats, "twilight_per_head_mean": spec["headwise_support_mean"],
        "gas_final_group_union": gas_stats,
        "same_support_less_kernel_equivalence": equivalence,
        "indices": "profile-sized synthetic, not actual model selection traces"}, indent=2) + "\n")
    timings = []
    for round_index in range(args.rounds):
        order = list(METHODS)
        random.Random(seed + 500009 + round_index).shuffle(order)
        for method in order:
            samples = cuda_event_samples(funcs[method], device, args.warmups, args.repeats)
            timings.extend({"length": length, "batch_size": batch, "method": method,
                            "round": round_index + 1, "repeat": index + 1, "latency_us": value,
                            "order": ">".join(order)} for index, value in enumerate(samples))
            print(f"DONE length={length} batch={batch} method={method} round={round_index+1}/{args.rounds}", flush=True)
    write_csv(timing_path, timings, list(timings[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768], choices=K_BY_LENGTH)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--correctness-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_sizes + [args.repeats, args.rounds, args.correctness_batch]) <= 0 or args.warmups < 0:
        parser.error("invalid benchmark dimensions or timing counts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output = args.output_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"protocol": "E6 matched union kernel replay", "lengths": args.lengths,
              "batch_sizes": args.batch_sizes, "K_by_length": {str(k): v for k, v in K_BY_LENGTH.items()},
              "dtype": "float16", "heads": 32, "kv_heads": 8, "head_dim": 128,
              "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
              "correctness_batch": args.correctness_batch, "seed": args.seed,
              "recent_ratio": 0.25, "methods": list(METHODS),
              "boundary": "fixed indices and QKV; kernel only; no selection, metadata, or model inference",
              "limitations": "synthetic masks; union proxy changes head supports; Less uses our kernel, not official"}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Different protocol in output directory; use a new output path")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    module = get_gas_module()
    all_rows = []
    for length in args.lengths:
        spec = load_profile_spec(length)
        for batch in args.batch_sizes:
            correct, timings, stats = paths(args.output, length, batch)
            if not (correct.exists() and timings.exists() and stats.exists()):
                run_shape(args, module, spec, length, batch)
            else:
                print(f"SKIP complete length={length} batch={batch}", flush=True)
            with timings.open(newline="", encoding="utf-8") as handle:
                data = list(csv.DictReader(handle))
            if len(data) != len(METHODS) * args.rounds * args.repeats:
                raise ValueError(f"Incomplete timing shard {timings}")
            for method in METHODS:
                values = [float(row["latency_us"]) for row in data if row["method"] == method]
                if len(values) != args.rounds * args.repeats:
                    raise ValueError(f"Incomplete {method} timings in {timings}")
                all_rows.append({"length": length, "batch_size": batch, "K": K_BY_LENGTH[length],
                                 "method": method, "samples": len(values),
                                 "median_us": statistics.median(values), "mean_us": statistics.fmean(values),
                                 "p10_us": quantile(values, 0.1), "p90_us": quantile(values, 0.9)})
    write_csv(args.output / "summary.csv", all_rows, list(all_rows[0]))
    print(f"COMPLETE matched E6 kernel replay: {args.output}", flush=True)


if __name__ == "__main__":
    main()
