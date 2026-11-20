#!/usr/bin/env python3
"""Experimental head-packed batch adapter for the upstream B=1 LessIsMore kernel.

This is not native batching: request and head dimensions are flattened and KV
is repacked offline. Selection, packing and wrapper planning are excluded.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import less_mask, load_extension
from bench_e6_qasper_trace import digest, fi_exact, reference, write_rows
from bench_gas_mixed_trace import dense_backends, make_batch

ROOT = Path(__file__).resolve().parents[2]


def official_setup(module, q, k, v, ids, lengths, output_tensor=None):
    """Flatten batch into head slots; each request uses its own KV-head slots."""
    batch, heads, dim = q.shape
    groups, max_len = k.shape[1:3]
    budget = ids.shape[1]
    if ids.shape != (batch, budget) or budget < 2 or any(int(ids[b, -1]) != lengths[b] - 1 for b in range(batch)):
        raise ValueError("each support must end in its request's latest token")
    # Upstream wrapper is hard-coded to B=1 and one global last page. The
    # appended page carries each request's own last token in its head slots.
    data = torch.empty(max_len + 1, 2, 1, batch * groups, dim, device=q.device, dtype=q.dtype)
    for j, src in enumerate((k, v)):
        data[:max_len, j, 0].view(max_len, batch, groups, dim).copy_(src.permute(2, 0, 1, 3))
        for b, length in enumerate(lengths):
            data[max_len, j, 0, b * groups:(b + 1) * groups].copy_(src[b, :, length - 1])
    indices = ids[:, None, :-1].expand(batch, heads, budget - 1).reshape(batch * heads, budget - 1).to(torch.int32).contiguous()
    indptr = torch.tensor([0, budget - 1], dtype=torch.int32, device=q.device)
    output = (torch.empty_like(q) if output_tensor is None else output_tensor).view(1, batch * heads, dim)
    query = q.view(1, batch * heads, dim)
    empty = torch.empty(0, device=q.device, dtype=q.dtype)
    wrapper = module.BatchDecodeWithPagedKVCachePyTorchWrapper(0)
    wrapper.begin_forward(indptr, batch * heads, batch * groups, dim, 1, empty)
    def call():
        wrapper.forward(query, output, data, indices, indptr, 1, max_len, empty, 1.0, 1e4)
        return output.view_as(q)
    return call, wrapper


def official_grouped_setup(module, q, k, v, ids, lengths, pack_size):
    """Call the upstream B1 wrapper once per head-packed request group."""
    output = torch.empty_like(q)
    entries = []
    try:
        for start in range(0, q.shape[0], pack_size):
            end = min(start + pack_size, q.shape[0])
            entries.append(official_setup(module, q[start:end], k[start:end], v[start:end],
                                          ids[start:end], lengths[start:end], output[start:end]))
    except Exception:
        for _, wrapper in entries:
            wrapper.end_forward()
        raise

    def call():
        for fn, _ in entries:
            fn()
        return output

    return call, [wrapper for _, wrapper in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", type=Path, default=Path("/data/tmp/torch_ext_e6_stream_adapters/e6_less_stream_decode/e6_less_stream_decode.so"))
    parser.add_argument("--module-name", default="e6_less_stream_decode")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--search-layer", type=int, default=2)
    parser.add_argument("--budget", type=int, help="fixed shared-token budget; otherwise match layer Twilight union")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    parser.add_argument("--pack-size", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.batch_sizes + [args.repeats, args.rounds, args.graph_iterations, args.pack_size]) < 1 or args.warmups < 0 or (args.budget is not None and args.budget < 2):
        parser.error("CUDA and positive measurements required")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests):
        raise ValueError("trace samples must be distinct")
    if any(m["config"] != manifests[0]["config"] or m["model"] != manifests[0]["model"] for m in manifests):
        raise ValueError("trace configurations differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    less_kernel_name = "less_official_headpacked" if args.module_name == "e6_less_stream_decode" else "less_patched_headpacked"
    config = {"samples": [m["sample_index"] for m in manifests], "steps": args.steps,
              "batch_sizes": args.batch_sizes, "layer": args.layer, "search_layer": args.search_layer,
              "fixed_budget": args.budget,
              "module": str(args.module.resolve()), "module_name": args.module_name, "module_sha256": digest(args.module),
              "trace_manifest_sha256": [digest(p / "manifest.json") for p in args.trace_dirs],
              "script_sha256": digest(Path(__file__)), "device": torch.cuda.get_device_name(),
              "source_dtype": manifests[0]["dtype"], "timed_dtype": "float16",
              "methods": ["fa2_dense", "fi_dense", "quest_fi_adapter", "twilight_fi_adapter", "gas_fi_adapter", "gas_inline8", "gas_groupsplit128", "less_fi_adapter", less_kernel_name],
              "gas_sources": {str(p): digest(ROOT / p) for p in ("csrc/src/gas_counted_decode.cu", "csrc/src/gas_cooperative_split.cu")},
              "rounds": args.rounds, "repeats": args.repeats, "graph_iterations": args.graph_iterations, "pack_size": args.pack_size,
              "limitations": f"{less_kernel_name} B1 attention kernel with groups of B packed into head axis; multiple kernel calls included; KV repacking, selection and plan excluded; not native batch; GAS-state search layer; K fixed or matched to mean Twilight group union"}
    path = args.output_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("output directory has different settings")
    path.write_text(json.dumps(config, indent=2) + "\n")
    module = load_extension(args.module, args.module_name)
    counted = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    groupsplit = load(name="gas_cooperative_split", sources=[str(ROOT / "csrc/src/gas_cooperative_split.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    rows = []
    for step in args.steps:
        names = [f"step{step:03d}_layer{args.layer:02d}.pt", f"step{step:03d}_layer{args.search_layer:02d}.pt"]
        for m in manifests:
            if any(n not in m["files"] for n in names):
                raise ValueError(f"missing trace for step {step}")
        traces = [torch.load(p / names[0], map_location="cpu", weights_only=True) for p in args.trace_dirs]
        searches = [torch.load(p / names[1], map_location="cpu", weights_only=True) for p in args.trace_dirs]
        for trace in traces:
            for key in ("q", "k", "v"):
                trace[key] = trace[key].to(torch.float16)
        lengths = [t["k"].shape[2] for t in traces]
        unions = [t["twilight"].view(1, 8, 4, -1).any(2).sum(-1).float().mean().item() for t in traces]
        budget = args.budget if args.budget is not None else max(2, round(statistics.fmean(unions)))
        if budget >= min(lengths):
            raise ValueError("matched budget exceeds shortest context")
        selected = []
        for search, length in zip(searches, lengths):
            if search["k"].shape[2] != length:
                raise ValueError("search and target layers have different sequence lengths")
            search_gpu = {key: search[key].to("cuda") for key in ("q", "k")}
            mask = less_mask(search_gpu, length, budget)
            selected.append(torch.nonzero(mask[0, 0], as_tuple=True)[0].cpu())
        for batch in args.batch_sizes:
            q, k, v, masks, shared, ex, _ = make_batch(traces, batch, "cuda")
            row_lengths = [lengths[i % len(lengths)] for i in range(batch)]
            ids = torch.stack([selected[i % len(selected)] for i in range(batch)]).to("cuda")
            less = torch.zeros_like(masks["gas"])
            less.scatter_(2, ids[:, None, :].expand(batch, 32, budget), True)
            expected_less = reference(q, k, v, less)
            expected_gas = reference(q, k, v, masks["gas"])
            expected_quest = reference(q, k, v, masks["candidate"])
            expected_dense = reference(q, k, v, masks["dense"])
            official, wrappers = official_grouped_setup(module, q, k, v, ids, row_lengths, args.pack_size)
            fa2, fi_dense = dense_backends(q, k, v, lengths)
            def gas_groupsplit():
                sc, ec = counted.count_prefix(shared, ex)
                return groupsplit.decode(q, k, v, shared, ex, sc, ec, 8, 8, 128)
            calls = {"fa2_dense": fa2, "fi_dense": fi_dense,
                     "quest_fi_adapter": fi_exact(q, k, v, masks["candidate"]),
                     "gas_inline8": lambda: counted.decode_prefix(q, k, v, shared, ex, 8),
                     "gas_groupsplit128": gas_groupsplit,
                     less_kernel_name: official,
                     "less_fi_adapter": fi_exact(q, k, v, less),
                     "gas_fi_adapter": fi_exact(q, k, v, masks["gas"]),
                     "twilight_fi_adapter": fi_exact(q, k, v, masks["twilight"])}
            expected_twi = reference(q, k, v, masks["twilight"])
            try:
                order = list(calls)
                random.Random(f"{step}:{batch}").shuffle(order)
                for name in order:
                    call = calls[name]
                    target = (expected_less if name.startswith("less_") else expected_gas if name.startswith("gas_") else
                              expected_quest if name.startswith("quest_") else expected_dense if name.endswith("_dense") else expected_twi)
                    output = call()
                    check = error_metrics(output, target, 0.02, 0.02)
                    if not check["pass"]:
                        raise ValueError(f"{step=} {batch=} {name}: {check}")
                    for _ in range(args.warmups):
                        call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.graph_iterations):
                            output = call()
                    output.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                    check = error_metrics(output, target, 0.02, 0.02)
                    if not check["pass"]:
                        raise ValueError(f"graph {step=} {batch=} {name}: {check}")
                    timings = []
                    for r in range(args.rounds):
                        measured = [t / args.graph_iterations for t in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                        timings.extend(measured)
                        write_rows(args.output_dir / "samples" / f"step{step}_b{batch}_{name}_{r}.csv", [{"latency_us": t} for t in measured])
                    row = {"step": step, "batch": batch, "method": name, "less_k": budget,
                           "twilight_union_mean_by_request": json.dumps(unions), "lengths": json.dumps(lengths),
                           "official_calls": len(wrappers) if name == less_kernel_name else 0,
                           "samples": len(timings), "median_us": statistics.median(timings), "max_abs_error": check["max_abs_error"]}
                    rows.append(row)
                    write_rows(args.output_dir / "shards" / f"step{step}_b{batch}_{name}.csv", [row])
                    print(f"step={step} B{batch} {name} K={budget}: {row['median_us']:.2f} us", flush=True)
                    del graph, output
            finally:
                for wrapper in wrappers:
                    wrapper.end_forward()
            del calls, official, q, k, v, masks, shared, ex, less, expected_less, expected_gas, expected_twi, expected_quest, expected_dense
            torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
