#!/usr/bin/env python3
"""Replay Quest B1 CUDA through a page-compatible head-packed adapter.

Head grouping, Q/K/V repacking and wrapper planning are excluded; every
upstream kernel call and output scatter is timed. Not native batch serving.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import load_extension
from bench_e6_qasper_trace import digest, fi_exact, reference, write_rows
from bench_gas_mixed_trace import make_batch

ROOT = Path(__file__).resolve().parents[2]


def page_groups(candidate, lengths):
    """Return exact full pages plus at most one partial final page per head."""
    batch, heads, max_len = candidate.shape
    if heads != 32 or len(lengths) != batch:
        raise ValueError("invalid mask or lengths")
    mask = candidate.cpu()
    buckets = defaultdict(list)
    for b, length in enumerate(lengths):
        if mask[b, :, length:].any():
            raise ValueError("mask selects padding")
        pages = (length + 15) // 16
        padded = torch.nn.functional.pad(mask[b, :, :length], (0, pages * 16 - length))
        counts = padded.view(heads, pages, 16).sum(-1)
        for h in range(heads):
            full = torch.nonzero(counts[h] == 16, as_tuple=True)[0].tolist()
            partial = torch.nonzero((counts[h] > 0) & (counts[h] < 16), as_tuple=True)[0].tolist()
            if len(partial) > 1 or (partial and (length % 16 == 0 or partial[0] != pages - 1 or int(counts[h, partial[0]]) != length % 16)):
                raise ValueError("Quest mask contains a non-tail partial page")
            if partial:
                rest, last, last_len = full, partial[0], length % 16
            elif full:
                rest, last, last_len = full[:-1], full[-1], 16
            else:
                raise ValueError("empty Quest support")
            if not rest:
                raise ValueError("upstream wrapper requires at least one non-last page")
            buckets[(len(rest), last_len)].append((b * heads + h, b * 8 + h // 4, rest, last))
    return buckets


def quest_setup(module, q, k, v, candidate, lengths, max_heads):
    batch, heads, dim = q.shape
    max_len = k.shape[2]
    pages = (max_len + 15) // 16
    buckets = page_groups(candidate, lengths)
    entries = []
    out = torch.empty_like(q).view(batch * heads, dim)
    try:
        for (n_remaining, last_len), values in sorted(buckets.items()):
            for start in range(0, len(values), max_heads):
                group = values[start:start + max_heads]
                head_ids = torch.tensor([x[0] for x in group], device=q.device, dtype=torch.int64)
                kv_ids = torch.tensor([x[1] for x in group], device=q.device, dtype=torch.int64)
                query = q.view(batch * heads, dim).index_select(0, head_ids).unsqueeze(0).contiguous()
                partial_out = torch.empty_like(query)
                data = torch.empty(pages + 1, 2, 16, len(group), dim, device=q.device, dtype=q.dtype)
                for j, src in enumerate((k, v)):
                    gathered = src.view(batch * 8, max_len, dim).index_select(0, kv_ids)
                    gathered = torch.nn.functional.pad(gathered, (0, 0, 0, pages * 16 - max_len))
                    data[:pages, j].copy_(gathered.view(len(group), pages, 16, dim).permute(1, 2, 0, 3))
                    for h, item in enumerate(group):
                        data[pages, j, :, h].copy_(data[item[3], j, :, h])
                indices = torch.tensor([x[2] for x in group], device=q.device, dtype=torch.int32)
                indptr = torch.tensor([0, n_remaining], device=q.device, dtype=torch.int32)
                empty = torch.empty(0, device=q.device, dtype=q.dtype)
                wrapper = module.BatchDecodeWithPagedKVCachePyTorchWrapper(0)
                wrapper.begin_forward(indptr, len(group), len(group), dim, 16, empty)
                entries.append((wrapper, query, partial_out, data, indices, indptr, last_len, head_ids))
    except Exception:
        for wrapper, *_ in entries:
            wrapper.end_forward()
        raise

    def call():
        for wrapper, query, partial, data, indices, indptr, last_len, head_ids in entries:
            wrapper.forward(query, partial, data, indices, indptr, last_len, pages, 1.0, 1e4)
            out.index_copy_(0, head_ids, partial.squeeze(0))
        return out.view_as(q)
    return call, [entry[0] for entry in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", type=Path, default=Path("/data/tmp/torch_ext_e6_stream_adapters/e6_quest_stream_decode/e6_quest_stream_decode.so"))
    parser.add_argument("--module-name", default="e6_quest_stream_decode")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--layers", type=int, nargs="+", default=[3])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--max-heads", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.batch_sizes + [args.max_heads, args.repeats, args.rounds, args.graph_iterations]) < 1 or args.warmups < 0:
        parser.error("CUDA and positive benchmark settings required")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests) or any(m["config"] != manifests[0]["config"] or m["model"] != manifests[0]["model"] for m in manifests):
        raise ValueError("samples must be distinct and use matching model/config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quest_name = "quest_official_grouped" if args.module_name == "e6_quest_stream_decode" else "quest_patched_grouped"
    config = {"samples": [m["sample_index"] for m in manifests], "steps": args.steps, "layers": args.layers,
              "batch_sizes": args.batch_sizes, "max_heads": args.max_heads, "dtype": "float16",
              "module_name": args.module_name, "module_sha256": digest(args.module), "script_sha256": digest(Path(__file__)),
              "trace_manifest_sha256": [digest(p / "manifest.json") for p in args.trace_dirs],
              "repeats": args.repeats, "rounds": args.rounds, "graph_iterations": args.graph_iterations,
              "device": torch.cuda.get_device_name(), "methods": [quest_name, "quest_fi_adapter", "gas_fi_adapter", "twilight_fi_adapter"],
              "limitations": f"{quest_name} B1 wrapper, grouped by page count/last page, duplicated KV and grouped Q offline, multiple calls plus output index_copy timed, selection/packing/plan excluded, GAS model trajectory"}
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("different benchmark settings in output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    module = load_extension(args.module, args.module_name)
    rows = []
    for step in args.steps:
        for layer in args.layers:
            filename = f"step{step:03d}_layer{layer:02d}.pt"
            if any(filename not in m["files"] for m in manifests):
                raise ValueError(f"missing {filename}")
            traces = [torch.load(p / filename, map_location="cpu", weights_only=True) for p in args.trace_dirs]
            for trace in traces:
                for key in ("q", "k", "v"):
                    trace[key] = trace[key].to(torch.float16)
            for batch in args.batch_sizes:
                q, k, v, masks, shared, ex, lengths = make_batch(traces, batch, "cuda")
                row_lengths = [lengths[i % len(lengths)] for i in range(batch)]
                expected = {name: reference(q, k, v, masks[name]) for name in ("candidate", "gas", "twilight")}
                official, wrappers = quest_setup(module, q, k, v, masks["candidate"], row_lengths, args.max_heads)
                calls = {quest_name: official,
                         "quest_fi_adapter": fi_exact(q, k, v, masks["candidate"]),
                         "gas_fi_adapter": fi_exact(q, k, v, masks["gas"]),
                         "twilight_fi_adapter": fi_exact(q, k, v, masks["twilight"])}
                try:
                    order = list(calls)
                    random.Random(f"{step}:{layer}:{batch}").shuffle(order)
                    for name in order:
                        target = expected["candidate" if name.startswith("quest_") else "gas" if name.startswith("gas_") else "twilight"]
                        output = calls[name]()
                        check = error_metrics(output, target, 0.02, 0.02)
                        if not check["pass"]:
                            raise ValueError(f"{step=} {layer=} {batch=} {name}: {check}")
                        for _ in range(args.warmups): calls[name]()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(args.graph_iterations): output = calls[name]()
                        output.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                        check = error_metrics(output, target, 0.02, 0.02)
                        if not check["pass"]:
                            raise ValueError(f"graph {step=} {layer=} {batch=} {name}: {check}")
                        times = []
                        for r in range(args.rounds):
                            samples = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                            times.extend(samples)
                            write_rows(args.output_dir / "samples" / f"s{step}_l{layer}_b{batch}_{name}_{r}.csv", [{"latency_us": x} for x in samples])
                        row = {"step": step, "layer": layer, "batch": batch, "method": name,
                               "quest_calls": len(wrappers) if name == quest_name else 0,
                               "lengths": json.dumps(lengths), "samples": len(times), "median_us": statistics.median(times),
                               "max_abs_error": check["max_abs_error"]}
                        rows.append(row)
                        write_rows(args.output_dir / "shards" / f"s{step}_l{layer}_b{batch}_{name}.csv", [row])
                        print(f"step={step} layer={layer} B{batch} {name}: {row['median_us']:.2f} us; calls={row['quest_calls']}", flush=True)
                        del graph, output
                finally:
                    for wrapper in wrappers: wrapper.end_forward()
                del calls, official, q, k, v, masks, shared, ex, expected
                torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
