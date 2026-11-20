#!/usr/bin/env python3
"""Quest-derived B1 wrapper on compact, per-head selected-page caches.

Selection, page packing, and wrapper planning are excluded. This is not native
batching or unmodified upstream Quest. Calls and output scatters are timed.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import load_extension
from bench_e6_qasper_trace import digest, write_rows
from bench_gas_mixed_trace import chunked_reference, make_batch
from bench_quest_official_headpacked import page_groups


def compact_setup(module, q, k, v, candidate, lengths, max_heads):
    batch, heads, dim = q.shape
    maxlen = k.shape[2]
    buckets = page_groups(candidate, lengths)
    out = torch.empty_like(q).view(batch * heads, dim)
    entries = []
    try:
        for (n_remaining, last_len), values in sorted(buckets.items()):
            for start in range(0, len(values), max_heads):
                group = values[start:start + max_heads]
                head_ids = torch.tensor([x[0] for x in group], device=q.device, dtype=torch.int64)
                kv_ids = torch.tensor([x[1] for x in group], device=q.device, dtype=torch.int64)
                pages = torch.tensor([x[2] + [x[3]] for x in group], device=q.device, dtype=torch.int64)
                tokens = (pages.unsqueeze(-1) * 16 + torch.arange(16, device=q.device)).reshape(len(group), -1)
                tokens.clamp_(max=maxlen - 1)
                query = q.view(batch * heads, dim).index_select(0, head_ids).unsqueeze(0).contiguous()
                partial = torch.empty_like(query)
                data = torch.empty(n_remaining + 1, 2, 16, len(group), dim, device=q.device, dtype=q.dtype)
                for j, source in enumerate((k, v)):
                    selected = source.view(batch * 8, maxlen, dim)[kv_ids[:, None], tokens]
                    data[:, j].copy_(selected.view(len(group), n_remaining + 1, 16, dim).permute(1, 2, 0, 3))
                    del selected
                indices = torch.arange(n_remaining, device=q.device, dtype=torch.int32).expand(len(group), -1).contiguous()
                indptr = torch.tensor([0, n_remaining], device=q.device, dtype=torch.int32)
                empty = torch.empty(0, device=q.device, dtype=q.dtype)
                wrapper = module.BatchDecodeWithPagedKVCachePyTorchWrapper(0)
                wrapper.begin_forward(indptr, len(group), len(group), dim, 16, empty)
                entries.append((wrapper, query, partial, data, indices, indptr, last_len, head_ids, n_remaining))
    except Exception:
        for wrapper, *_ in entries:
            wrapper.end_forward()
        raise

    def call():
        for wrapper, query, partial, data, indices, indptr, last_len, head_ids, last_page in entries:
            wrapper.forward(query, partial, data, indices, indptr, last_len, last_page, 1.0, 1e4)
            out.index_copy_(0, head_ids, partial.squeeze(0))
        return out.view_as(q)

    return call, [entry[0] for entry in entries]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--nominal-length", type=int, choices=(16384, 32768), required=True)
    p.add_argument("--module", type=Path, default=Path("/data/tmp/torch_ext_e6_stream_adapters/e6_quest_nopart_fixed/e6_quest_nopart_fixed.so"))
    p.add_argument("--max-heads", type=int, default=128)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--graph-iterations", type=int, default=8)
    p.add_argument("--reference-chunk", type=int, default=2)
    args = p.parse_args()
    if not torch.cuda.is_available() or min(args.batch, args.max_heads, args.repeats, args.rounds, args.graph_iterations, args.reference_chunk) < 1 or args.warmups < 0:
        p.error("CUDA and positive benchmark settings required")
    paths = [d / "step001_layer15.pt" for d in args.trace_dirs]
    manifests = [json.loads((d / "manifest.json").read_text()) for d in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests) or any(m["config"] != manifests[0]["config"] for m in manifests):
        raise ValueError("requests must be distinct and use the same configuration")
    cfg = manifests[0]["config"]
    if cfg["selector"].get("type") != "quest" or cfg["selector"].get("token_budget") != 8192 or cfg["selector"].get("chunk_size") != 16:
        raise ValueError("fixed Quest 8192/chunk16 required")
    traces = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    lengths = [t["k"].shape[2] for t in traces]
    if any(length <= args.nominal_length * 0.8 or length > args.nominal_length for length in lengths):
        raise ValueError("trace lengths do not match nominal length")
    for trace in traces:
        for key in ("q", "k", "v"):
            trace[key] = trace[key].to(torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = args.output_dir / "shards" / f"l{args.nominal_length}_b{args.batch}.csv"
    if shard.exists():
        raise FileExistsError(shard)
    meta = {"batch": args.batch, "nominal_length": args.nominal_length, "lengths": lengths,
            "trace_sha256": [digest(path) for path in paths], "config": cfg,
            "module_sha256": digest(args.module), "script_sha256": digest(Path(__file__)),
            "device": torch.cuda.get_device_name(), "dtype": "float16", "max_heads": args.max_heads,
            "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
            "graph_iterations": args.graph_iterations,
            "boundary": "patched B1 wrapper, per-head selected-page packing and planning excluded; grouped calls and output scatters timed; GPU-resident KV"}
    (args.output_dir / f"config_{args.nominal_length}_{args.batch}.json").write_text(json.dumps(meta, indent=2) + "\n")
    module = load_extension(args.module, "e6_quest_nopart_fixed")
    q, k, v, masks, _, _, lengths = make_batch(traces, args.batch, "cuda")
    expected = chunked_reference(q, k, v, masks["candidate"], args.reference_chunk)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row_lengths = [lengths[i % len(lengths)] for i in range(args.batch)]
    call, wrappers = compact_setup(module, q, k, v, masks["candidate"], row_lengths, args.max_heads)
    try:
        output = call()
        check = error_metrics(output, expected, 0.02, 0.02)
        if not check["pass"]:
            raise ValueError(f"incorrect output: {check}")
        for _ in range(args.warmups):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.graph_iterations):
                output = call()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check = error_metrics(output, expected, 0.02, 0.02)
        if not check["pass"]:
            raise ValueError(f"incorrect graph output: {check}")
        timings = []
        for r in range(args.rounds):
            samples = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
            timings.extend(samples)
            write_rows(args.output_dir / "samples" / f"l{args.nominal_length}_b{args.batch}_{r}.csv", [{"latency_us": t} for t in samples])
        row = {"nominal_length": args.nominal_length, "batch": args.batch, "method": "quest_patched_compact",
               "wrapper_calls": len(wrappers), "samples": len(timings), "median_us": statistics.median(timings),
               "max_abs_error": check["max_abs_error"], "max_allocated_bytes": torch.cuda.max_memory_allocated()}
        write_rows(shard, [row])
        print(f"DONE L{args.nominal_length} B{args.batch}: {row['median_us']:.2f} us, {len(wrappers)} calls", flush=True)
    finally:
        for wrapper in wrappers:
            wrapper.end_forward()


if __name__ == "__main__":
    main()
