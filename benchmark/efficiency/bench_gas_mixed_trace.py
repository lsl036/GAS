#!/usr/bin/env python3
"""Replay different captured requests together; selection and FI planning excluded."""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics, padded_tensor
from bench_e6_qasper_trace import digest, fi_exact, reference, write_rows

ROOT = Path(__file__).resolve().parents[2]


def choose_replay_policy(batch, shared_width):
    """Experimental policy using only host-visible padded index dimensions."""
    if shared_width < 256:
        return "inline8"
    if shared_width >= 1024:
        return "prefix_groupsplit128"
    return "inline16" if batch <= 64 else "prefix_groupsplit128"


def chunked_reference(q, k, v, mask, batch_chunk):
    if batch_chunk < 1:
        raise ValueError("reference batch chunk must be positive")
    return torch.cat([reference(q[i:i + batch_chunk], k[i:i + batch_chunk], v[i:i + batch_chunk],
                                None if mask is None else mask[i:i + batch_chunk])
                      for i in range(0, q.shape[0], batch_chunk)], dim=0)


def load_cached(shard, sample_paths, trace_paths, repeats):
    if not shard.exists():
        return None
    with shard.open(newline="", encoding="utf-8") as stream:
        cached = list(csv.DictReader(stream))
    if (len(cached) != 1 or int(cached[0]["samples"]) != repeats * len(sample_paths) or
            json.loads(cached[0]["trace_sha256"]) != [digest(p) for p in trace_paths]):
        raise ValueError(f"invalid existing shard {shard}")
    for sample_path in sample_paths:
        with sample_path.open(newline="", encoding="utf-8") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != repeats:
                raise ValueError(f"invalid sample shard {sample_path}")
    return cached[0]


def make_batch(traces, batch, device):
    """Pad real KV lengths; support indices never address the padded tail."""
    if not traces or batch < 1:
        raise ValueError("at least one trace and a positive batch required")
    lengths = [t["k"].shape[2] for t in traces]
    max_len = max(lengths)
    dtype = traces[0]["q"].dtype
    q = torch.empty(batch, 32, 128, dtype=dtype, device=device)
    k = torch.zeros(batch, 8, max_len, 128, dtype=dtype, device=device)
    v = torch.zeros_like(k)
    masks = {name: torch.zeros(batch, 32, max_len, dtype=torch.bool, device=device)
             for name in ("gas", "twilight", "candidate", "dense")}
    group_lists, exception_lists = [], []
    for i in range(batch):
        t = traces[i % len(traces)]
        length = lengths[i % len(traces)]
        if t["q"].shape != (1, 32, 128) or t["k"].shape != (1, 8, length, 128) or t["v"].shape != t["k"].shape:
            raise ValueError("unexpected trace Q/K/V shape")
        if any(t[name].shape != (1, 32, length) for name in ("shared", "gas", "twilight", "candidate")):
            raise ValueError("unexpected trace mask shape")
        if any(t[name].dtype != dtype for name in ("q", "k", "v")):
            raise ValueError("dtype mismatch")
        shared = t["shared"][0].bool().view(8, 4, length)
        gas = t["gas"][0].bool()
        if not torch.equal(shared, shared[:, :1].expand_as(shared)) or (shared.reshape(32, length) & ~gas).any():
            raise ValueError("invalid group-shared support")
        exception = gas & ~shared.reshape(32, length)
        if (exception.sum(-1) > 16).any() or (gas.sum(-1) == 0).any() or (t["twilight"].sum(-1) == 0).any():
            raise ValueError("invalid exception count or empty support")
        group_lists.extend(torch.nonzero(shared[g, 0], as_tuple=True)[0].tolist() for g in range(8))
        exception_lists.extend(torch.nonzero(exception[h], as_tuple=True)[0].tolist() for h in range(32))
        q[i].copy_(t["q"][0]); k[i, :, :length].copy_(t["k"][0]); v[i, :, :length].copy_(t["v"][0])
        for name in ("gas", "twilight", "candidate"):
            masks[name][i, :, :length].copy_(t[name][0])
        masks["dense"][i, :, :length] = True
    gw = max(1, max(map(len, group_lists)))
    ew = max(1, max(map(len, exception_lists)))
    shared = padded_tensor(group_lists, (batch, 8), device, width=gw)
    ex = padded_tensor(exception_lists, (batch, 32), device, width=ew)
    return q, k, v, masks, shared, ex, lengths


def pack_dense_pages(source, lengths, page):
    batch, groups, max_len, dim = source.shape
    pages = (max_len + page - 1) // page
    output = torch.zeros((batch, pages, groups, page, dim), dtype=source.dtype, device=source.device)
    for b, length in enumerate(lengths):
        full, tail = divmod(length, page)
        if full:
            output[b, :full].copy_(source[b, :, :full * page].view(groups, full, page, dim).permute(1, 0, 2, 3))
        if tail:
            output[b, full, :, :tail].copy_(source[b, :, full * page:length])
    return output.view(batch * pages, groups, page, dim)


def dense_backends(q, k, v, lengths, include_fa2=True, include_fi=True):
    """Dense batched GQA with actual per-request lengths, setup outside timing."""
    import flashinfer
    from flash_attn import flash_attn_with_kvcache

    batch, heads, dim = q.shape
    groups, max_len = k.shape[1:3]
    lens = [lengths[i % len(lengths)] for i in range(batch)]
    seqlens = torch.tensor(lens, device=q.device, dtype=torch.int32)
    qcache = q.unsqueeze(1).contiguous()
    kcache, vcache = ([x.permute(0, 2, 1, 3).contiguous() for x in (k, v)]
                      if include_fa2 else (None, None))
    def fa2():
        return flash_attn_with_kvcache(qcache, kcache, vcache, cache_seqlens=seqlens,
                                       causal=False, softmax_scale=1 / math.sqrt(dim)).squeeze(1)
    if not include_fi:
        return fa2 if include_fa2 else None, None
    page = 16
    pages = (max_len + page - 1) // page
    cache = (pack_dense_pages(k, lens, page), pack_dense_pages(v, lens, page))
    n_pages = [(length + page - 1) // page for length in lens]
    indptr = [0]
    ids = []
    for i, n in enumerate(n_pages):
        ids.extend(range(i * pages, i * pages + n))
        indptr.append(len(ids))
    indptr = torch.tensor(indptr, dtype=torch.int32, device=q.device)
    ids = torch.tensor(ids, dtype=torch.int32, device=q.device)
    last = torch.tensor([length - (n - 1) * page for length, n in zip(lens, n_pages)], dtype=torch.int32, device=q.device)
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="HND")
    wrapper.plan(indptr, ids, last, heads, groups, dim, page, pos_encoding_mode="NONE",
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return fa2 if include_fa2 else None, lambda: wrapper.run(q, cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 15, 31])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--dtype", choices=("source", "float16", "bfloat16"), default="source")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    parser.add_argument("--segment-sweep", action="store_true",
                        help="also time inline-prefix kernels with 4, 16 and 32 segments")
    parser.add_argument("--replay-policy", action="store_true",
                        help="time an experimental padded-width based dispatch")
    parser.add_argument("--tree-merge", action="store_true",
                        help="also time experimental tree-merged inline kernels")
    parser.add_argument("--async-sweep", action="store_true",
                        help="time experimental double-buffered inline kernels")
    parser.add_argument("--groupsplit-sweep", action="store_true",
                        help="also time 16/32-part group splits with 64/128-token targets")
    parser.add_argument("--compact-sweep", action="store_true",
                        help="time compact group work queues at 2x and 4x the SM count")
    parser.add_argument("--head-split-sweep", action="store_true",
                        help="compare per-head 4/8-way split, including prefix counting and merge")
    parser.add_argument("--group-warp-sweep", action="store_true",
                        help="compare warp-cooperative group split with 4/8 segments")
    parser.add_argument("--group-warp-partitions", type=int, nargs="+", choices=(4, 16, 32), default=[],
                        help="also compare 4-segment warp kernels with these partition caps")
    parser.add_argument("--group-warp-tiled", action="store_true",
                        help="compare tiled shared-token softmax at 8 and 16 partition caps")
    parser.add_argument("--sparse-only", action="store_true",
                        help="skip both dense backends and their separate KV layout copies")
    parser.add_argument("--methods", nargs="+", choices=("fa2_dense", "fi_dense", "fi_quest_adapter",
                        "fi_twilight_adapter", "fi_gas_adapter", "inline8", "twopass_inline8",
                        "prefix_groupsplit128"), help="measure only these base methods")
    parser.add_argument("--reference-batch-chunk", type=int, default=8,
                        help="limit FP32 reference KV conversion to this many requests")
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.batch_sizes + [args.repeats, args.rounds, args.graph_iterations, args.reference_batch_chunk]) < 1 or args.warmups < 0:
        parser.error("CUDA and positive measurement settings required")
    if len(args.group_warp_partitions) != len(set(args.group_warp_partitions)):
        parser.error("group-warp partition caps must be distinct")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests):
        raise ValueError("trace directories must use distinct Qasper samples")
    if any(m["config"] != manifests[0]["config"] or m["dtype"] != manifests[0]["dtype"] or
           m["model"] != manifests[0]["model"] for m in manifests):
        raise ValueError("trace model configuration/dtype mismatch")
    methods = ("fa2_dense", "fi_dense", "fi_quest_adapter", "fi_twilight_adapter", "fi_gas_adapter", "inline8", "twopass_inline8", "prefix_groupsplit128")
    if args.sparse_only:
        methods = tuple(m for m in methods if m not in ("fa2_dense", "fi_dense"))
    if args.segment_sweep:
        methods += ("inline4", "inline16", "inline32")
    if args.replay_policy:
        methods += ("width_dispatch",)
    if args.tree_merge:
        methods += ("tree8", "tree16")
    if args.async_sweep:
        methods += ("async4", "async8", "async16")
    if args.groupsplit_sweep:
        methods += tuple(f"groupsplit_p{parts}_c{chunk}" for parts in (16, 32) for chunk in (64, 128))
    if args.compact_sweep:
        methods += ("compact_2sm", "compact_4sm")
    if args.head_split_sweep:
        methods += ("head_split4", "head_split8")
    if args.group_warp_sweep:
        methods += ("group_warp4", "group_warp8")
    methods += tuple(f"group_warp4_p{parts}" for parts in args.group_warp_partitions)
    if args.group_warp_tiled:
        methods += ("group_warp_tiled_p8", "group_warp_tiled_p16")
    if args.methods:
        if len(set(args.methods)) != len(args.methods) or not set(args.methods).issubset(methods):
            parser.error("methods must be distinct and compatible with sparse-only")
        methods = tuple(args.methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"trace_dirs": [str(p.resolve()) for p in args.trace_dirs],
              "manifest_sha256": [digest(p / "manifest.json") for p in args.trace_dirs],
              "sample_indices": [m["sample_index"] for m in manifests],
              "sample_hashes": [m["sample_sha256"] for m in manifests],
              "steps": args.steps, "layers": args.layers, "batch_sizes": args.batch_sizes,
              "methods": methods, "warmups": args.warmups, "repeats": args.repeats,
              "dtype": args.dtype, "source_dtype": manifests[0]["dtype"],
              "sparse_only": args.sparse_only, "reference_batch_chunk": args.reference_batch_chunk,
              "rounds": args.rounds, "graph_iterations": args.graph_iterations,
              "sources": {str(p): digest(ROOT / p) for p in ("csrc/src/gas_counted_decode.cu", "csrc/src/gas_twopass_decode.cu", "csrc/src/gas_cooperative_split.cu")},
              "script_sha256": digest(Path(__file__)),
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
              "semantics": "distinct captured Qasper requests repeated round-robin; common layer/step; padded K/V; masks never select padding",
              "timing": "CUDA graph attention replay; selection, compaction and FI plan excluded; no end-to-end inference"}
    if args.head_split_sweep:
        config["sources"]["csrc/src/gas_counted_split.cu"] = digest(ROOT / "csrc/src/gas_counted_split.cu")
    if args.group_warp_sweep or args.group_warp_partitions or args.group_warp_tiled:
        config["sources"]["csrc/src/gas_group_warp_split.cu"] = digest(ROOT / "csrc/src/gas_group_warp_split.cu")
    config_path = args.output_dir / "config.json"
    config["methods"] = list(config["methods"])
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing output directory has a different protocol")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    old = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    new = load(name="gas_twopass_decode_guarded", sources=[str(ROOT / "csrc/src/gas_twopass_decode.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    groupsplit = load(name="gas_cooperative_split", sources=[str(ROOT / "csrc/src/gas_cooperative_split.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    headsplit = (load(name="gas_counted_split", sources=[str(ROOT / "csrc/src/gas_counted_split.cu")],
                      extra_cuda_cflags=["-O3"], verbose=False) if args.head_split_sweep else None)
    groupwarp = (load(name="gas_group_warp_split", sources=[str(ROOT / "csrc/src/gas_group_warp_split.cu")],
                      extra_cuda_cflags=["-O3"], verbose=False) if args.group_warp_sweep or args.group_warp_partitions or args.group_warp_tiled else None)
    rows = []
    for step in args.steps:
        for layer in args.layers:
            filename = f"step{step:03d}_layer{layer:02d}.pt"
            paths = [p / filename for p in args.trace_dirs]
            if not all(filename in m["files"] and p.is_file() for m, p in zip(manifests, paths)):
                raise ValueError(f"missing captured trace {filename}")
            traces = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
            if args.dtype != "source":
                for trace in traces:
                    for key in ("q", "k", "v"):
                        trace[key] = trace[key].to(getattr(torch, args.dtype))
            for batch in args.batch_sizes:
                q, k, v, masks, shared, ex, lengths = make_batch(traces, batch, "cuda")
                needed = {"dense" if method in ("fa2_dense", "fi_dense") else
                          "twilight" if method == "fi_twilight_adapter" else
                          "candidate" if method == "fi_quest_adapter" else "gas" for method in methods}
                expected = {name: chunked_reference(q, k, v, masks[name], args.reference_batch_chunk)
                            for name in needed}
                def prefix_groupsplit():
                    sc, ec = old.count_prefix(shared, ex)
                    return groupsplit.decode(q, k, v, shared, ex, sc, ec, 8, 8, 128)
                def groupsplit_variant(parts, chunk):
                    sc, ec = old.count_prefix(shared, ex)
                    return groupsplit.decode(q, k, v, shared, ex, sc, ec, 8, parts, chunk)
                def compact_variant(multiplier):
                    sc, ec = old.count_prefix(shared, ex)
                    return groupsplit.decode_compact(q, k, v, shared, ex, sc, ec, 8, 8, 128,
                                                     multiplier * config["sm_count"])
                def head_split_variant(parts):
                    sc, ec = old.count_prefix(shared, ex)
                    return headsplit.decode(q, k, v, shared, ex, sc, ec, 8, parts)
                def group_warp_variant(segments, parts=8, tiled=False):
                    sc, ec = old.count_prefix(shared, ex)
                    call = groupwarp.decode_tiled if tiled else groupwarp.decode
                    return call(q, k, v, shared, ex, sc, ec, segments, parts, 128)
                calls = {"inline8": lambda: old.decode_prefix(q, k, v, shared, ex, 8),
                         "twopass_inline8": lambda: new.decode_prefix(q, k, v, shared, ex, 8),
                         "prefix_groupsplit128": prefix_groupsplit}
                for method, mask in (("fi_gas_adapter", "gas"), ("fi_twilight_adapter", "twilight"),
                                     ("fi_quest_adapter", "candidate")):
                    if method in methods:
                        calls[method] = fi_exact(q, k, v, masks[mask])
                if "fa2_dense" in methods or "fi_dense" in methods:
                    fa2, fi_dense = dense_backends(q, k, v, lengths, "fa2_dense" in methods, "fi_dense" in methods)
                    if fa2 is not None:
                        calls["fa2_dense"] = fa2
                    if fi_dense is not None:
                        calls["fi_dense"] = fi_dense
                if args.segment_sweep:
                    for segments in (4, 16, 32):
                        calls[f"inline{segments}"] = lambda segments=segments: old.decode_prefix(q, k, v, shared, ex, segments)
                if args.replay_policy:
                    selected = choose_replay_policy(batch, shared.size(2))
                    if selected == "inline16":
                        calls["width_dispatch"] = lambda: old.decode_prefix(q, k, v, shared, ex, 16)
                    else:
                        calls["width_dispatch"] = calls[selected]
                if args.tree_merge:
                    calls["tree8"] = lambda: old.decode_prefix_tree(q, k, v, shared, ex, 8)
                    calls["tree16"] = lambda: old.decode_prefix_tree(q, k, v, shared, ex, 16)
                if args.async_sweep:
                    for segments in (4, 8, 16):
                        calls[f"async{segments}"] = lambda segments=segments: old.decode_prefix_async(q, k, v, shared, ex, segments)
                if args.groupsplit_sweep:
                    for parts in (16, 32):
                        for chunk in (64, 128):
                            calls[f"groupsplit_p{parts}_c{chunk}"] = lambda parts=parts, chunk=chunk: groupsplit_variant(parts, chunk)
                if args.compact_sweep:
                    calls["compact_2sm"] = lambda: compact_variant(2)
                    calls["compact_4sm"] = lambda: compact_variant(4)
                if args.head_split_sweep:
                    calls["head_split4"] = lambda: head_split_variant(4)
                    calls["head_split8"] = lambda: head_split_variant(8)
                if args.group_warp_sweep:
                    calls["group_warp4"] = lambda: group_warp_variant(4)
                    calls["group_warp8"] = lambda: group_warp_variant(8)
                for parts in args.group_warp_partitions:
                    calls[f"group_warp4_p{parts}"] = lambda parts=parts: group_warp_variant(4, parts)
                if args.group_warp_tiled:
                    calls["group_warp_tiled_p8"] = lambda: group_warp_variant(4, 8, True)
                    calls["group_warp_tiled_p16"] = lambda: group_warp_variant(4, 16, True)
                order = list(methods)
                random.Random(f"{step}:{layer}:{batch}").shuffle(order)
                for method in order:
                    shard = args.output_dir / "shards" / f"s{step}_l{layer}_b{batch}_{method}.csv"
                    sample_paths = [args.output_dir / "samples" / f"s{step}_l{layer}_b{batch}_{method}_{round_id}.csv"
                                    for round_id in range(args.rounds)]
                    cached = load_cached(shard, sample_paths, paths, args.repeats)
                    if cached is not None:
                        rows.append(cached)
                        print(f"SKIP {filename} B{batch} {method}", flush=True)
                        continue
                    target = expected["dense" if method in ("fa2_dense", "fi_dense") else
                                      "twilight" if method == "fi_twilight_adapter" else
                                      "candidate" if method == "fi_quest_adapter" else "gas"]
                    out = calls[method]()
                    check = error_metrics(out, target, 0.02, 0.02)
                    if not check["pass"]:
                        raise ValueError(f"{filename} B{batch} {method}: {check}")
                    for _ in range(args.warmups):
                        calls[method]()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.graph_iterations):
                            out = calls[method]()
                    out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                    check = error_metrics(out, target, 0.02, 0.02)
                    if not check["pass"]:
                        raise ValueError(f"graph {filename} B{batch} {method}: {check}")
                    samples = []
                    for round_id in range(args.rounds):
                        times = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                        samples.extend(times)
                        write_rows(args.output_dir / "samples" / f"s{step}_l{layer}_b{batch}_{method}_{round_id}.csv",
                                   [{"latency_us": x} for x in times])
                    row = {"step": step, "layer": layer, "batch": batch, "method": method,
                           "num_distinct_requests": len(traces), "lengths": json.dumps(lengths),
                           "trace_sha256": json.dumps([digest(p) for p in paths]),
                           "gas_per_head_tokens_mean": masks["gas"].sum(-1).float().mean().item(),
                           "gas_group_union_mean": masks["gas"].view(batch, 8, 4, -1).any(2).sum(-1).float().mean().item(),
                           "twi_per_head_tokens_mean": masks["twilight"].sum(-1).float().mean().item(),
                           "twi_group_union_mean": masks["twilight"].view(batch, 8, 4, -1).any(2).sum(-1).float().mean().item(),
                           "max_abs_error": check["max_abs_error"], "samples": len(samples), "median_us": statistics.median(samples)}
                    rows.append(row)
                    write_rows(shard, [row])
                    print(f"{filename} B{batch} {method}: {row['median_us']:.2f} us", flush=True)
                    del graph, out
                del calls, q, k, v, masks, expected, shared, ex
                if "fa2_dense" in methods or "fi_dense" in methods:
                    fa2, fi_dense = None, None
                torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
