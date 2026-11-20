#!/usr/bin/env python3
"""Memory-bounded real-trace attention replay with fixed Quest budget 8192.

All setup, including CPU-to-GPU transfers and layout conversion, is excluded.
Each method retains a complete B-request GPU KV cache during timed replay.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import less_mask, load_extension
from bench_e6_qasper_trace import digest, fi_exact, write_rows
from bench_gas_mixed_trace import chunked_reference, make_batch
from bench_less_official_headpacked import official_grouped_setup

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fa2_dense", "fi_dense", "quest_fi", "twilight_fi", "gas_fi",
           "gas_inline8", "less_fi", "less_patched")


def dense_call(method, q, k, v, lengths):
    batch, heads, dim = q.shape
    groups, maxlen = k.shape[1:3]
    lens = [lengths[i % len(lengths)] for i in range(batch)]
    if method == "fa2_dense":
        from flash_attn import flash_attn_with_kvcache
        cache_k = k.permute(0, 2, 1, 3).contiguous()
        cache_v = v.permute(0, 2, 1, 3).contiguous()
        seq = torch.tensor(lens, device=q.device, dtype=torch.int32)
        query = q.unsqueeze(1).contiguous()
        return lambda: flash_attn_with_kvcache(query, cache_k, cache_v, cache_seqlens=seq,
                                                causal=False, softmax_scale=1 / math.sqrt(dim)).squeeze(1)
    import flashinfer
    page = 16
    npages = (maxlen + page - 1) // page
    caches = []
    for source in (k, v):
        target = torch.zeros((batch, npages, groups, page, dim), device=q.device, dtype=q.dtype)
        for b, length in enumerate(lens):
            full = length // page
            if full:
                target[b, :full].copy_(source[b, :, :full * page].view(groups, full, page, dim).permute(1, 0, 2, 3))
            if length % page:
                target[b, full, :, :length % page].copy_(source[b, :, full * page:length])
        caches.append(target.view(batch * npages, groups, page, dim))
    page_counts = [(length + page - 1) // page for length in lens]
    indptr, ids = [0], []
    for b, count in enumerate(page_counts):
        ids.extend(range(b * npages, b * npages + count))
        indptr.append(len(ids))
    indptr = torch.tensor(indptr, device=q.device, dtype=torch.int32)
    ids = torch.tensor(ids, device=q.device, dtype=torch.int32)
    last = torch.tensor([length - (count - 1) * page for length, count in zip(lens, page_counts)],
                        device=q.device, dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="HND")
    wrapper.plan(indptr, ids, last, heads, groups, dim, page, pos_encoding_mode="NONE",
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q, tuple(caches))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--nominal-length", type=int, choices=(16384, 32768), required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--less-module", type=Path, default=Path("/data/tmp/torch_ext_e6_stream_adapters/e6_less_nopart_fixed/e6_less_nopart_fixed.so"))
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--graph-iterations", type=int, default=16)
    p.add_argument("--reference-chunk", type=int, default=4)
    args = p.parse_args()
    if not torch.cuda.is_available() or min(args.batch, args.repeats, args.rounds, args.graph_iterations, args.reference_chunk) < 1:
        p.error("CUDA and positive batch/timing parameters required")
    manifests = [json.loads((d / "manifest.json").read_text()) for d in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests):
        raise ValueError("trace requests must be distinct")
    for m in manifests:
        c = m["config"]
        if (m["model"] != manifests[0]["model"] or c != manifests[0]["config"] or
            c["selector"].get("type") != "quest" or c["selector"].get("token_budget") != 8192 or
            c["selector"].get("chunk_size") != 16 or c["weight_estimator"].get("type") != "min_max_quant" or
            c["weight_estimator"].get("quant_bit") != 4 or c["weight_pruner"].get("type") != "top_p" or
            c["weight_pruner"].get("threshold") != 0.85 or not c["gqa_refinement"].get("enabled") or
            c["gqa_refinement"].get("mode") != "group_max" or c["gqa_refinement"].get("threshold") != 0.8 or
            c["gqa_refinement"].get("exception_budget") != 16):
            raise ValueError("expected identical Quest 8192/Twilight 0.85/GAS 0.8+16 configurations")
    target_name, search_name = "step001_layer15.pt", "step001_layer13.pt"
    paths = [d / target_name for d in args.trace_dirs]
    search_paths = [d / search_name for d in args.trace_dirs]
    traces = [torch.load(f, map_location="cpu", weights_only=True) for f in paths]
    lengths = [t["k"].shape[2] for t in traces]
    if any(length <= args.nominal_length * 0.8 or length > args.nominal_length for length in lengths):
        raise ValueError(f"trace lengths {lengths} do not match nominal length {args.nominal_length}")
    for trace in traces:
        for key in ("q", "k", "v"):
            trace[key] = trace[key].to(torch.float16)
    unions = [t["twilight"].view(1, 8, 4, -1).any(2).sum(-1).float().mean().item() for t in traces]
    candidate_ranges = [[int(t["candidate"].sum(-1).min()), int(t["candidate"].sum(-1).max())] for t in traces]
    gas_unions = [t["gas"].view(1, 8, 4, -1).any(2).sum(-1).float().mean().item() for t in traces]
    per_head = {name: [t[name].sum(-1).float().mean().item() for t in traces]
                for name in ("candidate", "twilight", "gas")}
    budget = round(statistics.fmean(unions))
    if budget < 2 or budget >= min(lengths):
        raise ValueError("matched LessIsMore budget is outside context")
    less_ids = []
    for path, length in zip(search_paths, lengths):
        search = torch.load(path, map_location="cpu", weights_only=True)
        if search["k"].shape[2] != length:
            raise ValueError("search and target lengths differ")
        search_gpu = {key: search[key].to("cuda") for key in ("q", "k")}
        mask = less_mask(search_gpu, length, budget)
        less_ids.append(torch.nonzero(mask[0, 0], as_tuple=True)[0].cpu())
        del search_gpu, mask, search
    torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"nominal_length": args.nominal_length, "batch": args.batch, "lengths": lengths,
            "layer": 15, "step": 1, "search_layer": 13, "sample_indices": [m["sample_index"] for m in manifests],
            "trace_sha256": [digest(p) for p in paths], "search_sha256": [digest(p) for p in search_paths],
            "manifest_sha256": [digest(d / "manifest.json") for d in args.trace_dirs],
            "config": manifests[0]["config"], "less_k": budget, "twilight_union_by_request": unions,
            "less_support_sha256": [hashlib.sha256(indices.contiguous().numpy().tobytes()).hexdigest() for indices in less_ids],
            "gas_union_by_request": gas_unions, "candidate_minmax_by_request": candidate_ranges,
            "per_head_mean_by_request": per_head,
            "dtype": "float16", "source_dtype": manifests[0]["dtype"], "device": torch.cuda.get_device_name(),
            "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
            "graph_iterations": args.graph_iterations, "less_module_sha256": digest(args.less_module),
            "script_sha256": digest(Path(__file__)), "kernel_source_sha256": digest(ROOT / "csrc/src/gas_counted_decode.cu"),
            "timing": "resident full batch KV, CUDA graph/event attention; selection/layout/host transfer/planning excluded",
            "caveats": "repeated GAS-trajectory requests; per-head FI adapters; Less B1 patched/head-packed, not native batching; no quality or end-to-end claim"}
    meta_path = args.output_dir / f"config_{args.nominal_length}_{args.batch}.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("existing output protocol differs")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    q, k, v, masks, shared, ex, _ = make_batch(traces, args.batch, "cuda")
    ids = torch.stack([less_ids[i % len(less_ids)] for i in range(args.batch)]).to("cuda")
    less = torch.zeros_like(masks["gas"])
    less.scatter_(2, ids[:, None, :].expand(args.batch, 32, budget), True)
    masks["less"] = less
    module = None
    less_module = None
    for method in args.methods:
        shard = args.output_dir / "shards" / f"l{args.nominal_length}_b{args.batch}_{method}.csv"
        if shard.exists():
            with shard.open(newline="") as stream:
                cached = list(csv.DictReader(stream))
            if (len(cached) != 1 or cached[0]["method"] != method or
                    int(cached[0]["samples"]) != args.rounds * args.repeats):
                raise ValueError(f"incomplete or mismatched shard: {shard}")
            for r in range(args.rounds):
                sample_path = args.output_dir / "samples" / f"l{args.nominal_length}_b{args.batch}_{method}_{r}.csv"
                with sample_path.open(newline="") as stream:
                    if len(list(csv.DictReader(stream))) != args.repeats:
                        raise ValueError(f"incomplete samples: {sample_path}")
            print(f"SKIP {method} {shard}", flush=True)
            continue
        wrappers = []
        try:
            mask_name = ("dense" if method.endswith("dense") else "candidate" if method == "quest_fi" else
                         "twilight" if method == "twilight_fi" else "less" if method.startswith("less_") else "gas")
            expected = chunked_reference(q, k, v, masks[mask_name], args.reference_chunk)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if method.endswith("dense"):
                call = dense_call(method, q, k, v, lengths)
            elif method in ("quest_fi", "twilight_fi", "gas_fi", "less_fi"):
                call = fi_exact(q, k, v, masks[mask_name])
            elif method == "gas_inline8":
                if module is None:
                    module = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                                  extra_cuda_cflags=["-O3"], verbose=False)
                call = lambda: module.decode_prefix(q, k, v, shared, ex, 8)
            else:
                if less_module is None:
                    less_module = load_extension(args.less_module, "e6_less_nopart_fixed")
                row_lengths = [lengths[i % len(lengths)] for i in range(args.batch)]
                call, wrappers = official_grouped_setup(less_module, q, k, v, ids, row_lengths, 128)
            output = call()
            check = error_metrics(output, expected, 0.02, 0.02)
            if not check["pass"]:
                raise ValueError(f"{method} correctness: {check}")
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
                raise ValueError(f"{method} graph correctness: {check}")
            times = []
            for r in range(args.rounds):
                samples = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                times.extend(samples)
                write_rows(args.output_dir / "samples" / f"l{args.nominal_length}_b{args.batch}_{method}_{r}.csv",
                           [{"latency_us": x} for x in samples])
            row = {"nominal_length": args.nominal_length, "batch": args.batch, "method": method,
                   "less_k": budget, "samples": len(times), "median_us": statistics.median(times),
                   "max_abs_error": check["max_abs_error"], "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "less_wrapper_calls": len(wrappers), "timing_boundary": "gpu_resident_attention_only"}
            write_rows(shard, [row])
            print(f"DONE {method} L{args.nominal_length} B{args.batch} K{budget}: {row['median_us']:.2f} us", flush=True)
            del graph, output, expected, call
        finally:
            for wrapper in wrappers:
                wrapper.end_forward()
            gc.collect()
            torch.cuda.empty_cache()
    print(f"COMPLETE L{args.nominal_length} B{args.batch}", flush=True)


if __name__ == "__main__":
    main()
