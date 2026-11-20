#!/usr/bin/env python3
"""Exact Quest chunk replay with a page-16 FI adapter, not the official kernel."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_trace import digest, write_rows
from bench_gas_mixed_trace import chunked_reference, make_batch


def page16_call(q, k, v, mask, lengths):
    import flashinfer
    batch, heads, dim = q.shape
    maxlen = k.shape[2]
    page = 16
    npages = (maxlen + page - 1) // page
    padding = npages * page - maxlen
    page_counts = F.pad(mask.reshape(batch * heads, maxlen), (0, padding)).view(batch * heads, npages, page).sum(-1)
    lens = torch.tensor([lengths[b % len(lengths)] for b in range(batch)], device=q.device).repeat_interleave(heads)
    valid = (lens[:, None] - torch.arange(npages, device=q.device)[None, :] * page).clamp(0, page)
    if ((page_counts != 0) & (page_counts != valid)).any():
        raise ValueError("Quest candidate support contains a partially selected non-terminal chunk")
    rows, selected = (page_counts > 0).nonzero(as_tuple=True)
    counts = (page_counts > 0).sum(-1, dtype=torch.int32)
    if (counts == 0).any():
        raise ValueError("empty Quest support")
    indptr = F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    indices = ((rows // 4) * npages + selected).to(torch.int32)
    last_selected = selected[indptr[1:].long() - 1]
    last = (lens - last_selected * page).clamp(1, page).to(torch.int32)
    cache = tuple(F.pad(x, (0, 0, 0, padding)).view(-1, page, 1, dim) for x in (k, v))
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(indptr, indices, last, 1, 1, dim, page, pos_encoding_mode="NONE",
                 q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
    return lambda: wrapper.run(q.view(-1, 1, dim), cache).view_as(q)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--nominal-length", type=int, choices=(16384, 32768), required=True)
    p.add_argument("--reference-chunk", type=int, default=2)
    args = p.parse_args()
    meta = json.loads((args.output_dir / f"config_{args.nominal_length}_{args.batch}.json").read_text())
    paths = [d / "step001_layer15.pt" for d in args.trace_dirs]
    if [digest(f) for f in paths] != meta["trace_sha256"]:
        raise ValueError("trace hashes differ from the existing paired benchmark")
    method = "quest_fi_page16"
    shard = args.output_dir / "shards" / f"l{args.nominal_length}_b{args.batch}_{method}.csv"
    if shard.exists():
        raise FileExistsError(f"existing result must not be overwritten: {shard}")
    traces = [torch.load(f, map_location="cpu", weights_only=True) for f in paths]
    for trace in traces:
        for key in ("q", "k", "v"):
            trace[key] = trace[key].to(torch.float16)
    q, k, v, masks, _, _, lengths = make_batch(traces, args.batch, "cuda")
    expected = chunked_reference(q, k, v, masks["candidate"], args.reference_chunk)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    call = page16_call(q, k, v, masks["candidate"], lengths)
    output = call()
    check = error_metrics(output, expected, 0.02, 0.02)
    if not check["pass"]:
        raise ValueError(f"page16 correctness: {check}")
    for _ in range(meta["warmups"]):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(meta["graph_iterations"]):
            output = call()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check = error_metrics(output, expected, 0.02, 0.02)
    if not check["pass"]:
        raise ValueError(f"page16 graph correctness: {check}")
    times = []
    for r in range(meta["rounds"]):
        measured = [x / meta["graph_iterations"] for x in cuda_event_samples(graph.replay, q.device, meta["warmups"], meta["repeats"])]
        times.extend(measured)
        write_rows(args.output_dir / "samples" / f"l{args.nominal_length}_b{args.batch}_{method}_{r}.csv",
                   [{"latency_us": x} for x in measured])
    row = {"nominal_length": args.nominal_length, "batch": args.batch, "method": method,
           "less_k": meta["less_k"], "samples": len(times), "median_us": statistics.median(times),
           "max_abs_error": check["max_abs_error"], "max_allocated_bytes": torch.cuda.max_memory_allocated(),
           "less_wrapper_calls": 0, "timing_boundary": "gpu_resident_attention_only"}
    provenance = {"method": method, "page_size": 16, "script_sha256": digest(Path(__file__)),
                  "trace_sha256": meta["trace_sha256"], "main_config": f"config_{args.nominal_length}_{args.batch}.json",
                  "boundary": "exact candidate support, GPU-resident KV; packing/plan/selection excluded; not official Quest"}
    (args.output_dir / f"page16_{args.nominal_length}_{args.batch}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_rows(shard, [row])
    print(f"DONE {method} L{args.nominal_length} B{args.batch}: {row['median_us']:.2f} us", flush=True)


if __name__ == "__main__":
    main()
