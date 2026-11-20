#!/usr/bin/env python3
"""B1 diagnostic of upstream Quest/Less attention on a captured GAS trace."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import (UnsupportedSupport, less_mask, less_setup, load_extension,
                                        quest_setup, quest_setup_split)
from bench_e6_qasper_trace import digest, reference

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("quest_official_b1", "less_official_b1"), required=True)
    p.add_argument("--less-k", type=int, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--rounds", type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.less_k < 2 or min(args.repeats, args.rounds) < 1 or args.warmups < 0:
        p.error("positive timing settings and less-k >= 2 required")
    if args.layer in (2, 13) or not 3 <= args.layer <= 31:
        p.error("choose a non-search sparse layer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trace_path = args.trace_dir / f"step001_layer{args.layer:02d}.pt"
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    name = "quest" if args.method == "quest_official_b1" else "less"
    artifacts = ROOT / "benchmark/results_efficiency/e6_qasper_official_artifacts"
    extension_path = artifacts / f"e6_{name}_official_decode.so"
    object_path = artifacts / f"{name}_approx_attn.cu.o"
    module = load_extension(extension_path, f"e6_{name}_official_decode")
    q, k, v = [trace[key].to(device="cuda", dtype=torch.float16).contiguous() for key in ("q", "k", "v")]
    length = k.shape[2]
    search_hash = None
    if name == "quest":
        mask = trace["candidate"].to("cuda")
    else:
        search_path = args.trace_dir / f"step001_layer{2 if args.layer < 13 else 13:02d}.pt"
        search = torch.load(search_path, map_location="cpu", weights_only=True)
        search = {key: search[key].to(device="cuda", dtype=torch.float16) for key in ("q", "k")}
        mask = less_mask(search, length, args.less_k)
        search_hash = digest(search_path)
        del search
    expected = reference(q, k, v, mask)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    if name == "quest":
        try:
            call, wrapper = quest_setup(module, q, k, v, mask[0])
            call_groups = 1
        except UnsupportedSupport:
            call, wrapper, call_groups = quest_setup_split(module, q, k, v, mask[0])
    else:
        call, wrapper = less_setup(module, q, k, v, mask)
        call_groups = 1
    try:
        torch.cuda.synchronize()
        setup_bytes = torch.cuda.memory_allocated() - before
        result = call()
        check = error_metrics(result, expected, 0.02, 0.02)
        if not check["pass"]:
            raise ValueError(f"reference check failed: {check}")
        for _ in range(args.warmups):
            call()
        torch.cuda.synchronize()
        if torch.cuda.current_stream().cuda_stream != 0:
            raise RuntimeError("upstream B1 kernel requires direct timing on the default CUDA stream")
        rounds = [cuda_event_samples(call, q.device, args.warmups, args.repeats) for _ in range(args.rounds)]
        output = {"method": args.method, "batch": 1, "layer": args.layer, "kv_length": length,
                  "sample_index": manifest["sample_index"], "sample_sha256": manifest["sample_sha256"],
                  "trace_sha256": digest(trace_path), "manifest_sha256": digest(manifest_path),
                  "search_trace_sha256": search_hash, "extension_sha256": digest(extension_path),
                  "official_object_sha256": digest(object_path), "dtype": str(q.dtype), "torch": torch.__version__,
                  "device": torch.cuda.get_device_name(), "less_k": args.less_k if name == "less" else None,
                  "per_head_tokens_mean": float(mask.sum(-1).float().mean()),
                  "group_union_tokens_mean": float(mask.reshape(1, 8, 4, length).any(2).sum(-1).float().mean()),
                  "wrapper_calls": call_groups, "median_us": statistics.median(x for r in rounds for x in r),
                  "samples_us": rounds, "warmups": args.warmups, "repeats": args.repeats,
                  "rounds": args.rounds, "timing": "direct CUDA events on default stream",
                  "check": check, "setup_allocated_bytes": setup_bytes,
                  "peak_process_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "note": ("original B1 Quest CUDA with wrapper bridge, replicated MHA KV; partial pages may use grouped calls and timed output scatter; no native GQA/batching"
                           if name == "quest" else "original B1 Less CUDA with wrapper bridge; reconstructed shared selection from GAS search-layer state"),
                  "timing_excludes": "selection, KV replication/packing, data transfer and plan"}
    finally:
        wrapper.end_forward()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{args.method}: {output['median_us']:.3f} us, layer={args.layer}, length={length}", flush=True)


if __name__ == "__main__":
    main()
