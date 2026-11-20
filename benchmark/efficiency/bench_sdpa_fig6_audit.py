#!/usr/bin/env python3
"""Compare dense SDPA backends and sparse FI on one captured Qasper request.

Each invocation measures one method. KV layout conversion, explicit GQA
expansion, mask/index construction and FI planning are outside timing.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_trace import digest, fi_exact
from bench_gas_mixed_trace import chunked_reference, dense_backends, make_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("sdpa_flash", "sdpa_efficient_expanded", "fa2_kvcache", "gas_fi"), required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--graph-iterations", type=int, default=8)
    p.add_argument("--timing", choices=("graph", "direct"), default="graph")
    args = p.parse_args()
    if min(args.batch, args.repeats, args.rounds, args.graph_iterations) < 1 or args.warmups < 0:
        p.error("positive batch/repeats/rounds/graph-iterations required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    trace_path = args.trace_dir / f"step001_layer{args.layer:02d}.pt"
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    for key in ("q", "k", "v"):
        trace[key] = trace[key].to(torch.float16)
    with torch.inference_mode():
        q, k, v, masks, shared, ex, lengths = make_batch([trace], args.batch, "cuda")
        del shared, ex
        method = args.method
        target_mask = masks["gas"] if method == "gas_fi" else masks["dense"]
        expected = chunked_reference(q, k, v, target_mask, 1)
        del masks["candidate"], masks["twilight"]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        if method == "sdpa_flash":
            def call():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q.unsqueeze(2), k, v, dropout_p=0.0, enable_gqa=True).squeeze(2)
        elif method == "sdpa_efficient_expanded":
            # PyTorch 2.5 efficient backend requires equal Q/KV head counts.
            expanded_k, expanded_v = (x.repeat_interleave(4, dim=1) for x in (k, v))
            def call():
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(q.unsqueeze(2), expanded_k, expanded_v, dropout_p=0.0).squeeze(2)
        elif method == "fa2_kvcache":
            call, _ = dense_backends(q, k, v, lengths, include_fa2=True, include_fi=False)
        else:
            call = fi_exact(q, k, v, target_mask)
        torch.cuda.synchronize()
        setup_bytes = torch.cuda.memory_allocated() - before
        result = call()
        check = error_metrics(result, expected, 0.02, 0.02)
        if not check["pass"]:
            raise ValueError(f"reference check failed: {check}")
        for _ in range(args.warmups):
            call()
        torch.cuda.synchronize()
        graph_check = None
        if args.timing == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_iterations):
                    result = call()
            result.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            graph_check = error_metrics(result, expected, 0.02, 0.02)
            if not graph_check["pass"]:
                raise ValueError(f"graph check failed: {graph_check}")
            rounds = [[x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                      for _ in range(args.rounds)]
        else:
            if torch.cuda.current_stream().cuda_stream != 0:
                raise RuntimeError("direct B1 protocol requires the default CUDA stream")
            rounds = [cuda_event_samples(call, q.device, args.warmups, args.repeats) for _ in range(args.rounds)]
        times = [x for row in rounds for x in row]
        result_row = {
            "method": method, "sample_index": manifest["sample_index"], "sample_sha256": manifest["sample_sha256"],
            "trace_sha256": digest(trace_path), "manifest_sha256": digest(manifest_path), "trace_dir": str(args.trace_dir.resolve()),
            "layer": args.layer, "batch": args.batch, "kv_length": lengths[0], "dtype": str(q.dtype),
            "q_heads": q.shape[1], "kv_heads": k.shape[1], "head_dim": q.shape[-1],
            "support": "GAS per-head mask" if method == "gas_fi" else "dense",
            "gas_per_head_tokens_mean": float(target_mask.sum(-1).float().mean()) if method == "gas_fi" else None,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(),
            "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
            "timing": args.timing, "graph_iterations": args.graph_iterations if args.timing == "graph" else None,
            "median_us": statistics.median(times), "samples_us": rounds,
            "check": check, "graph_check": graph_check, "setup_allocated_bytes": setup_bytes,
            "peak_process_allocated_bytes": torch.cuda.max_memory_allocated(),
            "setup_excluded": "KV layouts, explicit head expansion, selection, masks, FI plan, data transfer",
            "note": "single original trace repeated across batch; SDPA efficient uses materialized 32-head K/V" if method == "sdpa_efficient_expanded" else "single original trace repeated across batch",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result_row, indent=2) + "\n")
    print(f"{method}: {result_row['median_us']:.3f} us, length={lengths[0]}, B={args.batch}", flush=True)


if __name__ == "__main__":
    main()
