#!/usr/bin/env python3
"""Replay LessIsMore sparse layers with one K and search-layer mask reuse.

Uses the isolated page-fixed B1 kernel with batch mapped to heads. Selection,
repacking and planning are excluded. Search-layer Q/K come from GAS traces.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import less_mask, load_extension
from bench_e6_qasper_trace import digest, fi_exact, write_rows
from bench_gas_mixed_trace import chunked_reference, load_cached, make_batch
from bench_less_official_headpacked import official_grouped_setup

ROOT = Path(__file__).resolve().parents[2]
SPARSE_LAYERS = tuple(x for x in range(3, 32) if x != 13)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--module", type=Path, default=Path("/data/tmp/torch_ext_e6_stream_adapters/e6_less_nopart_fixed/e6_less_nopart_fixed.so"))
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--budget-calibration", type=Path)
    p.add_argument("--steps", type=int, nargs="+", default=[1])
    p.add_argument("--layers", type=int, nargs="+", default=list(SPARSE_LAYERS))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    p.add_argument("--pack-size", type=int, default=128)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--graph-iterations", type=int, default=16)
    p.add_argument("--methods", nargs="+", choices=("less_patched_headpacked", "less_fi_adapter"),
                   default=["less_patched_headpacked", "less_fi_adapter"])
    p.add_argument("--reference-batch-chunk", type=int, default=8)
    args = p.parse_args()
    if not torch.cuda.is_available() or min(args.batch_sizes + [args.budget, args.pack_size, args.repeats, args.rounds, args.graph_iterations, args.reference_batch_chunk]) < 1 or args.budget < 2 or args.warmups < 0:
        p.error("CUDA, budget >= 2 and positive benchmark settings required")
    if len(set(args.layers)) != len(args.layers) or any(x not in SPARSE_LAYERS for x in args.layers):
        p.error("choose distinct sparse layers in 3..31 excluding search layer 13")
    manifests = [json.loads((d / "manifest.json").read_text()) for d in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests) or any(m["model"] != manifests[0]["model"] or m["config"] != manifests[0]["config"] for m in manifests):
        raise ValueError("distinct requests with matching model/config required")
    manifest_hashes = [digest(d / "manifest.json") for d in args.trace_dirs]
    if args.budget_calibration is not None:
        calibration = json.loads(args.budget_calibration.read_text())
        if (calibration["K"] != args.budget or calibration["trace_manifests_sha256"] != manifest_hashes or
                args.steps != [calibration["step"]]):
            raise ValueError("budget calibration does not match traces, step or requested K")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"samples": [m["sample_index"] for m in manifests], "trace_dirs": [str(d.resolve()) for d in args.trace_dirs],
              "manifest_sha256": manifest_hashes, "model": manifests[0]["model"],
              "steps": args.steps, "layers": args.layers, "batch_sizes": args.batch_sizes, "dtype": "float16",
              "budget": args.budget, "search_layers": [2, 13], "pack_size": args.pack_size,
              "budget_calibration_sha256": digest(args.budget_calibration) if args.budget_calibration else None,
              "methods": args.methods, "module_sha256": digest(args.module) if "less_patched_headpacked" in args.methods else None,
              "script_sha256": digest(Path(__file__)), "warmups": args.warmups, "repeats": args.repeats,
              "rounds": args.rounds, "graph_iterations": args.graph_iterations, "reference_batch_chunk": args.reference_batch_chunk, "device": torch.cuda.get_device_name(),
              "limitations": "patched B1 LessIsMore kernel with batch packed into heads; GAS-run search states; fixed K; search layers dense; selection, packing and plans excluded; not native batch or TPOT"}
    path = args.output_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("output directory has different protocol")
    path.write_text(json.dumps(config, indent=2) + "\n")
    module = load_extension(args.module, "e6_less_nopart_fixed") if "less_patched_headpacked" in args.methods else None
    rows = []
    for step in args.steps:
        selected = {}
        for search_layer in (2, 13):
            filename = f"step{step:03d}_layer{search_layer:02d}.pt"
            if any(filename not in m["files"] for m in manifests):
                raise ValueError(f"missing {filename}")
            selected[search_layer] = []
            for directory in args.trace_dirs:
                trace = torch.load(directory / filename, map_location="cpu", weights_only=True)
                length = trace["k"].shape[2]
                if args.budget >= length:
                    raise ValueError("budget must be below every source length")
                search = {key: trace[key].to("cuda") for key in ("q", "k")}
                mask = less_mask(search, length, args.budget)
                selected[search_layer].append(torch.nonzero(mask[0, 0], as_tuple=True)[0].cpu())
        for layer in args.layers:
            search_layer = 2 if layer < 13 else 13
            name = f"step{step:03d}_layer{layer:02d}.pt"
            search_name = f"step{step:03d}_layer{search_layer:02d}.pt"
            paths = [d / name for d in args.trace_dirs] + [d / search_name for d in args.trace_dirs]
            if any(name not in m["files"] for m in manifests):
                raise ValueError(f"missing {name}")
            traces = [torch.load(d / name, map_location="cpu", weights_only=True) for d in args.trace_dirs]
            for trace in traces:
                for key in ("q", "k", "v"):
                    trace[key] = trace[key].to(torch.float16)
            for batch in args.batch_sizes:
                q, k, v, masks, shared, ex, lengths = make_batch(traces, batch, "cuda")
                row_lengths = [lengths[i % len(lengths)] for i in range(batch)]
                ids = torch.stack([selected[search_layer][i % len(lengths)] for i in range(batch)]).to("cuda")
                less = torch.zeros_like(masks["gas"])
                less.scatter_(2, ids[:, None, :].expand(batch, 32, args.budget), True)
                expected = chunked_reference(q, k, v, less, args.reference_batch_chunk)
                wrappers = []
                official = None
                calls = {}
                if "less_patched_headpacked" in args.methods:
                    official, wrappers = official_grouped_setup(module, q, k, v, ids, row_lengths, args.pack_size)
                    calls["less_patched_headpacked"] = official
                if "less_fi_adapter" in args.methods:
                    calls["less_fi_adapter"] = fi_exact(q, k, v, less)
                try:
                    for method, call in calls.items():
                        shard = args.output_dir / "shards" / f"s{step}_l{layer}_b{batch}_{method}.csv"
                        sample_paths = [args.output_dir / "samples" / f"s{step}_l{layer}_b{batch}_{method}_{r}.csv" for r in range(args.rounds)]
                        cached = load_cached(shard, sample_paths, paths, args.repeats)
                        if cached is not None:
                            rows.append(cached); print(f"SKIP s{step} l{layer} B{batch} {method}", flush=True); continue
                        result = call()
                        check = error_metrics(result, expected, 0.02, 0.02)
                        if not check["pass"]:
                            raise ValueError(f"{step=} {layer=} {batch=} {method}: {check}")
                        for _ in range(args.warmups): call()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(args.graph_iterations): result = call()
                        result.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                        check = error_metrics(result, expected, 0.02, 0.02)
                        if not check["pass"]:
                            raise ValueError(f"graph {step=} {layer=} {batch=} {method}: {check}")
                        times = []
                        for r, sample_path in enumerate(sample_paths):
                            samples = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                            times.extend(samples)
                            write_rows(sample_path, [{"latency_us": x} for x in samples])
                        row = {"step": step, "layer": layer, "batch": batch, "method": method,
                               "search_layer": search_layer, "budget": args.budget, "calls": len(wrappers) if method.startswith("less_patched") else 0,
                               "trace_sha256": json.dumps([digest(p) for p in paths]), "samples": len(times),
                               "median_us": statistics.median(times), "max_abs_error": check["max_abs_error"]}
                        rows.append(row); write_rows(shard, [row])
                        print(f"s{step} l{layer} B{batch} {method}: {row['median_us']:.2f} us", flush=True)
                        del graph, result
                finally:
                    for wrapper in wrappers: wrapper.end_forward()
                # The last callable closes over packed KV; release it before the next batch.
                del call, calls, official, wrappers, q, k, v, masks, shared, ex, less, expected
                torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
