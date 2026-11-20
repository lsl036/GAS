#!/usr/bin/env python3
"""Ragged GAS attention and CPU planning on captured real Qasper traces."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_trace import digest, fi_exact, gas_indices, reference, write_rows
from gas_ragged_plan import build_plan, build_plan_gpu

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64, 96, 128])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--max-parts", type=int, default=8)
    parser.add_argument("--planner", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    parser.add_argument("--plan-repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if min(args.batch_sizes + args.chunks + [args.max_parts, args.repeats, args.rounds, args.graph_iterations, args.plan_repeats]) < 1 or args.warmups < 0:
        parser.error("invalid benchmark parameters")
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [args.trace_dir / name for name in manifest["files"]
             if (not args.steps or int(name[4:7]) in args.steps) and
             (not args.layers or int(name[13:15]) in args.layers)]
    if not files:
        raise ValueError("no matching traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"manifest_sha256": digest(manifest_path), "files": [p.name for p in files],
              "script_sha256": digest(Path(__file__)), "planner_sha256": digest(ROOT / "benchmark/efficiency/gas_ragged_plan.py"),
              "ragged_source_sha256": digest(ROOT / "csrc/src/gas_ragged_decode.cu"),
              "counted_source_sha256": digest(ROOT / "csrc/src/gas_counted_decode.cu"),
              "split_source_sha256": digest(ROOT / "csrc/src/gas_counted_split.cu"),
              "dtype": args.dtype, "source_dtype": manifest["dtype"], "batch_sizes": args.batch_sizes,
              "chunks": args.chunks, "max_parts": args.max_parts, "warmups": args.warmups,
              "planner": args.planner,
              "repeats": args.repeats, "rounds": args.rounds, "graph_iterations": args.graph_iterations,
              "plan_repeats": args.plan_repeats, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "batch": "physical copies of one captured request",
              "attention": "CUDA Graph events; compact indices, counts, FI plan and ragged plan precomputed",
              "plan": "synchronized wall time from GPU counts; CPU planner copies counts to host and uploads plan; GPU planner uses dynamic nonzero and may synchronize; selector/compaction excluded",
              "pipeline": "synchronized wall time of GPU prefix count, selected planner, transfers/synchronization, ragged attention; selector/compaction excluded",
              "fi": "per-head page-1 exact-support adapter, not author kernel or explicit group-shared load"}
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory contains a different protocol")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    def module(name, file):
        return load(name=name, sources=[str(ROOT / f"csrc/src/{file}")], extra_cuda_cflags=["-O3"], verbose=False)
    counted = module("gas_counted_decode", "gas_counted_decode.cu")
    split = module("gas_counted_split", "gas_counted_split.cu")
    ragged = module("gas_ragged_decode", "gas_ragged_decode.cu")
    device, dtype = torch.device("cuda:0"), getattr(torch, args.dtype)
    rows = []
    for path in files:
        trace = torch.load(path, map_location="cpu", weights_only=True)
        q0, k0, v0 = [trace[key].to(device=device, dtype=dtype).contiguous() for key in ("q", "k", "v")]
        masks = {name: trace[name].squeeze(0).bool() for name in ("shared", "gas", "twilight", "candidate")}
        refs = {name: reference(q0, k0, v0, masks[key].unsqueeze(0).to(device))
                for name, key in (("gas", "gas"), ("twilight", "twilight"), ("quest", "candidate"))}
        for batch in args.batch_sizes:
            q, k, v = [x.repeat(batch, 1, 1) if x.ndim == 3 else x.repeat(batch, 1, 1, 1) for x in (q0, k0, v0)]
            shared, ex = gas_indices(masks["shared"], masks["gas"], batch, device)
            sc, ec = counted.count_prefix(shared, ex)
            def create_plan(gs, hs, chunk):
                return (build_plan(gs, hs, shared.shape[-1], ex.shape[-1], chunk, args.max_parts, device)
                        if args.planner == "cpu" else build_plan_gpu(gs, hs, chunk, args.max_parts))
            plans, plan_samples = {}, {}
            for chunk in args.chunks:
                samples = []
                for trial in range(args.plan_repeats + 1):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    plan = create_plan(sc, ec, chunk)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1e6
                    if trial:
                        samples.append(elapsed)
                plans[chunk], plan_samples[chunk] = plan, samples
            def execute(plan, gs=sc, hs=ec):
                return ragged.decode(q, k, v, shared, ex, gs, hs, plan["jobs"], plan["parts"],
                                     plan["offsets"], plan["long_rows"], 8, plan["partial_count"])
            calls = {"counted8": lambda: counted.decode(q, k, v, shared, ex, sc, ec, 8),
                     "inline8": lambda: counted.decode_prefix(q, k, v, shared, ex, 8),
                     "split8p4": lambda: split.decode(q, k, v, shared, ex, sc, ec, 8, 4),
                     "fi_gas": fi_exact(q, k, v, masks["gas"].unsqueeze(0).to(device).repeat(batch, 1, 1)),
                     "fi_twilight": fi_exact(q, k, v, masks["twilight"].unsqueeze(0).to(device).repeat(batch, 1, 1)),
                     "fi_quest": fi_exact(q, k, v, masks["candidate"].unsqueeze(0).to(device).repeat(batch, 1, 1))}
            calls.update({f"ragged{chunk}": (lambda chunk=chunk: execute(plans[chunk])) for chunk in args.chunks})
            common = {"trace": path.name, "trace_sha256": digest(path), "batch": batch, "dtype": args.dtype}
            for chunk, samples in plan_samples.items():
                plan = plans[chunk]
                row = {**common, "stage": f"{args.planner}_plan", "method": f"plan{chunk}", "median_us": statistics.median(samples),
                       "samples": len(samples), "max_abs_error": 0.0, "jobs": plan["jobs"].shape[0],
                       "long_heads": plan["long_rows"].numel(), "partial_count": plan["partial_count"]}
                rows.append(row)
                write_rows(args.output_dir / "samples" / f"{path.stem}_B{batch}_plan{chunk}.csv", [{"latency_us": x} for x in samples])
            order = list(calls)
            random.Random(f"{path.name}:{batch}:{args.dtype}").shuffle(order)
            for name in order:
                method_ref = refs["twilight" if name == "fi_twilight" else "quest" if name == "fi_quest" else "gas"]
                for _ in range(args.warmups):
                    calls[name]()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.graph_iterations):
                        output = calls[name]()
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                check = error_metrics(output, method_ref.expand_as(output), 0.02, 0.02)
                if not check["pass"] or not bool(torch.isfinite(output).all()):
                    raise ValueError(f"{path.name} B{batch} {name} graph failed: {check}")
                samples = []
                for r in range(args.rounds):
                    measured = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, device, args.warmups, args.repeats)]
                    samples.extend(measured)
                    write_rows(args.output_dir / "samples" / f"{path.stem}_B{batch}_{name}_{r}.csv", [{"latency_us": x} for x in measured])
                plan = plans[int(name.removeprefix("ragged"))] if name.startswith("ragged") else None
                row = {**common, "stage": "attention_graph", "method": name, "median_us": statistics.median(samples),
                       "samples": len(samples), "max_abs_error": check["max_abs_error"],
                       "jobs": plan["jobs"].shape[0] if plan else "", "long_heads": plan["long_rows"].numel() if plan else "",
                       "partial_count": plan["partial_count"] if plan else ""}
                rows.append(row)
                print(f"{path.stem} B{batch} {name}: {row['median_us']:.2f} us", flush=True)
                del graph, output
            # A synchronized chain, measured separately from graph timing.
            chunk = 128 if 128 in args.chunks else args.chunks[0]
            samples = []
            for trial in range(args.plan_repeats + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                gs, hs = counted.count_prefix(shared, ex)
                plan = create_plan(gs, hs, chunk)
                output = execute(plan, gs, hs)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1e6
                check = error_metrics(output, refs["gas"].expand_as(output), 0.02, 0.02)
                if not check["pass"]:
                    raise ValueError(f"{path.name} B{batch} pipeline failed: {check}")
                if trial:
                    samples.append(elapsed)
            row = {**common, "stage": f"{args.planner}_pipeline", "method": f"pipeline{chunk}", "median_us": statistics.median(samples),
                   "samples": len(samples), "max_abs_error": check["max_abs_error"], "jobs": plan["jobs"].shape[0],
                   "long_heads": plan["long_rows"].numel(), "partial_count": plan["partial_count"]}
            rows.append(row)
            write_rows(args.output_dir / "samples" / f"{path.stem}_B{batch}_pipeline{chunk}.csv", [{"latency_us": x} for x in samples])
            print(f"{path.stem} B{batch} pipeline{chunk}: {row['median_us']:.2f} us", flush=True)
            del q, k, v, shared, ex, sc, ec, calls, plans, output
            torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
