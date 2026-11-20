#!/usr/bin/env python3
"""Graph replay of official B1 Quest/LessIsMore and exact-support GAS."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_official_b1 import (UnsupportedSupport, less_mask, less_setup,
                                        load_extension, quest_setup, quest_setup_split)
from bench_e6_qasper_trace import digest, fi_exact, reference, write_rows

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmark/results_efficiency/e6_qasper_official_artifacts"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("official_quest", "official_less", "gas_fi"), required=True)
    parser.add_argument("--stream-build-dir", type=Path, required=True,
                        help="directory containing independently built current-stream adapter manifests")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.warmups < 0 or min(args.repeats, args.rounds, args.graph_iterations) < 1:
        parser.error("invalid sampling parameters")
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [args.trace_dir / name for name in manifest["files"]
             if (not args.steps or int(name[4:7]) in args.steps)
             and (not args.layers or int(name[13:15]) in args.layers)]
    if not files:
        raise ValueError("no matching traces")
    manifests = {name: json.loads((args.stream_build_dir / f"{name}_stream_build.json").read_text()) for name in ("quest", "less")}
    quest_path, less_path = (Path(manifests[name]["module"]) for name in ("quest", "less"))
    if any(digest(path) != manifests[name]["module_sha256"] for name, path in (("quest", quest_path), ("less", less_path))):
        raise ValueError("stream-adapted extension hash mismatch")
    quest = load_extension(quest_path, "e6_quest_stream_decode") if args.method == "official_quest" else None
    less = load_extension(less_path, "e6_less_stream_decode") if args.method == "official_less" else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"method": args.method, "script_sha256": digest(Path(__file__)), "manifest_sha256": digest(manifest_path),
              "quest_extension_sha256": digest(quest_path), "less_extension_sha256": digest(less_path),
              "files": [p.name for p in files], "less_k": 91, "device": torch.cuda.get_device_name(0),
              "warmups": args.warmups, "repeats": args.repeats, "rounds": args.rounds,
              "graph_iterations": args.graph_iterations, "dtype": "FP16 from BF16 trace",
              "scope": "attention calls only; setup/selection/planning/packing excluded",
              "quest": "B1 upstream MHA core with current-stream wrapper, replicated KV, partial pages require timed head splitting and scatter",
              "less": "B1 upstream sparse core with current-stream wrapper, author quality-path selection on captured GAS search state; search layers excluded",
              "gas": "FlashInfer exact per-head mask, not an explicit shared-load kernel"}
    config_path = args.output_dir / f"{args.method}.config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing directory has different protocol")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    all_rows = []
    for path in files:
        trace = torch.load(path, map_location="cpu", weights_only=True)
        q, k, v = [trace[key].to(device="cuda", dtype=torch.float16).contiguous() for key in ("q", "k", "v")]
        step, layer = int(path.name[4:7]), int(path.name[13:15])
        for method in (args.method,):
            row = {"trace": path.name, "trace_sha256": digest(path), "method": method,
                   "status": "failed", "reason": "", "groups": "", "samples": 0,
                   "median_us": "", "max_abs_error": ""}
            wrapper = call = graph = output = None
            try:
                if method == "official_quest":
                    mask = trace["candidate"].to("cuda")
                    try:
                        call, wrapper = quest_setup(quest, q, k, v, mask[0])
                        row["groups"] = 1
                    except UnsupportedSupport:
                        call, wrapper, row["groups"] = quest_setup_split(quest, q, k, v, mask[0])
                elif method == "official_less":
                    if layer in (2, 13):
                        raise UnsupportedSupport("dense search layer")
                    search_layer = 2 if layer < 13 else 13
                    search = torch.load(args.trace_dir / f"step{step:03d}_layer{search_layer:02d}.pt", map_location="cpu", weights_only=True)
                    search = {key: value.to("cuda") for key, value in search.items() if key in ("q", "k")}
                    mask = less_mask(search, k.shape[-2], 91)
                    call, wrapper = less_setup(less, q, k, v, mask)
                else:
                    mask = trace["gas"].to("cuda")
                    call = fi_exact(q, k, v, mask)
                ref = reference(q, k, v, mask)
                for _ in range(args.warmups):
                    call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.graph_iterations):
                        output = call()
                # A stale output must never pass when an upstream kernel was
                # launched on a different stream and omitted from the graph.
                output.fill_(float("nan"))
                if method == "official_quest" and row["groups"] != 1:
                    for cell in call.__closure__ or ():
                        value = cell.cell_contents
                        if isinstance(value, list) and value and isinstance(value[0], tuple) and len(value[0]) == 8:
                            for entry in value:
                                entry[2].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                metrics = error_metrics(output, ref, 0.02, 0.02)
                if not metrics["pass"] or not metrics["finite"]:
                    raise ValueError(f"graph output differs from FP32 reference: {metrics}")
                row["max_abs_error"] = metrics["max_abs_error"]
                samples = []
                for round_id in range(args.rounds):
                    values = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, q.device, args.warmups, args.repeats)]
                    samples.extend(values)
                    write_rows(args.output_dir / "samples" / f"{path.stem}_{method}_{round_id}.csv", [{"latency_us": x} for x in values])
                row.update(status="available", median_us=statistics.median(samples), samples=len(samples))
            except UnsupportedSupport as exc:
                row.update(status="unavailable", reason=str(exc))
            except Exception as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
            finally:
                if wrapper is not None:
                    wrapper.end_forward()
            print(f"{path.stem} {method}: {row['status']} {row['median_us']} {row['reason']}", flush=True)
            all_rows.append(row)
            write_rows(args.output_dir / "shards" / f"{path.stem}_{method}.csv", [row])
            wrapper = call = graph = output = None
            torch.cuda.empty_cache()
    write_rows(args.output_dir / f"{args.method}.results.csv", all_rows)
    print(f"wrote {len(all_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
