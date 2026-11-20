#!/usr/bin/env python3
"""Compare GAS attention implementations using identical real-trace masks."""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics, padded_tensor
from bench_e6_kernel_baselines import get_gas_module
from bench_e6_qasper_trace import digest, fi_exact, gas_indices, reference, write_rows

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("old_fused", "old_per_head", "warp_fused", "head_parallel", "split8", "split16", "split32",
           "staged8", "staged16", "tiled_head", "token_parallel4", "token_parallel8", "token_parallel16",
           "vector16_4", "vector16_8", "vector16_16", "vector16_32", "vector16_adaptive", "gas_hybrid",
           "vector_tree8", "vector_tree16", "vector_tree32", "flashinfer_exact", "flashinfer_twilight", "flashinfer_quest")


def hybrid_backend(batch, shared_width):
    # Fixed from the nine-trace pilot, not tuned per measured trace.
    threshold = 96 if batch == 32 else 94
    return "flashinfer_exact" if batch == 1 or shared_width >= threshold else "vector16_8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64, 96, 128])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--graph-iterations", type=int, default=0,
                        help="capture this many calls in one CUDA graph and report time per call")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if min(args.batch_sizes + [args.repeats, args.rounds]) < 1 or args.warmups < 0 or args.graph_iterations < 0:
        parser.error("batch sizes, repeats and rounds must be positive; warmups nonnegative")
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [args.trace_dir / name for name in manifest["files"]
             if (not args.steps or int(name[4:7]) in args.steps)
             and (not args.layers or int(name[13:15]) in args.layers)]
    if not files:
        raise ValueError("no matching traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"manifest_sha256": digest(manifest_path), "files": [f.name for f in files],
              "script_sha256": digest(Path(__file__)),
              "source_sha256": digest(ROOT / "csrc/src/gas_warp_decode.cu"),
              "tree_source_sha256": digest(ROOT / "csrc/src/gas_vector_tree.cu") if any(m.startswith("vector_tree") for m in args.methods) else None,
              "baseline_sha256": digest(ROOT / "csrc/src/gqa_group_decode.cu"),
              "batch_sizes": args.batch_sizes, "warmups": args.warmups,
              "methods": args.methods,
              "repeats": args.repeats, "rounds": args.rounds,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "dtype": "FP16, cast from captured BF16 before timing",
              "timing": (f"CUDA graph of {args.graph_iterations} calls; event time divided by call count"
                         if args.graph_iterations else "individual CUDA event intervals; allocations included"),
              "graph_iterations": args.graph_iterations,
              "hybrid_policy": "B1 uses FlashInfer; B32 uses FlashInfer at width>=96; other B use FlashInfer at width>=94; otherwise vector16_8; selection and both backends' metadata preparation excluded",
              "batch": "physical copies of the same request"}
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory already contains a different protocol")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    old = get_gas_module()
    warp = load(name="gas_warp_decode_v1", sources=[str(ROOT / "csrc/src/gas_warp_decode.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)
    tree = load(name="gas_vector_tree", sources=[str(ROOT / "csrc/src/gas_vector_tree.cu")],
                extra_cuda_cflags=["-O3"], verbose=False) if any(m.startswith("vector_tree") for m in args.methods) else None
    device = torch.device("cuda:0")
    all_rows = []
    for path in files:
        trace = torch.load(path, map_location="cpu", weights_only=True)
        q0, k0, v0 = [trace[key].to(device=device, dtype=torch.float16).contiguous() for key in ("q", "k", "v")]
        shared, mask = [trace[key].squeeze(0).bool() for key in ("shared", "gas")]
        ref = reference(q0, k0, v0, mask.unsqueeze(0).to(device))
        candidate, twilight = [trace[key].squeeze(0).bool() for key in ("candidate", "twilight")]
        reference_quest = reference(q0, k0, v0, candidate.unsqueeze(0).to(device))
        reference_twilight = reference(q0, k0, v0, twilight.unsqueeze(0).to(device))
        for batch in args.batch_sizes:
            q, k, v = [x.repeat(batch, 1, 1) if x.ndim == 3 else x.repeat(batch, 1, 1, 1) for x in (q0, k0, v0)]
            indices, exceptions = gas_indices(shared, mask, batch, device)
            hybrid_selected = hybrid_backend(batch, indices.shape[2])
            head_lists = [torch.nonzero(mask[h], as_tuple=True)[0].tolist() for h in range(mask.shape[0])]
            head_ids = padded_tensor(head_lists * batch, (batch, mask.shape[0]), device)
            calls = {"old_fused": lambda: old.batched_gqa_group_exception_decode(q, k, v, indices, exceptions),
                     "old_per_head": lambda: old.batched_per_head_sparse_decode(q, k, v, head_ids),
                     "warp_fused": lambda: warp.decode(q, k, v, indices, exceptions),
                     "head_parallel": lambda: warp.decode(q, k, v, indices, exceptions, 1),
                     "split8": lambda: warp.decode(q, k, v, indices, exceptions, 2, 8),
                     "split16": lambda: warp.decode(q, k, v, indices, exceptions, 2, 16),
                     "split32": lambda: warp.decode(q, k, v, indices, exceptions, 2, 32),
                     "staged8": lambda: warp.decode(q, k, v, indices, exceptions, 3, 8),
                     "staged16": lambda: warp.decode(q, k, v, indices, exceptions, 3, 16),
                     "tiled_head": lambda: warp.decode(q, k, v, indices, exceptions, 4)}
            calls.update({"token_parallel4": lambda: warp.decode(q, k, v, indices, exceptions, 5),
                          "token_parallel8": lambda: warp.decode(q, k, v, indices, exceptions, 6),
                          "token_parallel16": lambda: warp.decode(q, k, v, indices, exceptions, 7),
                          "vector16_4": lambda: warp.decode(q, k, v, indices, exceptions, 8),
                          "vector16_8": lambda: warp.decode(q, k, v, indices, exceptions, 9),
                          "vector16_16": lambda: warp.decode(q, k, v, indices, exceptions, 10),
                          "vector16_32": lambda: warp.decode(q, k, v, indices, exceptions, 11),
                          "vector16_adaptive": lambda: warp.decode(q, k, v, indices, exceptions, 11 if batch == 1 else 9)})
            if tree is not None:
                calls.update({"vector_tree8": lambda: tree.decode(q, k, v, indices, exceptions, 8),
                              "vector_tree16": lambda: tree.decode(q, k, v, indices, exceptions, 16),
                              "vector_tree32": lambda: tree.decode(q, k, v, indices, exceptions, 32)})
            for method, selected in (("flashinfer_exact", mask), ("flashinfer_twilight", twilight),
                                     ("flashinfer_quest", candidate)):
                if method in args.methods or (method == "flashinfer_exact" and "gas_hybrid" in args.methods and hybrid_selected == method):
                    calls[method] = fi_exact(q, k, v, selected.unsqueeze(0).to(device).repeat(batch, 1, 1))
            if "gas_hybrid" in args.methods:
                calls["gas_hybrid"] = calls[hybrid_selected]
            order = list(args.methods)
            random.Random(f"{path.name}:{batch}").shuffle(order)
            for name in order:
                output = calls[name]()
                torch.cuda.synchronize()
                method_ref = reference_quest if name == "flashinfer_quest" else reference_twilight if name == "flashinfer_twilight" else ref
                check = error_metrics(output, method_ref.expand_as(output), 0.02, 0.02)
                if not check["pass"] or not bool(torch.isfinite(output).all()):
                    raise RuntimeError(f"{path.name} B{batch} {name} failed reference: {check}")
                graph = None
                timed = calls[name]
                if args.graph_iterations:
                    for _ in range(args.warmups):
                        calls[name]()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.graph_iterations):
                            graph_output = calls[name]()
                    graph_output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_check = error_metrics(graph_output, method_ref.expand_as(graph_output), 0.02, 0.02)
                    if not graph_check["pass"] or not bool(torch.isfinite(graph_output).all()):
                        raise RuntimeError(f"{path.name} B{batch} {name} graph reference failed: {graph_check}")
                    timed = graph.replay
                samples = []
                for round_id in range(args.rounds):
                    values = cuda_event_samples(timed, device, args.warmups, args.repeats)
                    if graph is not None:
                        values = [value / args.graph_iterations for value in values]
                    samples.extend(values)
                    write_rows(args.output_dir / "samples" / f"{path.stem}_B{batch}_{name}_{round_id}.csv",
                               [{"latency_us": value} for value in values])
                row = {"trace": path.name, "trace_sha256": digest(path), "batch": batch, "method": name,
                       "shared_width": indices.shape[2], "exception_width": exceptions.shape[2],
                       "backend": hybrid_selected if name == "gas_hybrid" else name,
                       "median_us": statistics.median(samples), "samples": len(samples),
                       "max_abs_error": check["max_abs_error"], "relative_l2_error": check["relative_l2_error"]}
                all_rows.append(row)
                write_rows(args.output_dir / "shards" / f"{path.stem}_B{batch}_{name}.csv", [row])
                print(f"{path.stem} B{batch} {name}: {row['median_us']:.2f} us", flush=True)
                del graph, timed
            del q, k, v, indices, exceptions, head_ids, calls, output
            torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", all_rows)
    print(f"wrote {len(all_rows)} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
