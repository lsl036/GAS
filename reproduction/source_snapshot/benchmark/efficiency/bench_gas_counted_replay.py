#!/usr/bin/env python3
"""Graph benchmark for counted GAS support on real Qasper traces."""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from bench_e4_kernel_evidence import cuda_event_samples, error_metrics
from bench_e6_qasper_trace import digest, fi_exact, gas_indices, reference, write_rows

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("counted4", "counted8", "counted16", "counted32", "inline4", "inline8", "inline16", "inline32", "twopass4", "twopass8", "twopass16", "twopass32", "twopass_inline8", "twopass_inline16", "twopass_inline32", "prefix_plus8", "prefix_plus32", "split8p2", "split8p4", "split8p8", "split16p4", "prefix_split8p4", "adaptive64", "adaptive128", "adaptive256", "prefix_adaptive128", "persistent64", "persistent128", "persistent256", "prefix_persistent128", "cooperative4", "cooperative8", "cooperative16", "prefix_cooperative8", "groupsplit64", "groupsplit128", "groupsplit256", "prefix_groupsplit128", "fi_gas", "fi_twilight", "fi_quest", "count_metadata", "count_prefix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64, 96, 128])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if min(args.batch_sizes + [args.repeats, args.rounds, args.graph_iterations]) < 1 or args.warmups < 0:
        parser.error("invalid benchmark parameters")
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [args.trace_dir / name for name in manifest["files"]
             if (not args.steps or int(name[4:7]) in args.steps)
             and (not args.layers or int(name[13:15]) in args.layers)]
    if not files:
        raise ValueError("no matching traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"manifest_sha256": digest(manifest_path), "files": [p.name for p in files],
              "source_sha256": digest(ROOT / "csrc/src/gas_counted_decode.cu"), "script_sha256": digest(Path(__file__)),
              "split_source_sha256": digest(ROOT / "csrc/src/gas_counted_split.cu") if any(m.startswith("split") or m.startswith("prefix_split") for m in args.methods) else None,
              "persistent_source_sha256": digest(ROOT / "csrc/src/gas_persistent_decode.cu") if any("persistent" in m for m in args.methods) else None,
              "adaptive_source_sha256": digest(ROOT / "csrc/src/gas_adaptive_split.cu") if any("adaptive" in m for m in args.methods) else None,
              "cooperative_source_sha256": digest(ROOT / "csrc/src/gas_cooperative_decode.cu") if any("cooperative" in m for m in args.methods) else None,
              "groupsplit_source_sha256": digest(ROOT / "csrc/src/gas_cooperative_split.cu") if any("groupsplit" in m for m in args.methods) else None,
              "twopass_source_sha256": digest(ROOT / "csrc/src/gas_twopass_decode.cu") if any("twopass" in m for m in args.methods) else None,
              "dtype": args.dtype, "source_dtype": manifest["dtype"], "batch_sizes": args.batch_sizes,
              "methods": args.methods, "warmups": args.warmups, "repeats": args.repeats,
              "rounds": args.rounds, "graph_iterations": args.graph_iterations,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "batch": "physical copies of the same captured request",
              "attention_timing": "CUDA graph average; precomputed compact indices/counts, FI plan and all selection excluded",
              "count_metadata": "separate CUDA graph: count nonnegative padded indices using PyTorch; count_prefix uses binary search on packed prefixes; neither is full selection/compaction/plan cost",
              "prefix_plus": "two GPU kernels in graph: count padded prefixes then decode; selection, compaction and dispatch excluded",
              "inline": "one GPU kernel includes prefix-length binary searches and decode; selection and compaction excluded",
              "persistent": "GPU counter initialization, job construction, queue consumption and merge included; precomputed counts, selection and compaction excluded; prefix_persistent includes count_prefix",
              "adaptive": "fixed-grid dynamic splitting and merge included; precomputed counts, selection and compaction excluded; prefix_adaptive includes count_prefix",
              "cooperative": "one CTA per KV group, shared K/V reused for four heads, per-head exceptions; precomputed counts, selection and compaction excluded; prefix_cooperative includes count_prefix",
              "groupsplit": "group-cooperative adaptive splitting and merge included; prefix_groupsplit also includes count_prefix; selection and compaction excluded",
              "twopass": "per-head CTA stages logits and computes stable softmax/PV; inline variant counts prefix inside kernel; selection and compaction excluded",
              "fi": "exact per-head page-1 adapter, not author Quest/Twilight or native group-shared kernel"}
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory has a different protocol")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    module = load(name="gas_counted_decode", sources=[str(ROOT / "csrc/src/gas_counted_decode.cu")],
                  extra_cuda_cflags=["-O3"], verbose=False)
    split = load(name="gas_counted_split", sources=[str(ROOT / "csrc/src/gas_counted_split.cu")],
                 extra_cuda_cflags=["-O3"], verbose=False) if any(m.startswith("split") or m.startswith("prefix_split") for m in args.methods) else None
    persistent = load(name="gas_persistent_decode", sources=[str(ROOT / "csrc/src/gas_persistent_decode.cu")],
                      extra_cuda_cflags=["-O3"], verbose=False) if any("persistent" in m for m in args.methods) else None
    adaptive = load(name="gas_adaptive_split", sources=[str(ROOT / "csrc/src/gas_adaptive_split.cu")],
                    extra_cuda_cflags=["-O3"], verbose=False) if any("adaptive" in m for m in args.methods) else None
    cooperative = load(name="gas_cooperative_decode", sources=[str(ROOT / "csrc/src/gas_cooperative_decode.cu")],
                       extra_cuda_cflags=["-O3"], verbose=False) if any("cooperative" in m for m in args.methods) else None
    groupsplit = load(name="gas_cooperative_split", sources=[str(ROOT / "csrc/src/gas_cooperative_split.cu")],
                      extra_cuda_cflags=["-O3"], verbose=False) if any("groupsplit" in m for m in args.methods) else None
    twopass = load(name="gas_twopass_decode_guarded", sources=[str(ROOT / "csrc/src/gas_twopass_decode.cu")],
                   extra_cuda_cflags=["-O3"], verbose=False) if any("twopass" in m for m in args.methods) else None
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
            shared, exceptions = gas_indices(masks["shared"], masks["gas"], batch, device)
            def count_metadata():
                return ((shared >= 0).sum(-1, dtype=torch.int32),
                        (exceptions >= 0).sum(-1, dtype=torch.int32))
            sc, ec = count_metadata()
            calls = {f"counted{n}": (lambda n=n: module.decode(q, k, v, shared, exceptions, sc, ec, n))
                     for n in (4, 8, 16, 32)}
            calls.update({f"inline{n}": (lambda n=n: module.decode_prefix(q, k, v, shared, exceptions, n))
                          for n in (4, 8, 16, 32)})
            def prefix_plus(n):
                group_counts, head_counts = module.count_prefix(shared, exceptions)
                return module.decode(q, k, v, shared, exceptions, group_counts, head_counts, n)
            calls.update({f"prefix_plus{n}": (lambda n=n: prefix_plus(n)) for n in (8, 32)})
            if twopass is not None:
                for segments in (4, 8, 16, 32):
                    calls[f"twopass{segments}"] = (lambda segments=segments:
                        twopass.decode(q, k, v, shared, exceptions, sc, ec, segments))
                for segments in (8, 16, 32):
                    calls[f"twopass_inline{segments}"] = (lambda segments=segments:
                        twopass.decode_prefix(q, k, v, shared, exceptions, segments))
            if groupsplit is not None:
                for chunk in (64, 128, 256):
                    calls[f"groupsplit{chunk}"] = (lambda chunk=chunk:
                        groupsplit.decode(q, k, v, shared, exceptions, sc, ec, 8, 8, chunk))
                def prefix_groupsplit():
                    group_counts, head_counts = module.count_prefix(shared, exceptions)
                    return groupsplit.decode(q, k, v, shared, exceptions, group_counts, head_counts, 8, 8, 128)
                calls["prefix_groupsplit128"] = prefix_groupsplit
            if cooperative is not None:
                for segments in (4, 8, 16):
                    calls[f"cooperative{segments}"] = (lambda segments=segments:
                        cooperative.decode(q, k, v, shared, exceptions, sc, ec, segments))
                def prefix_cooperative():
                    group_counts, head_counts = module.count_prefix(shared, exceptions)
                    return cooperative.decode(q, k, v, shared, exceptions, group_counts, head_counts, 8)
                calls["prefix_cooperative8"] = prefix_cooperative
            if adaptive is not None:
                for chunk in (64, 128, 256):
                    calls[f"adaptive{chunk}"] = (lambda chunk=chunk:
                        adaptive.decode(q, k, v, shared, exceptions, sc, ec, 8, 8, chunk))
                def prefix_adaptive():
                    group_counts, head_counts = module.count_prefix(shared, exceptions)
                    return adaptive.decode(q, k, v, shared, exceptions, group_counts, head_counts, 8, 8, 128)
                calls["prefix_adaptive128"] = prefix_adaptive
            if persistent is not None:
                blocks = min(torch.cuda.get_device_properties(device).multi_processor_count * 4, batch * q.shape[1] * 8)
                for chunk in (64, 128, 256):
                    calls[f"persistent{chunk}"] = (lambda chunk=chunk:
                        persistent.decode(q, k, v, shared, exceptions, sc, ec, chunk, 8, blocks))
                def prefix_persistent():
                    group_counts, head_counts = module.count_prefix(shared, exceptions)
                    return persistent.decode(q, k, v, shared, exceptions, group_counts, head_counts, 128, 8, blocks)
                calls["prefix_persistent128"] = prefix_persistent
            if split is not None:
                for segments, parts in ((8, 2), (8, 4), (8, 8), (16, 4)):
                    calls[f"split{segments}p{parts}"] = (lambda segments=segments, parts=parts:
                        split.decode(q, k, v, shared, exceptions, sc, ec, segments, parts))
                def prefix_split():
                    group_counts, head_counts = module.count_prefix(shared, exceptions)
                    return split.decode(q, k, v, shared, exceptions, group_counts, head_counts, 8, 4)
                calls["prefix_split8p4"] = prefix_split
            calls["count_metadata"] = count_metadata
            calls["count_prefix"] = lambda: module.count_prefix(shared, exceptions)
            for name, key in (("fi_gas", "gas"), ("fi_twilight", "twilight"), ("fi_quest", "candidate")):
                if name in args.methods:
                    calls[name] = fi_exact(q, k, v, masks[key].unsqueeze(0).to(device).repeat(batch, 1, 1))
            order = list(args.methods)
            random.Random(f"{path.name}:{batch}:{args.dtype}").shuffle(order)
            for name in order:
                result = calls[name]()
                torch.cuda.synchronize()
                method_ref = refs["twilight" if name == "fi_twilight" else "quest" if name == "fi_quest" else "gas"]
                if name in ("count_metadata", "count_prefix"):
                    if not torch.equal(result[0], sc) or not torch.equal(result[1], ec):
                        raise ValueError("count metadata mismatch")
                    max_error = 0.0
                else:
                    check = error_metrics(result, method_ref.expand_as(result), 0.02, 0.02)
                    if not check["pass"] or not bool(torch.isfinite(result).all()):
                        raise ValueError(f"{path.name} B{batch} {name} correctness failed: {check}")
                    max_error = check["max_abs_error"]
                for _ in range(args.warmups):
                    calls[name]()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.graph_iterations):
                        output = calls[name]()
                if name in ("count_metadata", "count_prefix"):
                    for tensor in output:
                        tensor.fill_(-1)
                else:
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if name in ("count_metadata", "count_prefix"):
                    if not torch.equal(output[0], sc) or not torch.equal(output[1], ec):
                        raise ValueError("graph metadata mismatch")
                else:
                    check = error_metrics(output, method_ref.expand_as(output), 0.02, 0.02)
                    if not check["pass"] or not bool(torch.isfinite(output).all()):
                        raise ValueError(f"{path.name} B{batch} {name} graph failed: {check}")
                    max_error = max(max_error, check["max_abs_error"])
                samples = []
                for round_id in range(args.rounds):
                    measured = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, device, args.warmups, args.repeats)]
                    samples.extend(measured)
                    write_rows(args.output_dir / "samples" / f"{path.stem}_B{batch}_{name}_{round_id}.csv",
                               [{"latency_us": x} for x in measured])
                row = {"trace": path.name, "trace_sha256": digest(path), "batch": batch, "dtype": args.dtype,
                       "method": name, "shared_width": shared.shape[2], "mean_shared_count": float(sc.float().mean()),
                       "max_shared_count": int(sc.max()), "mean_exception_count": float(ec.float().mean()),
                       "median_us": statistics.median(samples), "samples": len(samples), "max_abs_error": max_error}
                rows.append(row)
                write_rows(args.output_dir / "shards" / f"{path.stem}_B{batch}_{name}.csv", [row])
                print(f"{path.stem} {args.dtype} B{batch} {name}: {row['median_us']:.2f} us", flush=True)
                del graph, output, result
            del q, k, v, shared, exceptions, sc, ec, calls
            torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
