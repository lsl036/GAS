#!/usr/bin/env python3
"""Measure real-weight LLaMA common decode components on synthetic inputs.

Attention, selection and KV updates are excluded. Each layer is measured in
isolation; the entire model and batched KV need not fit simultaneously.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from bench_e4_kernel_evidence import cuda_event_samples
from bench_e6_qasper_trace import digest, write_rows
from bench_e7_v2_estimated_tpot import CheckpointWeights, common_callable, output_callable

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--graph-iterations", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.batch_sizes + [args.repeats, args.rounds, args.graph_iterations]) < 1 or args.warmups < 0:
        parser.error("CUDA and positive measurement settings required")
    model = json.loads((args.model / "config.json").read_text())
    if (model["hidden_size"], model["num_attention_heads"], model["num_key_value_heads"], model["num_hidden_layers"]) != (4096, 32, 8, 32):
        raise ValueError("expected LLaMA-3.1-8B dimensions")
    if any(layer < 0 or layer >= 32 for layer in args.layers) or len(set(args.layers)) != len(args.layers):
        raise ValueError("invalid layer selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {name: {"size": (args.model / name).stat().st_size, "mtime_ns": (args.model / name).stat().st_mtime_ns}
              for name in sorted(set(weight_map.values()))}
    config = {"model": str(args.model.resolve()), "model_config_sha256": digest(args.model / "config.json"),
              "weight_index_sha256": digest(args.model / "model.safetensors.index.json"), "script_sha256": digest(Path(__file__)),
              "weight_shards_stat": shards,
              "common_source_sha256": digest(ROOT / "benchmark/efficiency/bench_e7_v2_estimated_tpot.py"),
              "batch_sizes": args.batch_sizes, "layers": args.layers, "dtype": "float16",
              "repeats": args.repeats, "rounds": args.rounds, "warmups": args.warmups,
              "graph_iterations": args.graph_iterations, "device": torch.cuda.get_device_name(),
              "scope": "real checkpoint weights; synthetic FP16 hidden and attention output; per-layer norm/QKV/RoPE/W_O/residual/MLP; output embedding/final norm/LM head/argmax; attention, selection, cache update excluded; individual layer residency"}
    cfgpath = args.output_dir / "config.json"
    if cfgpath.exists() and json.loads(cfgpath.read_text()) != config:
        raise ValueError("existing output uses a different protocol")
    cfgpath.write_text(json.dumps(config, indent=2) + "\n")
    weights = CheckpointWeights(args.model)
    rows = []
    for layer in args.layers + ["output"]:
        actual_weights = ([weights.get(name) for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")]
                          if layer == "output" else weights.layer(layer))
        for batch in args.batch_sizes:
            torch.manual_seed(1000 + batch)
            x = torch.randn(batch, 4096, dtype=torch.float16, device="cuda")
            data = {"x": x, "attn_input": torch.randn_like(x),
                    "cos": torch.ones(batch, 128, dtype=torch.float16, device="cuda"),
                    "sin": torch.zeros(batch, 128, dtype=torch.float16, device="cuda"),
                    "token": torch.arange(batch, device="cuda", dtype=torch.int64) % model["vocab_size"]}
            call = output_callable(data, actual_weights, model["rms_norm_eps"]) if layer == "output" else common_callable(data, actual_weights, model["rms_norm_eps"], 32, 8)
            shard = args.output_dir / "shards" / f"layer{layer}_b{batch}.csv"
            if shard.exists():
                with shard.open(newline="", encoding="utf-8") as stream:
                    saved = list(csv.DictReader(stream))
                if len(saved) != 1 or int(saved[0]["samples"]) != args.repeats * args.rounds:
                    raise ValueError(f"invalid cached shard {shard}")
                for r in range(args.rounds):
                    p = args.output_dir / "samples" / f"layer{layer}_b{batch}_{r}.csv"
                    with p.open(newline="", encoding="utf-8") as stream:
                        if sum(1 for _ in csv.DictReader(stream)) != args.repeats:
                            raise ValueError(f"invalid samples {p}")
                rows.append(saved[0])
                print(f"SKIP layer={layer} B{batch}", flush=True)
                continue
            result = call()
            if not all(bool(torch.isfinite(t).all()) for t in result):
                raise ValueError(f"non-finite result layer={layer} B{batch}")
            for _ in range(args.warmups): call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_iterations): result = call()
            # An output must be overwritten on graph replay, not merely retained.
            result[-1].fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
            if not all(bool(torch.isfinite(t).all()) for t in result):
                raise ValueError(f"graph output invalid layer={layer} B{batch}")
            samples = []
            for r in range(args.rounds):
                times = [x / args.graph_iterations for x in cuda_event_samples(graph.replay, x.device, args.warmups, args.repeats)]
                samples.extend(times)
                write_rows(args.output_dir / "samples" / f"layer{layer}_b{batch}_{r}.csv", [{"latency_us": t} for t in times])
            row = {"layer": layer, "batch": batch, "median_us": statistics.median(samples), "samples": len(samples)}
            rows.append(row)
            write_rows(shard, [row])
            print(f"layer={layer} B{batch}: {row['median_us']:.2f} us", flush=True)
            del graph, result, x, data, call
        del actual_weights
        torch.cuda.empty_cache()
    write_rows(args.output_dir / "results.csv", rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
