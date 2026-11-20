#!/usr/bin/env python3
"""Validate a completed Fig.6 main phase before skipping it on resume."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from summarize_fig6 import BATCHES, LAYERS, read_rows

METHODS = ("fa2_dense", "fi_quest_adapter", "fi_twilight_adapter", "fi_gas_adapter", "inline8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trace-dirs", type=Path, nargs=2, required=True)
    args = p.parse_args()
    base = args.output_dir
    config = json.loads((base / "config.json").read_text())
    manifests = [d.resolve() / "manifest.json" for d in args.trace_dirs]
    script = Path(__file__).with_name("bench_gas_mixed_trace.py")
    if (config["manifest_sha256"] != [digest(p) for p in manifests] or
            config["script_sha256"] != digest(script) or config["steps"] != [1] or
            config["layers"] != list(LAYERS) or config["batch_sizes"] != list(BATCHES) or
            config["methods"] != list(METHODS) or config["dtype"] != "float16" or
            config["repeats"] != 10 or config["rounds"] != 2 or config["graph_iterations"] != 8):
        raise ValueError("main phase config, source, or trace manifests do not match")
    trace_hashes = {layer: [digest(directory / f"step001_layer{layer:02d}.pt") for directory in args.trace_dirs]
                    for layer in LAYERS}
    for method in METHODS:
        rows = read_rows(base / "results.csv", method, LAYERS)
        for (layer, batch), row in rows.items():
            if json.loads(row["trace_sha256"]) != trace_hashes[layer]:
                raise ValueError(f"trace mismatch: {method} layer {layer} batch {batch}")
            shard = base / "shards" / f"s1_l{layer}_b{batch}_{method}.csv"
            with shard.open(newline="") as stream:
                saved = list(csv.DictReader(stream))
            if len(saved) != 1 or saved[0] != row:
                raise ValueError(f"shard differs from aggregated result: {shard}")
            for round_id in range(2):
                path = base / "samples" / f"s1_l{layer}_b{batch}_{method}_{round_id}.csv"
                with path.open(newline="") as stream:
                    if sum(1 for _ in csv.DictReader(stream)) != 10:
                        raise ValueError(f"missing raw timings: {path}")
    print(f"verified complete main phase: {base}")


if __name__ == "__main__":
    main()
