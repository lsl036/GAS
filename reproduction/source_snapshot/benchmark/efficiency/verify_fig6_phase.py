#!/usr/bin/env python3
"""Verify completed dense-FI or Less replay shards before resuming Fig.6."""
import argparse
import csv
import json
from pathlib import Path

from summarize_fig6 import BATCHES, LAYERS, LESS_LAYERS, read_rows
from verify_fig6_main import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trace-dirs", type=Path, nargs=2, required=True)
    p.add_argument("--method", choices=("fi_dense", "less_fi_adapter", "less_patched_headpacked"), required=True)
    args = p.parse_args()
    base, method = args.output_dir, args.method
    config = json.loads((base / "config.json").read_text())
    layers = LAYERS if method == "fi_dense" else LESS_LAYERS
    if (config["manifest_sha256"] != [digest(d / "manifest.json") for d in args.trace_dirs] or
            config["steps"] != [1] or tuple(config["layers"]) != layers or
            config["batch_sizes"] != list(BATCHES) or config["methods"] != [method] or
            config["dtype"] != "float16" or config["repeats"] != 10 or
            config["rounds"] != 2 or config["graph_iterations"] != 8):
        raise ValueError(f"incompatible config: {base}")
    hashes = {layer: [digest(d / f"step001_layer{layer:02d}.pt") for d in args.trace_dirs] for layer in LAYERS}
    rows = read_rows(base / "results.csv", method, layers)
    for (layer, batch), row in rows.items():
        expected_hash = hashes[layer]
        if method != "fi_dense":
            expected_hash = expected_hash + hashes[2 if layer < 13 else 13]
        if json.loads(row["trace_sha256"]) != expected_hash:
            raise ValueError(f"trace mismatch: {method} layer {layer} batch {batch}")
        shard = base / "shards" / f"s1_l{layer}_b{batch}_{method}.csv"
        with shard.open(newline="") as stream:
            if list(csv.DictReader(stream)) != [row]:
                raise ValueError(f"shard mismatch: {shard}")
        for round_id in range(2):
            path = base / "samples" / f"s1_l{layer}_b{batch}_{method}_{round_id}.csv"
            with path.open(newline="") as stream:
                if sum(1 for _ in csv.DictReader(stream)) != 10:
                    raise ValueError(f"missing raw timings: {path}")
    print(f"verified complete {method} phase: {base}")


if __name__ == "__main__":
    main()
