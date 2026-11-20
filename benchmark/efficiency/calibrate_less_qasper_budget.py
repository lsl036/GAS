#!/usr/bin/env python3
"""Exploratory fixed K matched to mean Twilight group-union on supplied traces."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bench_e6_qasper_trace import digest

SPARSE_LAYERS = [layer for layer in range(3, 32) if layer != 13]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, default=1)
    args = p.parse_args()
    manifests = [json.loads((d / "manifest.json").read_text()) for d in args.trace_dirs]
    if len({m["sample_sha256"] for m in manifests}) != len(manifests) or any(m["config"] != manifests[0]["config"] or m["model"] != manifests[0]["model"] for m in manifests):
        raise ValueError("distinct samples with identical model/config required")
    values, files, lengths = [], [], []
    for directory, manifest in zip(args.trace_dirs, manifests):
        for layer in SPARSE_LAYERS:
            name = f"step{args.step:03d}_layer{layer:02d}.pt"
            if name not in manifest["files"]:
                raise ValueError(f"missing {name}")
            path = directory / name
            trace = torch.load(path, map_location="cpu", weights_only=True)
            mask = trace["twilight"]
            if mask.ndim != 3 or mask.shape[:2] != (1, 32):
                raise ValueError(f"invalid Twilight mask in {path}")
            values.extend(mask.view(1, 8, 4, -1).any(2).sum(-1).flatten().tolist())
            lengths.append(mask.shape[-1])
            files.append({"path": str(path.resolve()), "sha256": digest(path)})
    mean = statistics.fmean(values)
    budget = max(2, round(mean))
    if budget >= min(lengths):
        raise ValueError("matched K must be less than each context length")
    result = {"step": args.step, "samples": [m["sample_index"] for m in manifests],
              "layers": SPARSE_LAYERS, "group_counts": len(values), "mean_union": mean,
              "median_union": statistics.median(values), "min_union": min(values), "max_union": max(values),
              "K": budget, "trace_manifests_sha256": [digest(d / "manifest.json") for d in args.trace_dirs],
              "trace_files": files, "selection": "fixed K, 75% middle / 25% recent, layer 2 and 13 search",
              "caveat": "post-hoc match on evaluation traces, not an independent calibration set; source states are GAS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"K={budget}, mean Twilight group union={mean:.4f}, groups={len(values)}; wrote {args.output}")


if __name__ == "__main__":
    main()
