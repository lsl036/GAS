#!/usr/bin/env python3
"""Pair B1 stream-adapted official-kernel replay with GAS trace timings."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from bench_e6_qasper_trace import write_rows


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--gas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configs = {m: json.loads((args.official_dir / f"{m}.config.json").read_text())
               for m in ("official_quest", "official_less", "gas_fi")}
    gas_config = json.loads((args.gas_dir / "config.json").read_text())
    base = configs["gas_fi"]
    for config in configs.values():
        for key in ("manifest_sha256", "files", "warmups", "repeats", "rounds", "graph_iterations", "device"):
            if config[key] != base[key]:
                raise ValueError(f"official graph configuration mismatch: {key}")
    for key in ("manifest_sha256", "warmups", "repeats", "rounds", "graph_iterations", "device"):
        if gas_config[key] != base[key]:
            raise ValueError(f"GAS graph configuration mismatch: {key}")
    if gas_config["files"] != base["files"]:
        raise ValueError("trace list mismatch")
    expected = set(base["files"])
    official = {}
    for method in configs:
        rows = read(args.official_dir / f"{method}.results.csv")
        if len(rows) != len(expected) or {r["trace"] for r in rows} != expected:
            raise ValueError(f"incomplete/duplicate {method} results")
        if any(r["status"] == "failed" for r in rows):
            raise ValueError(f"failed {method} rows")
        official[method] = {r["trace"]: r for r in rows if r["status"] == "available"}
    rows = [r for r in read(args.gas_dir / "results.csv") if r["batch"] == "1" and r["method"] == "vector16_adaptive"]
    if len(rows) != len(expected) or {r["trace"] for r in rows} != expected:
        raise ValueError("incomplete GAS B1 results")
    vector = {r["trace"]: r for r in rows}
    for method, items in official.items():
        for trace, r in items.items():
            if r["trace_sha256"] != vector[trace]["trace_sha256"]:
                raise ValueError(f"trace digest mismatch: {method}/{trace}")
            if int(r["samples"]) != base["repeats"] * base["rounds"]:
                raise ValueError(f"sample count mismatch: {method}/{trace}")
    summary, paired = [], []
    for method, items in official.items():
        times = [float(r["median_us"]) for r in items.values()]
        summary.append({"method": method, "available": len(items), "total": len(expected),
                        "median_trace_us": statistics.median(times)})
        ratios = [float(r["median_us"]) / float(vector[t]["median_us"]) for t, r in items.items()]
        paired.append({"baseline": method, "candidate": "vector16_adaptive", "pair_count": len(ratios),
                       "median_baseline_over_candidate": statistics.median(ratios),
                       "candidate_faster_count": sum(r > 1 for r in ratios)})
    summary.append({"method": "vector16_adaptive", "available": len(vector), "total": len(expected),
                    "median_trace_us": statistics.median(float(r["median_us"]) for r in vector.values())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "summary.csv", summary)
    write_rows(args.output_dir / "paired.csv", paired)
    print(f"wrote {len(summary)} summary rows and {len(paired)} paired rows")


if __name__ == "__main__":
    main()
