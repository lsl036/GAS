#!/usr/bin/env python3
"""Validate hashed LessIsMore replay and compare with the original matrix."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    methods = ("less_fi", "less_patched")
    rows = []
    for length, tag in ((16384, "16k"), (32768, "32k")):
        audits = [json.loads((args.directory / "audit_masks" / f"{tag}_s{i}.json").read_text()) for i in (0, 1)]
        if any(any(a["violations"].values()) for a in audits):
            raise ValueError(f"mask audit failed: {tag}")
        expected_hashes = [a["less_support_sha256"] for a in audits]
        for batch in (32, 64, 96, 128):
            meta = json.loads((args.directory / f"config_{length}_{batch}.json").read_text())
            original = json.loads((args.original / f"config_{length}_{batch}.json").read_text())
            if meta["less_support_sha256"] != expected_hashes:
                raise ValueError(f"Less support mismatch: {length} B{batch}")
            for key in ("trace_sha256", "search_sha256", "config", "less_k", "dtype", "less_module_sha256",
                        "warmups", "repeats", "rounds", "graph_iterations"):
                if meta[key] != original[key]:
                    raise ValueError(f"original mismatch in {key}: {length} B{batch}")
            for method in methods:
                name = f"l{length}_b{batch}_{method}.csv"
                with (args.directory / "shards" / name).open(newline="") as stream:
                    shard = list(csv.DictReader(stream))
                with (args.original / "shards" / name).open(newline="") as stream:
                    old = list(csv.DictReader(stream))
                if len(shard) != 1 or len(old) != 1 or shard[0]["method"] != method:
                    raise ValueError(f"invalid shard: {name}")
                values = []
                for r in range(meta["rounds"]):
                    with (args.directory / "samples" / f"l{length}_b{batch}_{method}_{r}.csv").open(newline="") as stream:
                        times = [float(x["latency_us"]) for x in csv.DictReader(stream)]
                    if len(times) != meta["repeats"] or any(not math.isfinite(t) or t <= 0 for t in times):
                        raise ValueError(f"invalid timings: {name}, round {r}")
                    values.extend(times)
                row = shard[0]
                if int(row["samples"]) != len(values) or not math.isclose(float(row["median_us"]), statistics.median(values), rel_tol=1e-9):
                    raise ValueError(f"invalid median: {name}")
                row["original_median_us"] = old[0]["median_us"]
                row["relative_change_percent"] = (float(row["median_us"]) / float(old[0]["median_us"]) - 1) * 100
                rows.append(row)
    with (args.directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# LessIsMore Hashed Replay", "", "| Context | Batch | Method | Original median us | Replay median us | Change |",
             "|---:|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['nominal_length']} | {row['batch']} | {row['method']} | {float(row['original_median_us']):.3f} | {float(row['median_us']):.3f} | {float(row['relative_change_percent']):+.2f}% |")
    lines.extend(["", "For each length, both per-request selected-index hashes are identical across batches and match an independently reconstructed mask. The rerun uses the same Q/K/V traces, budgets, method protocol, and measurement counts as the original matrix. Each output passed its own reference and graph replay checks.",
                  "", "The original run did not record mask hashes, so the prior mask identity cannot be established retroactively. These are separate sequential runs with repeated requests and hot buffers; observed timing differences are not independent-request confidence intervals. Do not combine the runs as a simultaneous benchmark. Quest/Twilight/ Less adapter and no-end-to-end caveats in the paired matrix still apply.", ""])
    (args.directory / "Less_Hashed_Replay_Summary.md").write_text("\n".join(lines))
    print(f"validated {len(rows)} Less replay rows and four independent support hashes")


if __name__ == "__main__":
    main()
