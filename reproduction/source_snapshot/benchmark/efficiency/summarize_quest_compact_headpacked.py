#!/usr/bin/env python3
"""Validate eight Quest compact-wrapper replay points and their provenance."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--paired-directory", type=Path, required=True)
    args = p.parse_args()
    reference = {}
    with (args.paired_directory / "summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "quest_fi_page16":
                reference[int(row["nominal_length"]), int(row["batch"])] = float(row["median_us"])
    results, scripts, modules = [], set(), set()
    for batch in (32, 64, 96, 128):
        for length in (16384, 32768):
            config = json.loads((args.directory / f"config_{length}_{batch}.json").read_text())
            paired = json.loads((args.paired_directory / f"config_{length}_{batch}.json").read_text())
            if config["trace_sha256"] != paired["trace_sha256"] or config["config"] != paired["config"]:
                raise ValueError(f"trace/config mismatch for {length}/B{batch}")
            scripts.add(config["script_sha256"])
            modules.add(config["module_sha256"])
            path = args.directory / "shards" / f"l{length}_b{batch}.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or int(rows[0]["batch"]) != batch or int(rows[0]["nominal_length"]) != length:
                raise ValueError(f"invalid shard {path}")
            row = rows[0]
            measured = []
            for r in range(config["rounds"]):
                sample = args.directory / "samples" / f"l{length}_b{batch}_{r}.csv"
                with sample.open(newline="") as stream:
                    times = [float(x["latency_us"]) for x in csv.DictReader(stream)]
                if len(times) != config["repeats"] or any(not math.isfinite(x) or x <= 0 for x in times):
                    raise ValueError(f"incomplete timings: {sample}")
                measured.extend(times)
            if int(row["samples"]) != len(measured) or not math.isclose(float(row["median_us"]), statistics.median(measured), rel_tol=1e-9):
                raise ValueError(f"timing mismatch {path}")
            row["same_trace_fi_page16_us"] = reference[length, batch]
            results.append(row)
    if len(scripts) != 1 or len(modules) != 1:
        raise ValueError("script or module version differs between results")
    with (args.directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Quest Compact Head-Packed Replay", "", "Median CUDA Graph attention time, microseconds.", "",
             "| Batch | Context | Quest patched compact | Wrapper calls | Peak allocated GiB | Same-trace FI page-16 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['batch']} | {row['nominal_length']} | {float(row['median_us']):.3f} | {row['wrapper_calls']} | {int(row['max_allocated_bytes']) / 2**30:.2f} | {float(row['same_trace_fi_page16_us']):.3f} |")
    lines.extend(["", "```python", "import numpy as np", "quest_patched_compact_us = np.array(["])
    for batch in (32, 64, 96, 128):
        vals = [float(next(r["median_us"] for r in results if int(r["batch"]) == batch and int(r["nominal_length"]) == length)) for length in (16384, 32768)]
        lines.append(f"    [{vals[0]:.3f}, {vals[1]:.3f}],  # B{batch}")
    lines.extend(["]) ", "```", "", "- Quest-derived B1 wrapper is page-corrected and head-packed; it is not an unmodified official native-batch kernel.",
                  "- A compact cache is constructed per query head from selected pages outside timing; original full-cache sparsity/physical access pattern is not preserved. Selection, packing and planning are excluded; all grouped wrapper calls and output scatters are timed.",
                  "- FI page-16 results use the same Q/K/V and candidate support but different layout/backend and were measured in separate runs with a different number of samples. This is not an isolated kernel comparison.",
                  "- Two repeated GAS-trajectory RULER requests per length; no standalone quality or end-to-end performance is established.", ""])
    report = args.directory / "Quest_Compact_Result_Summary.md"
    report.write_text("\n".join(lines))
    print(f"validated {len(results)} points: {report}")


if __name__ == "__main__":
    main()
