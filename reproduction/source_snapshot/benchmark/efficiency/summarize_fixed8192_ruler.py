#!/usr/bin/env python3
"""Validate and summarize the complete fixed-8192 RULER replay matrix."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from bench_fixed8192_ruler import METHODS


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    results, configs, page16_scripts = [], {}, set()
    for length in (16384, 32768):
        for batch in (32, 64, 96, 128):
            config_path = args.directory / f"config_{length}_{batch}.json"
            config = json.loads(config_path.read_text())
            if config["nominal_length"] != length or config["batch"] != batch:
                raise ValueError(f"configuration mismatch: {config_path}")
            configs[length, batch] = config
            page16_meta = json.loads((args.directory / f"page16_{length}_{batch}.json").read_text())
            if page16_meta["trace_sha256"] != config["trace_sha256"] or page16_meta["page_size"] != 16:
                raise ValueError(f"page-16 provenance mismatch at {length}, B{batch}")
            page16_scripts.add(page16_meta["script_sha256"])
            for method in METHODS + ("quest_fi_page16",):
                path = args.directory / "shards" / f"l{length}_b{batch}_{method}.csv"
                with path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if (len(rows) != 1 or rows[0]["method"] != method or int(rows[0]["batch"]) != batch or
                    int(rows[0]["nominal_length"]) != length or int(rows[0]["less_k"]) != config["less_k"] or
                    int(rows[0]["samples"]) != config["rounds"] * config["repeats"]):
                    raise ValueError(f"mismatched shard: {path}")
                timings, round_medians = [], []
                for r in range(config["rounds"]):
                    sample = args.directory / "samples" / f"l{length}_b{batch}_{method}_{r}.csv"
                    with sample.open(newline="") as stream:
                        measured = [float(x["latency_us"]) for x in csv.DictReader(stream)]
                        if len(measured) != config["repeats"] or not all(math.isfinite(x) and x > 0 for x in measured):
                            raise ValueError(f"incomplete sample file: {sample}")
                        timings.extend(measured)
                        round_medians.append(statistics.median(measured))
                if not math.isclose(statistics.median(timings), float(rows[0]["median_us"]), rel_tol=1e-9):
                    raise ValueError(f"median mismatch: {path}")
                row = rows[0]
                row["p10_us"] = percentile(timings, 0.1)
                row["p90_us"] = percentile(timings, 0.9)
                row["round_medians_us"] = json.dumps(round_medians)
                results.append(row)
    if len(page16_scripts) != 1:
        raise ValueError("page-16 script differs between points")
    for length in (16384, 32768):
        first = configs[length, 32]
        for batch in (64, 96, 128):
            config = configs[length, batch]
            for key in ("trace_sha256", "search_sha256", "config", "less_k", "dtype", "script_sha256",
                        "kernel_source_sha256", "less_module_sha256", "warmups", "repeats", "rounds", "graph_iterations"):
                if config[key] != first[key]:
                    raise ValueError(f"inconsistent {key} at length {length}, batch {batch}")
    first, second = configs[16384, 32], configs[32768, 32]
    for key in ("config", "dtype", "script_sha256", "kernel_source_sha256", "less_module_sha256",
                "warmups", "repeats", "rounds", "graph_iterations"):
        if first[key] != second[key]:
            raise ValueError(f"inconsistent {key} across lengths")
    output = args.directory / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    values = {(int(x["batch"]), int(x["nominal_length"]), x["method"]): float(x["median_us"]) for x in results}
    lines = ["# Fixed-8192 RULER Attention Replay", "", "Units: microseconds per attention call; median of CUDA Graph event samples.", "",
             "| Batch | Context | FA2 | FI dense | Quest FI page 1 | Quest FI page 16 | Twilight FI | GAS FI | GAS inline8 | Less FI | Less patched |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = ("fa2_dense", "fi_dense", "quest_fi", "quest_fi_page16", "twilight_fi", "gas_fi", "gas_inline8", "less_fi", "less_patched")
    for batch in (32, 64, 96, 128):
        for length in (16384, 32768):
            lines.append(f"| {batch} | {length} | " + " | ".join(f"{values[batch, length, m]:.3f}" for m in order) + " |")
    lines.extend(["", "## Plot Arrays", "", "Columns are 16K and 32K; rows are B32, B64, B96 and B128.", "", "```python", "import numpy as np"])
    for method in order:
        lines.append(f"{method} = np.array([")
        for batch in (32, 64, 96, 128):
            lines.append(f"    [{values[batch, 16384, method]:.3f}, {values[batch, 32768, method]:.3f}],  # B{batch}")
        lines.append("])")
    lines.extend(["```", "", "## Interpretation", "",
                  "- Nominal lengths refer to RULER tasks; exact captured lengths, token-count ranges, input hashes, per-request Twilight unions and K are in the per-point config JSON files.",
                  "- Fixed Quest budget 8192 uses chunk size 16; boundary chunks can result in fewer than 8192 selected tokens per head.",
                  "- FI Quest/Twilight/GAS and FI Less are exact per-head mask adapters, not official Quest/Twilight kernels. Quest page 1 and page 16 use the same token support. Less patched uses a modified B1 kernel with batch packed into heads, not an official native-batch implementation.",
                  "- Less K is matched to mean Twilight group union on the same two requests and layer; per-head work and quality are not matched.",
                  "- Two requests per length are repeated to form each batch. Captured GAS-trajectory Q/K/V and masks do not measure standalone method quality or end-to-end model latency.",
                  "- Full-batch GPU-resident KV; allocation/layout, CPU-to-GPU transfer, selection, indices and planning excluded. This is attention-only replay, not CPU-offloaded serving or TPOT.",
                  "- These selected layer-15 results are a pilot, not independent or representative model-wide evidence. Per-method masks differ, and raw timing samples are retained.", ""])
    report = args.directory / "Fixed8192_RULER_Result_Summary.md"
    report.write_text("\n".join(lines))
    print(f"wrote {len(results)} validated rows to {output}; report: {report}")


if __name__ == "__main__":
    main()
