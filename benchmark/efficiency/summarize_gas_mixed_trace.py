#!/usr/bin/env python3
"""Check completeness and summarize mixed-request kernel replay."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.input_dir / "config.json").read_text())
    with (args.input_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(str(s), str(l), str(b), m) for s in config["steps"] for l in config["layers"]
                for b in config["batch_sizes"] for m in config["methods"]}
    actual = {(r["step"], r["layer"], r["batch"], r["method"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("incomplete or duplicate measurements")
    if any(int(r["samples"]) != config["rounds"] * config["repeats"] for r in rows):
        raise ValueError("incorrect sample count")
    values = {(r["step"], r["layer"], r["batch"], r["method"]): float(r["median_us"]) for r in rows}
    summary, pairs = [], []
    for batch in map(str, config["batch_sizes"]):
        for method in config["methods"]:
            values_m = [values[(str(s), str(l), batch, method)] for s in config["steps"] for l in config["layers"]]
            summary.append({"batch": batch, "method": method, "trace_count": len(values_m),
                            "median_trace_us": statistics.median(values_m), "min_trace_us": min(values_m),
                            "max_trace_us": max(values_m)})
        for candidate in ("inline8", "twopass_inline8", "prefix_groupsplit128"):
            for baseline in ("fa2_dense", "fi_dense", "fi_quest_adapter", "fi_twilight_adapter", "fi_gas_adapter"):
                ratios = [values[(str(s), str(l), batch, baseline)] / values[(str(s), str(l), batch, candidate)]
                          for s in config["steps"] for l in config["layers"]]
                pairs.append({"batch": batch, "baseline": baseline, "candidate": candidate,
                              "trace_count": len(ratios), "candidate_faster_count": sum(r > 1 for r in ratios),
                              "median_paired_ratio": statistics.median(ratios)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "summary.csv", summary)
    write(args.output_dir / "paired.csv", pairs)
    lookup = {(r["batch"], r["method"]): r["median_trace_us"] for r in summary}
    lines = ["# Mixed Qasper attention replay", "", f"Samples: {config['sample_indices']}; "
             "the four distinct requests are repeated round-robin within each batch; not independent serving batches. "
             "Latencies are microseconds, median across layer/step trace medians. Selection, compaction, layout conversion "
             "and FI plan construction are excluded. CUDA Graph replay and hot buffers.", "",
             "Quest and Twilight use exact-mask FlashInfer adapters, not their authors' kernels. "
             "The LessIsMore official batched backend is not measured; do not substitute an adapter for it. "
             "GAS, Quest and Twilight supports differ. This is not end-to-end TPOT.", "",
             "| Batch | " + " | ".join(config["methods"]) + " |", "|---:|" + "---:|" * len(config["methods"])]
    for batch in map(str, config["batch_sizes"]):
        lines.append("| " + batch + " | " + " | ".join(f"{lookup[(batch, m)]:.2f}" for m in config["methods"]) + " |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(summary)} summary and {len(pairs)} paired rows")


if __name__ == "__main__":
    main()
